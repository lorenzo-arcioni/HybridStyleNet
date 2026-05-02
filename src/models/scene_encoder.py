"""
scene_encoder.py
----------------
Scene Encoder — Componente 1 di RAG-ColorNet.

Responsabilità:
  1. Estrarre patch features semantiche tramite DINOv2-Small (frozen)
  2. Calcolare il color histogram soft in spazio CIE Lab (192-dim)
  3. Costruire i patch descriptor per il retrieval (sem + chroma, 416-dim)

Output principali:
  F_sem   : (B, N, 384)   patch tokens DINOv2
  h       : (B, 192)      color histogram Lab
  Q       : (B, N, 416)   query descriptor per retrieval

DINOv2 è SEMPRE frozen — i pesi non vengono mai aggiornati.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.color_utils import rgb_to_lab          # type: ignore[import]


# ---------------------------------------------------------------------------
# Costanti Lab per gli istogrammi
# ---------------------------------------------------------------------------
_LAB_RANGES = {
    "L": (0.0,   100.0),
    "a": (-128.0, 127.0),
    "b": (-128.0, 127.0),
}


# ---------------------------------------------------------------------------
# ColorHistogram
# ---------------------------------------------------------------------------

class ColorHistogram(nn.Module):
    """
    Soft color histogram differenziabile in spazio CIE Lab.

    Per ogni canale c ∈ {L*, a*, b*} calcola un istogramma soft con
    B_hist bin tramite kernel gaussiano:

        h_c(k) = softmax_k( -‖pixel_c - μ_k‖² / (2σ²) )

    Output: h ∈ ℝ^{B × 192}  (3 canali × 64 bin)

    Parameters
    ----------
    n_bins      : numero di bin per canale (default 64)
    sigma_scale : σ_bin = Δμ * sigma_scale
    """

    def __init__(self, n_bins: int = 64, sigma_scale: float = 0.5) -> None:
        super().__init__()
        self.n_bins      = n_bins
        self.sigma_scale = sigma_scale

        # Pre-calcola i centri dei bin per ogni canale (non trainable)
        for name, (vmin, vmax) in _LAB_RANGES.items():
            delta = (vmax - vmin) / n_bins
            centers = torch.linspace(
                vmin + delta / 2, vmax - delta / 2, n_bins
            )
            sigma = delta * sigma_scale
            self.register_buffer(f"centers_{name}", centers)
            self.register_buffer(f"sigma_{name}",  torch.tensor(sigma))

    # ------------------------------------------------------------------
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : (B, 3, H, W) sRGB float32 [0, 1]

        Returns
        -------
        h : (B, 192) — istogramma soft normalizzato
        """
        B = img.shape[0]
        img_lab = rgb_to_lab(img)                 # (B, 3, H, W)

        parts = []
        for c_idx, name in enumerate(["L", "a", "b"]):
            channel = img_lab[:, c_idx, :, :]     # (B, H, W)
            pixels  = channel.reshape(B, -1)       # (B, H*W)

            centers = getattr(self, f"centers_{name}")  # (n_bins,)
            sigma   = getattr(self, f"sigma_{name}")    # scalar

            # distanza²: (B, H*W, 1) vs (1, 1, n_bins)
            diff    = pixels.unsqueeze(-1) - centers.view(1, 1, -1)
            weights = torch.exp(-diff.pow(2) / (2 * sigma.pow(2)))  # (B, H*W, n_bins)

            # normalizza per pixel poi somma sui pixel
            hist_unnorm = weights.sum(dim=1)          # (B, n_bins) — somma sui pixel
            hist        = hist_unnorm / (hist_unnorm.sum(dim=-1, keepdim=True) + 1e-8)
            parts.append(hist)

        return torch.cat(parts, dim=1)             # (B, 192)


# ---------------------------------------------------------------------------
# ChromaticPatchFeatures
# ---------------------------------------------------------------------------

class ChromaticPatchFeatures(nn.Module):
    """
    Estrae statistiche cromatiche per ogni patch DINOv2 e le proietta
    a chroma_dim dimensioni tramite un layer lineare trainable.

    Per ogni patch 14×14:
      - media Lab (3 valori)
      - std  Lab (3 valori)
      - istogramma locale a 13 bin su ciascuno dei 3 canali (39 valori)
    Totale raw: 3 + 3 + 39 = 45 → proiettato a chroma_dim (default 32)

    Parameters
    ----------
    patch_size  : dimensione della patch DINOv2 (default 14)
    chroma_dim  : dimensione output dopo proiezione lineare
    """

    RAW_DIM = 45   # 3 mean + 3 std + 3*13 hist-locale

    def __init__(self, patch_size: int = 14, chroma_dim: int = 32) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.chroma_dim = chroma_dim
        self.proj       = nn.Linear(self.RAW_DIM, chroma_dim, bias=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        img:    torch.Tensor,   # (B, 3, H, W) sRGB [0,1]
        n_h:    int,            # numero di patch verticali
        n_w:    int,            # numero di patch orizzontali
    ) -> torch.Tensor:
        """
        Returns
        -------
        c_patch : (B, N, chroma_dim)  N = n_h * n_w
        """
        B = img.shape[0]
        img_lab = rgb_to_lab(img)                 # (B, 3, H, W)
        P = self.patch_size

        # Ritaglia l'immagine alle dimensioni divisibili per P
        H_crop = n_h * P
        W_crop = n_w * P
        img_lab = img_lab[:, :, :H_crop, :W_crop]

        # Reshape in patch: (B, 3, n_h, P, n_w, P)
        lab_patches = img_lab.reshape(B, 3, n_h, P, n_w, P)
        # → (B, n_h, n_w, 3, P, P)
        lab_patches = lab_patches.permute(0, 2, 4, 1, 3, 5)
        # → (B, N, 3, P*P)
        N = n_h * n_w
        lab_flat = lab_patches.reshape(B, N, 3, P * P)

        # Statistiche per patch
        mean = lab_flat.mean(-1)                  # (B, N, 3)
        std  = lab_flat.std(-1)                   # (B, N, 3)

        # Istogramma locale a 13 bin (differenziabile, approssimato)
        hist_parts = []
        for c in range(3):
            vmin, vmax = list(_LAB_RANGES.values())[c]
            bins   = torch.linspace(vmin, vmax, 13, device=img.device)
            delta  = bins[1] - bins[0]
            pixels = lab_flat[:, :, c, :]         # (B, N, P*P)
            diff   = pixels.unsqueeze(-1) - bins.view(1, 1, 1, -1)
            w      = torch.exp(-diff.pow(2) / (2 * (delta * 0.5).pow(2)))
            w      = w / (w.sum(-1, keepdim=True) + 1e-8)
            h      = w.mean(-2)                   # (B, N, 13)
            hist_parts.append(h)

        hist = torch.cat(hist_parts, dim=-1)      # (B, N, 39)

        raw = torch.cat([mean, std, hist], dim=-1)  # (B, N, 45)
        return self.proj(raw)                     # (B, N, chroma_dim)


# ---------------------------------------------------------------------------
# SceneEncoder
# ---------------------------------------------------------------------------

class SceneEncoder(nn.Module):
    """
    Scene Encoder completo.

    Combina DINOv2-Small (frozen), ColorHistogram e ChromaticPatchFeatures
    per produrre tutte le rappresentazioni necessarie al resto del modello.

    Parameters
    ----------
    embed_dim       : dimensione embedding DINOv2 (384 per vits14)
    patch_size      : patch size DINOv2 (14)
    n_bins          : bin per canale del color histogram
    sigma_scale     : sigma relativa per il kernel gaussiano
    chroma_dim      : dimensione output ChromaticPatchFeatures
    imagenet_mean   : media per normalizzazione ImageNet
    imagenet_std    : std  per normalizzazione ImageNet
    """

    def __init__(
        self,
        embed_dim:     int   = 384,
        patch_size:    int   = 14,
        n_bins:        int   = 64,
        sigma_scale:   float = 0.5,
        chroma_dim:    int   = 32,
        imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        imagenet_std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()

        self.embed_dim  = embed_dim
        self.patch_size = patch_size
        self.chroma_dim = chroma_dim
        self.desc_dim   = embed_dim + chroma_dim  # 416

        # --- DINOv2 backbone (frozen) ---
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        )
        self._freeze_backbone()

        # --- Sottomoduli trainable ---
        self.histogram = ColorHistogram(n_bins=n_bins, sigma_scale=sigma_scale)
        self.chroma_feat = ChromaticPatchFeatures(
            patch_size=patch_size, chroma_dim=chroma_dim
        )
        self.layer_norm = nn.LayerNorm(self.desc_dim)

        # Registra mean/std come buffer (non trainable, seguono il device)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(imagenet_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(imagenet_std).view(1, 3, 1, 1)
        )

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Congela tutti i parametri di DINOv2 — non viene mai aggiornato."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    # ------------------------------------------------------------------
    def _normalise_for_dino(self, img: torch.Tensor) -> torch.Tensor:
        """Normalizza con statistiche ImageNet per l'input a DINOv2."""
        return (img - self.imagenet_mean) / self.imagenet_std

    # ------------------------------------------------------------------
    def extract_patch_features(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Estrae i patch tokens da DINOv2.

        Returns
        -------
        F_sem : (B, N, 384)
        n_h   : numero di patch verticali
        n_w   : numero di patch orizzontali
        """
        B, C, H, W = img.shape
        # Assicura che H e W siano multipli del patch_size
        P    = self.patch_size
        new_H = (H // P) * P
        new_W = (W // P) * P

        n_h = new_H // P
        n_w = new_W // P

        with torch.no_grad():
            if new_H != H or new_W != W:
                img = F.interpolate(
                    img, size=(new_H, new_W),
                    mode="bilinear", align_corners=False, antialias=True,
                )
            img_norm = self._normalise_for_dino(img)
            out = self.backbone.forward_features(img_norm)

        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            F_sem = out["x_norm_patchtokens"]
        else:
            tokens = out if isinstance(out, torch.Tensor) else out["x"]
            F_sem  = tokens[:, 1:, :]

        return F_sem, n_h, n_w

    # ------------------------------------------------------------------
    def forward(
        self,
        img: torch.Tensor,
    ) -> dict:
        """
        Forward pass completo dello Scene Encoder.

        Parameters
        ----------
        img : (B, 3, H, W) sRGB float32 [0, 1]

        Returns
        -------
        dict con:
          "F_sem"   : (B, N, 384)   patch features DINOv2
          "h"       : (B, 192)      color histogram Lab
          "Q"       : (B, N, 416)   query descriptor (sem + chroma, LayerNorm)
          "n_h"     : int           patch grid height
          "n_w"     : int           patch grid width
        """
        # 1. Patch features semantiche (DINOv2 frozen, no_grad interno)
        F_sem, n_h, n_w = self.extract_patch_features(img)   # (B, N, 384)

        # 2. Color histogram
        h = self.histogram(img)                               # (B, 192)

        # 3. Chromatic patch features (trainable)
        c_patch = self.chroma_feat(img, n_h, n_w)            # (B, N, chroma_dim)

        # 4. Descriptor: concatenazione + LayerNorm
        Q = self.layer_norm(
            torch.cat([F_sem, c_patch], dim=-1)
        )                                                     # (B, N, 416)

        return {
            "F_sem": F_sem,
            "h":     h,
            "Q":     Q,
            "n_h":   n_h,
            "n_w":   n_w,
        }

    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "SceneEncoder":
        """Override: DINOv2 rimane sempre in eval mode."""
        super().train(mode)
        self.backbone.eval()   # backbone sempre eval
        return self

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "SceneEncoder":
        enc_cfg  = cfg["encoder"]
        hist_cfg = cfg["histogram"]
        desc_cfg = cfg["descriptor"]
        return cls(
            embed_dim     = enc_cfg["embed_dim"],
            patch_size    = enc_cfg["patch_size"],
            n_bins        = hist_cfg["n_bins"],
            sigma_scale   = hist_cfg["sigma_scale"],
            chroma_dim    = desc_cfg["chroma_dim"],
            imagenet_mean = tuple(enc_cfg["imagenet_mean"]),
            imagenet_std  = tuple(enc_cfg["imagenet_std"]),
        )
