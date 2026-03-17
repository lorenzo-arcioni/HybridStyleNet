"""
bilateral_grid.py
-----------------
Bilateral Grid Renderer — Componente 4 di RAG-ColorNet.

Converte il retrieved edit (spazio feature DINOv2) in trasformazioni
affini pixel-wise applicate tramite bilateral grid slicing edge-aware.

Struttura:
  GridNet          : decoder leggero → coefficienti per due bilateral grid
                     (globale 8×8×8, locale 16×16×8)
  SemanticGuide    : MLP che produce una guida ibrida chroma+semantica
  bilateral_slice  : interpolazione trilineare differenziabile
  BilateralGridRenderer : modulo che orchestra tutto

Inizializzazione all'identità: entrambi i branch producono la trasformazione
identità all'inizio del training → nessuna modifica dell'immagine a t=0.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.color_utils import rgb_to_lab          # type: ignore[import]


# ---------------------------------------------------------------------------
# SemanticGuide
# ---------------------------------------------------------------------------

class SemanticGuide(nn.Module):
    """
    Produce la guida ibrida g(i,j) ∈ [0,1] per il bilateral slicing.

    g = α · g_chroma + (1-α) · g_sem

    g_chroma : 0.5·L* + 0.25·|a*| + 0.25·|b*| normalizzato
    g_sem    : σ(MLP(f_sem_patch))  — valore scalare per patch

    Parameters
    ----------
    dino_dim    : dimensione embedding DINOv2 patch (384)
    hidden_dim  : dimensione hidden del MLP (64)
    alpha       : peso di g_chroma vs g_sem (0.5)
    patch_size  : patch size DINOv2 (14)
    """

    def __init__(
        self,
        dino_dim:   int   = 384,
        hidden_dim: int   = 64,
        alpha:      float = 0.5,
        patch_size: int   = 14,
    ) -> None:
        super().__init__()
        self.alpha      = alpha
        self.patch_size = patch_size

        self.mlp = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def _chroma_guide(self, img: torch.Tensor) -> torch.Tensor:
        """
        Guida cromatica basata su Lab.
        g_chroma(i,j) = (0.5·L* + 0.25·|a*| + 0.25·|b*|) / 114.0

        Returns
        -------
        g_chroma : (B, H, W) ∈ [0, 1]
        """
        lab = rgb_to_lab(img)                     # (B, 3, H, W)
        L   = lab[:, 0, :, :]
        a   = lab[:, 1, :, :].abs()
        b   = lab[:, 2, :, :].abs()
        g   = (0.5 * L + 0.25 * a + 0.25 * b) / 114.0
        return g.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        img:   torch.Tensor,   # (B, 3, H, W) sRGB [0,1]
        F_sem: torch.Tensor,   # (B, N, 384)  patch tokens DINOv2
        n_h:   int,
        n_w:   int,
    ) -> torch.Tensor:
        """
        Returns
        -------
        g : (B, H, W)  guida ibrida ∈ [0, 1]
        """
        B, C, H, W = img.shape

        # Guida cromatica
        g_chroma = self._chroma_guide(img)        # (B, H, W)

        # Guida semantica: MLP per patch → upsample a pixel
        g_sem_patch = self.mlp(F_sem)             # (B, N, 1)
        g_sem_patch = g_sem_patch.reshape(B, n_h, n_w, 1).permute(0, 3, 1, 2)
        # Upsample a piena risoluzione
        g_sem = F.interpolate(
            g_sem_patch, size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                              # (B, H, W)

        return self.alpha * g_chroma + (1.0 - self.alpha) * g_sem


# ---------------------------------------------------------------------------
# bilateral_slice  (differenziabile)
# ---------------------------------------------------------------------------

def bilateral_slice(
    grid:  torch.Tensor,    # (B, n_coeff, s_spatial, s_spatial, s_lum)
    img:   torch.Tensor,    # (B, 3, H, W)
    guide: torch.Tensor,    # (B, H, W) ∈ [0, 1]
) -> torch.Tensor:
    B, n_coeff, s_sp, _, s_lum = grid.shape
    _, _, H, W = img.shape

    # Coordinate normalizzate in [-1, 1]
    y_coords = torch.linspace(-1, 1, H, device=img.device)
    x_coords = torch.linspace(-1, 1, W, device=img.device)
    yy, xx   = torch.meshgrid(y_coords, x_coords, indexing="ij")

    zz = guide * 2.0 - 1.0                       # (B, H, W)

    xx_exp = xx.unsqueeze(0).expand(B, -1, -1)    # (B, H, W)
    yy_exp = yy.unsqueeze(0).expand(B, -1, -1)    # (B, H, W)

    # sample_coords: (B, H, W, 3)
    sample_coords = torch.stack([xx_exp, yy_exp, zz], dim=-1)
    # Per grid_sample 5D: (B, 1, H, W, 3)
    sample_coords_5d = sample_coords.unsqueeze(1)

    # grid_5d: (B, n_coeff, s_lum, s_sp, s_sp)
    grid_5d = grid.permute(0, 1, 4, 2, 3)

    # Interpola ogni canale separatamente per evitare il problema del batch
    # grid_sample 5D: input (B, C, D, H, W), grid (B, 1, H, W, 3)
    coeffs = F.grid_sample(
        grid_5d,
        sample_coords_5d.expand(-1, 1, -1, -1, -1),
        mode="bilinear", align_corners=True, padding_mode="border",
    )
    # coeffs: (B, n_coeff, 1, H, W) → (B, n_coeff, H, W)
    coeffs = coeffs.squeeze(2)

    # Applica trasformazione affine 3x4
    A = coeffs[:, :9, :, :].reshape(B, 3, 3, H, W)
    b = coeffs[:, 9:, :, :]                        # (B, 3, H, W)

    out = torch.einsum("bckhw,bkhw->bchw", A, img) + b
    return out.clamp(0.0, 1.0) # range ampio ma finito

# ---------------------------------------------------------------------------
# GridNet
# ---------------------------------------------------------------------------

class GridNet(nn.Module):
    """
    Decoder leggero che produce i coefficienti delle due bilateral grid
    a partire dal retrieved edit e dalle feature DINOv2.

    Branch globale  → G_global (B, 12, 8,  8,  8)
    Branch locale   → G_local  (B, 12, 16, 16, 8)

    Entrambi i branch sono inizializzati per produrre la trasformazione
    identità: A = I_3, b = 0.

    Parameters
    ----------
    retrieval_dim : dimensione del retrieved edit (d_r = 256)
    dino_dim      : dimensione embedding DINOv2 (384)
    dino_proj_dim : proiezione DINOv2 nell'encoder (128)
    fusion_dim    : dimensione after fusion (256)
    global_s      : spatial resolution della grid globale (8)
    global_l      : luminance bins grid globale (8)
    local_s       : spatial resolution della grid locale (16)
    local_l       : luminance bins grid locale (8)
    n_affine      : coefficienti affini per bin (12 = 3×3 + 3)
    """

    def __init__(
        self,
        retrieval_dim: int = 256,
        dino_dim:      int = 384,
        dino_proj_dim: int = 128,
        fusion_dim:    int = 256,
        global_s:      int = 8,
        global_l:      int = 8,
        local_s:       int = 16,
        local_l:       int = 8,
        n_affine:      int = 12,
    ) -> None:
        super().__init__()
        self.global_s = global_s
        self.global_l = global_l
        self.local_s  = local_s
        self.local_l  = local_l
        self.n_affine = n_affine

        # Proiezione DINOv2 384 → 128
        self.dino_proj = nn.Conv2d(dino_dim, dino_proj_dim, kernel_size=1)

        # Fusione: retrieval (d_r=256) + dino_proj (128) → fusion_dim (256)
        self.fusion_conv = nn.Conv2d(
            retrieval_dim + dino_proj_dim, fusion_dim, kernel_size=1
        )

        # Branch globale: GAP → MLP → reshape
        global_out = n_affine * global_s * global_s * global_l
        self.global_branch = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, global_out),
        )

        # Branch locale: AdaptiveAvgPool(16×16) → Conv1x1 → reshape
        local_out = n_affine * local_l
        self.local_pool = nn.AdaptiveAvgPool2d((local_s, local_s))
        self.local_conv = nn.Conv2d(fusion_dim, local_out, kernel_size=1)

        self._init_identity()

    # ------------------------------------------------------------------
    def _init_identity(self) -> None:
        """
        Inizializza l'ultimo layer di ogni branch per produrre
        la trasformazione identità: A = I_3, b = 0.

        identity_coeffs = [1,0,0, 0,1,0, 0,0,1, 0,0,0] × (s×s×l)
        """
        identity = torch.tensor(
            [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
            dtype=torch.float32,
        )

        # Global branch — ultimo Linear
        last_global = self.global_branch[-1]
        nn.init.zeros_(last_global.weight)
        total_global = self.global_s * self.global_s * self.global_l
        last_global.bias.data = identity.repeat(total_global)

        # Local branch — local_conv
        nn.init.zeros_(self.local_conv.weight)
        total_local = self.local_l
        self.local_conv.bias.data = identity.repeat(total_local)

    # ------------------------------------------------------------------
    def forward(
        self,
        R_spatial: torch.Tensor,   # (B, d_r,      n_h, n_w)
        F_sem:     torch.Tensor,   # (B, dino_dim, n_h, n_w) — già in forma spaziale
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        G_global : (B, n_affine, global_s, global_s, global_l)
        G_local  : (B, n_affine, local_s,  local_s,  local_l)
        """
        B = R_spatial.shape[0]

        # F_sem in forma spaziale: (B, dino_dim, n_h, n_w)
        F_sem_proj = self.dino_proj(F_sem)        # (B, 128, n_h, n_w)

        # Fusione retrieved edit + DINOv2
        F_fused = self.fusion_conv(
            torch.cat([R_spatial, F_sem_proj], dim=1)
        )                                         # (B, 256, n_h, n_w)

        # --- Branch globale ---
        f_gap = F_fused.mean(dim=[-2, -1])        # (B, 256) global avg pool
        G_global_flat = self.global_branch(f_gap) # (B, 12 * 8 * 8 * 8)
        G_global = G_global_flat.reshape(
            B, self.n_affine, self.global_s, self.global_s, self.global_l
        )

        # --- Branch locale ---
        F_local_pooled = self.local_pool(F_fused) # (B, 256, 16, 16)
        G_local_flat   = self.local_conv(F_local_pooled)  # (B, 12*8, 16, 16)
        # reshape: (B, 12*8, 16, 16) → (B, 12, 16, 16, 8)
        G_local = G_local_flat.reshape(
            B, self.n_affine, self.local_l, self.local_s, self.local_s
        ).permute(0, 1, 3, 4, 2).contiguous()

        return G_global, G_local

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "GridNet":
        bcfg = cfg["bilateral_grid"]
        rcfg = cfg["retrieval"]
        ecfg = cfg["encoder"]
        return cls(
            retrieval_dim = rcfg["d_r"],
            dino_dim      = ecfg["embed_dim"],
            dino_proj_dim = bcfg["dino_proj_dim"],
            fusion_dim    = bcfg["fusion_dim"],
            global_s      = bcfg["global_s"],
            global_l      = bcfg["global_l"],
            local_s       = bcfg["local_s"],
            local_l       = bcfg["local_l"],
            n_affine      = bcfg["n_affine_coeffs"],
        )


# ---------------------------------------------------------------------------
# BilateralGridRenderer  (modulo principale del componente 4)
# ---------------------------------------------------------------------------

class BilateralGridRenderer(nn.Module):
    """
    Orchestrazione completa del rendering via bilateral grid.

    Espone:
      - GridNet per predire i coefficienti
      - SemanticGuide per la guida ibrida
      - bilateral_slice per applicare le trasformazioni

    Returns
    -------
    I_global : (B, 3, H, W)  immagine trasformata dalla grid globale
    I_local  : (B, 3, H, W)  immagine trasformata dalla grid locale
    G_global : (B, 12, 8,  8,  8)  coefficienti grid globale (per TV loss)
    G_local  : (B, 12, 16, 16, 8)  coefficienti grid locale  (per TV loss)
    g        : (B, H, W)     guida di slicing (per debug)
    """

    def __init__(
        self,
        grid_net:   GridNet,
        guide:      SemanticGuide,
    ) -> None:
        super().__init__()
        self.grid_net = grid_net
        self.guide    = guide

    # ------------------------------------------------------------------
    def forward(
        self,
        R_spatial: torch.Tensor,   # (B, d_r, n_h, n_w)
        F_sem:     torch.Tensor,   # (B, N,  dino_dim) — patch tokens
        img:       torch.Tensor,   # (B, 3, H, W)
        n_h:       int,
        n_w:       int,
    ) -> dict:
        B, _, H, W = img.shape

        # F_sem (B, N, D) → forma spaziale (B, D, n_h, n_w) per GridNet
        F_sem_spatial = F_sem.permute(0, 2, 1).reshape(
            B, F_sem.shape[-1], n_h, n_w
        )

        # Predici i coefficienti delle bilateral grid
        G_global, G_local = self.grid_net(R_spatial, F_sem_spatial)

        # Guida ibrida
        g = self.guide(img, F_sem, n_h, n_w)      # (B, H, W)

        # Bilateral slicing
        I_global = bilateral_slice(G_global, img, g)
        I_local  = bilateral_slice(G_local,  img, g)

        return {
            "I_global": I_global,
            "I_local":  I_local,
            "G_global": G_global,
            "G_local":  G_local,
            "guide":    g,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "BilateralGridRenderer":
        grid_net = GridNet.from_config(cfg)
        bcfg     = cfg["bilateral_grid"]
        ecfg     = cfg["encoder"]
        guide    = SemanticGuide(
            dino_dim   = ecfg["embed_dim"],
            hidden_dim = bcfg["guide_hidden"],
            alpha      = bcfg["guide_alpha"],
            patch_size = ecfg["patch_size"],
        )
        return cls(grid_net=grid_net, guide=guide)
