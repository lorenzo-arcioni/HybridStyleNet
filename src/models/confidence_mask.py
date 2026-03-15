"""
confidence_mask.py
------------------
Confidence Mask — Componente 5 di RAG-ColorNet.

MaskNet produce una maschera α ∈ [0,1]^{B×1×H×W} che pesa pixel
per pixel il contributo del ramo locale vs globale:

    I_pred = α · I_local + (1-α) · I_global

α alto  → il retrieval locale è affidabile → usa I_local
α basso → incertezza alta → ricade su I_global (trasformazione media)

Input del MaskNet:
  - F_sem upsampliato:  (B, 128, H/4, W/4)   — cosa c'è in ogni zona
  - Divergenza D:       (B, 3,   H/4, W/4)   — |I_local - I_global|
    (alta divergenza = i due branch disaccordano = bassa confidenza)

La maschera è calcolata a risoluzione ridotta (H/4 × W/4) e poi
upsampliata a piena risoluzione con bilinear interpolation.

L'entropy loss (- Σ α·log(α) + (1-α)·log(1-α)) la forza ad essere
binaria nel tempo, evitando valori medi indefiniti.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.raw_pipeline import gamma_encode        # type: ignore[import]


# ---------------------------------------------------------------------------
# MaskNet
# ---------------------------------------------------------------------------

class MaskNet(nn.Module):
    """
    Rete leggera per la confidence mask.

    Parameters
    ----------
    dino_dim         : dimensione embedding DINOv2 (384)
    upsample_dim     : proiezione DINOv2 per l'input della mask (128)
    hidden_channels  : canali hidden del conv layer (64)
    upsample_factor  : fattore di downscale rispetto alla risoluzione piena (4)
                       la mask è calcolata a H/4 × W/4
    """

    def __init__(
        self,
        dino_dim:        int = 384,
        upsample_dim:    int = 128,
        hidden_channels: int = 64,
        upsample_factor: int = 4,
    ) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor

        # Proiezione DINOv2 → upsample_dim
        self.dino_proj = nn.Conv2d(dino_dim, upsample_dim, kernel_size=1)

        # Stima della maschera:
        # input: upsample_dim (F_sem) + 3 (divergenza D)
        self.conv1 = nn.Conv2d(
            upsample_dim + 3, hidden_channels,
            kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        Inizializzazione che produce α ≈ 0.5 all'inizio del training.
        Il bias di conv2 a 0 → sigmoid(0) = 0.5.
        """
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)             # sigmoid(0) = 0.5

    # ------------------------------------------------------------------
    def forward(
        self,
        F_sem:    torch.Tensor,   # (B, N, dino_dim) patch tokens DINOv2
        I_local:  torch.Tensor,   # (B, 3, H, W)
        I_global: torch.Tensor,   # (B, 3, H, W)
        n_h:      int,
        n_w:      int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        F_sem    : patch features DINOv2
        I_local  : immagine prodotta dalla bilateral grid locale
        I_global : immagine prodotta dalla bilateral grid globale
        n_h, n_w : dimensioni griglia patch

        Returns
        -------
        alpha : (B, 1, H, W)  confidence mask ∈ [0, 1]
        """
        B, _, H, W = I_local.shape

        # 1. Divergenza tra i due rami: (B, 3, H, W)
        D = (I_local - I_global).abs()

        # 2. F_sem → forma spaziale (B, dino_dim, n_h, n_w)
        F_sem_sp = F_sem.permute(0, 2, 1).reshape(B, F_sem.shape[-1], n_h, n_w)

        # 3. Proietta DINOv2 e upsampla a H/upsample_factor
        H_low = H // self.upsample_factor
        W_low = W // self.upsample_factor

        F_proj = self.dino_proj(F_sem_sp)          # (B, upsample_dim, n_h, n_w)
        F_up   = F.interpolate(
            F_proj, size=(H_low, W_low),
            mode="bilinear", align_corners=False,
        )                                          # (B, upsample_dim, H/4, W/4)

        # 4. Downsampla D alla stessa risoluzione ridotta
        D_down = F.interpolate(
            D, size=(H_low, W_low),
            mode="bilinear", align_corners=False,
        )                                          # (B, 3, H/4, W/4)

        # 5. Concatena e stima la maschera
        x    = torch.cat([F_up, D_down], dim=1)   # (B, upsample_dim+3, H/4, W/4)
        x    = F.relu(self.conv1(x), inplace=True)
        x    = self.conv2(x)                       # (B, 1, H/4, W/4)
        alpha_low = torch.sigmoid(x)               # (B, 1, H/4, W/4)

        # 6. Upsampla a piena risoluzione
        alpha = F.interpolate(
            alpha_low, size=(H, W),
            mode="bilinear", align_corners=False,
        )                                          # (B, 1, H, W)

        return alpha


# ---------------------------------------------------------------------------
# ConfidenceMaskBlender
# ---------------------------------------------------------------------------

class ConfidenceMaskBlender(nn.Module):
    """
    Combina MaskNet + blending finale + gamma encoding.

    Output:
      I_pred = α · I_local + (1-α) · I_global       (lineare)
      I_out  = gamma_sRGB(clip(I_pred, 0, 1))        (display)

    Returns dict con tutte le quantità intermedie per la loss.
    """

    def __init__(self, mask_net: MaskNet) -> None:
        super().__init__()
        self.mask_net = mask_net

    # ------------------------------------------------------------------
    def forward(
        self,
        F_sem:    torch.Tensor,   # (B, N, 384)
        I_local:  torch.Tensor,   # (B, 3, H, W)
        I_global: torch.Tensor,   # (B, 3, H, W)
        n_h:      int,
        n_w:      int,
    ) -> dict:
        """
        Returns
        -------
        dict con:
          "alpha"  : (B, 1, H, W)  confidence mask
          "I_pred" : (B, 3, H, W)  blended (lineare, prima della gamma)
          "I_out"  : (B, 3, H, W)  output finale sRGB ∈ [0,1]
        """
        alpha = self.mask_net(F_sem, I_local, I_global, n_h, n_w)

        I_pred = alpha * I_local + (1.0 - alpha) * I_global
        I_pred = I_pred.clamp(0.0, 1.0)

        I_out  = gamma_encode(I_pred)

        return {
            "alpha":  alpha,
            "I_pred": I_pred,
            "I_out":  I_out,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "ConfidenceMaskBlender":
        mcfg = cfg["mask"]
        ecfg = cfg["encoder"]
        mask_net = MaskNet(
            dino_dim        = ecfg["embed_dim"],
            upsample_dim    = mcfg["upsample_dim"],
            hidden_channels = mcfg["hidden_channels"],
            upsample_factor = mcfg["upsample_factor"],
        )
        return cls(mask_net=mask_net)
