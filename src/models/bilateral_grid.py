"""
models/bilateral_grid.py

Bilateral Grid + Trilinear Slicing  (§6.3).

Struttura:
  G ∈ R^(B, 12, H_g, W_g, L_b)
    12  = coefficienti affini 3×3 + bias 3
    H_g × W_g = risoluzione spaziale della grid (8×8 o 32×32)
    L_b = 8  bin di luminanza

Il bilateral slicing opera a risoluzione piena (H, W) usando
coordinate normalizzate → resolution-agnostic.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPS = 1e-8

# Pesi luminanza BT.601 (§6.3)
_LUM_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


# ── Bilateral Grid ────────────────────────────────────────────────────────────

class BilateralGrid(nn.Module):
    """
    Bilateral Grid differenziabile con slicing trilineare.

    La grid codifica una trasformazione affine cromatica per ogni cella
    (x, y, luminanza). La guida di luminanza rende la trasformazione
    edge-aware: pixel con luminanza simile ricevono la stessa
    trasformazione indipendentemente dalla loro posizione.

    Args:
        grid_h:    Risoluzione spaziale verticale della grid.
        grid_w:    Risoluzione spaziale orizzontale della grid.
        luma_bins: Numero di bin di luminanza (L_b).
        in_ch:     Canali input (default 3 = RGB).
        out_ch:    Canali output (default 3 = RGB).
    """

    def __init__(
        self,
        grid_h: int = 8,
        grid_w: int = 8,
        luma_bins: int = 8,
        in_ch: int = 3,
        out_ch: int = 3,
    ) -> None:
        super().__init__()
        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.luma_bins = luma_bins
        self.in_ch     = in_ch
        self.out_ch    = out_ch
        # Coefficienti: matrice (out_ch × in_ch) + bias (out_ch)
        self.n_coeffs  = out_ch * in_ch + out_ch   # = 12 per RGB→RGB

    def apply(
        self,
        grid: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applica la bilateral grid a un'immagine tramite slicing trilineare.

        Args:
            grid:  (B, n_coeffs, grid_h, grid_w, luma_bins)
                   Coefficienti della trasformazione affine.
            image: (B, 3, H, W)  immagine sorgente in [0,1].
                   Usata sia come input della trasformazione
                   che come guida di luminanza.

        Returns:
            (B, out_ch, H, W)  immagine trasformata in [0,1].
        """
        B, C, H, W = image.shape
        assert C == self.in_ch, \
            f"BilateralGrid.apply: atteso {self.in_ch} canali, got {C}"

        # ── 1. Guida di luminanza (§6.3.2) ───────────────────────────────────
        lum_w = _LUM_WEIGHTS.to(image.device)              # (3,)
        guide = (image * lum_w[None, :, None, None]).sum(1)  # (B, H, W)

        # ── 2. Coordinate normalizzate nella grid ─────────────────────────────
        # x_g ∈ [0, grid_w-1], y_g ∈ [0, grid_h-1], l_g ∈ [0, luma_bins-1]
        # Tutte in coordinate continue per l'interpolazione trilineare

        # Griglia di coordinate pixel (H, W) normalizzate in [0,1]
        # align_corners=True → 0 mappa al primo bin, 1 all'ultimo
        grid_y = torch.linspace(0, 1, H, device=image.device)  # (H,)
        grid_x = torch.linspace(0, 1, W, device=image.device)  # (W,)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij") # (H, W)

        # Scala alle dimensioni della grid
        # grid_sample usa [-1, 1] → convertiamo
        gx_n = gx * 2.0 - 1.0  # (H, W)  ∈ [-1, 1]
        gy_n = gy * 2.0 - 1.0  # (H, W)  ∈ [-1, 1]

        # Coordinate di luminanza normalizzate in [-1, 1]
        gl_n = guide * 2.0 - 1.0  # (B, H, W)

        # ── 3. Slicing trilineare ─────────────────────────────────────────────
        # Usiamo F.grid_sample con griglia 5-D (B, H, W, 1, 3) per 3-D input
        # La grid 5-D ha shape (B, H, W, 1, 3): (x, y, z) coordinate

        # Espandi le coordinate spaziali al batch
        gx_b = gx_n[None, :, :].expand(B, H, W)   # (B, H, W)
        gy_b = gy_n[None, :, :].expand(B, H, W)   # (B, H, W)

        # Griglia campionamento: (B, H, W, 1, 3) — (x_spatial, y_spatial, luma)
        sample_grid = torch.stack(
            [gx_b, gy_b, gl_n], dim=-1
        ).unsqueeze(3)  # (B, H, W, 1, 3)

        # grid ha shape (B, n_coeffs, luma_bins, grid_h, grid_w)
        # grid_sample 3-D: input (B, C, D, H, W), grid (B, H_out, W_out, D_out, 3)
        # Riorganizza grid: (B, n_coeffs, luma_bins, grid_h, grid_w)
        G = grid  # già nel formato corretto

        # sample_grid deve avere shape (B, H, W, 1, 3) con ordine (x, y, z)
        # dove x = dimensione W della grid, y = dimensione H, z = luma
        # F.grid_sample 5D: input (B,C,D,H,W) grid (B,Ho,Wo,Do,3) → (B,C,Ho,Wo,Do)
        coeffs = F.grid_sample(
            G,                                  # (B, n_coeffs, luma_bins, grid_h, grid_w)
            sample_grid,                        # (B, H, W, 1, 3)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # → (B, n_coeffs, H, W, 1)

        coeffs = coeffs.squeeze(-1)             # (B, n_coeffs, H, W)

        # ── 4. Trasformazione affine pixel-wise (§6.3.4) ──────────────────────
        out_ch  = self.out_ch
        in_ch   = self.in_ch
        n_mat   = out_ch * in_ch

        A = coeffs[:, :n_mat, :, :]              # (B, out*in, H, W)
        b = coeffs[:, n_mat:, :, :]              # (B, out_ch, H, W)

        # Reshape A: (B, out_ch, in_ch, H, W)
        A = A.view(B, out_ch, in_ch, H, W)

        # image: (B, in_ch, H, W) → (B, 1, in_ch, H, W)
        img_exp = image.unsqueeze(1)             # (B, 1, in_ch, H, W)

        # Moltiplicazione matrice-vettore per pixel
        # (B, out_ch, in_ch, H, W) * (B, 1, in_ch, H, W) → somma su in_ch
        out = (A * img_exp).sum(dim=2) + b       # (B, out_ch, H, W)

        return out.clamp(0.0, 1.0)

    def forward(
        self,
        grid: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Alias di apply() per uso come modulo nn."""
        return self.apply(grid, image)


# ── Grid Predictor ────────────────────────────────────────────────────────────

class GlobalGridPredictor(nn.Module):
    """
    Predice la Bilateral Grid globale (8×8×L_b) da feature globali.

    Input:  vettore globale f ∈ R^(B, in_features)  (dopo GAP)
    Output: G_global ∈ R^(B, 12, L_b, grid_h, grid_w)

    Args:
        in_features: Dimensione del vettore input (es. 512 da P5 GAP).
        grid_h:      Risoluzione verticale grid.
        grid_w:      Risoluzione orizzontale grid.
        luma_bins:   Numero bin luminanza.
        hidden_dim:  Dimensione layer nascosto MLP.
    """

    def __init__(
        self,
        in_features: int = 512,
        grid_h: int = 8,
        grid_w: int = 8,
        luma_bins: int = 8,
        hidden_dim: int = 256,
        out_ch: int = 3,
        in_ch: int = 3,
    ) -> None:
        super().__init__()

        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.luma_bins = luma_bins
        self.n_coeffs  = out_ch * in_ch + out_ch

        n_out = self.n_coeffs * grid_h * grid_w * luma_bins

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_out),
        )

        # Inizializza l'ultimo layer vicino a zero
        # → la grid parte vicino all'identità
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Bias identità: matrice 3×3 = I, bias = 0
        # Coefficienti in ordine: [R→R, R→G, R→B, G→R, G→G, G→B, B→R, B→G, B→B,
        #                          bias_R, bias_G, bias_B]
        # Inizializziamo i bias della matrice all'identità
        with torch.no_grad():
            identity_coeffs = torch.zeros(self.n_coeffs)
            identity_coeffs[0] = 1.0   # R→R
            identity_coeffs[4] = 1.0   # G→G
            identity_coeffs[8] = 1.0   # B→B
            # Replicate su tutti le celle della grid
            bias_val = identity_coeffs.repeat(
                grid_h * grid_w * luma_bins
            )
            self.mlp[-1].bias.data = bias_val

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (B, in_features)

        Returns:
            (B, n_coeffs, luma_bins, grid_h, grid_w)
        """
        B = f.shape[0]
        out = self.mlp(f)   # (B, n_coeffs * grid_h * grid_w * luma_bins)
        return out.view(B, self.n_coeffs, self.luma_bins,
                        self.grid_h, self.grid_w)


class LocalGridPredictor(nn.Module):
    """
    Predice la Bilateral Grid locale (32×32×L_b) da feature spaziali.

    Input:  feature map x ∈ R^(B, in_channels, H', W')  (da SPADEResBlock)
    Output: G_local ∈ R^(B, 12, L_b, 32, 32)

    Usa AdaptiveAvgPool per portare qualsiasi (H', W') a (32, 32)
    → resolution-agnostic.

    Args:
        in_channels: Canali input (es. 256 da x3).
        grid_h:      Risoluzione verticale grid (default 32).
        grid_w:      Risoluzione orizzontale grid (default 32).
        luma_bins:   Numero bin luminanza.
    """

    def __init__(
        self,
        in_channels: int = 256,
        grid_h: int = 32,
        grid_w: int = 32,
        luma_bins: int = 8,
        out_ch: int = 3,
        in_ch: int = 3,
    ) -> None:
        super().__init__()

        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.luma_bins = luma_bins
        self.n_coeffs  = out_ch * in_ch + out_ch

        # AdaptiveAvgPool → (32, 32) resolution-agnostic
        self.pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))

        # Convoluzione 1×1 per predire i coefficienti
        # Output: (B, n_coeffs * luma_bins, grid_h, grid_w)
        n_out_ch = self.n_coeffs * luma_bins
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, n_out_ch, kernel_size=1, bias=True),
        )

        # Inizializzazione identità
        nn.init.zeros_(self.conv[0].weight)
        with torch.no_grad():
            identity_coeffs = torch.zeros(self.n_coeffs)
            identity_coeffs[0] = 1.0
            identity_coeffs[4] = 1.0
            identity_coeffs[8] = 1.0
            bias_val = identity_coeffs.repeat(luma_bins)
            self.conv[0].bias.data = bias_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H', W')  — qualsiasi H', W'

        Returns:
            (B, n_coeffs, luma_bins, grid_h, grid_w)
        """
        B = x.shape[0]
        pooled = self.pool(x)   # (B, in_channels, grid_h, grid_w)
        out    = self.conv(pooled)  # (B, n_coeffs*luma_bins, grid_h, grid_w)
        return out.view(B, self.n_coeffs, self.luma_bins,
                        self.grid_h, self.grid_w)