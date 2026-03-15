"""
structural_losses.py
--------------------
Loss strutturali di RAG-ColorNet.

Tre loss che non misurano la fedeltà cromatica ma la qualità
strutturale dell'output e la regolarità dei parametri intermedi:

  TotalVariationLoss    — smoothness dei coefficienti della bilateral grid
  EntropyMaskLoss       — spinge la confidence mask verso valori binari
  LuminancePreservationLoss — preserva la struttura di luminanza dell'src

Tutte sono pensate come termini di regolarizzazione con pesi piccoli
nel curriculum (λ ≈ 0.01 – 0.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.color_utils import rgb_to_lab          # type: ignore[import]


# ---------------------------------------------------------------------------
# Total Variation Loss  (sulla bilateral grid)
# ---------------------------------------------------------------------------

class TotalVariationLoss(nn.Module):
    """
    Total Variation sui coefficienti affini della bilateral grid.

    Penalizza variazioni brusche nei coefficienti adiacenti nelle
    dimensioni spaziali (x, y) — garantisce che la trasformazione
    cromatica sia localmente smooth, evitando artefatti a blocchi.

    Applicata a entrambe le grid G_global e G_local.

    L_TV = (1/|G|) · Σᵢⱼ (‖A_{i+1,j} - A_{i,j}‖_F + ‖A_{i,j+1} - A_{i,j}‖_F)

    Input: grid (B, n_affine, s, s, l)
           dove s = dimensione spaziale, l = bin di luminanza
    """

    def _tv_single(self, grid: torch.Tensor) -> torch.Tensor:
        """
        TV su una singola bilateral grid.

        grid : (B, n_affine, s_h, s_w, s_l)
        """
        # TV nelle direzioni spaziali (dim 2 e 3)
        diff_h = (grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]).abs()
        diff_w = (grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]).abs()
        return diff_h.mean() + diff_w.mean()

    def forward(
        self,
        G_global: torch.Tensor,   # (B, 12, 8,  8,  8)
        G_local:  torch.Tensor,   # (B, 12, 16, 16, 8)
    ) -> torch.Tensor:
        return self._tv_single(G_global) + self._tv_single(G_local)


# ---------------------------------------------------------------------------
# Entropy Mask Loss
# ---------------------------------------------------------------------------

class EntropyMaskLoss(nn.Module):
    """
    Entropy loss sulla confidence mask α.

    Massimizza l'entropia binaria negativa — forza α verso 0 o 1
    invece di valori medi (~0.5) che sarebbero indefiniti:

    L_entropy = -(1/H·W) · Σᵢⱼ [α·log(α+ε) + (1-α)·log(1-α+ε)]

    α = 0 o α = 1 → L_entropy = 0  (maschera binaria, ottimale)
    α = 0.5       → L_entropy = log(2) ≈ 0.69  (massima incertezza)

    Nel curriculum questa loss ha peso piccolo (0.01) — serve solo
    a evitare che la maschera rimanga bloccata a 0.5 per tutto il training.

    Parameters
    ----------
    eps : stabilità numerica per i logaritmi
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        alpha : (B, 1, H, W) ∈ [0, 1]
        """
        term_pos = alpha       * torch.log(alpha        + self.eps)
        term_neg = (1 - alpha) * torch.log(1 - alpha    + self.eps)
        return -(term_pos + term_neg).mean()


# ---------------------------------------------------------------------------
# Luminance Preservation Loss
# ---------------------------------------------------------------------------

class LuminancePreservationLoss(nn.Module):
    """
    Penalizza alterazioni della luminanza dell'immagine sorgente.

    Il fotografo cambia i colori ma di solito preserva la struttura
    tonale generale. Questa loss confronta L* di I_pred con L* di I_src
    (non con L* di I_tgt), per evitare che il modello «inventi»
    variazioni di esposizione non supportate dai dati.

    L_lum = (1/H·W) · Σᵢⱼ |L*_pred(i,j) - L*_src(i,j)|

    Attiva con peso 0.3 in tutte le fasi del curriculum.
    """

    def forward(
        self,
        pred: torch.Tensor,   # (B, 3, H, W) — output del modello
        src:  torch.Tensor,   # (B, 3, H, W) — immagine sorgente originale
    ) -> torch.Tensor:
        L_pred = rgb_to_lab(pred)[:, 0]           # (B, H, W)
        L_src  = rgb_to_lab(src)[:, 0]            # (B, H, W)
        return F.l1_loss(L_pred, L_src)
