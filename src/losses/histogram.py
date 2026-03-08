"""
losses/histogram.py

Color Histogram Loss con Earth Mover's Distance  (§6.4.2).

L_hist = (1/3) Σ_{c∈{L*,a*,b*}} Σ_k |CDF^pred_c(k) - CDF^tgt_c(k)|

Proprietà: invariante a permutazioni spaziali dei pixel
→ complementare a L_ΔE (pixel-wise).

Il soft histogram usa kernel gaussiani differenziabili.
"""

import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab

EPS = 1e-8


def soft_histogram(
    img_channel: torch.Tensor,
    bins: int = 64,
    value_range: tuple = (-128.0, 128.0),
    sigma_bin: float = None,
) -> torch.Tensor:
    """
    Calcola l'istogramma soft (differenziabile) per un singolo canale.

    h(k) = (1/N) Σ_i exp(-(x_i - μ_k)² / (2σ²))

    Args:
        img_channel: (B, H, W) — valori di un canale.
        bins:        Numero di bin.
        value_range: (min_val, max_val) dell'asse dei valori.
        sigma_bin:   Larghezza gaussiana. Default: (range/bins) * 0.5.

    Returns:
        h: (B, bins) — istogramma normalizzato (somma a 1 per batch elem).
    """
    B = img_channel.shape[0]
    v_min, v_max = value_range

    # Centri dei bin uniformemente spaziati
    centers = torch.linspace(v_min, v_max, bins,
                              dtype=img_channel.dtype,
                              device=img_channel.device)  # (bins,)

    if sigma_bin is None:
        sigma_bin = (v_max - v_min) / (2.0 * bins)

    # Flatten spaziale: (B, N) dove N = H*W
    x = img_channel.flatten(1)  # (B, N)

    # Distanza da ogni pixel a ogni bin: (B, N, bins)
    diff = x.unsqueeze(-1) - centers.unsqueeze(0).unsqueeze(0)
    weights = torch.exp(-diff ** 2 / (2.0 * sigma_bin ** 2))  # (B, N, bins)

    # Somma su pixel → (B, bins)
    h = weights.sum(dim=1)

    # Normalizza a distribuzione di probabilità
    h = h / (h.sum(dim=1, keepdim=True) + EPS)

    return h  # (B, bins)


def histogram_emd(
    h_pred: torch.Tensor,
    h_tgt: torch.Tensor,
) -> torch.Tensor:
    """
    Earth Mover's Distance (Wasserstein-1) tra due istogrammi 1D.

    EMD(h1, h2) = Σ_k |CDF1(k) - CDF2(k)|

    Args:
        h_pred: (B, bins) — istogramma predetto (normalizzato).
        h_tgt:  (B, bins) — istogramma target (normalizzato).

    Returns:
        (B,) — EMD per elemento del batch.
    """
    cdf_pred = torch.cumsum(h_pred, dim=1)  # (B, bins)
    cdf_tgt  = torch.cumsum(h_tgt,  dim=1)  # (B, bins)
    return torch.abs(cdf_pred - cdf_tgt).sum(dim=1)  # (B,)


# Ranges per i canali Lab
_LAB_RANGES = {
    0: (0.0,    100.0),   # L*
    1: (-128.0, 127.0),   # a*
    2: (-128.0, 127.0),   # b*
}


class ColorHistogramLoss(nn.Module):
    """
    Loss basata su Earth Mover's Distance tra istogrammi Lab.

    L_hist = (1/3) Σ_{c∈{L*,a*,b*}} EMD(h^pred_c, h^tgt_c)

    Args:
        bins:      Numero di bin per canale (default 64).
        sigma_bin: Larghezza gaussiana soft histogram.
                   None → calcolata automaticamente come range/(2*bins).
    """

    def __init__(
        self,
        bins: int = 64,
        sigma_bin: float = None,
    ) -> None:
        super().__init__()
        self.bins      = bins
        self.sigma_bin = sigma_bin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [0,1] — sRGB predetto.
            target: (B, 3, H, W) in [0,1] — sRGB target.

        Returns:
            Scalare — EMD medio sui 3 canali Lab e sul batch.
        """
        # Converti in Lab: (B, H, W, 3)
        pred_lab   = rgb_to_lab(pred.permute(0, 2, 3, 1))
        target_lab = rgb_to_lab(target.permute(0, 2, 3, 1))

        total_emd = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for c in range(3):
            vrange = _LAB_RANGES[c]

            # Estrai canale c: (B, H, W)
            pred_ch   = pred_lab[..., c]
            target_ch = target_lab[..., c]

            # Calcola soft histogram
            h_pred = soft_histogram(
                pred_ch, self.bins, vrange, self.sigma_bin
            )
            h_tgt = soft_histogram(
                target_ch, self.bins, vrange, self.sigma_bin
            )

            # EMD
            emd = histogram_emd(h_pred, h_tgt)  # (B,)
            total_emd = total_emd + emd.mean()

        return total_emd / 3.0