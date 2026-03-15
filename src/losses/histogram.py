"""
losses/histogram.py

Color Histogram Loss — distribuzione spettrale globale dei colori.

Implementa la Wasserstein-1 distance (Earth Mover's Distance) tra
istogrammi di colore soft (differenziabili via kernel gaussiano) calcolati
separatamente sui tre canali CIE L*a*b*.

Riferimento tesi: §6.5.2
Formula: L_hist = (1/3) Σ_c Σ_k |CDF^pred_c(k) - CDF^tgt_c(k)|

Proprietà chiave:
    - Invariante alla posizione spaziale dei pixel (cattura distribuzione globale)
    - Complementare a L_ΔE che è pixel-wise
    - Differenziabile grazie al kernel gaussiano soft invece dell'assegnamento hard
    - EMD ≡ norma L1 tra CDF per distribuzioni 1D (formula di Vallender)
"""

import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab

EPS = 1e-8


def soft_histogram(
    img_lab_channel: torch.Tensor,
    n_bins: int,
    val_min: float,
    val_max: float,
    sigma_factor: float = 0.5,
) -> torch.Tensor:
    """
    Calcola un istogramma soft differenziabile per un singolo canale Lab.

    Ogni pixel contribuisce a tutti i bin con peso gaussiano proporzionale
    alla vicinanza al centro del bin. L'output è normalizzato a distribuzione
    di probabilità (somma = 1).

    Args:
        img_lab_channel: Tensore float32 shape (B, H, W), valori nel range
                         [val_min, val_max].
        n_bins:          Numero di bin (default 64 in ColorHistogramLoss).
        val_min:         Valore minimo del range del canale.
        val_max:         Valore massimo del range del canale.
        sigma_factor:    Larghezza del kernel come frazione del passo bin.
                         sigma = sigma_factor * (val_max - val_min) / n_bins.
                         Default 0.5 → overlap controllato tra bin adiacenti.

    Returns:
        Tensore float32 shape (B, n_bins), distribuzione di probabilità
        con somma = 1 lungo la dimensione dei bin.
    """
    
    B, H, W = img_lab_channel.shape
    N = H * W

    # Centri dei bin uniformemente distribuiti in [val_min, val_max]
    bin_step = (val_max - val_min) / n_bins
    # Centri: val_min + (k + 0.5) * bin_step  per k=0,...,n_bins-1
    centers = torch.linspace(
        val_min + 0.5 * bin_step,
        val_max - 0.5 * bin_step,
        n_bins,
        dtype=img_lab_channel.dtype,
        device=img_lab_channel.device,
    )  # (n_bins,)

    sigma = sigma_factor * bin_step + EPS

    # Flatten pixel: (B, N)
    pixels = img_lab_channel.reshape(B, N)  # (B, N)

    # Broadcasting: (B, N, 1) - (1, 1, n_bins) → (B, N, n_bins)
    diff = pixels.unsqueeze(-1) - centers.unsqueeze(0).unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)  # (B, N, n_bins)

    # Somma sui pixel → (B, n_bins)
    hist = weights.sum(dim=1)

    # Normalizza a distribuzione di probabilità
    hist = hist / (hist.sum(dim=-1, keepdim=True) + EPS)

    return hist


def histogram_emd(
    hist1: torch.Tensor,
    hist2: torch.Tensor,
) -> torch.Tensor:
    """
    Earth Mover's Distance (Wasserstein-1) tra due istogrammi 1D discreti.

    Per distribuzioni 1D: EMD = Σ_k |CDF1(k) - CDF2(k)|
    (formula di Vallender, invariante alla scala dei bin).

    Args:
        hist1, hist2: Tensori float32 shape (B, n_bins), distribuzioni
                      di probabilità (somma = 1 per campione).

    Returns:
        Tensore float32 shape (B,), EMD per campione del batch.
    """
    cdf1 = torch.cumsum(hist1, dim=-1)  # (B, n_bins)
    cdf2 = torch.cumsum(hist2, dim=-1)
    return torch.abs(cdf1 - cdf2).sum(dim=-1)  # (B,)


class ColorHistogramLoss(nn.Module):
    """
    Loss basata sulla distribuzione globale dei colori in spazio CIE Lab.

    Calcola la Wasserstein-1 distance tra gli istogrammi soft dei tre canali
    L*, a*, b* dell'immagine predetta e del target, mediata sul batch.

    Parametri degli istogrammi (§6.5.2):
        - n_bins = 64
        - sigma_factor = 0.5 (metà passo bin → overlap controllato)
        - Range per canale: L* ∈ [0, 100], a* ∈ [-128, 127], b* ∈ [-128, 127]

    Example:
        >>> loss_fn = ColorHistogramLoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> tgt  = torch.rand(2, 3, 384, 512)
        >>> loss = loss_fn(pred, tgt)   # scalar tensor
    """

    # Range dei canali CIE Lab per definire i centri dei bin
    _LAB_RANGES = [
        (0.0,    100.0),   # L*
        (-128.0, 127.0),   # a*
        (-128.0, 127.0),   # b*
    ]

    def __init__(
        self,
        n_bins: int = 64,
        sigma_factor: float = 0.5,
    ):
        """
        Args:
            n_bins:       Numero di bin per canale. Default 64.
            sigma_factor: Larghezza del kernel gaussiano come frazione
                          del passo bin. Default 0.5.
        """
        super().__init__()
        self.n_bins       = n_bins
        self.sigma_factor = sigma_factor

    def _to_lab(self, img: torch.Tensor) -> torch.Tensor:
        """
        Converti (B, 3, H, W) sRGB [0,1] → (B, H, W, 3) Lab float32.
        """
        return rgb_to_lab(img.float().permute(0, 2, 3, 1))  # (B, H, W, 3)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   sRGB (B, 3, H, W) in [0,1].
            target: sRGB (B, 3, H, W) in [0,1].

        Returns:
            Scalare: media dell'EMD sui tre canali e sul batch.
        """
        lab_pred = self._to_lab(pred)    # (B, H, W, 3)
        lab_tgt  = self._to_lab(target)

        total_emd = torch.zeros(1, device=pred.device, dtype=torch.float32)

        for c, (val_min, val_max) in enumerate(self._LAB_RANGES):
            ch_pred = lab_pred[..., c]   # (B, H, W)
            ch_tgt  = lab_tgt[..., c]

            hist_pred = soft_histogram(
                ch_pred, self.n_bins, val_min, val_max, self.sigma_factor
            )  # (B, n_bins)
            hist_tgt = soft_histogram(
                ch_tgt, self.n_bins, val_min, val_max, self.sigma_factor
            )

            emd = histogram_emd(hist_pred, hist_tgt)  # (B,)
            total_emd = total_emd + emd.mean()

        # Media sui 3 canali
        return total_emd / 3.0
