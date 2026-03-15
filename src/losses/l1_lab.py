"""
losses/l1_lab.py

L1 Lab Loss — loss cromatica di warm-up nelle prime epoche del curriculum.

Nelle epoche 1-5, CIEDE2000 è disabilitata perché i suoi gradienti sono
instabili nelle zone di bassa chroma (C* ≈ 0) e nelle zone di transizione
dell'angolo hue. La L1 in spazio CIE Lab ha gradienti uniformi e stabili
su tutto il range cromatico, guidando la convergenza rapida verso la
regione corretta dello spazio delle soluzioni.

A partire dall'epoca 6, questa loss viene gradualmente disattivata
(peso da 0.4 → 0.0) mentre ΔE viene introdotta con peso crescente.
Il trasferimento graduale garantisce continuità del segnale di training.

Riferimento tesi: §6.5.1 (motivazione warm-up), §6.6 (curriculum)
Formula: L_L1Lab = (1/HW) Σ ‖I^pred_Lab(i,j) - I^tgt_Lab(i,j)‖₁
"""

import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab

class L1LabLoss(nn.Module):
    """
    MAE (L1) tra immagine predetta e target nello spazio CIE L*a*b*.

    Complementare a DeltaELoss: stessi input, formula più semplice con
    gradienti più stabili → usata come loss primaria nelle prime epoche.

    Formula:
        L_L1Lab = (1/HW) Σ_{i,j} |I^pred_Lab(i,j) - I^tgt_Lab(i,j)|

    Dove la norma L1 è sulla somma dei tre canali L*, a*, b*.

    Example:
        >>> loss_fn = L1LabLoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> tgt  = torch.rand(2, 3, 384, 512)
        >>> loss = loss_fn(pred, tgt)   # scalar tensor
    """

    def __init__(self):
        super().__init__()

    def _to_lab_hwc(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) sRGB → (B, H, W, 3) Lab float32."""
        return rgb_to_lab(img.float().permute(0, 2, 3, 1))

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
            Scalare: MAE media sui tre canali Lab e su tutti i pixel.
        """
        lab_pred = self._to_lab_hwc(pred)    # (B, H, W, 3)
        lab_tgt  = self._to_lab_hwc(target)

        # L1 su tutti e 3 i canali Lab contemporaneamente
        return torch.abs(lab_pred - lab_tgt).mean()
