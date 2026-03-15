"""
losses/chroma.py

Chroma Consistency Loss — saturazione isolata e hue circolare.

Separa i contributi cromatici che ΔE₀₀ combina in un unico scalare:
    L_sat:  MAE sulla chroma C* = sqrt(a*² + b*²)     — intensità del colore
    L_hue:  distanza circolare sull'angolo h* = atan2(b*, a*)  — direzione del colore

Formula combinata (§6.5.4):
    L_chroma = L_sat + 0.5 · L_hue

Il fattore 0.5 su L_hue riflette che l'hue è percettivamente rilevante
solo in zone con saturazione sufficiente: nei pixel quasi-acromatici
(C* ≈ 0) l'angolo hue è mal definito e irrilevante per il color grading.

Complementarità con L_ΔE:
    L_ΔE    → misura combinata L* + C* + h* in spazio percettivo
    L_sat   → supervisione esplicita su C* isolata
    L_hue   → distanza circolare su h* (gestisce periodicità ±π)
    L_hist  → distribuzione globale (invariante posizione)
"""

import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab, lab_chroma, lab_hue, circular_distance

EPS = 1e-8


class ChromaConsistencyLoss(nn.Module):
    """
    Loss di consistenza cromatica: saturazione + hue circolare.

    Formula (§6.5.4):
        L_sat  = (1/HW) Σ |C*_pred - C*_tgt|
        L_hue  = (1/HW) Σ d_circ(h*_pred, h*_tgt)
        L_chroma = L_sat + hue_weight · L_hue

    Dove:
        C* = sqrt(a*² + b*² + ε)          chroma (ε-smoothed)
        h* = atan2(b*, a*)                 hue in radianti
        d_circ(h1, h2) = |atan2(sin(h1-h2), cos(h1-h2))|  ∈ [0, π]

    Args:
        hue_weight: Peso del termine hue rispetto alla saturazione.
                    Default 0.5 (cfr. §6.5.4).

    Example:
        >>> loss_fn = ChromaConsistencyLoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> tgt  = torch.rand(2, 3, 384, 512)
        >>> loss = loss_fn(pred, tgt)   # scalar tensor
    """

    def __init__(self, hue_weight: float = 0.5):
        super().__init__()
        self.hue_weight = hue_weight

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
            Scalare: L_sat + hue_weight · L_hue.
        """
        lab_pred = self._to_lab_hwc(pred)    # (B, H, W, 3)
        lab_tgt  = self._to_lab_hwc(target)

        # ── Saturazione ───────────────────────────────────────────────────────
        # lab_chroma: (..., 3) → (...,)
        C_pred = lab_chroma(lab_pred)   # (B, H, W)
        C_tgt  = lab_chroma(lab_tgt)
        loss_sat = torch.abs(C_pred - C_tgt).mean()

        # ── Hue circolare ─────────────────────────────────────────────────────
        h_pred = lab_hue(lab_pred)      # (B, H, W), in (-π, π]
        h_tgt  = lab_hue(lab_tgt)
        loss_hue = circular_distance(h_pred, h_tgt).mean()

        return loss_sat + self.hue_weight * loss_hue
