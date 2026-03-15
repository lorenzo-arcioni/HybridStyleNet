"""
losses/delta_e.py

CIEDE2000 (ΔE₀₀) loss per color grading photographer-specific.

Implementa la formula CIEDE2000 completa (CIE 142-2001) come funzione
differenziabile in PyTorch, con ε-smoothing sulle radici quadrate per
evitare gradienti infiniti nelle zone acromatiche (C* ≈ 0).

Riferimento tesi: §6.5.1
Formula: L_ΔE = (1/HW) Σ ΔE₀₀(I^pred_Lab(i,j), I^tgt_Lab(i,j))

NOTA sul calcolo in fp32:
    Tutti i tensori in input vengono promossi a float32 prima del calcolo
    della loss (via .float()), indipendentemente dalla precisione del
    forward pass (fp16/bf16). Questo previene underflow nei termini
    logaritmici e nelle radici quadrate di CIEDE2000. Il costo aggiuntivo
    è trascurabile rispetto al forward pass del modello.
"""

import torch
import torch.nn as nn
import math

# Importa le conversioni dal modulo utils — percorso adattabile
from utils.color_space import rgb_to_lab

EPS = 1e-8

# Costanti angolari in radianti (le formule CIEDE2000 usano gradi)
_DEG_30  = math.radians(30.0)
_DEG_6   = math.radians(6.0)
_DEG_63  = math.radians(63.0)
_DEG_275 = math.radians(275.0)
_DEG_25  = math.radians(25.0)


# ── Kernel CIEDE2000 ──────────────────────────────────────────────────────────

def ciede2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """
    Calcola ΔE₀₀ tra due tensori CIE L*a*b*.

    Implementazione vettorizzata della formula completa CIE 142-2001.
    Tutti i calcoli sono eseguiti in float32.

    Args:
        lab1: Tensore float32 shape (..., 3), canali [L*, a*, b*].
        lab2: Tensore float32 shape (..., 3), canali [L*, a*, b*].

    Returns:
        Tensore float32 shape (...,), valori ΔE₀₀ ≥ 0.
        ΔE₀₀ < 1  → differenza impercettibile
        ΔE₀₀ < 2  → eccellente per color grading
        ΔE₀₀ < 5  → accettabile
        ΔE₀₀ > 10 → differenza evidente
        ΔE₀₀ > 15 → differenza inaccettabile
    """
    
    lab1 = lab1.float()
    lab2 = lab2.float()

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # ── Passo 1: C*_ab e media ────────────────────────────────────────────────
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + EPS)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + EPS)
    C_bar = (C1 + C2) / 2.0

    # ── Passo 2: a' corretto e C' ─────────────────────────────────────────────
    # G = 0.5 * (1 - sqrt(C_bar^7 / (C_bar^7 + 25^7)))
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - torch.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7 + EPS)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + EPS)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + EPS)

    # ── Passo 3: h' (angolo in radianti) ─────────────────────────────────────
    h1p = torch.atan2(b1, a1p)  # ∈ (-π, π]
    h2p = torch.atan2(b2, a2p)
    # Normalizza in [0, 2π)
    h1p = torch.where(h1p < 0, h1p + 2 * math.pi, h1p)
    h2p = torch.where(h2p < 0, h2p + 2 * math.pi, h2p)

    # ── Passo 4: ΔL', ΔC', Δh', ΔH' ─────────────────────────────────────────
    dLp = L2 - L1
    dCp = C2p - C1p

    # Δh' con wrap-around circolare
    dh_raw = h2p - h1p
    dh_cond1 = torch.abs(h2p - h1p) <= math.pi
    dh_cond2 = (h2p - h1p) > math.pi
    dhp = torch.where(dh_cond1, dh_raw,
           torch.where(dh_cond2, dh_raw - 2 * math.pi,
                                 dh_raw + 2 * math.pi))

    # ΔH' = 2·sqrt(C1'·C2')·sin(Δh'/2)
    dHp = 2.0 * torch.sqrt(C1p * C2p + EPS) * torch.sin(dhp / 2.0)

    # ── Passo 5: medie L̄', C̄', h̄' ──────────────────────────────────────────
    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    # h̄' con wrap-around
    h_sum  = h1p + h2p
    h_diff_abs = torch.abs(h1p - h2p)
    hp_bar = torch.where(
        h_diff_abs <= math.pi,
        h_sum / 2.0,
        torch.where(h_sum < 2 * math.pi,
                    (h_sum + 2 * math.pi) / 2.0,
                    (h_sum - 2 * math.pi) / 2.0),
    )

    # ── Passo 6: T, S_L, S_C, S_H ────────────────────────────────────────────
    T = (1.0
         - 0.17 * torch.cos(hp_bar - _DEG_30)
         + 0.24 * torch.cos(2.0 * hp_bar)
         + 0.32 * torch.cos(3.0 * hp_bar + _DEG_6)
         - 0.20 * torch.cos(4.0 * hp_bar - _DEG_63))

    Lp_bar_50 = Lp_bar - 50.0
    S_L = 1.0 + 0.015 * (Lp_bar_50 ** 2) / torch.sqrt(20.0 + Lp_bar_50 ** 2 + EPS)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T

    # ── Passo 7: R_T (termine di interazione hue-chroma) ─────────────────────
    Cp_bar7 = Cp_bar ** 7
    R_C = 2.0 * torch.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7 + EPS))

    d_theta = _DEG_30 * torch.exp(
        -((hp_bar - _DEG_275) ** 2) / (_DEG_25 ** 2)
    )
    R_T = -R_C * torch.sin(2.0 * d_theta)

    # ── Passo 8: ΔE₀₀ ────────────────────────────────────────────────────────
    term_L = (dLp / S_L) ** 2
    term_C = (dCp / S_C) ** 2
    term_H = (dHp / S_H) ** 2
    term_R = R_T * (dCp / S_C) * (dHp / S_H)

    return torch.sqrt(term_L + term_C + term_H + term_R + EPS)


# ── Loss class ────────────────────────────────────────────────────────────────

class DeltaELoss(nn.Module):
    """
    Loss CIEDE2000 media su tutti i pixel.

    Accetta immagini sRGB [0,1] in formato (B, 3, H, W) oppure
    Lab (B, H, W, 3) se already_lab=True.

    Tutti i calcoli interni sono in float32 per stabilità numerica.

    Example:
        >>> loss_fn = DeltaELoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> tgt  = torch.rand(2, 3, 384, 512)
        >>> loss = loss_fn(pred, tgt)   # scalar tensor
    """

    def __init__(self, already_lab: bool = False):
        """
        Args:
            already_lab: Se True, gli input sono già in CIE Lab
                         shape (B, H, W, 3). Se False (default),
                         vengono convertiti da sRGB (B, 3, H, W).
        """
        super().__init__()
        self.already_lab = already_lab

    def _to_lab_hwc(self, img: torch.Tensor) -> torch.Tensor:
        """Converti (B, 3, H, W) sRGB → (B, H, W, 3) Lab float32."""
        # (B, 3, H, W) → (B, H, W, 3)
        img_hwc = img.float().permute(0, 2, 3, 1)
        return rgb_to_lab(img_hwc)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   sRGB (B, 3, H, W) in [0,1], oppure Lab (B, H, W, 3)
                    se already_lab=True.
            target: stessa shape e spazio di pred.

        Returns:
            Scalare: media di ΔE₀₀ su tutti i pixel del batch.
        """
        if self.already_lab:
            lab_pred = pred.float()
            lab_tgt  = target.float()
        else:
            lab_pred = self._to_lab_hwc(pred)    # (B, H, W, 3)
            lab_tgt  = self._to_lab_hwc(target)

        delta_e = ciede2000(lab_pred, lab_tgt)   # (B, H, W)
        return delta_e.mean()
