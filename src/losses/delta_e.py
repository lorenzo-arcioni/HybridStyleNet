"""
losses/delta_e.py

CIEDE2000 Loss  (§6.4.1).

Misura la distanza cromatica percettiva tra immagine predetta e target
nello spazio CIE L*a*b*, usando la formula completa CIEDE2000 con
ε-smoothing per differenziabilità.

Target: ΔE₀₀ < 5 (accettabile), < 2 (eccellente).
"""

import math
import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab

EPS = 1e-8
DEG_TO_RAD = math.pi / 180.0


def ciede2000(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Calcola CIEDE2000 per ogni pixel tra due immagini Lab.

    Implementazione completamente vettorizzata e differenziabile
    (§6.4.1 — formula completa con tutti i fattori di peso).

    Args:
        lab1: (B, H, W, 3) o (..., 3) — L*, a*, b* predette.
        lab2: (B, H, W, 3) o (..., 3) — L*, a*, b* target.
        eps:  Epsilon per stabilità numerica sotto le radici quadrate.

    Returns:
        ΔE₀₀: (...,) — distanza per pixel, stessa shape senza l'ultima dim.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # ── Passo 1: C* e C_bar ──────────────────────────────────────────────────
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
    C_bar = (C1 + C2) * 0.5

    # ── Passo 2: Fattore G e a' corretto ──────────────────────────────────────
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - torch.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7 + eps)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    # ── Passo 3: C' e h' ──────────────────────────────────────────────────────
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + eps)

    # h' in [0, 2π): atan2 → [−π, π], poi shift
    h1p = torch.atan2(b1, a1p) % (2 * math.pi)
    h2p = torch.atan2(b2, a2p) % (2 * math.pi)

    # ── Passo 4: ΔL', ΔC', Δh', ΔH' ─────────────────────────────────────────
    dLp = L2 - L1
    dCp = C2p - C1p

    # Differenza di hue circolare
    dh_raw = h2p - h1p
    dh_abs = torch.abs(dh_raw)

    dhp = torch.where(
        dh_abs <= math.pi,
        dh_raw,
        torch.where(dh_raw > math.pi, dh_raw - 2 * math.pi, dh_raw + 2 * math.pi),
    )

    dHp = 2.0 * torch.sqrt(C1p * C2p + eps) * torch.sin(dhp * 0.5)

    # ── Passo 5: medie L_bar, C_bar', h_bar' ─────────────────────────────────
    Lp_bar = (L1 + L2) * 0.5
    Cp_bar = (C1p + C2p) * 0.5

    # h_bar' (media circolare)
    hp_sum = h1p + h2p
    hp_bar = torch.where(
        dh_abs <= math.pi,
        hp_sum * 0.5,
        torch.where(
            hp_sum < 2 * math.pi,
            (hp_sum + 2 * math.pi) * 0.5,
            (hp_sum - 2 * math.pi) * 0.5,
        ),
    )

    # ── Passo 6: fattori T, S_L, S_C, S_H ────────────────────────────────────
    T = (
        1.0
        - 0.17 * torch.cos(hp_bar - 30.0 * DEG_TO_RAD)
        + 0.24 * torch.cos(2.0 * hp_bar)
        + 0.32 * torch.cos(3.0 * hp_bar + 6.0 * DEG_TO_RAD)
        - 0.20 * torch.cos(4.0 * hp_bar - 63.0 * DEG_TO_RAD)
    )

    SL = 1.0 + 0.015 * (Lp_bar - 50.0) ** 2 / torch.sqrt(
        20.0 + (Lp_bar - 50.0) ** 2 + eps
    )
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T

    # ── Passo 7: fattore di rotazione R_T ────────────────────────────────────
    Cp_bar7 = Cp_bar ** 7
    RC = 2.0 * torch.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7 + eps))
    d_theta = 30.0 * DEG_TO_RAD * torch.exp(
        -((hp_bar - 275.0 * DEG_TO_RAD) / (25.0 * DEG_TO_RAD)) ** 2
    )
    RT = -RC * torch.sin(2.0 * d_theta)

    # ── Passo 8: ΔE₀₀ ────────────────────────────────────────────────────────
    term_L = (dLp / SL) ** 2
    term_C = (dCp / SC) ** 2
    term_H = (dHp / SH) ** 2
    term_R = RT * (dCp / SC) * (dHp / SH)

    de00 = torch.sqrt(term_L + term_C + term_H + term_R + eps)
    return de00


class DeltaELoss(nn.Module):
    """
    Loss basata su CIEDE2000 media su tutti i pixel.

    L_ΔE = (1/HW) Σ_{i,j} ΔE₀₀(I^pred_Lab(i,j), I^tgt_Lab(i,j))

    Args:
        eps: Epsilon per stabilità numerica nelle radici quadrate.
    """

    def __init__(self, eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [0,1] — immagine predetta sRGB.
            target: (B, 3, H, W) in [0,1] — immagine target sRGB.

        Returns:
            Scalare — ΔE₀₀ media.
        """
        # (B,3,H,W) → (B,H,W,3)
        pred_hw   = pred.permute(0, 2, 3, 1)
        target_hw = target.permute(0, 2, 3, 1)

        # Converti in Lab
        pred_lab   = rgb_to_lab(pred_hw)    # (B, H, W, 3)
        target_lab = rgb_to_lab(target_hw)  # (B, H, W, 3)

        # CIEDE2000 per pixel
        de = ciede2000(pred_lab, target_lab, eps=self.eps)  # (B, H, W)

        return de.mean()