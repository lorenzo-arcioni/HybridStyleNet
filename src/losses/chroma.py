"""
losses/chroma.py

Cosine Similarity Loss  (§6.4.5)
Chroma Consistency Loss (§6.4.6)

Cosine Loss:
    L_cos = 1 - (1/HW) Σ_{i,j}
            v^pred(i,j)^T v^tgt(i,j) /
            (‖v^pred(i,j)‖ · ‖v^tgt(i,j)‖ + ε)

    dove v(i,j) = (a*(i,j), b*(i,j)) — vettore cromatico nel piano Lab.

Chroma Consistency Loss:
    L_chroma = L_sat + 0.5 · L_hue

    L_sat = (1/HW) Σ |C*^pred - C*^tgt|
    L_hue = (1/HW) Σ d_circ(h*^pred, h*^tgt)
"""

import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab, lab_chroma, lab_hue, circular_distance

EPS = 1e-8


class CosineLoss(nn.Module):
    """
    Penalizza l'errore di direzione (hue) nel piano cromatico (a*, b*)  (§6.4.5).

    L_cos = 1 - mean_pixels[ cos(angle(v^pred, v^tgt)) ]

    Range: [0, 2] — 0 = hue perfetto, 2 = hue opposto.

    Args:
        eps: Epsilon per evitare divisione per zero su pixel acromatici.
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
            pred:   (B, 3, H, W) in [0,1].
            target: (B, 3, H, W) in [0,1].

        Returns:
            Scalare — cosine loss.
        """
        # Converti in Lab
        pred_lab   = rgb_to_lab(pred.permute(0, 2, 3, 1))    # (B,H,W,3)
        target_lab = rgb_to_lab(target.permute(0, 2, 3, 1))  # (B,H,W,3)

        # Vettori cromatici (a*, b*)
        v_pred = pred_lab[..., 1:3]    # (B, H, W, 2)
        v_tgt  = target_lab[..., 1:3]  # (B, H, W, 2)

        # Prodotto scalare
        dot = (v_pred * v_tgt).sum(dim=-1)  # (B, H, W)

        # Norme con epsilon
        norm_pred = v_pred.norm(dim=-1).clamp(min=self.eps)
        norm_tgt  = v_tgt.norm(dim=-1).clamp(min=self.eps)

        # Coseno dell'angolo tra vettori cromatici
        cos_sim = dot / (norm_pred * norm_tgt)  # (B, H, W) ∈ [-1, 1]

        # Loss: 1 - cos_sim (0 = perfetto, 2 = opposto)
        loss = 1.0 - cos_sim.mean()

        return loss


class ChromaConsistencyLoss(nn.Module):
    """
    Chroma Consistency Loss: errori separati di saturazione e hue  (§6.4.6).

    L_chroma = L_sat + hue_weight · L_hue

    L_sat = mean |C*^pred - C*^tgt|                (MAE sulla chroma)
    L_hue = mean d_circ(h*^pred, h*^tgt)           (distanza circolare hue)

    Args:
        hue_weight: Peso relativo di L_hue in L_chroma (default 0.5).
        eps:        Epsilon per stabilità numerica nella chroma.
    """

    def __init__(
        self,
        hue_weight: float = 0.5,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.hue_weight = hue_weight
        self.eps        = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [0,1].
            target: (B, 3, H, W) in [0,1].

        Returns:
            Scalare — chroma consistency loss.
        """
        # Converti in Lab
        pred_lab   = rgb_to_lab(pred.permute(0, 2, 3, 1))    # (B,H,W,3)
        target_lab = rgb_to_lab(target.permute(0, 2, 3, 1))  # (B,H,W,3)

        # ── Saturazione (chroma C*) ───────────────────────────────────────────
        c_pred = lab_chroma(pred_lab)    # (B, H, W)
        c_tgt  = lab_chroma(target_lab)  # (B, H, W)
        l_sat  = torch.abs(c_pred - c_tgt).mean()

        # ── Hue circolare ─────────────────────────────────────────────────────
        h_pred = lab_hue(pred_lab)    # (B, H, W) ∈ (-π, π]
        h_tgt  = lab_hue(target_lab)  # (B, H, W)
        l_hue  = circular_distance(h_pred, h_tgt).mean()  # ∈ [0, π]

        # Normalizza L_hue ∈ [0, π] → [0, 1] dividendo per π
        l_hue_norm = l_hue / torch.pi

        return l_sat + self.hue_weight * l_hue_norm