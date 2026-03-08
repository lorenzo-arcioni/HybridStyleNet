"""
losses/identity.py

Identity Loss  (§6.4.7).

Previene l'overediting: con probabilità p_id al mini-batch,
il target viene sostituito con il sorgente stesso e la rete
viene penalizzata per qualsiasi modifica.

L_id = (1/HW) Σ_{i,j} ‖I^pred(i,j) - I^src(i,j)‖₁

Effetto: costringe la bilateral grid verso l'identità (A=I, b=0).
"""

import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    """
    Loss L1 tra predizione e sorgente, applicata con probabilità p_id.

    Durante il training, il chiamante deve decidere se attivare questa loss
    sostituendo il target con il sorgente. Questo modulo calcola sempre
    la MAE tra pred e src (il chiamante controlla quando applicarla).

    Args:
        p_id: Probabilità di attivazione per mini-batch (default 0.2).
              Non usata internamente: il chiamante può ignorarla o usarla.
    """

    def __init__(self, p_id: float = 0.2) -> None:
        super().__init__()
        self.p_id = p_id

    def forward(
        self,
        pred: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola la MAE tra predizione e sorgente.

        Args:
            pred: (B, 3, H, W) in [0,1] — immagine predetta.
            src:  (B, 3, H, W) in [0,1] — immagine sorgente originale.

        Returns:
            Scalare — L1 media.
        """
        return torch.abs(pred - src).mean()

    def should_apply(self) -> bool:
        """
        Campiona se applicare la loss in questo step.

        Returns:
            True con probabilità p_id.
        """
        return torch.rand(1).item() < self.p_id