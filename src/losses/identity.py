"""
losses/identity.py

Identity Loss — prevenzione dell'overediting e stabilità del training.

Con probabilità p_id=0.2 per mini-batch, il target viene sostituito con
l'immagine sorgente: in questi casi la rete deve imparare a lasciare
l'immagine invariata (I^pred ≈ I^src), vincolando i coefficienti affini
della bilateral grid verso la trasformazione identità.

Riferimento tesi: §6.5.5
Formula: L_id = (1/HW) Σ ‖I^pred - I^src‖₁

NOTA: Questa loss è applicata solo nei "mini-batch identità" (p_id=0.2).
      Il modulo riceve già la pred calcolata con target=src, quindi il
      confronto è sempre tra pred e src.

Ruolo nel few-shot regime:
    Con 100-200 coppie la rete può apprendere trasformazioni aggressive
    che funzionano sul training set ma divergono su scene mai viste.
    Il 20% di campioni identità è un prior implicito sulla conservatività
    della trasformazione: la rete apprende che la trasformazione di default
    in assenza di segnale è l'identità.

Perché L1 e non L2:
    La norma ℓ₁ è più robusta agli outlier: alcuni pixel (specchi, luci
    saturate) hanno grandi errori per ragioni valide e la ℓ₂ li
    penalizzerebbe quadraticamente.
"""

import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    """
    Loss L1 tra immagine predetta e sorgente per prevenire overediting.

    Deve essere applicata solo sui mini-batch identità (dove target=src).
    La selezione casuale dei batch identità è gestita dal training loop
    (cfr. training/adapt.py), non da questa classe.

    Formula:
        L_id = (1/HW) Σ_{i,j} ‖I^pred(i,j) - I^src(i,j)‖₁

    Example:
        >>> loss_fn = IdentityLoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> src  = torch.rand(2, 3, 384, 512)   # stesso src usato come target
        >>> loss = loss_fn(pred, src)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   Immagine predetta (B, 3, H, W) in [0,1].
            source: Immagine sorgente (B, 3, H, W) in [0,1].
                    In un mini-batch identità coincide con il target.

        Returns:
            Scalare: MAE pixel-wise tra pred e source.
        """
        return torch.abs(pred.float() - source.float()).mean()
