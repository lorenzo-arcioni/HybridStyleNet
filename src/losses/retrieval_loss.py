"""
retrieval_loss.py
-----------------
Loss specifiche del meccanismo di retrieval di RAG-ColorNet.

Due loss che supervisionano la qualità dell'apprendimento delle
proiezioni W^Q, W^K, W^V e dell'assignment del ClusterNet:

  RetrievalQualityLoss  — leave-one-out: il retrieved edit deve approssimare
                          la vera edit signature della coppia lasciata fuori
  ClusterAssignmentLoss — cross-entropy tra soft assignment p e hard
                          assignment K-Means (attiva solo nelle prime 5 epoche)

RetrievalQualityLoss è il cuore dell'apprendimento delle proiezioni:
forza W^Q, W^K a produrre uno spazio in cui patch semanticamente simili
(stessa scena, stessa luce) sono vicine, e W^V a produrre edit realistici.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Retrieval Quality Loss
# ---------------------------------------------------------------------------

class RetrievalQualityLoss(nn.Module):
    """
    Leave-one-out retrieval quality loss.

    Per ogni coppia i nel batch di training, usa il database senza la
    coppia i stessa (leave-one-out) e misura quanto il retrieved edit
    approssima la vera edit signature E_i:

    L_retrieval = (1/N) · Σᵢ (1/Nᵢ) · Σₙ ‖R_i(n) - E_i(n)‖²

    dove:
      R_i(n) = retrieved edit per la patch n dell'immagine i
               usando D_φ \ {i} come database
      E_i(n) = vera edit signature (DINOv2(tgt_i) - DINOv2(src_i))

    In pratica, durante il training si simula il leave-one-out usando
    il retrieved edit già calcolato nel forward pass — la coppia corrente
    è esclusa dal database durante l'adaptation (il database è costruito
    prima del training e non contiene le coppie di validazione).

    Parameters
    ----------
    reduction : "mean" | "sum"
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    # ------------------------------------------------------------------
    def forward(
        self,
        R_spatial:   torch.Tensor,   # (B, d_r, n_h, n_w) — retrieved edit
        edit_target: torch.Tensor,   # (B, d_r, n_h, n_w) — edit target proiettato
    ) -> torch.Tensor:
        """
        Versione semplificata: MSE tra il retrieved edit e l'edit target
        nello spazio delle proiezioni (già in d_r dimensioni).

        edit_target è la proiezione W^V applicata alla vera edit signature:
          edit_target = W^V · E_i  (pre-calcolato nel training loop)

        Parameters
        ----------
        R_spatial    : retrieved edit come mappa spaziale
        edit_target  : edit target nello stesso spazio di proiezione
        """
        return F.mse_loss(R_spatial, edit_target, reduction=self.reduction)

    # ------------------------------------------------------------------
    def forward_from_patches(
        self,
        R_patches:       torch.Tensor,   # (B, N, d_r) — retrieved edit per patch
        edit_signatures: torch.Tensor,   # (B, N, edit_dim) — E_i vera
        W_V:             torch.Tensor,   # (edit_dim, d_r) — peso W^V
    ) -> torch.Tensor:
        """
        Variante che proietta le edit signatures con W^V prima del confronto.
        Usata quando si vuole supervisionare direttamente nello spazio originale.

        edit_target = E_i @ W_V^T  (proiezione con lo stesso W^V del retrieval)
        """
        edit_target = F.linear(edit_signatures, W_V)  # (B, N, d_r)
        return F.mse_loss(R_patches, edit_target, reduction=self.reduction)


# ---------------------------------------------------------------------------
# Cluster Assignment Loss
# ---------------------------------------------------------------------------

class ClusterAssignmentLoss(nn.Module):
    """
    Cross-entropy tra soft assignment p e hard assignment K-Means.

    Supervisiona il ClusterNet nelle prime N epoche del curriculum
    (assignment_loss_epochs = 5) in modo che l'output del MLP sia
    coerente con la struttura K-Means inizializzata sul fotografo.

    L_cluster = -Σ_k z_k · log(p_k + ε)

    dove z_k ∈ {0,1} è l'assegnazione hard determinata da K-Means.

    Dopo le prime 5 epoche λ_cluster = 0 (vedi composite_loss.py) —
    il ClusterNet è libero di raffinarsi senza supervisione rigida.

    Parameters
    ----------
    eps : stabilità numerica per il logaritmo
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    # ------------------------------------------------------------------
    def forward(
        self,
        p: torch.Tensor,   # (B, K)  soft assignment (post-softmax)
        z: torch.Tensor,   # (B,)    hard assignment (long tensor, class indices)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        p : soft cluster probabilities da ClusterNet
        z : hard cluster indices da K-Means (long tensor)
        """
        # Usa cross-entropy standard di PyTorch con logit-equivalenti
        # p è già post-softmax → convertiamo in log-probs
        log_p = torch.log(p + self.eps)           # (B, K)
        return F.nll_loss(log_p, z)
