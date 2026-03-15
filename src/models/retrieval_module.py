"""
retrieval_module.py
-------------------
Local Retrieval Module — Componente 3 di RAG-ColorNet (cuore architetturale).

Implementa la Cross-Image Local Attention che, per ogni patch della nuova
immagine, recupera il trattamento cromatico applicato dal fotografo alle
patch semanticamente simili nelle sue foto precedenti.

Pipeline per ogni cluster k (pesato da p_k):
  1. Proietta Q, K_i, V_i con W^Q, W^K, W^V trainable
  2. Calcola A_i = Softmax(Q̃ · K̃_i^T / √d_r)       (N_new × N_i)
  3. Aggrega R_i = A_i · Ṽ_i                          (N_new × d_r)
  4. Pesa per similarità globale s̃_i
  5. Somma pesata sui cluster: R_final = Σ_k p_k · R^(k)

Il database (chiavi e valori pre-calcolati) è gestito da memory/database.py
e passato come argomento — il retrieval module non ha stato proprio sul DB.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RetrievalModule
# ---------------------------------------------------------------------------

class RetrievalModule(nn.Module):
    """
    Cross-Image Local Attention per il retrieval di edit signatures.

    Parameters
    ----------
    desc_dim    : dimensione del query descriptor (416 = 384 + 32)
    edit_dim    : dimensione dell'edit signature DINOv2 (384)
    d_r         : dimensione dello spazio di proiezione per Q, K, V (256)
    skip_thresh : p_k sotto questa soglia → cluster skippato
    """

    def __init__(
        self,
        desc_dim:    int   = 416,
        edit_dim:    int   = 384,
        d_r:         int   = 256,
        skip_thresh: float = 0.01,
    ) -> None:
        super().__init__()
        self.desc_dim    = desc_dim
        self.edit_dim    = edit_dim
        self.d_r         = d_r
        self.skip_thresh = skip_thresh
        self.scale       = d_r ** -0.5

        # Proiezioni lineari trainable
        self.W_Q = nn.Linear(desc_dim, d_r, bias=False)
        self.W_K = nn.Linear(desc_dim, d_r, bias=False)
        self.W_V = nn.Linear(edit_dim, d_r, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(module.weight)

    # ------------------------------------------------------------------
    def _attend_single_image(
        self,
        Q_proj:  torch.Tensor,  # (B, N_new, d_r)
        K_proj:  torch.Tensor,  # (N_i, d_r)
        V_proj:  torch.Tensor,  # (N_i, d_r)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola attention tra le query della nuova immagine e
        chiavi/valori di un'immagine di training.

        Returns
        -------
        R_i   : (B, N_new, d_r)  retrieved edit per questa immagine
        s_i   : (B,)             similarità globale (media dei max attention)
        """
        # (B, N_new, d_r) × (d_r, N_i) → (B, N_new, N_i)
        logits = torch.einsum("bnd,md->bnm", Q_proj, K_proj) * self.scale
        A_i    = F.softmax(logits, dim=-1)        # (B, N_new, N_i)

        # Retrieved edit: (B, N_new, N_i) × (N_i, d_r) → (B, N_new, d_r)
        R_i = torch.einsum("bnm,md->bnd", A_i, V_proj)

        # Similarità globale: media sui patch dei max attention weights
        s_i = A_i.max(dim=-1).values.mean(dim=-1)  # (B,)

        return R_i, s_i

    # ------------------------------------------------------------------
    def forward_cluster(
        self,
        Q:           torch.Tensor,          # (B, N_new, 416)
        db_keys:     torch.Tensor,          # (M, N_i, 416)  top-M immagini
        db_values:   torch.Tensor,          # (M, N_i, 384)
    ) -> torch.Tensor:
        """
        Retrieval per un singolo cluster con top-M immagini.

        Parameters
        ----------
        Q          : query descriptor della nuova immagine
        db_keys    : patch descriptors delle top-M immagini nel cluster
        db_values  : edit signatures delle top-M immagini nel cluster

        Returns
        -------
        R_k : (B, N_new, d_r)  retrieved edit per questo cluster
        """
        B, N_new, _ = Q.shape
        M = db_keys.shape[0]

        Q_proj = self.W_Q(Q)                      # (B, N_new, d_r)

        R_list:  List[torch.Tensor] = []
        s_list:  List[torch.Tensor] = []

        for i in range(M):
            K_i     = db_keys[i]                  # (N_i, 416)
            V_i     = db_values[i]                # (N_i, 384)

            K_proj  = self.W_K(K_i)               # (N_i, d_r)
            V_proj  = self.W_V(V_i)               # (N_i, d_r)

            R_i, s_i = self._attend_single_image(Q_proj, K_proj, V_proj)
            R_list.append(R_i)
            s_list.append(s_i)

        # Pesi di similarità globale: s̃_i = s_i / Σ s_i
        s_stack = torch.stack(s_list, dim=-1)     # (B, M)
        s_norm  = s_stack / (s_stack.sum(dim=-1, keepdim=True) + 1e-8)

        # Aggregazione: Σ_i s̃_i · R_i
        R_stack = torch.stack(R_list, dim=1)      # (B, M, N_new, d_r)
        R_k = (R_stack * s_norm.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        return R_k                                # (B, N_new, d_r)

    # ------------------------------------------------------------------
    def forward(
        self,
        Q:          torch.Tensor,           # (B, N_new, 416)
        cluster_db: Dict[int, Dict],        # {k: {"keys": Tensor, "values": Tensor}}
        p:          torch.Tensor,           # (B, K)  cluster probabilities
        n_h:        int,
        n_w:        int,
    ) -> torch.Tensor:
        """
        Forward pass completo: mixture across tutti i cluster.

        Parameters
        ----------
        Q          : query descriptor della nuova immagine
        cluster_db : dizionario per cluster, ognuno con "keys" e "values"
                     "keys"  : (M_k, N_i, 416) top-M immagini del cluster
                     "values": (M_k, N_i, 384)
        p          : soft assignment ai cluster
        n_h, n_w   : dimensioni della griglia di patch

        Returns
        -------
        R_spatial : (B, d_r, n_h, n_w)  retrieved edit come mappa spaziale
        """
        B, N_new, _ = Q.shape
        K            = p.shape[1]

        R_final = torch.zeros(B, N_new, self.d_r, device=Q.device, dtype=Q.dtype)

        for k in range(K):
            # Skip cluster con probabilità trascurabile
            if p[:, k].max().item() < self.skip_thresh:
                continue

            if k not in cluster_db or cluster_db[k] is None:
                continue

            db_keys   = cluster_db[k]["keys"].to(Q.device)    # (M, N_i, 416)
            db_values = cluster_db[k]["values"].to(Q.device)  # (M, N_i, 384)

            if db_keys.shape[0] == 0:
                continue

            R_k = self.forward_cluster(Q, db_keys, db_values)  # (B, N_new, d_r)

            # Somma pesata per probabilità del cluster: p_k · R_k
            # p[:, k]: (B,) → (B, 1, 1)
            R_final = R_final + p[:, k].view(B, 1, 1) * R_k

        # Reshape a mappa spaziale: (B, N_new, d_r) → (B, d_r, n_h, n_w)
        R_spatial = R_final.reshape(B, n_h, n_w, self.d_r)
        R_spatial = R_spatial.permute(0, 3, 1, 2).contiguous()

        return R_spatial                          # (B, d_r, n_h, n_w)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "RetrievalModule":
        rcfg = cfg["retrieval"]
        dcfg = cfg["descriptor"]
        return cls(
            desc_dim    = dcfg["output_dim"],
            edit_dim    = cfg["encoder"]["embed_dim"],
            d_r         = rcfg["d_r"],
            skip_thresh = rcfg["skip_threshold"],
        )
