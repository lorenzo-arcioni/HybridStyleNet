"""
cluster_net.py
--------------
Cluster Assignment — Componente 2 di RAG-ColorNet.

ClusterNet è un MLP leggero (2 layer, ~100K parametri) che mappa
il color histogram h ∈ ℝ^192 a una distribuzione di probabilità
p ∈ Δ^{K-1} sui K cluster stilistici del fotografo.

Workflow:
  1. Inizializzazione: centroidi K-Means su {h_i} delle coppie di training
  2. Training: supervisione con L_cluster (cross-entropy vs assignment K-Means)
               solo nelle prime N epoche (curriculum)
  3. Inference: soft assignment p = softmax(MLP(h))

Il numero K di cluster è determinato automaticamente con il criterio
del gomito (implementato in utils/kmeans_init.py) e varia per fotografo.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ClusterNet
# ---------------------------------------------------------------------------

class ClusterNet(nn.Module):
    """
    MLP 2-layer per soft cluster assignment.

        p = Softmax( W2 · ReLU(W1 · h + b1) + b2 )

    Parameters
    ----------
    input_dim  : dimensione input (192 = 3 × 64 bin)
    hidden_dim : dimensione layer nascosto (256)
    n_clusters : K — numero di cluster del fotografo
    """

    def __init__(
        self,
        input_dim:  int = 192,
        hidden_dim: int = 256,
        n_clusters: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters

        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_clusters)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Inizializzazione Xavier uniforme per stabilità in few-shot."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, input_dim)  color histogram Lab

        Returns
        -------
        p : (B, K)  probabilità sul simplex, sum = 1
        """
        x = F.relu(self.fc1(h))
        return F.softmax(self.fc2(x), dim=-1)

    # ------------------------------------------------------------------
    def hard_assignment(self, h: torch.Tensor) -> torch.Tensor:
        """
        Restituisce l'indice del cluster con probabilità massima.

        Returns
        -------
        z : (B,) long tensor con indici in [0, K-1]
        """
        return self.forward(h).argmax(dim=-1)

    # ------------------------------------------------------------------
    def reinitialise_from_centroids(
        self,
        centroids: torch.Tensor,
        freeze_after: bool = False,
    ) -> None:
        """
        Reinizializza il layer di output con i centroidi K-Means.

        I pesi fc2 vengono impostati in modo che, all'inizio del
        few-shot adaptation, l'output del ClusterNet sia allineato
        con l'assignment K-Means calcolato sui color histograms.

        Parameters
        ----------
        centroids    : (K, input_dim) centroidi K-Means sui histograms
        freeze_after : se True, congela i parametri dopo la reinizializzazione
                       (utile se si vuole mantenere l'assignment fisso)
        """
        K, D = centroids.shape
        assert K == self.n_clusters, (
            f"Centroids K={K} ≠ ClusterNet n_clusters={self.n_clusters}"
        )

        # Proietta i centroidi nello spazio nascosto tramite fc1 (frozen al momento)
        # e imposta fc2 per avvicinarsi all'output K-Means
        with torch.no_grad():
            # Stategia semplice: inizializza fc2 con centroidi normalizzati
            # come proxy di "distanza dal centroide"
            centroids_norm = F.normalize(centroids.float(), dim=-1)
            self.fc2.weight.data = centroids_norm
            self.fc2.bias.data   = torch.zeros(K)

        if freeze_after:
            for p in self.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def rebuild_for_k(self, new_k: int) -> "ClusterNet":
        """
        Restituisce un nuovo ClusterNet con K diverso.
        Utile dopo un re-clustering che cambia K*.
        I pesi fc1 vengono trasferiti, fc2 viene reinizializzato.
        """
        new_net = ClusterNet(
            input_dim  = self.input_dim,
            hidden_dim = self.hidden_dim,
            n_clusters = new_k,
        )
        # Copia fc1
        new_net.fc1.weight.data = self.fc1.weight.data.clone()
        new_net.fc1.bias.data   = self.fc1.bias.data.clone()
        return new_net

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, n_clusters: int) -> "ClusterNet":
        ccfg = cfg["cluster"]
        return cls(
            input_dim  = cfg["histogram"]["n_bins"] * 3,   # 192
            hidden_dim = ccfg["hidden_dim"],
            n_clusters = n_clusters,
        )
