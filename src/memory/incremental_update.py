"""
incremental_update.py
---------------------
Gestione dell'aggiornamento incrementale del database del fotografo.

Tre responsabilità:

1. Preprocessing iniziale
   Calcola e cacha le rappresentazioni DINOv2 di tutte le coppie
   di training prima dell'adaptation. Operazione una-tantum, ~0.6s/pair.

2. Aggiornamento incrementale (zero retraining)
   Aggiunge nuove coppie al database live:
     a. DINOv2 forward sulla nuova coppia (src + tgt)
     b. Assegna al cluster con ||h - C_k|| minimo
     c. Inserisce (key, value) nel ClusterMemory corretto
     d. Aggiorna l'indice FAISS
   Tempo totale per coppia: ~0.6s su RTX 3080.

3. Re-clustering periodico
   Ogni `recluster_every` nuove coppie:
     a. Ricalcola K* con elbow criterion su tutti gli h_i aggiornati
     b. Se K* è cambiato, esegue K-Means e ri-assegna
     c. Fine-tune ClusterNet per N epoche (solo ClusterNet, ~2 min)
     d. Ricostruisce gli indici FAISS

Interfaccia principale:
  updater = IncrementalUpdater(model, database, faiss_manager, cfg)
  updater.preprocess_all(photographer_dataset)
  updater.add_pair(src_tensor, tgt_tensor, meta={})
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .database    import PhotographerDatabase
from .faiss_index import FAISSIndexManager
from utils.kmeans_init import elbow_kmeans       # type: ignore[import]


# ---------------------------------------------------------------------------
# IncrementalUpdater
# ---------------------------------------------------------------------------

class IncrementalUpdater:
    """
    Gestisce l'intero ciclo di vita del database del fotografo:
    preprocessing, aggiornamento live, re-clustering.

    Parameters
    ----------
    model        : RAGColorNet — usato solo per encode_only()
    database     : PhotographerDatabase da aggiornare
    faiss_mgr    : FAISSIndexManager associato al database
    cfg          : config dict (base + photographer merged)
    device       : device per i forward DINOv2
    """

    def __init__(
        self,
        model:     nn.Module,
        database:  PhotographerDatabase,
        faiss_mgr: FAISSIndexManager,
        cfg:       dict,
        device:    str = "cuda",
    ) -> None:
        self.model     = model
        self.database  = database
        self.faiss_mgr = faiss_mgr
        self.cfg       = cfg
        self.device    = device

        # Stato interno per il re-clustering
        self._all_histograms: List[np.ndarray] = []   # (192,) per ogni coppia
        self._n_since_recluster: int = 0
        self._recluster_every = cfg.get("incremental", {}).get(
            "recluster_every", 50
        )

    # ------------------------------------------------------------------
    # 1. PREPROCESSING INIZIALE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def preprocess_all(
        self,
        dataset,                               # PhotographerDataset
        cluster_assignments: np.ndarray,       # (N,) int — da K-Means
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> None:
        """
        Pre-calcola e cacha le rappresentazioni di tutte le coppie.

        Popola database.clusters con tutte le coppie assegnate al
        rispettivo cluster. Aggiorna anche gli indici FAISS.

        Parameters
        ----------
        dataset             : PhotographerDataset con tutte le coppie
        cluster_assignments : array (N,) con l'indice di cluster per ogni coppia
        """
        self.model.eval()
        n_pairs = len(dataset)

        t0 = time.time()
        for idx in range(n_pairs):
            src, tgt = dataset.load_original(idx)
            src = src.unsqueeze(0).to(self.device)
            tgt = tgt.unsqueeze(0).to(self.device)

            key, value, h = self._compute_representations(src, tgt)

            cluster_id = int(cluster_assignments[idx])
            meta = {
                "src_file":   dataset.src_path(idx).name,
                "tgt_file":   dataset.tgt_path(idx).name,
                "pair_idx":   idx,
                "cluster_id": cluster_id,
            }

            self.database.add_pair(key, value, cluster_id, meta)
            self._all_histograms.append(h.cpu().numpy().squeeze(0))
            self.faiss_mgr.update_cluster(cluster_id, key.cpu().numpy())

            if show_progress and (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate    = (idx + 1) / elapsed
                eta     = (n_pairs - idx - 1) / rate
                print(
                    f"  Preprocessing: {idx+1}/{n_pairs} "
                    f"[{elapsed:.0f}s, ~{eta:.0f}s remaining]"
                )

        print(
            f"Preprocessing completato: {n_pairs} coppie in "
            f"{time.time()-t0:.1f}s"
        )
        print(f"  Occupazione database: {self.database.memory_usage_mb():.1f} MB")
        print(f"  Stato cluster: {self.database.cluster_sizes()}")

    # ------------------------------------------------------------------
    # 2. AGGIORNAMENTO INCREMENTALE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def add_pair(
        self,
        src:  torch.Tensor,          # (1, 3, H, W) o (3, H, W)
        tgt:  torch.Tensor,          # (1, 3, H, W) o (3, H, W)
        meta: Optional[dict] = None,
    ) -> dict:
        """
        Aggiunge una nuova coppia al database live. Zero retraining.

        Steps:
          1. DINOv2 forward su src e tgt           (~0.3s each)
          2. Calcola key (descriptor) e value (edit signature)
          3. Assegna al cluster tramite color histogram
          4. Inserisce nel ClusterMemory
          5. Aggiorna indice FAISS
          6. Controlla se serve re-clustering

        Returns
        -------
        dict con: cluster_id, n_total, recluster_triggered
        """
        # Normalizza dimensioni
        if src.dim() == 3:
            src = src.unsqueeze(0)
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(0)

        src = src.to(self.device)
        tgt = tgt.to(self.device)

        self.model.eval()
        t0 = time.time()

        # Calcola rappresentazioni
        key, value, h = self._compute_representations(src, tgt)

        # Assegnazione al cluster: argmin ||h - C_k||
        cluster_id = self._assign_cluster(h)

        # Aggiorna database e indice
        self.database.add_pair(key, value, cluster_id, meta or {})
        self.faiss_mgr.update_cluster(cluster_id, key.cpu().numpy())
        self._all_histograms.append(h.cpu().numpy().squeeze(0))
        self._n_since_recluster += 1

        elapsed = time.time() - t0

        # Controlla se serve re-clustering
        recluster_triggered = False
        if self._n_since_recluster >= self._recluster_every:
            recluster_triggered = True
            self._n_since_recluster = 0

        return {
            "cluster_id":           cluster_id,
            "n_total":              len(self.database),
            "elapsed_s":            elapsed,
            "recluster_triggered":  recluster_triggered,
        }

    # ------------------------------------------------------------------
    # 3. RE-CLUSTERING PERIODICO
    # ------------------------------------------------------------------

    def recluster(
        self,
        cluster_net,                    # ClusterNet da aggiornare
        fine_tune_epochs: int = 5,
        fine_tune_lr:     float = 1e-4,
    ) -> Tuple[int, bool]:
        """
        Ricalcola i cluster e ri-addestra il ClusterNet se K* è cambiato.

        Returns
        -------
        new_k       : nuovo numero di cluster K*
        k_changed   : True se K* è diverso dal precedente
        """
        if len(self._all_histograms) < 2:
            return self.database.n_clusters, False

        histograms = np.stack(self._all_histograms, axis=0)  # (N, 192)

        # K-Means con elbow criterion
        k_max   = self.cfg["cluster"]["k_max"]
        tau     = self.cfg["cluster"]["elbow_tau"]
        new_k, new_centroids, new_assignments = elbow_kmeans(
            histograms, k_max=k_max, tau=tau
        )

        k_changed = (new_k != self.database.n_clusters)

        if k_changed:
            print(f"Re-clustering: K {self.database.n_clusters} → {new_k}")
            self._rebuild_database(new_assignments, new_k)
            self._rebuild_faiss()

            # Aggiorna il ClusterNet per il nuovo K
            new_cluster_net = cluster_net.rebuild_for_k(new_k)
            new_cluster_net.reinitialise_from_centroids(
                torch.from_numpy(new_centroids).float()
            )
            self._fine_tune_cluster_net(
                new_cluster_net, histograms, new_assignments,
                epochs=fine_tune_epochs, lr=fine_tune_lr,
            )
            # Aggiorna il cluster_net nel modello
            self.model.replace_cluster_net(new_cluster_net)

        else:
            # K* invariato: aggiorna solo i centroidi
            self.database.centroids = new_centroids.astype(np.float32)

        return new_k, k_changed

    # ------------------------------------------------------------------
    def _rebuild_database(
        self,
        new_assignments: np.ndarray,
        new_k:           int,
    ) -> None:
        """
        Ri-assegna tutte le coppie ai nuovi cluster.
        Crea un nuovo database con K* cluster e ri-popola.
        """
        # Raccoglie tutte le coppie nell'ordine originale
        all_keys:   List[np.ndarray] = []
        all_values: List[np.ndarray] = []
        all_meta:   List[dict]       = []

        for k in range(self.database.n_clusters):
            mem = self.database.clusters[k]
            all_keys   += mem._keys
            all_values += mem._values
            all_meta   += mem._meta

        # Crea nuovo database
        new_db = PhotographerDatabase(
            n_clusters      = new_k,
            desc_dim        = self.database.desc_dim,
            edit_dim        = self.database.edit_dim,
            photographer_id = self.database.photographer_id,
        )

        for i, (key, value, meta) in enumerate(
            zip(all_keys, all_values, all_meta)
        ):
            new_db.add_pair(
                torch.from_numpy(key.astype(np.float32)),
                torch.from_numpy(value.astype(np.float32)),
                cluster_id=int(new_assignments[i]),
                meta=meta,
            )

        # Sostituisce il database
        self.database.clusters      = new_db.clusters
        self.database.n_clusters    = new_k
        self.faiss_mgr.n_clusters   = new_k

    # ------------------------------------------------------------------
    def _rebuild_faiss(self) -> None:
        """Ricostruisce tutti gli indici FAISS dopo il re-clustering."""
        # Ricrea il manager con il nuovo n_clusters
        from .faiss_index import FAISSIndexManager
        new_mgr = FAISSIndexManager.from_config(
            self.cfg, self.database.n_clusters
        )
        new_mgr.build_from_database(self.database)
        self.faiss_mgr.indices = new_mgr.indices
        self.faiss_mgr.n_clusters = self.database.n_clusters

    # ------------------------------------------------------------------
    def _fine_tune_cluster_net(
        self,
        cluster_net,
        histograms:      np.ndarray,   # (N, 192)
        assignments:     np.ndarray,   # (N,) int
        epochs:          int   = 5,
        lr:              float = 1e-4,
    ) -> None:
        """
        Fine-tune leggero del ClusterNet (solo ClusterNet, ~2 min).
        Cross-entropy tra soft assignment e hard assignment K-Means.
        """
        cluster_net.train()
        cluster_net.to(self.device)
        optimizer = torch.optim.Adam(cluster_net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        h_tensor = torch.from_numpy(histograms).float().to(self.device)
        z_tensor = torch.from_numpy(assignments).long().to(self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = cluster_net.fc2(
                torch.relu(cluster_net.fc1(h_tensor))
            )                                    # (N, K) — logits pre-softmax
            loss = criterion(logits, z_tensor)
            loss.backward()
            optimizer.step()

        cluster_net.eval()

    # ------------------------------------------------------------------
    # Helper interni
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_representations(
        self,
        src: torch.Tensor,    # (1, 3, H, W)
        tgt: torch.Tensor,    # (1, 3, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola key, value e histogram per una singola coppia.

        key   = query descriptor di src   (N_patches, desc_dim)
        value = edit signature            (N_patches, edit_dim)
                = DINOv2(tgt) - DINOv2(src)
        h     = color histogram di src    (1, 192)
        """
        enc_src = self.model.encode_only(src)
        enc_tgt = self.model.scene_encoder.extract_patch_features(tgt)

        F_sem_src = enc_src["F_sem"].squeeze(0)   # (N_patches, 384)
        F_sem_tgt = enc_tgt[0].squeeze(0)          # (N_patches, 384)
        Q_src     = enc_src["Q"].squeeze(0)        # (N_patches, 416)
        h         = enc_src["h"]                   # (1, 192)

        # Edit signature: differenza nello spazio delle feature
        value = (F_sem_tgt - F_sem_src)            # (N_patches, 384)

        return Q_src, value, h

    def _assign_cluster(self, h: torch.Tensor) -> int:
        """
        Assegna il color histogram al cluster più vicino (distanza L2).

        Usa i centroidi nel database se disponibili,
        altrimenti delega al ClusterNet.
        """
        if self.database.centroids is not None:
            centroids = torch.from_numpy(
                self.database.centroids
            ).float().to(self.device)              # (K, 192)
            h_vec = h.squeeze(0)                   # (192,)
            dists = ((centroids - h_vec) ** 2).sum(dim=-1)
            return int(dists.argmin().item())
        else:
            # Fallback al ClusterNet
            with torch.no_grad():
                return int(
                    self.model.cluster_net.hard_assignment(h).item()
                )
