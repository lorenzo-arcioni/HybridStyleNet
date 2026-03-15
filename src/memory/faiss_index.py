"""
faiss_index.py
--------------
FAISS IVF-PQ index per retrieval approssimato efficiente.

Senza FAISS il retrieval ha costo O(N · N_new): per N=300 immagini
con ~972 patch ognuna si tratta di ~291K confronti per ogni forward pass.
Con IVF-PQ il costo scende a O(√N · N_new) con perdita trascurabile
di qualità di retrieval.

Strategia:
  - Un indice per cluster (gli cluster hanno distribuzioni diverse)
  - Indice IVF-PQ: Inverted File con Product Quantization
  - La chiave di retrieval è il centroide delle patch di ogni immagine
    (vettore 416-dim), non le singole patch
  - Top-M image retrieval: l'indice restituisce gli M indici di immagine,
    poi si caricano le relative patch keys/values dal database

Quando N_k < n_train_min (cluster troppo piccolo per FAISS) si usa
brute-force exact search come fallback.

Dipendenza opzionale: se faiss non è installato, si usa sempre
il fallback brute-force (più lento ma corretto).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import faiss                     # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    warnings.warn(
        "faiss non trovato. Retrieval usa brute-force (più lento). "
        "Installa con: pip install faiss-gpu  oppure  pip install faiss-cpu",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_N_TRAIN_MIN   = 40     # minimo campioni per addestrare l'indice IVF
_DEFAULT_NLIST = 16     # celle IVF
_DEFAULT_NPROBE= 4      # celle visitate a query time
_DEFAULT_PQ_M  = 8      # sub-quantizzatori PQ (desc_dim deve essere divisibile)


# ---------------------------------------------------------------------------
# ClusterIndex  (indice per un singolo cluster)
# ---------------------------------------------------------------------------

class ClusterIndex:
    """
    FAISS IVF-PQ index per un singolo cluster di immagini.

    Indicizza i centroidi delle immagini (media delle patch keys)
    per un retrieval globale rapido. Restituisce indici di immagine,
    non di singole patch.

    Parameters
    ----------
    desc_dim  : dimensione del descrittore (416)
    nlist     : numero di celle IVF
    nprobe    : celle visitate a query time
    pq_m      : sub-quantizzatori PQ
    """

    def __init__(
        self,
        desc_dim: int = 416,
        nlist:    int = _DEFAULT_NLIST,
        nprobe:   int = _DEFAULT_NPROBE,
        pq_m:     int = _DEFAULT_PQ_M,
    ) -> None:
        self.desc_dim   = desc_dim
        self.nlist      = nlist
        self.nprobe     = nprobe
        self.pq_m       = min(pq_m, desc_dim)  # pq_m ≤ desc_dim
        self._index     = None
        self._centroids: Optional[np.ndarray] = None   # (N_k, desc_dim)
        self._n_images  = 0
        self._use_faiss = False

    # ------------------------------------------------------------------
    def build(self, image_centroids: np.ndarray) -> None:
        """
        Costruisce l'indice dai centroidi delle immagini nel cluster.

        Parameters
        ----------
        image_centroids : (N_k, desc_dim) float32
                          centroide = media delle patch keys per ogni immagine
        """
        self._centroids = image_centroids.astype(np.float32)
        self._n_images  = len(image_centroids)

        if not _FAISS_AVAILABLE or self._n_images < _N_TRAIN_MIN:
            self._use_faiss = False
            return

        # Adatta nlist: non può superare il numero di campioni
        actual_nlist = min(self.nlist, self._n_images)

        # IVF-PQ: desc_dim deve essere divisibile per pq_m
        # Se non lo è, si trova il pq_m più grande che divide desc_dim
        pq_m = self._find_valid_pq_m(self.desc_dim, self.pq_m)

        # Ogni sub-vettore deve avere almeno 1 bit → 8 bit per sub-centroide
        # PQ richiede almeno 256 campioni di training; se non bastano, fallback
        if self._n_images < 256:
            self._use_faiss = False
            return

        quantizer = faiss.IndexFlatL2(self.desc_dim)
        index     = faiss.IndexIVFPQ(
            quantizer, self.desc_dim,
            actual_nlist, pq_m, 8        # 8 bit per sub-quantizzatore
        )
        index.train(self._centroids)
        index.add(self._centroids)
        index.nprobe = min(self.nprobe, actual_nlist)

        self._index     = index
        self._use_faiss = True

    # ------------------------------------------------------------------
    @staticmethod
    def _find_valid_pq_m(desc_dim: int, pq_m: int) -> int:
        """Trova il pq_m più grande ≤ pq_m che divide desc_dim."""
        for m in range(pq_m, 0, -1):
            if desc_dim % m == 0:
                return m
        return 1

    # ------------------------------------------------------------------
    def search(
        self,
        query: np.ndarray,   # (desc_dim,) float32  — centroide query image
        top_m: int = 10,
    ) -> np.ndarray:
        """
        Restituisce gli indici delle top-M immagini più vicine alla query.

        Returns
        -------
        indices : (top_m,) int array — indici in [0, N_k-1]
        """
        actual_m = min(top_m, self._n_images)
        if self._n_images == 0:
            return np.array([], dtype=np.int64)

        if self._use_faiss and self._index is not None:
            q = query.reshape(1, -1).astype(np.float32)
            _, indices = self._index.search(q, actual_m)
            return indices[0]

        # Brute-force fallback: L2 distance tra query e centroidi
        return self._brute_force_search(query, actual_m)

    # ------------------------------------------------------------------
    def _brute_force_search(
        self, query: np.ndarray, top_m: int
    ) -> np.ndarray:
        """Ricerca esatta tramite distanza L2."""
        diffs  = self._centroids - query.reshape(1, -1)
        dists  = (diffs ** 2).sum(axis=1)
        return np.argsort(dists)[:top_m]

    # ------------------------------------------------------------------
    def update(self, new_centroid: np.ndarray) -> None:
        """
        Aggiunge un nuovo centroide all'indice (aggiornamento incrementale).

        Se l'indice è FAISS, lo ricostruisce da zero (necessario per IVF-PQ).
        Se è brute-force, appende semplicemente.
        """
        if self._centroids is None:
            self._centroids = new_centroid.reshape(1, -1).astype(np.float32)
        else:
            self._centroids = np.vstack(
                [self._centroids, new_centroid.reshape(1, -1).astype(np.float32)]
            )
        self._n_images += 1

        # Ricostruisce l'indice solo se si supera la soglia FAISS
        if _FAISS_AVAILABLE and self._n_images >= _N_TRAIN_MIN:
            self.build(self._centroids)

    # ------------------------------------------------------------------
    @property
    def n_images(self) -> int:
        return self._n_images

    @property
    def is_faiss(self) -> bool:
        return self._use_faiss


# ---------------------------------------------------------------------------
# FAISSIndexManager  (gestore di tutti gli indici per cluster)
# ---------------------------------------------------------------------------

class FAISSIndexManager:
    """
    Gestisce un ClusterIndex per ogni cluster del fotografo.

    È l'interfaccia principale usata da IncrementalUpdater e
    dal PhotographerDatabase per il retrieval rapido.

    Parameters
    ----------
    n_clusters : K
    desc_dim   : dimensione del descrittore (416)
    nlist      : IVF nlist
    nprobe     : IVF nprobe
    pq_m       : PQ sub-quantizzatori
    """

    def __init__(
        self,
        n_clusters: int,
        desc_dim:   int = 416,
        nlist:      int = _DEFAULT_NLIST,
        nprobe:     int = _DEFAULT_NPROBE,
        pq_m:       int = _DEFAULT_PQ_M,
    ) -> None:
        self.n_clusters = n_clusters
        self.desc_dim   = desc_dim

        self.indices: Dict[int, ClusterIndex] = {
            k: ClusterIndex(desc_dim=desc_dim, nlist=nlist, nprobe=nprobe, pq_m=pq_m)
            for k in range(n_clusters)
        }

    # ------------------------------------------------------------------
    def build_from_database(self, database) -> None:
        """
        Costruisce tutti gli indici partendo da un PhotographerDatabase.

        Calcola il centroide (media patch keys) per ogni immagine in ogni
        cluster e lo aggiunge all'indice corrispondente.
        """
        for k, mem in database.clusters.items():
            if len(mem) == 0:
                continue

            centroids = np.stack(
                [keys.astype(np.float32).mean(axis=0)
                 for keys in mem._keys],
                axis=0
            )                                    # (N_k, desc_dim)

            self.indices[k].build(centroids)

    # ------------------------------------------------------------------
    def search_cluster(
        self,
        cluster_id:  int,
        query_key:   torch.Tensor,   # (desc_dim,) o (N_patches, desc_dim)
        top_m:       int = 10,
    ) -> np.ndarray:
        """
        Cerca le top-M immagini in un cluster.

        Se query_key ha più patch, ne calcola la media (centroide query).

        Returns
        -------
        indices : (top_m,) int array
        """
        if query_key.dim() == 2:
            q = query_key.float().mean(dim=0).cpu().numpy()
        else:
            q = query_key.float().cpu().numpy()

        return self.indices[cluster_id].search(q, top_m)

    # ------------------------------------------------------------------
    def update_cluster(
        self,
        cluster_id:  int,
        new_key:     np.ndarray,   # (N_patches, desc_dim)
    ) -> None:
        """Aggiunge un nuovo centroide immagine all'indice del cluster."""
        centroid = new_key.astype(np.float32).mean(axis=0)
        self.indices[cluster_id].update(centroid)

    # ------------------------------------------------------------------
    def status(self) -> Dict[int, dict]:
        """Restituisce lo stato di ogni indice (n_images, is_faiss)."""
        return {
            k: {
                "n_images": idx.n_images,
                "is_faiss": idx.is_faiss,
            }
            for k, idx in self.indices.items()
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, n_clusters: int) -> "FAISSIndexManager":
        rcfg = cfg["retrieval"]
        return cls(
            n_clusters = n_clusters,
            desc_dim   = cfg["descriptor"]["output_dim"],
            nlist      = rcfg["faiss_nlist"],
            nprobe     = rcfg["faiss_nprobe"],
            pq_m       = rcfg["faiss_pq_m"],
        )
