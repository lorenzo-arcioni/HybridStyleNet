"""
database.py
-----------
Database non-parametrico del fotografo.

Mantiene in RAM (CPU, fp16) le rappresentazioni pre-calcolate di tutte
le coppie di training del fotografo, organizzate per cluster:

  Per ogni cluster k:
    keys  : (N_k, N_patches, 416) float16  — query descriptors delle patch
    values: (N_k, N_patches, 384) float16  — edit signatures (ΔF_sem)

L'edit signature di una coppia è definita come:
  E_i = DINOv2(I_tgt_i) - DINOv2(I_src_i)
cioè il delta nello spazio delle feature semantiche tra sorgente e target.

Il database è completamente stateless rispetto al modello:
aggiungere una coppia non richiede nessun retraining — solo un
forward pass DINOv2 (~0.6s) e l'inserimento nella struttura dati.

Struttura su disco:
  db_root/
    cluster_0/
      keys.npy       (N_0, N_patches, 416) float16
      values.npy     (N_0, N_patches, 384) float16
      meta.json      [{filename, date, cluster_id, ...}]
    cluster_1/
      ...
    centroids.npy    (K, 192) float32  — centroidi K-Means
    manifest.json    {n_pairs, k, patch_size, desc_dim, edit_dim, created}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# ClusterMemory  (memoria per un singolo cluster)
# ---------------------------------------------------------------------------

class ClusterMemory:
    """
    Struttura dati per le coppie di un singolo cluster.

    Mantiene keys e values come liste di array numpy fp16
    (una entry per immagine), convertiti in tensor solo al momento
    del retrieval per minimizzare l'uso di memoria.

    Parameters
    ----------
    cluster_id : indice del cluster
    desc_dim   : dimensione del query descriptor (416)
    edit_dim   : dimensione dell'edit signature (384)
    """

    def __init__(
        self,
        cluster_id: int,
        desc_dim:   int = 416,
        edit_dim:   int = 384,
    ) -> None:
        self.cluster_id = cluster_id
        self.desc_dim   = desc_dim
        self.edit_dim   = edit_dim

        self._histograms: List[np.ndarray] = []   # (192,) per immagine — per ranking globale

        # Una entry per immagine: shape (N_patches_i, dim)
        self._keys:   List[np.ndarray] = []   # float16
        self._values: List[np.ndarray] = []   # float16
        self._meta:   List[dict]       = []

    # ------------------------------------------------------------------
    def add(
        self,
        key:   torch.Tensor,   # (N_patches, desc_dim)
        value: torch.Tensor,   # (N_patches, edit_dim)
        meta:  Optional[dict] = None,
        histogram: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Aggiunge una coppia al cluster.

        key e value vengono convertiti in numpy fp16 e mantenuti in RAM.
        """
        self._keys.append(
            key.detach().cpu().to(torch.float16).numpy()
        )
        self._values.append(
            value.detach().cpu().to(torch.float16).numpy()
        )
        self._meta.append(meta or {})

        # Salva l'istogramma se fornito, altrimenti usa i primi 192 della media delle chiavi
        if histogram is not None:
            self._histograms.append(histogram.detach().cpu().float().numpy())
        else:
            self._histograms.append(
                key.detach().cpu().float().mean(dim=0).numpy()[:192]
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    # ------------------------------------------------------------------
    def get_top_m(
        self,
        query_hist: torch.Tensor,       # (desc_dim,) — histogram della nuova img
        m:          int = 10,
        device:     str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Restituisce le top-M coppie più simili alla query.

        La similarità globale è calcolata come media coseno tra il
        color histogram della nuova immagine e i key-centroids di ogni
        immagine nel cluster (media delle patch keys).

        Returns
        -------
        dict con:
          "keys"   : (M, N_patches, desc_dim) float32
          "values" : (M, N_patches, edit_dim) float32
        """
        n = len(self._keys)
        if n == 0:
            return {"keys": torch.empty(0), "values": torch.empty(0)}

        actual_m = min(m, n)

        if n <= actual_m:
            # Restituisce tutto senza ranking
            selected_keys   = self._keys
            selected_values = self._values
        else:
            # Calcola similarità globale tramite media delle chiavi
            scores = self._compute_global_scores(query_hist)
            top_idx = np.argsort(scores)[-actual_m:][::-1]
            selected_keys   = [self._keys[i]   for i in top_idx]
            selected_values = [self._values[i] for i in top_idx]

        # Pad le patch alla lunghezza massima (per batch uniforme)
        max_patches = max(k.shape[0] for k in selected_keys)
        keys_padded   = self._pad_and_stack(selected_keys,   max_patches, self.desc_dim)
        values_padded = self._pad_and_stack(selected_values, max_patches, self.edit_dim)

        return {
            "keys":   torch.from_numpy(keys_padded).to(torch.float32).to(device),
            "values": torch.from_numpy(values_padded).to(torch.float32).to(device),
        }

    # ------------------------------------------------------------------
    def _compute_global_scores(
        self, query_hist: torch.Tensor
    ) -> np.ndarray:
        """
        Similarità coseno tra query_hist e la media delle chiavi di ogni img.
        query_hist: (desc_dim,) tensor
        """
        q = query_hist.cpu().float().numpy()
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        scores = np.zeros(len(self._keys), dtype=np.float32)
        for i, hist in enumerate(self._histograms):
            h_norm    = hist / (np.linalg.norm(hist) + 1e-8)
            scores[i] = np.dot(q_norm, h_norm)

        return scores

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_and_stack(
        arrays: List[np.ndarray],
        max_patches: int,
        dim: int,
    ) -> np.ndarray:
        """
        Stack di array con padding a zero per allineare il numero di patch.
        Returns: (M, max_patches, dim) float16
        """
        M = len(arrays)
        out = np.zeros((M, max_patches, dim), dtype=np.float16)
        for i, arr in enumerate(arrays):
            n = arr.shape[0]
            out[i, :n, :] = arr
        return out

    # ------------------------------------------------------------------
    def save(self, cluster_dir: Path) -> None:
        """Salva il cluster su disco in formato numpy."""
        cluster_dir.mkdir(parents=True, exist_ok=True)

        if len(self._keys) == 0:
            return

        # Salva come array di oggetti (lunghezze diverse per patch)
        np.save(cluster_dir / "keys.npy",   np.array(self._keys,   dtype=object))
        np.save(cluster_dir / "values.npy", np.array(self._values, dtype=object))

        with open(cluster_dir / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        cluster_dir: Path,
        cluster_id:  int,
        desc_dim:    int = 416,
        edit_dim:    int = 384,
    ) -> "ClusterMemory":
        mem = cls(cluster_id=cluster_id, desc_dim=desc_dim, edit_dim=edit_dim)

        keys_path   = cluster_dir / "keys.npy"
        values_path = cluster_dir / "values.npy"
        meta_path   = cluster_dir / "meta.json"

        if not keys_path.exists():
            return mem                             # cluster vuoto

        keys_arr   = np.load(keys_path,   allow_pickle=True)
        values_arr = np.load(values_path, allow_pickle=True)

        mem._keys   = list(keys_arr)
        mem._values = list(values_arr)

        if meta_path.exists():
            with open(meta_path) as f:
                mem._meta = json.load(f)
        else:
            mem._meta = [{} for _ in mem._keys]

        return mem


# ---------------------------------------------------------------------------
# PhotographerDatabase  (database completo del fotografo)
# ---------------------------------------------------------------------------

class PhotographerDatabase:
    """
    Database completo del fotografo: K cluster di ClusterMemory.

    È l'interfaccia principale usata dal RetrievalModule durante
    il forward pass e dall'IncrementalUpdater durante l'aggiornamento.

    Parameters
    ----------
    n_clusters  : K — numero di cluster (determinato da K-Means)
    desc_dim    : dimensione query descriptor (416)
    edit_dim    : dimensione edit signature (384)
    photographer_id : identificatore del fotografo
    """

    def __init__(
        self,
        n_clusters:      int,
        desc_dim:        int = 416,
        edit_dim:        int = 384,
        photographer_id: str = "unknown",
    ) -> None:
        self.n_clusters      = n_clusters
        self.desc_dim        = desc_dim
        self.edit_dim        = edit_dim
        self.photographer_id = photographer_id

        self.clusters: Dict[int, ClusterMemory] = {
            k: ClusterMemory(cluster_id=k, desc_dim=desc_dim, edit_dim=edit_dim)
            for k in range(n_clusters)
        }
        self.centroids: Optional[np.ndarray] = None   # (K, 192) float32

    # ------------------------------------------------------------------
    def add_pair(
        self,
        key:        torch.Tensor,     # (N_patches, desc_dim)
        value:      torch.Tensor,     # (N_patches, edit_dim)
        cluster_id: int,
        meta:       Optional[dict] = None,
        histogram:  Optional[torch.Tensor] = None
    ) -> None:
        """Aggiunge una coppia pre-calcolata al cluster indicato."""
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = ClusterMemory(
                cluster_id=cluster_id,
                desc_dim=self.desc_dim,
                edit_dim=self.edit_dim,
            )
        self.clusters[cluster_id].add(key, value, meta)

    # ------------------------------------------------------------------
    def get_cluster_db(
        self,
        query_hist: torch.Tensor,   # (B, 192) color histogram
        top_m:      int = 10,
        device:     str = "cpu",
    ) -> Dict[int, Optional[Dict[str, torch.Tensor]]]:
        """
        Prepara il cluster_db per il RetrievalModule.

        Per ogni cluster k, recupera le top-M coppie più simili
        alla query (usando il color histogram come proxy globale).

        Returns
        -------
        {k: {"keys": (M, N_patches, desc_dim), "values": (M, N_patches, edit_dim)}}
        """
        # Usa il primo elemento del batch come rappresentante
        # (in inference normalmente B=1; in training la similarità
        #  globale viene calcolata per patch nel retrieval module)
        q_hist = query_hist[0] if query_hist.dim() == 2 else query_hist

        cluster_db: Dict[int, Optional[Dict]] = {}
        for k, mem in self.clusters.items():
            if len(mem) == 0:
                cluster_db[k] = None
                continue
            cluster_db[k] = mem.get_top_m(q_hist, m=top_m, device=device)

        return cluster_db

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Numero totale di coppie nel database."""
        return sum(len(mem) for mem in self.clusters.values())

    def cluster_sizes(self) -> Dict[int, int]:
        return {k: len(mem) for k, mem in self.clusters.items()}

    # ------------------------------------------------------------------
    def save(self, db_root: str | Path) -> None:
        """
        Salva l'intero database su disco.

        Layout:
          db_root/
            cluster_0/ keys.npy  values.npy  meta.json
            ...
            centroids.npy
            manifest.json
        """
        db_root = Path(db_root)
        db_root.mkdir(parents=True, exist_ok=True)

        for k, mem in self.clusters.items():
            mem.save(db_root / f"cluster_{k}")

        if self.centroids is not None:
            np.save(db_root / "centroids.npy", self.centroids)

        manifest = {
            "photographer_id": self.photographer_id,
            "n_clusters":      self.n_clusters,
            "n_pairs":         len(self),
            "desc_dim":        self.desc_dim,
            "edit_dim":        self.edit_dim,
            "cluster_sizes":   self.cluster_sizes(),
            "created":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(db_root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, db_root: str | Path) -> "PhotographerDatabase":
        """Carica un database dal disco."""
        db_root = Path(db_root)

        with open(db_root / "manifest.json") as f:
            manifest = json.load(f)

        db = cls(
            n_clusters      = manifest["n_clusters"],
            desc_dim        = manifest["desc_dim"],
            edit_dim        = manifest["edit_dim"],
            photographer_id = manifest.get("photographer_id", "unknown"),
        )

        for k in range(manifest["n_clusters"]):
            cluster_dir = db_root / f"cluster_{k}"
            if cluster_dir.exists():
                db.clusters[k] = ClusterMemory.load(
                    cluster_dir, cluster_id=k,
                    desc_dim=manifest["desc_dim"],
                    edit_dim=manifest["edit_dim"],
                )

        centroids_path = db_root / "centroids.npy"
        if centroids_path.exists():
            db.centroids = np.load(centroids_path)

        return db

    # ------------------------------------------------------------------
    def memory_usage_mb(self) -> float:
        """Stima dell'occupazione RAM in MB (fp16)."""
        total_elements = 0
        for mem in self.clusters.values():
            for key in mem._keys:
                total_elements += key.size
            for val in mem._values:
                total_elements += val.size
        # fp16 = 2 bytes per elemento
        return total_elements * 2 / (1024 ** 2)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg:             dict,
        n_clusters:      int,
        photographer_id: str = "unknown",
    ) -> "PhotographerDatabase":
        return cls(
            n_clusters      = n_clusters,
            desc_dim        = cfg["descriptor"]["output_dim"],
            edit_dim        = cfg["encoder"]["embed_dim"],
            photographer_id = photographer_id,
        )
