"""
grade.py
--------
Inferenza su singola immagine — entry point principale per uso programmatico.

Espone la classe Grader che carica modello + database una sola volta
e può essere richiamata su più immagini senza ricaricare i pesi.

Uso tipico:
    from inference.grade import Grader

    grader = Grader(
        checkpoint = "checkpoints/photographer_01_adapted.pth",
        db_path    = "memory/photographer_01/",
        device     = "cuda",
    )

    pred, meta = grader.grade("photo.jpg")
    pred, meta = grader.grade(tensor_3hw)   # accetta anche tensor
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


# ---------------------------------------------------------------------------
# GradingResult
# ---------------------------------------------------------------------------

class GradingResult:
    """
    Risultato di una singola operazione di grading.

    Attributi
    ---------
    pred        : (3, H, W) float32 [0,1]  immagine gradata
    alpha       : (1, H, W) float32 [0,1]  confidence mask
    p           : (K,) float32             soft cluster assignment
    dominant_k  : int                      cluster dominante
    elapsed_ms  : float                    tempo di inferenza in ms
    """

    def __init__(
        self,
        pred:       torch.Tensor,
        alpha:      torch.Tensor,
        p:          torch.Tensor,
        elapsed_ms: float,
    ) -> None:
        self.pred       = pred
        self.alpha      = alpha
        self.p          = p
        self.dominant_k = int(p.argmax().item())
        self.elapsed_ms = elapsed_ms

    def __repr__(self) -> str:
        H, W = self.pred.shape[-2], self.pred.shape[-1]
        return (
            f"GradingResult("
            f"size={H}×{W}, "
            f"cluster={self.dominant_k}({self.p[self.dominant_k]:.2f}), "
            f"time={self.elapsed_ms:.0f}ms)"
        )


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """
    Grader: carica modello e database una volta, gradua N immagini.

    Pensato per uso programmatico (notebook, API, batch script).
    Per uso CLI vedere infer.py nella root del progetto.

    Parameters
    ----------
    checkpoint : path al checkpoint .pth adattato al fotografo
    db_path    : path alla directory del database del fotografo
    config     : path al config base (default: configs/base.yaml)
    device     : "cuda" | "cpu"
    fp16       : usa fp16 per l'inferenza
    top_m      : top-M immagini per cluster nel retrieval
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        db_path:    Union[str, Path],
        config:     Union[str, Path] = "configs/base.yaml",
        device:     str  = "cuda" if torch.cuda.is_available() else "cpu",
        fp16:       bool = True,
        top_m:      int  = 10,
    ) -> None:
        self.device = device
        self.fp16   = fp16
        self.top_m  = top_m

        self._cfg        = self._load_config(config)
        self._model      = self._load_model(checkpoint)
        self._database, self._faiss_mgr = self._load_database(db_path)

    # ------------------------------------------------------------------
    def grade(
        self,
        image: Union[str, Path, torch.Tensor],
        return_intermediates: bool = False,
    ) -> GradingResult:
        """
        Gradua una singola immagine.

        Parameters
        ----------
        image                : path (str/Path) o tensor (3,H,W) sRGB [0,1]
        return_intermediates : se True, il GradingResult include anche
                               I_global, I_local, guide nel campo .extras

        Returns
        -------
        GradingResult
        """
        from utils.image_io import load_tensor

        # Carica se necessario
        if not isinstance(image, torch.Tensor):
            img_tensor = load_tensor(image, device="cpu")
        else:
            img_tensor = image.cpu()

        t0  = time.perf_counter()
        out = self._forward(img_tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = GradingResult(
            pred       = out["I_out"][0].float().cpu(),
            alpha      = out["alpha"][0].float().cpu(),
            p          = out["p"][0].float().cpu(),
            elapsed_ms = elapsed_ms,
        )

        if return_intermediates:
            result.extras = {
                "I_global": out["I_global"][0].float().cpu(),
                "I_local":  out["I_local"][0].float().cpu(),
                "guide":    out["guide"][0].float().cpu(),
                "h":        out["h"][0].float().cpu(),
            }

        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, img_tensor: torch.Tensor) -> dict:
        """Forward pass con cluster_db costruito dalla query."""
        from training.pretrain import build_cluster_db_for_batch

        src = img_tensor.unsqueeze(0).to(self.device)

        h = self._model.scene_encoder.histogram(src)
        cluster_db = build_cluster_db_for_batch(
            query_hist = h,
            database   = self._database,
            faiss_mgr  = self._faiss_mgr,
            top_m      = self.top_m,
            device     = self.device,
        )

        with autocast(enabled=self.fp16):
            return self._model(src, cluster_db)

    # ------------------------------------------------------------------
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _load_model(self, checkpoint: Union[str, Path]) -> nn.Module:
        from models.rag_colornet import RAGColorNet

        ckpt   = torch.load(checkpoint, map_location=self.device)
        k_star = ckpt.get("k_star", 8)
        model  = RAGColorNet.from_config(self._cfg, n_clusters=k_star)
        model.load_state_dict(ckpt["model_state"])
        model  = model.to(self.device).eval()
        return model

    def _load_database(self, db_path: Union[str, Path]) -> tuple:
        from memory.database    import PhotographerDatabase
        from memory.faiss_index import FAISSIndexManager

        database  = PhotographerDatabase.load(db_path)
        faiss_mgr = FAISSIndexManager.from_config(
            self._cfg, n_clusters=database.n_clusters
        )
        faiss_mgr.build_from_database(database)
        return database, faiss_mgr

    # ------------------------------------------------------------------
    @property
    def n_pairs(self) -> int:
        """Numero di coppie nel database del fotografo."""
        return len(self._database)

    @property
    def k_star(self) -> int:
        """Numero di cluster K*."""
        return self._database.n_clusters

    @property
    def photographer_id(self) -> str:
        return self._database.photographer_id
