"""
benchmark.py
------------
Confronto quantitativo con i metodi dello stato dell'arte.

Espone una classe BenchmarkRunner che valuta RAG-ColorNet
e (opzionalmente) altri metodi sullo stesso test set,
producendo una tabella comparativa.

Metodi baseline riferimento (dalla documentazione):
  HDRNet (2017)       — bilateral grid, nessuna personalizzazione
  Deep Preset (2020)  — preset-based, parzialmente personalizzabile
  CSRNet (2020)       — color style transfer leggero
  PromptIR (2023)     — prompt-based, few-shot parziale

I metodi baseline sono chiamati tramite un'interfaccia uniforme
BaselineModel che i notebook possono implementare con i loro checkpoint.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_all     # type: ignore[import]


# ---------------------------------------------------------------------------
# BaselineModel  (interfaccia)
# ---------------------------------------------------------------------------

class BaselineModel(ABC):
    """
    Interfaccia uniforme per i metodi baseline.

    I notebook implementano questa classe per ogni metodo
    da confrontare, caricando i relativi checkpoint.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome del metodo (es. 'HDRNet', 'Deep Preset')."""
        ...

    @abstractmethod
    def grade(self, src: torch.Tensor) -> torch.Tensor:
        """
        Applica il grading all'immagine sorgente.

        Parameters
        ----------
        src : (B,3,H,W) sRGB [0,1]

        Returns
        -------
        pred : (B,3,H,W) sRGB [0,1]
        """
        ...


# ---------------------------------------------------------------------------
# RAGColorNetWrapper  (wrapper per l'interfaccia uniforme)
# ---------------------------------------------------------------------------

class RAGColorNetWrapper(BaselineModel):
    """
    Wrapper di RAGColorNet per il benchmark.

    Parameters
    ----------
    model      : RAGColorNet già adattato
    cluster_db : database del fotografo
    device     : device
    """

    def __init__(
        self,
        model:      nn.Module,
        cluster_db: dict,
        device:     str = "cuda",
        label:      str = "RAG-ColorNet",
    ) -> None:
        self._model      = model.eval()
        self._cluster_db = cluster_db
        self._device     = device
        self._label      = label

    @property
    def name(self) -> str:
        return self._label

    @torch.no_grad()
    def grade(self, src: torch.Tensor) -> torch.Tensor:
        src = src.to(self._device)
        out = self._model(src, self._cluster_db)
        return out["I_out"].cpu()


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Valuta più metodi sullo stesso test set e produce una tabella comparativa.

    Parameters
    ----------
    test_loader  : DataLoader del test set del fotografo
    device       : device
    compute_nima : se True calcola anche il punteggio NIMA (lento)
    """

    def __init__(
        self,
        test_loader:  DataLoader,
        device:       str  = "cuda",
        compute_nima: bool = False,
    ) -> None:
        self.test_loader  = test_loader
        self.device       = device
        self.compute_nima = compute_nima
        self._results: Dict[str, Dict[str, float]] = {}
        self._lpips_model = None

    # ------------------------------------------------------------------
    def add_method(
        self,
        method: BaselineModel,
    ) -> Dict[str, float]:
        """
        Valuta un metodo sul test set e memorizza i risultati.

        Returns
        -------
        dict con metriche: delta_e, ssim_L, lpips, inference_ms
        """
        print(f"Valutazione: {method.name} ...")

        all_preds, all_tgts, all_srcs = [], [], []
        inference_times: List[float] = []

        for batch in self.test_loader:
            src = batch["src"]
            tgt = batch["tgt"]

            t0   = time.perf_counter()
            pred = method.grade(src)
            inference_times.append((time.perf_counter() - t0) * 1000)

            all_preds.append(pred)
            all_tgts.append(tgt)
            all_srcs.append(src)

        pred_all = torch.cat(all_preds).to(self.device)
        tgt_all  = torch.cat(all_tgts).to(self.device)
        src_all  = torch.cat(all_srcs).to(self.device)

        # Carica LPIPS una sola volta
        if self._lpips_model is None:
            self._lpips_model = _try_load_lpips(self.device)

        metrics = compute_all(
            pred         = pred_all,
            tgt          = tgt_all,
            src          = src_all,
            lpips_model  = self._lpips_model,
            compute_nima = self.compute_nima,
        )

        metrics["inference_ms"] = sum(inference_times) / len(inference_times)
        metrics["method"]       = method.name
        self._results[method.name] = metrics

        self._print_row(method.name, metrics)
        return metrics

    # ------------------------------------------------------------------
    def compare_all(
        self,
        methods: List[BaselineModel],
    ) -> Dict[str, Dict[str, float]]:
        """
        Valuta tutti i metodi in sequenza.

        Returns
        -------
        {method_name: metrics_dict}
        """
        for method in methods:
            self.add_method(method)
        return self._results

    # ------------------------------------------------------------------
    def summary_table(self) -> str:
        """
        Tabella comparativa in formato testo.

        Formato:
          Metodo           ΔE₀₀   SSIM_L   LPIPS  ms/img
          ─────────────────────────────────────────────────
          HDRNet           6.500   0.960   0.120    15
          RAG-ColorNet     2.900   0.975   0.065  1400
        """
        if not self._results:
            return "Nessun risultato. Esegui add_method() o compare_all() prima."

        header = (
            f"{'Metodo':<22} {'ΔE₀₀':>7} {'SSIM_L':>8} "
            f"{'LPIPS':>7} {'ms/img':>8}"
        )
        sep  = "─" * 56
        rows = [header, sep]

        # Ordina per ΔE₀₀ crescente
        sorted_results = sorted(
            self._results.items(),
            key=lambda x: x[1].get("delta_e", 99),
        )

        for name, r in sorted_results:
            rows.append(
                f"{name:<22} {r.get('delta_e', 0):>7.3f} "
                f"{r.get('ssim_L', 0):>8.4f} "
                f"{r.get('lpips', 0):>7.4f} "
                f"{r.get('inference_ms', 0):>8.0f}"
            )

        if self.compute_nima and any("nima_delta" in r for r in self._results.values()):
            rows.append(sep)
            rows.append(f"{'Metodo':<22} {'ΔNIMA':>7}")
            rows.append(sep)
            for name, r in sorted_results:
                if "nima_delta" in r:
                    rows.append(f"{name:<22} {r['nima_delta']:>7.3f}")

        return "\n".join(rows)

    # ------------------------------------------------------------------
    @staticmethod
    def _print_row(name: str, metrics: Dict) -> None:
        de = metrics.get("delta_e", 0)
        ss = metrics.get("ssim_L", 0)
        lp = metrics.get("lpips", 0)
        ms = metrics.get("inference_ms", 0)
        print(f"  {name:<20} ΔE₀₀={de:.3f}  SSIM={ss:.4f}  LPIPS={lp:.4f}  {ms:.0f}ms")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _try_load_lpips(device: str):
    """Carica il modello LPIPS se disponibile."""
    try:
        import lpips
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except ImportError:
        return None
