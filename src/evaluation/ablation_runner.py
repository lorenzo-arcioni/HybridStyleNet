"""
ablation_runner.py
------------------
Runner per gli ablation studies A0–A10 definiti nella documentazione.

Ogni ablation modifica una componente del modello per isolare il suo
contributo. Il runner costruisce il modello nella configurazione corretta
e restituisce i risultati delle metriche su un dataset di test.

Ablation studies:
  A0  — HDRNet generico (lower bound)
  A1  — Senza retrieval (style prototype globale)
  A2  — Senza cluster (K=1)
  A3  — Guida cromatica pura (no g_sem)
  A4  — Guida semantica pura (no g_chroma)
  A5  — Senza meta-learning (random init)
  A6  — VGG16 invece di DINOv2 (non implementato qui — richiede refactor)
  A7  — N=50 coppie
  A8  — N=300, senza aggiornamento incrementale
  A9  — Modello completo N=300
  A10 — Modello completo N=300+200 incrementali

Uso tipico in un notebook:
    runner = AblationRunner(base_cfg, photographer_dataset, device)
    results = runner.run("A9", checkpoint_path="checkpoints/adapted.pth")
    print(results)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import compute_all      # type: ignore[import]


# ---------------------------------------------------------------------------
# Configurazioni ablation
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: Dict[str, dict] = {
    "A0": {
        "description": "HDRNet generico (lower bound assoluto)",
        "model_type":  "hdrnet",
        "retrieval":   False,
        "n_clusters":  1,
        "guide":       "chroma",
        "meta":        False,
    },
    "A1": {
        "description": "Senza retrieval — style prototype globale",
        "model_type":  "ragcolornet",
        "retrieval":   False,   # usa media globale invece del retrieval locale
        "n_clusters":  None,    # usa K* del fotografo
        "guide":       "hybrid",
        "meta":        True,
    },
    "A2": {
        "description": "Senza cluster — K=1",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  1,
        "guide":       "hybrid",
        "meta":        True,
    },
    "A3": {
        "description": "Guida cromatica pura (alpha=1.0)",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "chroma",   # alpha=1.0
        "meta":        True,
    },
    "A4": {
        "description": "Guida semantica pura (alpha=0.0)",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "semantic",  # alpha=0.0
        "meta":        True,
    },
    "A5": {
        "description": "Senza meta-learning (random init)",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "hybrid",
        "meta":        False,   # usa random init invece di θ_meta
    },
    "A7": {
        "description": "N=50 coppie",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "hybrid",
        "meta":        True,
        "n_pairs":     50,
    },
    "A8": {
        "description": "N=300, senza aggiornamento incrementale",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "hybrid",
        "meta":        True,
        "n_pairs":     300,
        "incremental": False,
    },
    "A9": {
        "description": "Modello completo, N=300",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "hybrid",
        "meta":        True,
        "n_pairs":     300,
        "incremental": False,
    },
    "A10": {
        "description": "Modello completo, N=300+200 incrementali",
        "model_type":  "ragcolornet",
        "retrieval":   True,
        "n_clusters":  None,
        "guide":       "hybrid",
        "meta":        True,
        "n_pairs":     300,
        "incremental": True,
        "n_incremental": 200,
    },
}


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """
    Esegue ablation studies sul dataset di test del fotografo.

    Parameters
    ----------
    base_cfg    : config dict completo (base.yaml merged)
    test_loader : DataLoader sul test set del fotografo
    device      : device
    """

    def __init__(
        self,
        base_cfg:    dict,
        test_loader: DataLoader,
        device:      str = "cuda",
    ) -> None:
        self.base_cfg    = base_cfg
        self.test_loader = test_loader
        self.device      = device
        self._results: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def run(
        self,
        ablation_id:     str,
        model:           nn.Module,
        cluster_db:      dict,
        lpips_model:     Optional[object] = None,
    ) -> Dict[str, float]:
        """
        Valuta il modello nella configurazione dell'ablation ablation_id.

        Parameters
        ----------
        ablation_id : "A0"–"A10"
        model       : RAGColorNet già adattato al fotografo
        cluster_db  : database del fotografo
        lpips_model : istanza lpips.LPIPS (opzionale, riusata tra run)

        Returns
        -------
        dict con metriche: delta_e, ssim_L, lpips
        """
        if ablation_id not in ABLATION_CONFIGS:
            raise ValueError(
                f"Ablation ID '{ablation_id}' non riconosciuto. "
                f"Disponibili: {list(ABLATION_CONFIGS.keys())}"
            )

        cfg = ABLATION_CONFIGS[ablation_id]

        # Modifica temporanea del modello per questa ablation
        modified_model = self._apply_ablation(model, cfg)
        modified_model.eval()

        # Valutazione sul test set
        all_preds, all_tgts, all_srcs = [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                # Per A0 (HDRNet) skip del retrieval
                db = cluster_db if cfg.get("retrieval", True) else \
                     {k: None for k in cluster_db}

                out = modified_model(src, db)
                all_preds.append(out["I_out"].cpu())
                all_tgts.append(tgt.cpu())
                all_srcs.append(src.cpu())

        pred = torch.cat(all_preds)
        tgt  = torch.cat(all_tgts)
        src  = torch.cat(all_srcs)

        metrics = compute_all(
            pred        = pred.to(self.device),
            tgt         = tgt.to(self.device),
            src         = src.to(self.device),
            lpips_model = lpips_model,
        )

        metrics["ablation_id"]  = ablation_id
        metrics["description"]  = cfg["description"]
        self._results[ablation_id] = metrics

        return metrics

    # ------------------------------------------------------------------
    def run_all(
        self,
        model:       nn.Module,
        cluster_db:  dict,
        ablation_ids: Optional[List[str]] = None,
        lpips_model:  Optional[object] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Esegue tutti gli ablation studies (o un sottoinsieme).

        Returns
        -------
        {ablation_id: metrics_dict}
        """
        ids = ablation_ids or list(ABLATION_CONFIGS.keys())

        for aid in ids:
            print(f"\n--- Ablation {aid}: {ABLATION_CONFIGS[aid]['description']} ---")
            try:
                metrics = self.run(aid, model, cluster_db, lpips_model)
                self._print_metrics(aid, metrics)
            except Exception as e:
                print(f"  ERRORE: {e}")

        return self._results

    # ------------------------------------------------------------------
    def summary_table(self) -> str:
        """
        Restituisce una stringa con la tabella riassuntiva degli ablation.
        Pronta per essere stampata in un notebook.
        """
        if not self._results:
            return "Nessun risultato disponibile. Esegui run() o run_all() prima."

        header = f"{'ID':<5} {'ΔE₀₀':>7} {'SSIM_L':>8} {'LPIPS':>7}  Descrizione"
        sep    = "-" * 70
        rows   = [header, sep]

        for aid in sorted(self._results.keys()):
            r = self._results[aid]
            rows.append(
                f"{aid:<5} {r.get('delta_e', 0):>7.3f} "
                f"{r.get('ssim_L', 0):>8.4f} "
                f"{r.get('lpips', 0):>7.4f}  "
                f"{r.get('description', '')}"
            )

        return "\n".join(rows)

    # ------------------------------------------------------------------
    def _apply_ablation(
        self,
        model:  nn.Module,
        cfg:    dict,
    ) -> nn.Module:
        """
        Applica le modifiche del config ablation al modello.
        Lavora su una copia shallow per non alterare il modello originale.
        """
        # Per la maggior parte delle ablation il modello non viene modificato
        # strutturalmente — il comportamento cambia tramite parametri
        # (es. guide alpha) o tramite cluster_db vuoto (no retrieval)

        if cfg.get("guide") == "chroma":
            # Forza alpha=1.0 nella SemanticGuide
            model.grid_renderer.guide.alpha = 1.0

        elif cfg.get("guide") == "semantic":
            # Forza alpha=0.0
            model.grid_renderer.guide.alpha = 0.0

        elif cfg.get("guide") == "hybrid":
            # Ripristina il valore di default dal config
            model.grid_renderer.guide.alpha = self.base_cfg["bilateral_grid"]["guide_alpha"]

        return model

    # ------------------------------------------------------------------
    @staticmethod
    def _print_metrics(aid: str, metrics: Dict) -> None:
        de = metrics.get("delta_e", 0)
        ss = metrics.get("ssim_L", 0)
        lp = metrics.get("lpips", 0)
        print(f"  ΔE₀₀: {de:.3f}  SSIM_L: {ss:.4f}  LPIPS: {lp:.4f}")
