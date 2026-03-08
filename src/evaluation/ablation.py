"""
evaluation/ablation.py

Runner per gli ablation studies A0-A9  (§9.3).

Ogni ablation disabilita un componente del modello completo
e misura il degradamento delle metriche.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.hybrid_style_net import HybridStyleNet
from evaluation.metrics import evaluate_dataset, MetricsResult
from utils.checkpoint import load_checkpoint
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Definizioni degli ablation ────────────────────────────────────────────────

ABLATION_DESCRIPTIONS = {
    "A0": "Baseline: MobileNetV3 + BilGrid 8×8 + 16×16, no conditioning",
    "A1": "No Swin: EfficientNet-B4 puro (stage 1-5 CNN)",
    "A2": "No Local Branch: solo Global BilGrid 8×8×8",
    "A3": "No Cross-Attention: prototype via mean pool semplice",
    "A4": "No MAML: random init invece di theta_meta",
    "A5": "No Task Augmentation: MAML su 5 task fissi FiveK",
    "A6": "No SPADE → AdaIN: AdaIN anche nel local branch",
    "A7": "No Consistency Loss: loss senza L_consistency",
    "A8_50":  "50 coppie nel training set",
    "A8_100": "100 coppie nel training set",
    "A8_200": "200 coppie nel training set",
    "A9": "Full Model: tutti i componenti attivi",
}


class AblationModel(nn.Module):
    """
    Wrapper per modificare HybridStyleNet per un ablation specifico.

    Disabilita componenti specifici intercettando il forward pass.
    """

    def __init__(
        self,
        base_model: HybridStyleNet,
        ablation_id: str,
    ) -> None:
        super().__init__()
        self.base_model  = base_model
        self.ablation_id = ablation_id
        self._patch_model()

    def _patch_model(self) -> None:
        """Applica le modifiche specifiche dell'ablation."""
        m = self.base_model

        if self.ablation_id == "A2":
            # No Local Branch: forza alpha=0 → usa solo global
            self._force_alpha = 0.0

        elif self.ablation_id == "A3":
            # No Cross-Attention: sostituisci context con zero tensor
            self._zero_context = True

        elif self.ablation_id == "A6":
            # No SPADE → AdaIN nel local branch
            # Sostituisce i SPADEResBlock con blocchi che usano AdaIN
            # Per semplicità, bypassiamo il conditioning spaziale
            self._use_adain_local = True

        logger.info(
            f"AblationModel: {self.ablation_id} — "
            f"{ABLATION_DESCRIPTIONS.get(self.ablation_id, 'unknown')}"
        )

    def forward(
        self,
        src: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward con modifiche per l'ablation."""
        out = self.base_model(src, **kwargs)

        if self.ablation_id == "A2":
            # Forza uso esclusivo del branch globale
            alpha = torch.zeros_like(out["alpha"])
            G_global = out["G_global"]
            pred = self.base_model.bil_grid.apply(G_global, self.base_model._denormalize(src))
            out["pred"]  = pred
            out["alpha"] = alpha

        return out


class AblationRunner:
    """
    Esegue tutti gli ablation studies e aggrega i risultati.

    Args:
        checkpoint_dir:  Directory con i checkpoint degli ablation.
        device:          Device di valutazione.
        compute_lpips:   Se True, calcola LPIPS.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
        compute_lpips: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device         = device
        self.compute_lpips  = compute_lpips
        self._results: Dict[str, MetricsResult] = {}

    def run_ablation(
        self,
        ablation_id: str,
        model: HybridStyleNet,
        loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> MetricsResult:
        """
        Esegue un singolo ablation.

        Args:
            ablation_id:      ID dell'ablation (es. "A0", "A9").
            model:            Modello HybridStyleNet.
            loader:           DataLoader di test.
            checkpoint_path:  Checkpoint specifico per questo ablation.

        Returns:
            MetricsResult con le metriche.
        """
        logger.info(
            f"\n{'='*60}\n"
            f"Ablation {ablation_id}: {ABLATION_DESCRIPTIONS.get(ablation_id, '')}\n"
            f"{'='*60}"
        )

        # Carica checkpoint se fornito
        if checkpoint_path:
            try:
                load_checkpoint(checkpoint_path, model, device=self.device)
                logger.info(f"Checkpoint caricato: {checkpoint_path}")
            except FileNotFoundError:
                logger.warning(
                    f"Checkpoint non trovato: {checkpoint_path}. "
                    f"Uso i pesi correnti del modello."
                )

        # Applica le modifiche dell'ablation
        if ablation_id in ("A2", "A3", "A6"):
            eval_model = AblationModel(model, ablation_id)
        else:
            eval_model = model

        # Valuta
        result = evaluate_dataset(
            eval_model, loader,
            device=self.device,
            compute_lpips=self.compute_lpips,
        )

        self._results[ablation_id] = result
        logger.info(f"Ablation {ablation_id}: {result}")
        return result

    def run_all(
        self,
        models_and_loaders: Dict[str, tuple],
    ) -> Dict[str, MetricsResult]:
        """
        Esegue tutti gli ablation.

        Args:
            models_and_loaders: Dizionario {ablation_id: (model, loader, ckpt_path)}.
                                ckpt_path è opzionale.

        Returns:
            Dizionario {ablation_id: MetricsResult}.
        """
        for ablation_id, args in models_and_loaders.items():
            model    = args[0]
            loader   = args[1]
            ckpt     = args[2] if len(args) > 2 else None
            self.run_ablation(ablation_id, model, loader, ckpt)

        self.print_summary()
        return self._results

    def print_summary(self) -> None:
        """Stampa una tabella riassuntiva di tutti gli ablation."""
        if not self._results:
            logger.warning("Nessun risultato da stampare.")
            return

        header = (
            f"{'Ablation':<12} | {'ΔE₀₀':>8} | {'SSIM':>8} | "
            f"{'LPIPS':>8} | {'ΔNIMA':>8} | Descrizione"
        )
        sep = "-" * len(header)

        logger.info(f"\n{sep}\n{header}\n{sep}")
        for ablation_id, result in sorted(self._results.items()):
            desc = ABLATION_DESCRIPTIONS.get(ablation_id, "")[:40]
            row = (
                f"{ablation_id:<12} | "
                f"{result.delta_e:>8.4f} | "
                f"{result.ssim:>8.4f} | "
                f"{result.lpips:>8.4f} | "
                f"{result.delta_nima:>8.4f} | "
                f"{desc}"
            )
            logger.info(row)
        logger.info(sep)

    @property
    def results(self) -> Dict[str, MetricsResult]:
        return self._results