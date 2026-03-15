"""
early_stopping.py
-----------------
Early stopping con patience sul validation set.

Monitora una metrica (default: val_delta_e) e interrompe il training
se non migliora per N epoche consecutive. Salva automaticamente il
checkpoint migliore.

Usato nella fase 3 (few-shot adaptation) con:
  - holdout 20% delle coppie del fotografo
  - patience 5 epoche
  - monitor: val_delta_e (minimizzare)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitora una metrica di validazione e segnala quando fermarsi.

    Parameters
    ----------
    patience    : epoche senza miglioramento prima di fermarsi
    min_delta   : miglioramento minimo considerato significativo
    mode        : "min" (la metrica deve scendere) | "max" (deve salire)
    checkpoint_path : se fornito, salva il miglior modello qui
    verbose     : stampa messaggi di stato
    """

    def __init__(
        self,
        patience:         int   = 5,
        min_delta:        float = 0.01,
        mode:             str   = "min",
        checkpoint_path:  Optional[str | Path] = None,
        verbose:          bool  = True,
    ) -> None:
        assert mode in ("min", "max"), f"mode deve essere 'min' o 'max', got '{mode}'"

        self.patience        = patience
        self.min_delta       = min_delta
        self.mode            = mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.verbose         = verbose

        self._counter:    int   = 0
        self._best:       float = float("inf") if mode == "min" else float("-inf")
        self._best_epoch: int   = 0
        self.should_stop: bool  = False

    # ------------------------------------------------------------------
    def step(
        self,
        metric:  float,
        model:   Optional[nn.Module] = None,
        epoch:   int = 0,
        extras:  Optional[dict] = None,   # dati aggiuntivi da salvare nel ckpt
    ) -> bool:
        """
        Aggiorna lo stato dell'early stopping.

        Parameters
        ----------
        metric  : valore della metrica di validazione per questa epoca
        model   : modello da salvare se migliora (opzionale)
        epoch   : epoca corrente (per il checkpoint)
        extras  : dict aggiuntivo da includere nel checkpoint

        Returns
        -------
        improved : True se la metrica è migliorata
        """
        improved = self._is_improvement(metric)

        if improved:
            self._best       = metric
            self._best_epoch = epoch
            self._counter    = 0

            if model is not None and self.checkpoint_path is not None:
                self._save_checkpoint(model, epoch, metric, extras)

            if self.verbose:
                print(
                    f"  ✓ Val metric migliorata: {metric:.4f} "
                    f"(epoch {epoch})"
                )
        else:
            self._counter += 1
            if self.verbose:
                print(
                    f"  · No improvement ({self._counter}/{self.patience}) "
                    f"— best: {self._best:.4f} @ epoch {self._best_epoch}"
                )

            if self._counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(
                        f"  ✗ Early stopping triggered dopo "
                        f"{self.patience} epoche senza miglioramento."
                    )

        return improved

    # ------------------------------------------------------------------
    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self._best - self.min_delta
        else:
            return metric > self._best + self.min_delta

    # ------------------------------------------------------------------
    def _save_checkpoint(
        self,
        model:   nn.Module,
        epoch:   int,
        metric:  float,
        extras:  Optional[dict],
    ) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch":        epoch,
            "metric":       metric,
            "model_state":  model.state_dict(),
        }
        if extras:
            payload.update(extras)

        torch.save(payload, self.checkpoint_path)

    # ------------------------------------------------------------------
    def load_best(self, model: nn.Module) -> dict:
        """
        Carica il miglior checkpoint nel modello.

        Returns
        -------
        checkpoint dict (contiene epoch, metric, e eventuali extras)
        """
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint non trovato: {self.checkpoint_path}"
            )

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        if self.verbose:
            print(
                f"  Caricato best checkpoint: epoch {ckpt['epoch']}, "
                f"metric {ckpt['metric']:.4f}"
            )
        return ckpt

    # ------------------------------------------------------------------
    @property
    def best_metric(self) -> float:
        return self._best

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def counter(self) -> int:
        return self._counter

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Ripristina lo stato — utile tra fasi di training diverse."""
        self._counter    = 0
        self._best       = float("inf") if self.mode == "min" else float("-inf")
        self._best_epoch = 0
        self.should_stop = False

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg:              dict,
        checkpoint_path:  Optional[str | Path] = None,
    ) -> "EarlyStopping":
        escfg = cfg.get("early_stopping", {})
        return cls(
            patience        = escfg.get("patience",   5),
            min_delta       = escfg.get("min_delta",  0.01),
            mode            = escfg.get("mode",       "min"),
            checkpoint_path = checkpoint_path,
            verbose         = True,
        )
