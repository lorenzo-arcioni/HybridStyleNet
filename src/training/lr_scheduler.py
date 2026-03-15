"""
lr_scheduler.py
---------------
Learning rate scheduler con warmup lineare + cosine annealing.

Espone una funzione factory build_scheduler() che costruisce il
scheduler appropriato dal config dict, usabile dai notebook e
dai moduli di training.

Il warmup lineare stabilizza il training nelle prime epoche
(particolarmente importante con fp16 e gradienti piccoli).
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


# ---------------------------------------------------------------------------
# WarmupCosineScheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup seguito da cosine annealing.

    Epoche 1..warmup_epochs    : lr cresce linearmente da 0 a base_lr
    Epoche warmup+1..max_epochs: lr scende con cosine fino a eta_min

    Parameters
    ----------
    optimizer     : ottimizzatore PyTorch
    warmup_epochs : durata del warmup (epoche)
    max_epochs    : durata totale del training (epoche)
    eta_min       : lr minimo al termine del cosine decay
    last_epoch    : epoch di partenza (-1 = inizio)
    """

    def __init__(
        self,
        optimizer:     Optimizer,
        warmup_epochs: int,
        max_epochs:    int,
        eta_min:       float = 1e-7,
        last_epoch:    int   = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    # ------------------------------------------------------------------
    def get_lr(self) -> List[float]:
        epoch = self.last_epoch   # 0-based internamente

        if epoch < self.warmup_epochs:
            # Warmup lineare: 0 → base_lr
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * scale for base_lr in self.base_lrs]

        # Cosine annealing
        progress = (epoch - self.warmup_epochs) / max(
            self.max_epochs - self.warmup_epochs, 1
        )
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# StepSchedulerWrapper (per l'inner loop Reptile)
# ---------------------------------------------------------------------------

class ConstantLR(_LRScheduler):
    """LR costante — placeholder per l'inner loop SGD di Reptile."""

    def get_lr(self) -> List[float]:
        return list(self.base_lrs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: Optimizer,
    cfg:       dict,
    phase:     str,          # "pretrain" | "meta" | "adapt_step1" | "adapt_step2"
    n_epochs:  int,
) -> _LRScheduler:
    """
    Costruisce il scheduler appropriato dal config e dalla fase.

    Parameters
    ----------
    optimizer : ottimizzatore da schedulare
    cfg       : config dict completo (base.yaml merged)
    phase     : fase di training
    n_epochs  : numero totale di epoche per questa fase

    Returns
    -------
    scheduler PyTorch pronto all'uso
    """
    scfg = cfg.get("scheduler", {})
    warmup = scfg.get("warmup_epochs", 2)
    eta_min = scfg.get("eta_min", 1e-7)

    if phase in ("pretrain", "adapt_step2"):
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs = warmup,
            max_epochs    = n_epochs,
            eta_min       = eta_min,
        )

    elif phase == "adapt_step1":
        # Step 1: warmup breve poi cosine su 10 epoche
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs = 1,
            max_epochs    = n_epochs,
            eta_min       = eta_min,
        )

    elif phase == "meta":
        # Nel meta-training si usa un lr costante gestito dall'outer loop
        return ConstantLR(optimizer)

    else:
        raise ValueError(f"Phase non riconosciuta: '{phase}'")
