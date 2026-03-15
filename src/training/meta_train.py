"""
meta_train.py
-------------
Funzioni per la Fase 2 (meta-training Reptile). Notebook-friendly.

Espone funzioni granulari che i notebook possono chiamare in loop:

  meta_train_step(model, task_sampler, loss_fn, cfg, device)
      → un outer step Reptile completo con M task

  evaluate_on_task(model, task, loss_fn, cfg, device)
      → valutazione del modello adattato su un task specifico

  build_meta_optimizer(model, cfg)
      → Adam per l'outer loop (usato solo per tracking del lr)

Nota: Reptile non usa un ottimizzatore standard per l'outer update
(l'update è diretto sui parametri) — l'optimizer qui serve solo
per il lr scheduling se si vuole decadere ε nel tempo.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from data.task_sampler   import Task, TaskSampler  # type: ignore[import]
from losses.composite_loss import CompositeLoss    # type: ignore[import]
from .reptile            import reptile_step
from .pretrain           import build_empty_cluster_db, val_step


# ---------------------------------------------------------------------------
# meta_train_step
# ---------------------------------------------------------------------------

def meta_train_step(
    model:        nn.Module,
    task_sampler: TaskSampler,
    loss_fn:      CompositeLoss,
    cfg:          dict,
    device:       str = "cuda",
    iteration:    int = 0,
) -> dict:
    """
    Singolo outer step di Reptile (M task → inner loops → outer update).

    Parameters
    ----------
    model        : RAGColorNet con parametri θ correnti
    task_sampler : TaskSampler che genera i task
    loss_fn      : CompositeLoss in modalità "meta"
    cfg          : config dict completo
    device       : device
    iteration    : iterazione corrente (per logging)

    Returns
    -------
    dict con:
      "meta_loss"  : loss media inner loop
      "n_tasks"    : numero task processati
      "iteration"  : iterazione corrente
    """
    rcfg  = cfg["reptile"]
    tasks = task_sampler.sample_batch(M=rcfg["n_tasks_per_batch"])

    # Aggiorna curriculum in modalità meta
    loss_fn.update_curriculum("meta", epoch=1)

    result = reptile_step(
        model    = model,
        tasks    = tasks,
        loss_fn  = loss_fn,
        n_inner  = rcfg["n_inner_steps"],
        inner_lr = rcfg["inner_lr"],
        outer_lr = rcfg["outer_lr"],
        device   = device,
        use_fp16 = cfg["hardware"]["fp16"],
    )

    result["iteration"] = iteration
    return result


# ---------------------------------------------------------------------------
# evaluate_on_task
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_task(
    model:   nn.Module,
    task:    Task,
    loss_fn: CompositeLoss,
    device:  str = "cuda",
) -> dict:
    """
    Valuta il modello sul query set di un task (senza adaptation).

    Utile per monitorare la meta-loss durante il meta-training:
    misura quanto bene θ_meta si comporta senza nessun fine-tuning
    sul nuovo task.

    Returns
    -------
    dict con loss media sul query set
    """
    model.eval()
    loss_fn.update_curriculum("meta", epoch=1)

    cluster_db = build_empty_cluster_db(model.cluster_net.n_clusters)
    query_loader = task.query_loader(batch_size=1)

    accum: Dict[str, float] = {}
    n = 0
    for batch in query_loader:
        breakdown = val_step(
            model=model, batch=batch,
            loss_fn=loss_fn, cluster_db=cluster_db, device=device,
        )
        for k, v in breakdown.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# build_meta_optimizer  (opzionale — solo per lr scheduling dell'outer step)
# ---------------------------------------------------------------------------

def build_meta_optimizer(
    model: nn.Module,
    cfg:   dict,
) -> torch.optim.Optimizer:
    """
    Adam per il tracking del lr nell'outer loop di Reptile.

    L'outer update di Reptile è manuale (non passa per .step()),
    ma avere un optimizer permette di usare i lr_scheduler PyTorch
    per decadere ε nel tempo.
    """
    rcfg = cfg["reptile"]
    return torch.optim.Adam(
        list(model.trainable_parameters()),
        lr=rcfg["outer_lr"],
    )
