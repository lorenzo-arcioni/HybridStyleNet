"""
adapt.py
--------
Funzioni per la Fase 3 (few-shot adaptation). Notebook-friendly.

Espone la logica dei due step di adaptation in funzioni granulari:

  setup_adaptation(model, database, faiss_mgr, cfg, device)
      → prepara il modello per l'adaptation (freeze/unfreeze, preprocessing)
        restituisce (optimizer_step1, scaler)

  adaptation_step1_epoch(model, loader, ...)
      → un'epoca con freeze parziale (ClusterNet + proiezioni + ultimi layer)

  adaptation_step2_epoch(model, loader, ...)
      → un'epoca con full fine-tuning

  switch_to_step2(model, optimizer, cfg)
      → sblocca tutti i parametri trainable, costruisce nuovo optimizer

Flusso tipico in un notebook:
  opt1, scaler = setup_adaptation(model, db, faiss_mgr, cfg)
  for epoch in range(1, 11):
      train_metrics = adaptation_step1_epoch(model, train_loader, opt1, ...)
      val_metrics   = validate(model, val_loader, ...)
      if early_stopper.step(val_metrics["loss/total"], model, epoch):
          ...

  opt2 = switch_to_step2(model, opt1, cfg)
  for epoch in range(11, 31):
      train_metrics = adaptation_step2_epoch(model, train_loader, opt2, ...)
      ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from losses.composite_loss import CompositeLoss                 # type: ignore[import]
from memory.database       import PhotographerDatabase          # type: ignore[import]
from memory.faiss_index    import FAISSIndexManager             # type: ignore[import]
from memory.incremental_update import IncrementalUpdater        # type: ignore[import]
from .pretrain             import run_epoch, build_optimizer, build_cluster_db_for_batch
from .lr_scheduler         import build_scheduler


# ---------------------------------------------------------------------------
# setup_adaptation
# ---------------------------------------------------------------------------

def setup_adaptation(
    model:        nn.Module,
    cfg:          dict,
    device:       str = "cuda",
) -> Tuple[torch.optim.Optimizer, GradScaler]:
    """
    Prepara il modello per lo Step 1 dell'adaptation:
      - Carica il checkpoint θ_meta
      - Imposta i moduli trainable per lo Step 1 (freeze parziale)
      - Costruisce optimizer e scaler

    Parameters
    ----------
    model   : RAGColorNet (già istanziato con il K* del fotografo)
    cfg     : config dict completo (base + photographer merged)
    device  : device

    Returns
    -------
    optimizer : AdamW per lo Step 1
    scaler    : GradScaler fp16
    """
    # Carica θ_meta
    init_ckpt = cfg.get("init_checkpoint")
    if init_ckpt and Path(init_ckpt).exists():
        ckpt = torch.load(init_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        # Carica solo i parametri compatibili (K potrebbe essere diverso)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Parametri non trovati nel checkpoint: {len(missing)}")
        print(f"  Caricato θ_meta da: {init_ckpt}")
    else:
        print("  Nessun checkpoint θ_meta trovato — partenza da random init.")

    model.to(device)

    # Freeze parziale per Step 1
    model.set_adaptation_mode(step=1)

    optimizer = build_optimizer(model, cfg, phase="adapt_step1")
    scaler    = GradScaler(enabled=cfg["hardware"]["fp16"])

    n_train = model.count_trainable_params()
    print(f"  Step 1 — parametri trainable: {n_train:,}")

    return optimizer, scaler


# ---------------------------------------------------------------------------
# switch_to_step2
# ---------------------------------------------------------------------------

def switch_to_step2(
    model:     nn.Module,
    cfg:       dict,
    device:    str = "cuda",
) -> torch.optim.Optimizer:
    """
    Sblocca tutti i parametri trainable e costruisce l'optimizer
    per lo Step 2 (full fine-tuning).

    Da chiamare dopo le epoche dello Step 1.

    Returns
    -------
    nuovo AdamW per lo Step 2
    """
    model.set_adaptation_mode(step=2)
    optimizer = build_optimizer(model, cfg, phase="adapt_step2")

    n_train = model.count_trainable_params()
    print(f"  Step 2 — parametri trainable: {n_train:,}")

    return optimizer


# ---------------------------------------------------------------------------
# adaptation_step1_epoch
# ---------------------------------------------------------------------------

def adaptation_step1_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    loss_fn:     CompositeLoss,
    scaler:      GradScaler,
    database:    PhotographerDatabase,
    faiss_mgr:   FAISSIndexManager,
    cfg:         dict,
    epoch:       int,
    device:      str = "cuda",
    cluster_labels_map: Optional[Dict[int, int]] = None,
) -> dict:
    """
    Un'epoca di training dello Step 1 dell'adaptation.

    Costruisce il cluster_db per ogni batch usando FAISS,
    poi delega a run_epoch per la logica di training.
    """
    top_m = cfg["retrieval"]["top_m"]

    # Wrapper del loader che inietta il cluster_db nel batch
    metrics = _run_adaptation_epoch(
        model      = model,
        loader     = loader,
        optimizer  = optimizer,
        loss_fn    = loss_fn,
        scaler     = scaler,
        database   = database,
        faiss_mgr  = faiss_mgr,
        top_m      = top_m,
        phase      = "adapt",
        epoch      = epoch,
        device     = device,
        cluster_labels_map = cluster_labels_map,
        is_train   = True,
    )
    return metrics


# ---------------------------------------------------------------------------
# adaptation_step2_epoch
# ---------------------------------------------------------------------------

def adaptation_step2_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    loss_fn:     CompositeLoss,
    scaler:      GradScaler,
    database:    PhotographerDatabase,
    faiss_mgr:   FAISSIndexManager,
    cfg:         dict,
    epoch:       int,
    device:      str = "cuda",
    cluster_labels_map: Optional[Dict[int, int]] = None,
) -> dict:
    """Un'epoca di training dello Step 2 (full fine-tuning)."""
    top_m = cfg["retrieval"]["top_m"]
    return _run_adaptation_epoch(
        model      = model,
        loader     = loader,
        optimizer  = optimizer,
        loss_fn    = loss_fn,
        scaler     = scaler,
        database   = database,
        faiss_mgr  = faiss_mgr,
        top_m      = top_m,
        phase      = "adapt",
        epoch      = epoch,
        device     = device,
        cluster_labels_map = cluster_labels_map,
        is_train   = True,
    )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    loss_fn:   CompositeLoss,
    database:  PhotographerDatabase,
    faiss_mgr: FAISSIndexManager,
    cfg:       dict,
    epoch:     int,
    device:    str = "cuda",
) -> dict:
    """Validazione su un loader con il database del fotografo."""
    top_m = cfg["retrieval"]["top_m"]
    return _run_adaptation_epoch(
        model      = model,
        loader     = loader,
        optimizer  = None,
        loss_fn    = loss_fn,
        scaler     = None,
        database   = database,
        faiss_mgr  = faiss_mgr,
        top_m      = top_m,
        phase      = "adapt",
        epoch      = epoch,
        device     = device,
        is_train   = False,
    )


# ---------------------------------------------------------------------------
# _run_adaptation_epoch  (implementazione comune)
# ---------------------------------------------------------------------------

def _run_adaptation_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   Optional[torch.optim.Optimizer],
    loss_fn:     CompositeLoss,
    scaler:      Optional[GradScaler],
    database:    PhotographerDatabase,
    faiss_mgr:   FAISSIndexManager,
    top_m:       int,
    phase:       str,
    epoch:       int,
    device:      str,
    cluster_labels_map: Optional[Dict[int, int]] = None,
    is_train:    bool = True,
) -> dict:
    """Logica comune per train e val nell'adaptation."""
    from .pretrain import train_step, val_step

    weights = loss_fn.update_curriculum(phase, epoch)

    if is_train:
        model.train()
        model.scene_encoder.backbone.eval()
    else:
        model.eval()

    accum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        src = batch["src"].to(device)

        # Costruisce cluster_db per questo batch
        with torch.no_grad():
            h = model.scene_encoder.histogram(src)   # (B, 192)

        cluster_db = build_cluster_db_for_batch(
            query_hist = h,
            database   = database,
            faiss_mgr  = faiss_mgr,
            top_m      = top_m,
            device     = device,
        )

        # Recupera cluster labels se disponibili
        cluster_labels = None
        if cluster_labels_map is not None and weights.cluster > 0:
            idxs = batch.get("idx")
            if idxs is not None:
                cluster_labels = torch.tensor(
                    [cluster_labels_map.get(int(i), 0) for i in idxs],
                    dtype=torch.long, device=device,
                )

        if is_train and optimizer is not None and scaler is not None:
            breakdown = train_step(
                model          = model,
                batch          = batch,
                optimizer      = optimizer,
                loss_fn        = loss_fn,
                scaler         = scaler,
                cluster_db     = cluster_db,
                device         = device,
                cluster_labels = cluster_labels,
            )
        else:
            breakdown = val_step(
                model      = model,
                batch      = batch,
                loss_fn    = loss_fn,
                cluster_db = cluster_db,
                device     = device,
            )

        for k, v in breakdown.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}
    metrics["n_batches"] = n_batches
    return metrics
