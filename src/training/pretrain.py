"""
pretrain.py
-----------
Funzioni riutilizzabili per la Fase 1 (pre-training) e Fase 3
(few-shot adaptation). Progettate per essere chiamate dai notebook.

NON contiene un training loop monolitico — espone funzioni granulari:

  train_step(model, batch, optimizer, loss_fn, scaler, cluster_db)
      → LossBreakdown (un singolo batch)

  val_step(model, batch, loss_fn, cluster_db)
      → LossBreakdown (no grad)

  run_epoch(model, loader, optimizer, loss_fn, scaler, cluster_db, ...)
      → dict con metriche medie dell'epoca

  build_optimizer(model, cfg, phase, lr_override)
      → AdamW configurato per la fase corrente

  build_cluster_db_for_batch(batch_indices, database, faiss_mgr, query_hists)
      → cluster_db pronto per il forward pass

I notebook orchestrano chiamando run_epoch in loop con early stopping,
logging e visualizzazione intercalate.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from losses.composite_loss import CompositeLoss, LossBreakdown  # type: ignore[import]
from memory.database       import PhotographerDatabase           # type: ignore[import]
from memory.faiss_index    import FAISSIndexManager              # type: ignore[import]


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model:       nn.Module,
    cfg:         dict,
    phase:       str,           # "pretrain" | "adapt_step1" | "adapt_step2"
    lr_override: Optional[float] = None,
) -> torch.optim.Optimizer:
    """
    Costruisce un ottimizzatore AdamW per la fase indicata.

    Parameters
    ----------
    model       : RAGColorNet
    cfg         : config dict completo
    phase       : determina il lr di default se lr_override è None
    lr_override : sovrascrive il lr dal config

    Returns
    -------
    AdamW configurato sui soli parametri trainable del modello
    """
    ocfg = cfg["optimizer"]

    # Learning rate per fase
    if lr_override is not None:
        lr = lr_override
    elif phase == "pretrain":
        lr = cfg.get("training", {}).get("lr", 1e-4)
    elif phase == "adapt_step1":
        lr = cfg["adaptation"]["lr_step1"]
    elif phase == "adapt_step2":
        lr = cfg["adaptation"]["lr_step2"]
    else:
        lr = 1e-4

    # Parametri trainable — esclude DINOv2 frozen
    params = list(model.trainable_parameters())

    return torch.optim.AdamW(
        params,
        lr           = lr,
        betas        = tuple(ocfg.get("betas", [0.9, 0.999])),
        eps          = ocfg.get("eps",  1e-8),
        weight_decay = ocfg.get("weight_decay", 2e-3),
    )


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def train_step(
    model:       nn.Module,
    batch:       dict,
    optimizer:   torch.optim.Optimizer,
    loss_fn:     CompositeLoss,
    scaler:      GradScaler,
    cluster_db:  dict,
    device:      str  = "cuda",
    grad_clip:   float = 1.0,
    cluster_labels: Optional[torch.Tensor] = None,
    edit_target:    Optional[torch.Tensor] = None,
) -> LossBreakdown:
    """
    Singolo step di training su un batch.

    Parameters
    ----------
    model         : RAGColorNet in training mode
    batch         : {"src": Tensor, "tgt": Tensor, ...}
    optimizer     : ottimizzatore
    loss_fn       : CompositeLoss con curriculum aggiornato
    scaler        : GradScaler per fp16
    cluster_db    : {k: {"keys": ..., "values": ...}} per il retrieval
    device        : device
    grad_clip     : norma massima per gradient clipping
    cluster_labels: (B,) per ClusterAssignmentLoss (opzionale)
    edit_target   : (B, d_r, n_h, n_w) per RetrievalQualityLoss (opzionale)

    Returns
    -------
    LossBreakdown con total e tutti i termini individuali
    """
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)

    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=scaler.is_enabled()):
        model_out = model(src, cluster_db)
        breakdown = loss_fn(
            model_output    = model_out,
            batch           = {"src": src, "tgt": tgt},
            cluster_labels  = cluster_labels,
            edit_target     = edit_target,
        )

    scaler.scale(breakdown.total).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return breakdown


# ---------------------------------------------------------------------------
# val_step
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_step(
    model:      nn.Module,
    batch:      dict,
    loss_fn:    CompositeLoss,
    cluster_db: dict,
    device:     str = "cuda",
) -> LossBreakdown:
    """
    Singolo step di validazione (no grad, no optimizer).

    Parameters
    ----------
    model      : RAGColorNet in eval mode
    batch      : {"src": Tensor, "tgt": Tensor}
    loss_fn    : CompositeLoss
    cluster_db : database del fotografo
    device     : device

    Returns
    -------
    LossBreakdown
    """
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)

    with autocast(enabled=True):
        model_out = model(src, cluster_db)
        breakdown = loss_fn(
            model_output = model_out,
            batch        = {"src": src, "tgt": tgt},
        )

    return breakdown


# ---------------------------------------------------------------------------
# run_epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    loss_fn:     CompositeLoss,
    cluster_db:  dict,
    device:      str  = "cuda",
    optimizer:   Optional[torch.optim.Optimizer] = None,
    scaler:      Optional[GradScaler]            = None,
    grad_clip:   float = 1.0,
    phase:       str   = "pretrain",
    epoch:       int   = 1,
    cluster_labels_map: Optional[Dict[int, int]] = None,
) -> dict:
    """
    Esegue un'epoca completa di training o validazione.

    Se optimizer è None, esegue solo la validazione (no grad).
    Chiama loss_fn.update_curriculum(phase, epoch) automaticamente.

    Parameters
    ----------
    model       : RAGColorNet
    loader      : DataLoader del dataset corrente
    loss_fn     : CompositeLoss
    cluster_db  : database del fotografo (può essere vuoto in pretrain)
    device      : device
    optimizer   : se None → modalità validazione
    scaler      : GradScaler (creato qui se None in modalità training)
    grad_clip   : norma massima per gradient clipping
    phase       : fase corrente (aggiorna il curriculum)
    epoch       : epoca corrente (aggiorna il curriculum)
    cluster_labels_map : {pair_idx: cluster_id} per ClusterAssignmentLoss

    Returns
    -------
    dict con metriche medie:
      "loss/total", "loss/delta_e", ..., "n_batches"
    """
    is_train = optimizer is not None

    # Aggiorna il curriculum per questa epoca
    weights = loss_fn.update_curriculum(phase, epoch)

    if is_train:
        model.train()
        model.scene_encoder.backbone.eval()       # DINOv2 always eval
        if scaler is None:
            scaler = GradScaler(enabled=True)
    else:
        model.eval()

    # Accumulatori
    accum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        # Recupera cluster labels se disponibili
        cluster_labels = None
        if cluster_labels_map is not None and weights.cluster > 0:
            idxs = batch.get("idx")
            if idxs is not None:
                cluster_labels = torch.tensor(
                    [cluster_labels_map.get(int(i), 0) for i in idxs],
                    dtype=torch.long, device=device,
                )

        if is_train:
            breakdown = train_step(
                model           = model,
                batch           = batch,
                optimizer       = optimizer,
                loss_fn         = loss_fn,
                scaler          = scaler,
                cluster_db      = cluster_db,
                device          = device,
                grad_clip       = grad_clip,
                cluster_labels  = cluster_labels,
            )
        else:
            breakdown = val_step(
                model      = model,
                batch      = batch,
                loss_fn    = loss_fn,
                cluster_db = cluster_db,
                device     = device,
            )

        # Accumula
        for k, v in breakdown.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

    # Media sull'epoca
    metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}
    metrics["n_batches"] = n_batches
    return metrics


# ---------------------------------------------------------------------------
# build_empty_cluster_db  (utile in pretrain, dove non c'è database)
# ---------------------------------------------------------------------------

def build_empty_cluster_db(n_clusters: int) -> dict:
    """
    Restituisce un cluster_db vuoto con K cluster None.
    Usato in pretrain/meta-training dove non esiste un database specifico.
    """
    return {k: None for k in range(n_clusters)}


# ---------------------------------------------------------------------------
# build_cluster_db_for_batch
# ---------------------------------------------------------------------------

def build_cluster_db_for_batch(
    query_hist:  torch.Tensor,          # (B, 192) color histograms del batch
    database:    PhotographerDatabase,
    faiss_mgr:   FAISSIndexManager,
    top_m:       int = 10,
    device:      str = "cuda",
) -> dict:
    """
    Costruisce il cluster_db per un batch usando FAISS per il top-M retrieval.

    Versione ottimizzata: usa l'indice FAISS per trovare le top-M immagini
    per cluster invece di passare tutto il database.

    Parameters
    ----------
    query_hist  : color histogram del batch (usa il primo elemento)
    database    : PhotographerDatabase del fotografo
    faiss_mgr   : FAISSIndexManager associato
    top_m       : numero di immagini da recuperare per cluster
    device      : device per i tensori restituiti

    Returns
    -------
    cluster_db : {k: {"keys": Tensor, "values": Tensor}} per RetrievalModule
    """
    return database.get_cluster_db(
        query_hist = query_hist,
        top_m      = top_m,
        device     = device,
    )
