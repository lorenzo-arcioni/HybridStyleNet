"""
reptile.py
----------
Algoritmo Reptile (Nichol et al., 2018) — implementazione pura.

Reptile è un algoritmo di meta-learning del primo ordine compatibile
con fp16. Rispetto a MAML non calcola Hessiani — è più lento a
convergere in teoria, ma in pratica con DINOv2 frozen la differenza
è trascurabile perché la parte più difficile (la rappresentazione)
è già risolta dal backbone.

Update rule:
  Inner loop (k passi, lr α):
    θ_T = θ - α · ∇_θ L_T^sup(θ)  (k volte)

  Outer loop (M task, lr ε):
    θ ← θ + (ε/M) · Σ_m (θ̃_m - θ)

Questo modulo espone:
  inner_loop(model, task, optimizer, loss_fn, steps, device)
      → θ_T  (state_dict dei parametri aggiornati)

  outer_update(model, task_deltas, outer_lr)
      → aggiorna θ in-place

  reptile_step(model, tasks, ...) → loss scalare medio
      → esegue inner loop + outer update in un'unica chiamata

È progettato per essere chiamato dal notebook di meta-training
o da meta_train.py.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data.task_sampler import Task              # type: ignore[import]


# ---------------------------------------------------------------------------
# inner_loop
# ---------------------------------------------------------------------------

def inner_loop(
    model:       nn.Module,
    task:        Task,
    loss_fn:     nn.Module,
    n_steps:     int   = 5,
    lr:          float = 1e-3,
    device:      str   = "cuda",
    use_fp16:    bool  = True,
    batch_size:  int   = 1,
) -> Tuple[dict, float]:
    """
    Esegue l'inner loop di Reptile su un singolo task.

    Crea una copia temporanea dei parametri trainable, li aggiorna
    per n_steps sul support set del task, e restituisce il nuovo
    state_dict senza modificare il modello originale.

    Parameters
    ----------
    model      : modello con parametri θ (non viene modificato)
    task       : episodio con support_loader e query_loader
    loss_fn    : CompositeLoss (chiamata con phase="meta")
    n_steps    : passi SGD dell'inner loop
    lr         : α — inner loop learning rate
    device     : device per il forward pass
    use_fp16   : abilita autocast
    batch_size : batch size per il support loader

    Returns
    -------
    adapted_state : state_dict con i parametri θ_T (solo parametri trainable)
    support_loss  : loss media sull'inner loop (per logging)
    """
    # Clona i parametri trainable — l'originale non viene toccato
    adapted_params = {
        name: param.clone().detach().requires_grad_(param.requires_grad)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # SGD sull'inner loop (standard in Reptile)
    optimizer = torch.optim.SGD(
        list(adapted_params.values()), lr=lr, momentum=0.0
    )
    scaler = GradScaler(enabled=use_fp16)

    support_loader = task.support_loader(batch_size=batch_size)
    # Cicla sul support set per n_steps (con wraparound se necessario)
    support_iter   = _infinite_loader(support_loader)

    total_loss = 0.0
    model.eval()   # DINOv2 always eval; altri moduli in train via parametri

    for step in range(n_steps):
        batch = next(support_iter)
        src   = batch["src"].to(device)
        tgt   = batch["tgt"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_fp16):
            # Forward pass con i parametri adattati
            out = _forward_with_params(model, adapted_params, src, task)
            breakdown = loss_fn(out, {"src": src, "tgt": tgt})
            loss = breakdown.total

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(adapted_params.values()), max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    adapted_state = {name: p.detach() for name, p in adapted_params.items()}
    return adapted_state, total_loss / n_steps


# ---------------------------------------------------------------------------
# outer_update
# ---------------------------------------------------------------------------

def outer_update(
    model:        nn.Module,
    task_deltas:  List[dict],   # lista di (θ̃_m - θ) per ogni task
    outer_lr:     float = 0.01,
) -> None:
    """
    Applica l'outer update di Reptile in-place al modello.

    θ ← θ + (ε/M) · Σ_m (θ̃_m - θ)

    Parameters
    ----------
    model       : modello da aggiornare
    task_deltas : lista di dizionari {param_name: delta_tensor}
                  dove delta_tensor = θ̃_m - θ
    outer_lr    : ε — outer loop step size
    """
    M = len(task_deltas)
    if M == 0:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Media dei delta sui task
            mean_delta = sum(
                d[name] for d in task_deltas if name in d
            ) / M
            param.add_(outer_lr * mean_delta)


# ---------------------------------------------------------------------------
# reptile_step  (inner + outer in una chiamata)
# ---------------------------------------------------------------------------

def reptile_step(
    model:      nn.Module,
    tasks:      List[Task],
    loss_fn:    nn.Module,
    n_inner:    int   = 5,
    inner_lr:   float = 1e-3,
    outer_lr:   float = 1e-2,
    device:     str   = "cuda",
    use_fp16:   bool  = True,
) -> dict:
    """
    Esegue un passo completo di Reptile (inner loop + outer update).

    Parametri
    ----------
    tasks : lista di M task per questo outer step

    Returns
    -------
    dict con:
      "meta_loss"      : loss media sull'inner loop (per logging)
      "n_tasks"        : numero di task processati
    """
    # Stato corrente dei parametri trainable (per calcolare i delta)
    theta = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    task_deltas: List[dict] = []
    inner_losses: List[float] = []

    for task in tasks:
        adapted_state, inner_loss = inner_loop(
            model=model,
            task=task,
            loss_fn=loss_fn,
            n_steps=n_inner,
            lr=inner_lr,
            device=device,
            use_fp16=use_fp16,
        )

        # Delta: θ̃_m - θ
        delta = {
            name: adapted_state[name].to(theta[name].device) - theta[name]
            for name in theta
            if name in adapted_state
        }
        task_deltas.append(delta)
        inner_losses.append(inner_loss)

    outer_update(model, task_deltas, outer_lr=outer_lr)

    return {
        "meta_loss": sum(inner_losses) / len(inner_losses),
        "n_tasks":   len(tasks),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _infinite_loader(loader):
    """Generatore infinito che riavvolge il DataLoader quando esaurito."""
    while True:
        for batch in loader:
            yield batch


def _forward_with_params(
    model:          nn.Module,
    adapted_params: dict,
    src:            torch.Tensor,
    task:           Task,
) -> dict:
    """
    Forward pass temporaneo con i parametri adattati.

    Sostituisce temporaneamente i parametri del modello con
    adapted_params, esegue il forward, poi li ripristina.

    Nota: questa è un'approssimazione semplificata — per una
    implementazione completa si userebbe higher o functorch.
    In Reptile di primo ordine questa approssimazione è valida
    perché non si calcolano gradienti di secondo ordine.
    """
    # Salva e sostituisce i parametri
    original_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in adapted_params:
                original_params[name] = param.data.clone()
                param.data.copy_(adapted_params[name])

    # Forward pass
    # cluster_db vuoto durante il meta-training (nessun database specifico)
    cluster_db = {k: None for k in range(model.cluster_net.n_clusters)}
    try:
        out = model(src, cluster_db)
    finally:
        # Ripristina sempre i parametri originali
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    return out
