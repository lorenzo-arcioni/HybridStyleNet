"""
utils/checkpoint.py

Salvataggio e caricamento di checkpoint PyTorch.
Gestisce rotazione automatica dei checkpoint (keep_last_n).
"""

import os
import glob
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    keep_last_n: int = 3,
) -> str:
    """
    Salva un checkpoint su disco con rotazione automatica.

    Args:
        state:           Dizionario con model, optimizer, epoch, metrics, ecc.
        checkpoint_dir:  Directory di destinazione.
        filename:        Nome file (es. "epoch_010.pth").
        is_best:         Se True, salva anche una copia come "best.pth".
        keep_last_n:     Numero massimo di checkpoint da mantenere
                         (0 = nessun limite). Non influenza "best.pth".

    Returns:
        Percorso assoluto del file salvato.
    """
    dir_path = Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    save_path = dir_path / filename
    torch.save(state, save_path)
    logger.info(f"Checkpoint salvato: {save_path}")

    if is_best:
        best_path = dir_path / "best.pth"
        torch.save(state, best_path)
        logger.info(f"Best checkpoint aggiornato: {best_path}")

    # ── Rotazione: rimuovi i checkpoint più vecchi ────────────────────────────
    if keep_last_n > 0:
        # Considera solo i checkpoint numerati (esclude best.pth e latest.pth)
        pattern = str(dir_path / "epoch_*.pth")
        existing = sorted(glob.glob(pattern))
        to_remove = existing[:-keep_last_n] if len(existing) > keep_last_n else []
        for old in to_remove:
            os.remove(old)
            logger.debug(f"Checkpoint rimosso: {old}")

    # ── Aggiorna latest.pth come symlink o copia ─────────────────────────────
    latest_path = dir_path / "latest.pth"
    # Usa copia diretta per compatibilità cross-platform
    torch.save(state, latest_path)

    return str(save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Carica un checkpoint da disco.

    Args:
        checkpoint_path: Percorso al file .pth.
        model:           Se fornito, carica i parametri nel modello.
        optimizer:       Se fornito, ripristina lo stato dell'ottimizzatore.
        scheduler:       Se fornito, ripristina lo stato dello scheduler.
        device:          Device di destinazione per il mapping dei tensori.
        strict:          Se True, richiede corrispondenza esatta delle chiavi
                         nel state_dict del modello.

    Returns:
        Il dizionario di stato completo (utile per recuperare epoch, metriche, ecc.).

    Raises:
        FileNotFoundError: Se il file non esiste.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    logger.info(f"Caricamento checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    if model is not None and "model_state_dict" in state:
        missing, unexpected = model.load_state_dict(
            state["model_state_dict"], strict=strict
        )
        if missing:
            logger.warning(f"Chiavi mancanti nel modello: {missing}")
        if unexpected:
            logger.warning(f"Chiavi inattese nel checkpoint: {unexpected}")
        logger.info("Parametri modello ripristinati.")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        # Sposta lo stato dell'ottimizzatore sul device corretto
        for opt_state in optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(device)
        logger.info("Stato ottimizzatore ripristinato.")

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Stato scheduler ripristinato.")

    return state


def build_checkpoint_state(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Costruisce il dizionario di stato da salvare.

    Args:
        model:     Modello PyTorch.
        optimizer: Ottimizzatore (opzionale).
        scheduler: Scheduler LR (opzionale).
        epoch:     Epoca corrente.
        metrics:   Dizionario di metriche da salvare (es. val_delta_e).
        config:    Configurazione esperimento (opzionale, per riproducibilità).

    Returns:
        Dizionario pronto per torch.save.
    """
    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        state["metrics"] = metrics

    if config is not None:
        state["config"] = config

    return state