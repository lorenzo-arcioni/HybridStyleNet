"""
utils/logging_utils.py

Logger strutturato e wrapper TensorBoard.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


# ── Logger Python standard ───────────────────────────────────────────────────

def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Crea (o recupera) un logger con handler su console e, opzionalmente, su file.

    Args:
        name:    Nome del logger (tipicamente __name__ del modulo chiamante).
        log_dir: Directory per il file di log. Se None, nessun file handler.
        level:   Livello di logging (default: INFO).
        console: Se True, aggiunge un handler su stdout.

    Returns:
        Logger configurato.
    """
    logger = logging.getLogger(name)

    # Evita di aggiungere handler duplicati se il logger esiste già
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Evita propagazione al root logger per non duplicare i messaggi
    logger.propagate = False

    return logger


# ── Wrapper TensorBoard ──────────────────────────────────────────────────────

class TensorBoardLogger:
    """
    Wrapper leggero attorno a SummaryWriter di TensorBoard.

    Gestisce la creazione lazy dello scrittore e fornisce metodi
    di convenienza per le metriche tipiche del training.
    """

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        comment: str = "",
    ) -> None:
        """
        Args:
            log_dir:  Directory per i file di evento TensorBoard.
            enabled:  Se False, tutti i metodi sono no-op (disabilita senza
                      modificare il codice chiamante).
            comment:  Suffisso opzionale aggiunto alla directory.
        """
        self.enabled = enabled
        self._writer = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir, comment=comment)
                self._logger = get_logger("tensorboard")
                self._logger.info(f"TensorBoard log dir: {log_dir}")
            except ImportError:
                self.enabled = False
                logging.getLogger(__name__).warning(
                    "tensorboard non installato. Logging disabilitato."
                )

    # ── Scalari ──────────────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Registra un singolo scalare."""
        if self.enabled and self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int
    ) -> None:
        """Registra un dizionario di scalari sotto un unico tag."""
        if self.enabled and self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Registra un dizionario di metriche, opzionalmente con prefisso.

        Args:
            metrics: Es. {"delta_e": 3.2, "ssim": 0.97}.
            step:    Step globale.
            prefix:  Es. "train/" o "val/".
        """
        for k, v in metrics.items():
            tag = f"{prefix}{k}" if prefix else k
            self.log_scalar(tag, v, step)

    # ── Istogrammi ───────────────────────────────────────────────────────────

    def log_histogram(
        self, tag: str, values: torch.Tensor, step: int
    ) -> None:
        """Registra la distribuzione di un tensore come istogramma."""
        if self.enabled and self._writer is not None:
            self._writer.add_histogram(tag, values, global_step=step)

    # ── Immagini ─────────────────────────────────────────────────────────────

    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        max_images: int = 4,
        dataformats: str = "NCHW",
    ) -> None:
        """
        Registra una griglia di immagini.

        Args:
            tag:        Tag TensorBoard.
            images:     Tensore shape (N, C, H, W) in [0, 1].
            step:       Step globale.
            max_images: Massimo numero di immagini da loggare.
            dataformats: Formato dimensioni (default NCHW).
        """
        if self.enabled and self._writer is not None:
            imgs = images[:max_images].clamp(0.0, 1.0).float()
            self._writer.add_images(tag, imgs, global_step=step,
                                    dataformats=dataformats)

    def log_comparison(
        self,
        src: torch.Tensor,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        step: int,
        max_images: int = 4,
    ) -> None:
        """
        Logga una tripla (sorgente, predizione, target) affiancata.

        Args:
            src:  Immagini sorgente  (N, 3, H, W) in [0,1].
            pred: Immagini predette  (N, 3, H, W) in [0,1].
            tgt:  Immagini target    (N, 3, H, W) in [0,1].
            step: Step globale.
        """
        if not (self.enabled and self._writer is not None):
            return

        n = min(max_images, src.shape[0])
        # Concatena orizzontalmente: [src | pred | tgt]
        row = torch.cat([src[:n], pred[:n], tgt[:n]], dim=3)  # (n, 3, H, 3W)
        self.log_images("comparison/src_pred_tgt", row, step,
                        max_images=n, dataformats="NCHW")

    # ── Learning rate ────────────────────────────────────────────────────────

    def log_lr(
        self,
        optimizer: torch.optim.Optimizer,
        step: int,
        prefix: str = "train/",
    ) -> None:
        """Logga il learning rate corrente di tutti i param group."""
        if self.enabled and self._writer is not None:
            for i, pg in enumerate(optimizer.param_groups):
                self._writer.add_scalar(
                    f"{prefix}lr_group{i}", pg["lr"], global_step=step
                )

    # ── Loss breakdown ───────────────────────────────────────────────────────

    def log_loss_breakdown(
        self,
        loss_dict: Dict[str, Union[float, torch.Tensor]],
        step: int,
        prefix: str = "train/",
    ) -> None:
        """
        Logga ogni componente della loss composita separatamente.

        Args:
            loss_dict: Es. {"delta_e": 2.1, "hist": 0.5, "perc": 0.3, ...}.
            step:      Step globale.
            prefix:    Prefisso tag TensorBoard.
        """
        if not (self.enabled and self._writer is not None):
            return

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self._writer.add_scalar(f"{prefix}loss/{k}", val, global_step=step)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Forza scrittura buffer su disco."""
        if self.enabled and self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Chiude lo scrittore TensorBoard."""
        if self.enabled and self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()