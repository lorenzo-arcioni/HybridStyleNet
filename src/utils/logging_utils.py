"""
logging_utils.py
----------------
Logger unificato per tensorboard e/o wandb.

Espone una classe Logger con API identica indipendentemente dal backend,
così i notebook non dipendono direttamente da tensorboard o wandb.

Uso tipico in un notebook:
    logger = Logger.from_config(cfg, run_name="adapt_ph01")
    logger.log_scalars(metrics, step=epoch)
    logger.log_images({"pred": img_tensor}, step=epoch)
    logger.close()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    """
    Logger unificato tensorboard / wandb / console.

    Parameters
    ----------
    log_dir     : directory per i log tensorboard
    backend     : "tensorboard" | "wandb" | "both" | "none"
    run_name    : nome della run (usato da wandb)
    project     : nome del progetto wandb
    config      : dict di iperparametri da loggare (wandb)
    """

    def __init__(
        self,
        log_dir:  Union[str, Path] = "logs/",
        backend:  str  = "tensorboard",
        run_name: str  = "run",
        project:  str  = "rag-colornet",
        config:   Optional[dict] = None,
    ) -> None:
        self.log_dir  = Path(log_dir)
        self.backend  = backend
        self.run_name = run_name
        self._tb      = None
        self._wandb   = None
        self._step    = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if backend in ("tensorboard", "both"):
            self._init_tensorboard(run_name)

        if backend in ("wandb", "both"):
            self._init_wandb(run_name, project, config)

    # ------------------------------------------------------------------
    def _init_tensorboard(self, run_name: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir    = self.log_dir / run_name
            self._tb  = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            print("tensorboard non disponibile — log tensorboard disabilitato.")

    def _init_wandb(self, run_name: str, project: str, config: Optional[dict]) -> None:
        try:
            import wandb
            self._wandb = wandb.init(
                project = project,
                name    = run_name,
                config  = config or {},
                reinit  = True,
            )
        except ImportError:
            print("wandb non disponibile — log wandb disabilitato.")

    # ------------------------------------------------------------------
    def log_scalars(
        self,
        metrics: Dict[str, float],
        step:    Optional[int] = None,
    ) -> None:
        """
        Logga un dizionario di scalari.

        Parameters
        ----------
        metrics : {"loss/total": 0.42, "loss/delta_e": 1.2, ...}
        step    : step globale; se None usa il contatore interno
        """
        s = step if step is not None else self._step

        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, global_step=s)

        if self._wandb is not None:
            self._wandb.log({**metrics, "step": s})

        self._step = s + 1

    # ------------------------------------------------------------------
    def log_images(
        self,
        images:   Dict[str, torch.Tensor],
        step:     Optional[int] = None,
        max_imgs: int = 4,
    ) -> None:
        """
        Logga immagini.

        Parameters
        ----------
        images  : {"pred": (B,3,H,W), "tgt": (B,3,H,W), ...}
        step    : step globale
        max_imgs: max immagini del batch
        """
        s = step if step is not None else self._step

        if self._tb is not None:
            for tag, img in images.items():
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                self._tb.add_images(tag, img[:max_imgs].clamp(0, 1), global_step=s)

        if self._wandb is not None:
            import wandb
            wandb_imgs = {}
            for tag, img in images.items():
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                imgs = img[:max_imgs].clamp(0, 1)
                wandb_imgs[tag] = [
                    wandb.Image(imgs[i].permute(1, 2, 0).cpu().numpy())
                    for i in range(imgs.shape[0])
                ]
            self._wandb.log({**wandb_imgs, "step": s})

    # ------------------------------------------------------------------
    def log_histogram(
        self,
        tag:    str,
        values: torch.Tensor,
        step:   Optional[int] = None,
    ) -> None:
        """Logga un istogramma di valori (solo tensorboard)."""
        s = step if step is not None else self._step
        if self._tb is not None:
            self._tb.add_histogram(tag, values.detach().cpu(), global_step=s)

    # ------------------------------------------------------------------
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Logga testo libero."""
        s = step if step is not None else self._step
        if self._tb is not None:
            self._tb.add_text(tag, text, global_step=s)
        if self._wandb is not None:
            self._wandb.log({tag: text, "step": s})

    # ------------------------------------------------------------------
    def print_metrics(
        self,
        metrics: Dict[str, float],
        prefix:  str = "",
        epoch:   Optional[int] = None,
    ) -> None:
        """Stampa le metriche su stdout in formato leggibile."""
        ep_str = f"[ep {epoch:3d}] " if epoch is not None else ""
        parts  = [f"{k.split('/')[-1]}: {v:.4f}" for k, v in metrics.items()
                  if isinstance(v, (int, float))]
        print(f"{ep_str}{prefix}{' | '.join(parts)}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Chiude i writer."""
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg:      dict,
        run_name: str,
        config:   Optional[dict] = None,
    ) -> "Logger":
        lcfg = cfg.get("logging", {})
        return cls(
            log_dir  = cfg["paths"]["logs_dir"],
            backend  = lcfg.get("backend", "tensorboard"),
            run_name = run_name,
            project  = "rag-colornet",
            config   = config,
        )
