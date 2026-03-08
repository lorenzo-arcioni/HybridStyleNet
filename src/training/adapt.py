"""
training/adapt.py

Fase 3: Few-shot adaptation al fotografo target  (§5.4, §8.4).

Schema Freeze-Then-Unfreeze:
    Fase 3A (epoche 1-10):  congela CNN stage 1-2 + Swin stage 4
    Fase 3B (epoche 11-30): scongela tutto, LR ridotto + cosine annealing

Early stopping su val_delta_e con patience=5.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import CustomDataset, collate_paired
from data.raw_pipeline import RawPipeline
from data.augmentation import PairAugmentation
from models.hybrid_style_net import HybridStyleNet
from losses.composite import ColorAestheticLoss, LossWeights
from training.scheduler import (
    CosineAnnealingScheduler,
    LossCurriculumScheduler,
    EarlyStopping,
)
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    build_checkpoint_state,
)
from utils.logging_utils import get_logger, TensorBoardLogger

logger = get_logger(__name__)


class FewShotAdapter:
    """
    Few-shot adaptation al fotografo target.

    Args:
        model:            HybridStyleNet con pesi theta_meta.
        custom_root:      Directory con src/ e tgt/ del fotografo.
        cfg:              Configurazione adapt.yaml.
        device:           Device di training.
        checkpoint_dir:   Directory per i checkpoint.
        log_dir:          Directory TensorBoard.
    """

    def __init__(
        self,
        model: HybridStyleNet,
        custom_root: str,
        cfg: Optional[Dict] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/adapt",
        log_dir: str = "logs/adapt",
    ) -> None:
        self.model          = model
        self.custom_root    = custom_root
        self.cfg            = cfg or {}
        self.device         = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        self.model.to(self.device)

        # Config fasi
        self.phase_a_cfg = self.cfg.get("phase_a", {})
        self.phase_b_cfg = self.cfg.get("phase_b", {})
        self.es_cfg      = self.cfg.get("early_stopping", {})
        self.aug_cfg     = self.cfg.get("augmentation", {})

        # Loss
        self.loss_fn = ColorAestheticLoss()
        self.loss_fn.to(self.device)

        # AMP
        self.use_amp = (
            self.cfg.get("hardware", {}).get("amp", True)
            and str(self.device) == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logging
        self.tb = TensorBoardLogger(log_dir)

        self._best_val_delta_e = float("inf")
        self._best_epoch       = 0
        self._global_step      = 0

        logger.info(
            f"FewShotAdapter: custom_root={custom_root}, "
            f"device={device}, amp={self.use_amp}"
        )

    # ── Build DataLoaders ─────────────────────────────────────────────────────

    def _build_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
    ) -> tuple:
        """
        Costruisce DataLoader di training e validazione.

        Returns:
            (train_loader, val_loader)
        """
        pipeline = RawPipeline(target_long_side=768)

        aug = PairAugmentation(
            horizontal_flip_prob=float(
                self.aug_cfg.get("horizontal_flip_prob", 0.5)
            ),
            random_crop_scale=tuple(
                self.aug_cfg.get("random_crop_scale", [0.7, 1.0])
            ),
            rotation_degrees=float(
                self.aug_cfg.get("rotation_degrees", 5.0)
            ),
            exposure_perturb_prob=float(
                self.aug_cfg.get("exposure_perturb", {}).get("prob", 0.3)
            ),
            exposure_range=tuple(
                self.aug_cfg.get("exposure_perturb", {}).get("range", [0.9, 1.1])
            ),
            noise_prob=float(
                self.aug_cfg.get("noise_perturb", {}).get("prob", 0.3)
            ),
            noise_sigma=float(
                self.aug_cfg.get("noise_perturb", {}).get("sigma", 0.01)
            ),
        )

        data_cfg    = self.cfg.get("data", {})
        val_split   = float(data_cfg.get("val_split", 0.2))
        max_pairs   = data_cfg.get("max_pairs", None)

        train_ds = CustomDataset(
            custom_root=self.custom_root,
            split="train",
            val_split=val_split,
            pipeline=pipeline,
            augmentation=aug,
            max_pairs=max_pairs,
        )
        val_ds = CustomDataset(
            custom_root=self.custom_root,
            split="val",
            val_split=val_split,
            pipeline=pipeline,
            augmentation=None,
        )

        logger.info(
            f"Dataset: {len(train_ds)} train, {len(val_ds)} val"
        )

        if len(train_ds) == 0:
            raise RuntimeError(
                f"Dataset vuoto in {self.custom_root}. "
                "Verifica la struttura src/tgt."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_paired,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_paired,
        )

        return train_loader, val_loader

    # ── Calcolo Style Cache ───────────────────────────────────────────────────

    def _build_style_cache(self, train_loader: DataLoader) -> None:
        """
        Calcola il style prototype e la cache del training set.
        """
        logger.info("Calcolo style cache dal training set...")

        src_list, tgt_list = [], []
        for batch in train_loader:
            for i in range(batch["src"].shape[0]):
                src_list.append(batch["src"][i])
                tgt_list.append(batch["tgt"][i])

        self.model.set_style_cache(src_list, tgt_list, batch_size=8)
        logger.info(f"Style cache pronta: {len(src_list)} coppie.")

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        resume_from: Optional[str] = None,
        num_workers: int = 4,
    ) -> str:
        """
        Esegue le fasi 3A e 3B di adaptation.

        Args:
            resume_from: Checkpoint da cui riprendere.
            num_workers: Worker DataLoader.

        Returns:
            Percorso del best checkpoint.
        """
        # ── Fase 3A ──────────────────────────────────────────────────────────
        pa      = self.phase_a_cfg
        epochs_a  = int(pa.get("epochs",     10))
        lr_a      = float(pa.get("lr",       5e-5))
        wd_a      = float(pa.get("weight_decay", 2e-3))
        bs_a      = int(pa.get("batch_size", 4))

        # Congela i parametri per la fase 3A
        self.model.freeze_for_adaptation()

        optimizer_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_a,
            weight_decay=wd_a,
        )

        train_loader, val_loader = self._build_dataloaders(bs_a, num_workers)
        self._build_style_cache(train_loader)

        curriculum_a = LossCurriculumScheduler(
            loss_fn=self.loss_fn,
            verbose=True,
        )

        early_stop = EarlyStopping(
            patience=int(self.es_cfg.get("patience", 5)),
            mode=self.es_cfg.get("mode", "min"),
        )

        best_ckpt_path = ""

        logger.info(f"=== Fase 3A: epoche 1-{epochs_a} (freeze) ===")
        for epoch in range(1, epochs_a + 1):
            curriculum_a.step_epoch(epoch)
            train_loss = self._train_epoch(
                train_loader, optimizer_a, epoch
            )
            val_metrics = self._validate(val_loader, epoch)

            self._log_epoch(epoch, train_loss, val_metrics,
                            optimizer_a, prefix="phase_a")

            # Salva best
            val_de = val_metrics.get("delta_e", float("inf"))
            is_best = val_de < self._best_val_delta_e
            if is_best:
                self._best_val_delta_e = val_de
                self._best_epoch       = epoch

            state = build_checkpoint_state(
                self.model, optimizer_a,
                epoch=epoch,
                metrics={"val_delta_e": val_de, "train_loss": train_loss},
            )
            path = save_checkpoint(
                state, self.checkpoint_dir,
                filename=f"epoch_{epoch:04d}.pth",
                is_best=is_best,
            )
            if is_best:
                best_ckpt_path = str(
                    Path(self.checkpoint_dir) / "best.pth"
                )

            if early_stop.step(val_de, epoch):
                logger.info("Early stopping nella Fase 3A.")
                break

        # ── Fase 3B ──────────────────────────────────────────────────────────
        pb       = self.phase_b_cfg
        epochs_b = int(pb.get("epochs",     20))
        lr_b     = float(pb.get("lr",       2.5e-5))
        wd_b     = float(pb.get("weight_decay", 2e-3))
        bs_b     = int(pb.get("batch_size", 4))
        t_max    = int(pb.get("scheduler", {}).get("t_max",  epochs_b))
        eta_min  = float(pb.get("scheduler", {}).get("eta_min", 1e-6))

        # Scongela tutto
        self.model.unfreeze_all()

        optimizer_b = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr_b,
            weight_decay=wd_b,
        )
        cosine_sched = CosineAnnealingScheduler(
            optimizer_b, base_lr=lr_b, T_max=t_max, eta_min=eta_min
        )

        # Ricrea i loader con il nuovo batch size
        if bs_b != bs_a:
            train_loader, val_loader = self._build_dataloaders(
                bs_b, num_workers
            )

        curriculum_b = LossCurriculumScheduler(loss_fn=self.loss_fn)
        early_stop_b = EarlyStopping(
            patience=int(self.es_cfg.get("patience", 5)),
            mode=self.es_cfg.get("mode", "min"),
        )

        total_epochs = epochs_a + epochs_b
        logger.info(
            f"=== Fase 3B: epoche {epochs_a+1}-{total_epochs} (unfreeze) ==="
        )

        for local_epoch in range(1, epochs_b + 1):
            global_epoch = epochs_a + local_epoch
            curriculum_b.step_epoch(global_epoch)

            train_loss = self._train_epoch(
                train_loader, optimizer_b, global_epoch
            )
            val_metrics = self._validate(val_loader, global_epoch)
            cosine_sched.step_epoch()

            self._log_epoch(global_epoch, train_loss, val_metrics,
                            optimizer_b, prefix="phase_b")

            val_de  = val_metrics.get("delta_e", float("inf"))
            is_best = val_de < self._best_val_delta_e
            if is_best:
                self._best_val_delta_e = val_de
                self._best_epoch       = global_epoch

            state = build_checkpoint_state(
                self.model, optimizer_b,
                epoch=global_epoch,
                metrics={"val_delta_e": val_de, "train_loss": train_loss},
            )
            path = save_checkpoint(
                state, self.checkpoint_dir,
                filename=f"epoch_{global_epoch:04d}.pth",
                is_best=is_best,
            )
            if is_best:
                best_ckpt_path = str(
                    Path(self.checkpoint_dir) / "best.pth"
                )

            if early_stop_b.step(val_de, global_epoch):
                logger.info("Early stopping nella Fase 3B.")
                break

        self.tb.close()
        logger.info(
            f"Adaptation completata. "
            f"Best epoch={self._best_epoch}, "
            f"best ΔE={self._best_val_delta_e:.4f}"
        )
        return best_ckpt_path

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """Esegue un'epoca di training. Restituisce la loss media."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out  = self.model(src)
                pred = out["pred"]
                loss_dict = self.loss_fn(pred, tgt, src=src)
                loss      = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.get("optimizer", {}).get("grad_clip", 1.0),
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1
            self._global_step += 1

            if self._global_step % 10 == 0:
                self.tb.log_loss_breakdown(
                    {k: v.item() for k, v in loss_dict.items()},
                    self._global_step,
                    prefix="adapt/",
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Esegue la validazione. Restituisce dizionario di metriche."""
        from losses.delta_e import DeltaELoss

        self.model.eval()
        l_de_fn   = DeltaELoss().to(self.device)
        total_de  = 0.0
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out  = self.model(src)
                pred = out["pred"]
                loss_dict = self.loss_fn(pred, tgt)
                de        = l_de_fn(pred, tgt)

            total_loss += loss_dict["total"].item()
            total_de   += de.item()
            n_batches  += 1

        n = max(n_batches, 1)
        metrics = {
            "loss":    total_loss / n,
            "delta_e": total_de   / n,
        }

        # Log immagini di confronto (primo batch)
        if loader.dataset and len(loader.dataset) > 0:
            sample = loader.dataset[0]
            src_t  = sample["src"].unsqueeze(0).to(self.device)
            tgt_t  = sample["tgt"].unsqueeze(0).to(self.device)
            out_t  = self.model(src_t)
            pred_t = out_t["pred"]
            self.tb.log_comparison(
                src_t.cpu(), pred_t.cpu(), tgt_t.cpu(), epoch
            )

        return metrics

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        optimizer: torch.optim.Optimizer,
        prefix: str = "",
    ) -> None:
        """Logga le metriche dell'epoca."""
        val_de = val_metrics.get("delta_e", float("inf"))
        logger.info(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"val_ΔE={val_de:.4f}"
        )
        self.tb.log_scalar(f"{prefix}/train_loss", train_loss, epoch)
        self.tb.log_metrics(val_metrics, epoch, prefix=f"{prefix}/val_")
        self.tb.log_lr(optimizer, epoch, prefix=f"{prefix}/")