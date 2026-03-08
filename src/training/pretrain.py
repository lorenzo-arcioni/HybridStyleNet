"""
training/pretrain.py

Fase 1: Pre-training su FiveK mixed  (§8.2).

Obiettivo: imparare la struttura generale di una trasformazione
fotografica su tutti e 5 gli expert, senza conditioning sullo stile.

Loss semplificata: L_ΔE + 0.5 * L_perc
Conditioning disabilitato (prototype = zero vector).
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import FiveKDataset, collate_paired
from data.raw_pipeline import RawPipeline
from data.augmentation import PairAugmentation
from losses.delta_e import DeltaELoss
from losses.perceptual import PerceptualLoss
from models.hybrid_style_net import HybridStyleNet
from utils.checkpoint import save_checkpoint, build_checkpoint_state
from utils.logging_utils import get_logger, TensorBoardLogger

logger = get_logger(__name__)


class Pretrainer:
    """
    Fase 1: Pre-training su FiveK mixed.

    Args:
        model:         Istanza HybridStyleNet.
        fivek_root:    Radice dataset FiveK.
        experts:       Lista expert da includere nel mix.
        cfg:           Dizionario di configurazione (da default.yaml).
        device:        Device di training.
        checkpoint_dir: Directory per i checkpoint.
        log_dir:       Directory per TensorBoard.
    """

    def __init__(
        self,
        model: HybridStyleNet,
        fivek_root: str,
        experts=None,
        cfg: Optional[Dict] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/pretrain",
        log_dir: str = "logs/pretrain",
    ) -> None:
        self.model          = model
        self.fivek_root     = fivek_root
        self.experts        = experts or ["A", "B", "C", "D", "E"]
        self.cfg            = cfg or {}
        self.device         = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        self.model.to(self.device)

        # Loss semplificata (§8.2)
        self.l_delta_e = DeltaELoss().to(self.device)
        self.l_perc    = PerceptualLoss().to(self.device)
        self.perc_weight = 0.5

        # Ottimizzatore
        opt_cfg = self.cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg.get("lr", 1e-4)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-8)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
        )

        self.grad_clip = float(opt_cfg.get("grad_clip", 1.0))

        # AMP
        self.use_amp = self.cfg.get("hardware", {}).get("amp", True) \
                       and device == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logger
        self.tb = TensorBoardLogger(log_dir)
        self._global_step = 0

        logger.info(
            f"Pretrainer: device={device}, experts={self.experts}, "
            f"amp={self.use_amp}"
        )

    def _build_dataloader(self, batch_size: int, num_workers: int) -> DataLoader:
        """Costruisce il DataLoader con il mix di tutti gli expert."""
        pipeline  = RawPipeline(target_long_side=768)
        augment   = PairAugmentation()

        datasets = []
        for expert in self.experts:
            try:
                ds = FiveKDataset(
                    fivek_root=self.fivek_root,
                    expert=expert,
                    split="train",
                    pipeline=pipeline,
                    augmentation=augment,
                )
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Expert {expert} saltato: {e}")

        if not datasets:
            raise RuntimeError("Nessun dataset FiveK disponibile per il pre-training.")

        mixed_ds = ConcatDataset(datasets)
        logger.info(f"Pretrainer dataset: {len(mixed_ds)} coppie totali")

        return DataLoader(
            mixed_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_paired,
            drop_last=True,
        )

    def _zero_prototype(self, batch_size: int) -> torch.Tensor:
        """Prototype zero (disabilita il conditioning durante il pre-training)."""
        dim = self.model.set_transformer.output_dim
        return torch.zeros(batch_size, dim, device=self.device)

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        num_workers: int = 4,
        save_every: int = 5,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Esegue il pre-training.

        Args:
            epochs:      Numero di epoche.
            batch_size:  Dimensione batch.
            num_workers: Worker DataLoader.
            save_every:  Salva checkpoint ogni N epoche.
            resume_from: Percorso checkpoint da cui riprendere.
        """
        start_epoch = 1

        if resume_from:
            from utils.checkpoint import load_checkpoint
            state = load_checkpoint(
                resume_from, self.model, self.optimizer,
                device=str(self.device)
            )
            start_epoch = state.get("epoch", 0) + 1
            logger.info(f"Pretrainer: ripreso dall'epoca {start_epoch}")

        loader = self._build_dataloader(batch_size, num_workers)

        for epoch in range(start_epoch, epochs + 1):
            epoch_loss = self._train_epoch(loader, epoch)

            logger.info(f"Pretrain Epoch {epoch}/{epochs} — loss={epoch_loss:.4f}")
            self.tb.log_scalar("pretrain/epoch_loss", epoch_loss, epoch)

            if epoch % save_every == 0 or epoch == epochs:
                state = build_checkpoint_state(
                    self.model, self.optimizer,
                    epoch=epoch,
                    metrics={"train_loss": epoch_loss},
                )
                save_checkpoint(
                    state,
                    self.checkpoint_dir,
                    filename=f"epoch_{epoch:04d}.pth",
                    is_best=(epoch == epochs),
                )

        self.tb.close()
        logger.info("Pre-training completato.")

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Esegue un'epoca di training. Restituisce la loss media."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            B   = src.shape[0]

            # Prototype zero per disabilitare il conditioning
            prototype = self._zero_prototype(B)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out  = self.model(src, prototype=prototype)
                pred = out["pred"]

                # Loss semplificata §8.2
                l_de   = self.l_delta_e(pred, tgt)
                l_perc = self.l_perc(pred, tgt)
                loss   = l_de + self.perc_weight * l_perc

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1
            self._global_step += 1

            if self._global_step % 10 == 0:
                self.tb.log_scalar("pretrain/step_loss", loss.item(),
                                   self._global_step)

        return total_loss / max(n_batches, 1)