"""
training/meta_train.py

Fase 2: Meta-training MAML con Task Augmentation  (§5.3, §8.3).

Implementa il loop bi-level MAML:
    Inner loop: adattamento rapido al task (T_inner step)
    Outer loop: meta-aggiornamento su batch di M task

Con first_order=False usa i gradienti del secondo ordine (MAML completo).
Con first_order=True usa FOMAML (più veloce, risultati simili).
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.hybrid_style_net import HybridStyleNet
from losses.composite import ColorAestheticLoss, LossWeights
from training.task_sampler import TaskSampler, MetaTask
from training.scheduler import LRScheduler, LossCurriculumScheduler
from utils.checkpoint import save_checkpoint, load_checkpoint, build_checkpoint_state
from utils.logging_utils import get_logger, TensorBoardLogger

logger = get_logger(__name__)


class MetaTrainer:
    """
    MAML meta-trainer per photographer-specific color grading.

    Args:
        model:            HybridStyleNet inizializzato con pesi pre-trained.
        task_sampler:     TaskSampler per campionare task FiveK + sintetici.
        cfg:              Dizionario configurazione (meta_train.yaml).
        device:           Device di training.
        checkpoint_dir:   Directory checkpoint.
        log_dir:          Directory TensorBoard.
    """

    def __init__(
        self,
        model: HybridStyleNet,
        task_sampler: TaskSampler,
        cfg: Optional[Dict] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/meta",
        log_dir: str = "logs/meta",
    ) -> None:
        self.model         = model
        self.task_sampler  = task_sampler
        self.cfg           = cfg or {}
        self.device        = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        self.model.to(self.device)

        # Iperparametri MAML (§5.3)
        maml_cfg = self.cfg.get("maml", {})
        self.inner_lr        = float(maml_cfg.get("inner_lr",        1e-3))
        self.meta_lr         = float(maml_cfg.get("meta_lr",         5e-5))
        self.inner_steps     = int(maml_cfg.get("inner_steps",       5))
        self.tasks_per_batch = int(maml_cfg.get("tasks_per_batch",   3))
        self.meta_iterations = int(maml_cfg.get("meta_iterations",   10000))
        self.first_order     = bool(maml_cfg.get("first_order",      False))

        # Meta-ottimizzatore
        self.meta_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.meta_lr,
            betas=(0.9, 0.999),
            weight_decay=float(
                self.cfg.get("optimizer", {}).get("weight_decay", 1e-4)
            ),
        )

        self.grad_clip = float(
            self.cfg.get("optimizer", {}).get("grad_clip", 1.0)
        )

        # LR scheduler con warmup
        sch_cfg = self.cfg.get("scheduler", {})
        self.lr_scheduler = LRScheduler(
            optimizer=self.meta_optimizer,
            base_lr=self.meta_lr,
            warmup_steps=int(sch_cfg.get("warmup_steps", 200)),
            T_max=self.meta_iterations,
            eta_min=float(sch_cfg.get("min_lr", 1e-6)),
        )

        # Loss con curriculum §6.6 fase 6-10
        self.loss_fn = ColorAestheticLoss(
            weights=LossWeights.curriculum_6_10()
        )
        self.loss_fn.to(self.device)

        self.curriculum = LossCurriculumScheduler(
            loss_fn=self.loss_fn,
            verbose=True,
        )

        # AMP
        self.use_amp = (
            self.cfg.get("hardware", {}).get("amp", True)
            and device == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logging
        self.tb = TensorBoardLogger(log_dir)
        self._iteration = 0

        logger.info(
            f"MetaTrainer: inner_lr={self.inner_lr}, meta_lr={self.meta_lr}, "
            f"inner_steps={self.inner_steps}, tasks_per_batch={self.tasks_per_batch}, "
            f"first_order={self.first_order}, amp={self.use_amp}"
        )

    def train(
        self,
        resume_from: Optional[str] = None,
        save_every: int = 500,
    ) -> None:
        """
        Esegue il meta-training completo.

        Args:
            resume_from: Checkpoint da cui riprendere.
            save_every:  Salva ogni N iterazioni.
        """
        if resume_from:
            state = load_checkpoint(
                resume_from, self.model, self.meta_optimizer,
                device=str(self.device),
            )
            self._iteration = state.get("iteration", 0)
            logger.info(f"MetaTrainer: ripreso dall'iterazione {self._iteration}")

        logger.info(
            f"Meta-training: {self.meta_iterations} iterazioni totali, "
            f"riparte da {self._iteration}"
        )

        while self._iteration < self.meta_iterations:
            # Curriculum loss (mappa iterazione → epoch-like per il curriculum)
            pseudo_epoch = self._iteration // 100 + 1
            self.curriculum.step_epoch(pseudo_epoch)

            # Campiona batch di task
            tasks = self.task_sampler.sample_batch(self.tasks_per_batch)

            # Meta-update
            meta_loss, loss_breakdown = self._meta_step(tasks)

            # LR step
            lr = self.lr_scheduler.step()

            self._iteration += 1

            # Logging
            if self._iteration % 10 == 0:
                self.tb.log_scalar("meta/loss",    meta_loss, self._iteration)
                self.tb.log_scalar("meta/lr",      lr,        self._iteration)
                self.tb.log_loss_breakdown(
                    loss_breakdown, self._iteration, prefix="meta/"
                )

            if self._iteration % 100 == 0:
                logger.info(
                    f"Meta iter {self._iteration}/{self.meta_iterations} — "
                    f"loss={meta_loss:.4f}, lr={lr:.2e}"
                )

            if self._iteration % save_every == 0:
                state = build_checkpoint_state(
                    self.model, self.meta_optimizer,
                    epoch=self._iteration,
                    metrics={"meta_loss": meta_loss},
                )
                state["iteration"] = self._iteration
                save_checkpoint(
                    state,
                    self.checkpoint_dir,
                    filename=f"iter_{self._iteration:06d}.pth",
                    is_best=False,
                )

        # Salva checkpoint finale
        state = build_checkpoint_state(
            self.model, self.meta_optimizer,
            epoch=self._iteration,
            metrics={"meta_loss": meta_loss},
        )
        state["iteration"] = self._iteration
        save_checkpoint(
            state, self.checkpoint_dir,
            filename="final.pth", is_best=True,
        )

        self.tb.close()
        logger.info("Meta-training completato.")

    def _meta_step(
        self,
        tasks: List[MetaTask],
    ) -> tuple:
        """
        Esegue un singolo meta-update su un batch di task.

        Args:
            tasks: Lista di MetaTask (lunghezza = tasks_per_batch).

        Returns:
            (meta_loss_scalar, loss_breakdown_dict)
        """
        self.meta_optimizer.zero_grad()

        meta_loss_total     = torch.tensor(0.0, device=self.device)
        loss_breakdown_accum: Dict[str, float] = {}

        for task in tasks:
            # Inner loop: adatta i parametri al task
            adapted_params = self._inner_loop(task)

            # Outer loop: calcola la loss sul query set con i parametri adattati
            query_loss, breakdown = self._compute_query_loss(
                task, adapted_params
            )
            meta_loss_total = meta_loss_total + query_loss / len(tasks)

            for k, v in breakdown.items():
                loss_breakdown_accum[k] = (
                    loss_breakdown_accum.get(k, 0.0) + v / len(tasks)
                )

        # Meta-gradiente
        meta_loss_total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.meta_optimizer.step()

        return meta_loss_total.item(), loss_breakdown_accum

    def _inner_loop(self, task: MetaTask) -> Dict[str, torch.Tensor]:
        """
        Inner loop MAML: T_inner passi di gradiente sul support set.

        Args:
            task: MetaTask con support_{src,tgt}.

        Returns:
            Dizionario {nome_param: tensore_adattato} (θ_T*)
        """
        # Copia i parametri correnti del modello
        fast_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        src = task.support_src.to(self.device)
        tgt = task.support_tgt.to(self.device)

        for _ in range(self.inner_steps):
            # Forward con i parametri veloci
            out  = self._forward_with_params(src, fast_params)
            pred = out["pred"]

            # Loss sul support set
            loss_dict = self.loss_fn(pred, tgt, src=src)
            loss      = loss_dict["total"]

            # Gradiente rispetto ai parametri veloci
            grads = torch.autograd.grad(
                loss,
                list(fast_params.values()),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Aggiorna i parametri veloci
            fast_params = {
                name: param - self.inner_lr * (
                    grad if grad is not None else torch.zeros_like(param)
                )
                for (name, param), grad in zip(fast_params.items(), grads)
            }

        return fast_params

    def _compute_query_loss(
        self,
        task: MetaTask,
        adapted_params: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Calcola la loss sul query set usando i parametri adattati.

        Returns:
            (query_loss, breakdown_dict)
        """
        src = task.query_src.to(self.device)
        tgt = task.query_tgt.to(self.device)

        out       = self._forward_with_params(src, adapted_params)
        pred      = out["pred"]
        loss_dict = self.loss_fn(pred, tgt, src=src)

        breakdown = {k: v.item() for k, v in loss_dict.items() if k != "total"}
        return loss_dict["total"], breakdown

    def _forward_with_params(
        self,
        src: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Esegue il forward pass del modello con parametri custom
        (necessario per MAML con higher-order gradients).

        Usa torch.nn.utils.parametrize o monkey-patching temporaneo.
        """
        # Salva i parametri originali
        original_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Sostituisce con i parametri adattati
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in params:
                param.data = params[name]

        # Forward
        B = src.shape[0]
        prototype = torch.zeros(
            B, self.model.set_transformer.output_dim,
            device=self.device,
        )
        out = self.model(src, prototype=prototype)

        # Ripristina i parametri originali
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in original_params:
                param.data = original_params[name]

        return out