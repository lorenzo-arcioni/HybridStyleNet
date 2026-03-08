"""
training/task_sampler.py

Task Sampler per MAML con Task Augmentation  (§5.3).

MetaTask:
    Struttura dati per un singolo task (support + query set).

TaskSampler:
    Campiona task reali (FiveK per-photographer) e task sintetici
    (interpolazione di stili in spazio CIE Lab).
"""

import logging
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import FiveKDataset, PairedDataset
from data.raw_pipeline import RawPipeline
from utils.color_space import rgb_to_lab, lab_to_rgb

logger = logging.getLogger(__name__)


@dataclass
class MetaTask:
    """
    Struttura dati per un singolo task MAML.

    Attributes:
        support_src:  (K_s, 3, H, W) — sorgenti support set.
        support_tgt:  (K_s, 3, H, W) — target support set.
        query_src:    (K_q, 3, H, W) — sorgenti query set.
        query_tgt:    (K_q, 3, H, W) — target query set.
        task_id:      Identificatore del task (es. "expert_A" o "synth_AB_0.3").
    """
    support_src: torch.Tensor
    support_tgt: torch.Tensor
    query_src:   torch.Tensor
    query_tgt:   torch.Tensor
    task_id:     str = ""


def interpolate_styles_lab(
    tgt_a: torch.Tensor,
    tgt_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Interpola due immagini target in spazio CIE Lab  (§5.3).

    I^tgt_λ = Lab^{-1}(λ · Lab(I^tgt_a) + (1-λ) · Lab(I^tgt_b))

    L'interpolazione in Lab produce uno stile percettivamente intermedio
    (§5.3 — motivazione della scelta di Lab vs RGB).

    Args:
        tgt_a: (N, 3, H, W) in [0,1] — target fotografo A.
        tgt_b: (N, 3, H, W) in [0,1] — target fotografo B.
        lam:   Parametro interpolazione ∈ [0, 1].

    Returns:
        (N, 3, H, W) in [0,1] — target sintetico interpolato.
    """
    # (B,3,H,W) → (B,H,W,3) per le funzioni color_space
    a_hw = tgt_a.permute(0, 2, 3, 1)
    b_hw = tgt_b.permute(0, 2, 3, 1)

    lab_a = rgb_to_lab(a_hw)   # (B, H, W, 3)
    lab_b = rgb_to_lab(b_hw)   # (B, H, W, 3)

    # Interpolazione lineare in Lab
    lab_mix = lam * lab_a + (1.0 - lam) * lab_b  # (B, H, W, 3)

    # Riconversione in sRGB
    rgb_mix = lab_to_rgb(lab_mix)                 # (B, H, W, 3) in [0,1]

    return rgb_mix.permute(0, 3, 1, 2)            # (B, 3, H, W)


class TaskSampler:
    """
    Campiona task per il meta-training MAML  (§5.3).

    Gestisce:
      - Task reali: coppie (support, query) da FiveK per-expert.
      - Task sintetici: interpolazione Lab tra coppie di expert.

    Args:
        fivek_root:       Radice dataset FiveK.
        experts:          Lista di expert da usare (es. ['A','B','C']).
        support_size:     K_s — coppie nel support set.
        query_size:       K_q — coppie nel query set.
        synthetic_prob:   Probabilità di campionare un task sintetico.
        lambda_min:       Estremo inferiore dell'interpolazione.
        lambda_max:       Estremo superiore dell'interpolazione.
        seed:             Seme per riproducibilità.
        pipeline:         Istanza RawPipeline (None → crea default).
        max_pairs_per_expert: Massimo coppie per expert.
    """

    def __init__(
        self,
        fivek_root: str,
        experts: List[str],
        support_size: int = 15,
        query_size: int = 5,
        synthetic_prob: float = 0.5,
        lambda_min: float = 0.1,
        lambda_max: float = 0.9,
        seed: int = 42,
        pipeline: Optional[RawPipeline] = None,
        max_pairs_per_expert: int = 1000,
    ) -> None:
        self.experts        = experts
        self.support_size   = support_size
        self.query_size     = query_size
        self.synthetic_prob = synthetic_prob
        self.lambda_min     = lambda_min
        self.lambda_max     = lambda_max
        self._rng           = random.Random(seed)

        if pipeline is None:
            pipeline = RawPipeline(target_long_side=256)

        # Carica i dataset per ogni expert
        self._datasets: Dict[str, PairedDataset] = {}
        for expert in experts:
            try:
                ds = FiveKDataset(
                    fivek_root=fivek_root,
                    expert=expert,
                    split="train",
                    pipeline=pipeline,
                    augmentation=None,
                    max_pairs=max_pairs_per_expert,
                )
                self._datasets[expert] = ds
                logger.info(
                    f"TaskSampler: expert {expert} → {len(ds)} coppie"
                )
            except Exception as e:
                logger.warning(
                    f"TaskSampler: impossibile caricare expert {expert}: {e}"
                )

        if not self._datasets:
            raise RuntimeError(
                "TaskSampler: nessun dataset FiveK disponibile. "
                "Verifica fivek_root e la struttura delle directory."
            )

        # Coppie di expert per task sintetici
        self._expert_pairs = list(combinations(list(self._datasets.keys()), 2))
        logger.info(
            f"TaskSampler inizializzato: {len(self._datasets)} expert reali, "
            f"{len(self._expert_pairs)} coppie sintetiche"
        )

    def sample_batch(self, n_tasks: int = 3) -> List[MetaTask]:
        """
        Campiona un batch di task per un'iterazione MAML.

        Args:
            n_tasks: M — numero di task nel batch.

        Returns:
            Lista di MetaTask.
        """
        tasks = []
        for _ in range(n_tasks):
            if (
                len(self._expert_pairs) > 0
                and self._rng.random() < self.synthetic_prob
            ):
                task = self._sample_synthetic_task()
            else:
                task = self._sample_real_task()
            tasks.append(task)
        return tasks

    def _sample_real_task(self) -> MetaTask:
        """Campiona un task reale da un expert casuale."""
        expert = self._rng.choice(list(self._datasets.keys()))
        ds     = self._datasets[expert]
        n      = len(ds)
        k_tot  = self.support_size + self.query_size

        if n < k_tot:
            # Campionamento con rimpiazzo se il dataset è troppo piccolo
            indices = [self._rng.randint(0, n - 1) for _ in range(k_tot)]
        else:
            indices = self._rng.sample(range(n), k_tot)

        sup_idx = indices[:self.support_size]
        qry_idx = indices[self.support_size:]

        sup_src, sup_tgt = self._load_samples(ds, sup_idx)
        qry_src, qry_tgt = self._load_samples(ds, qry_idx)

        return MetaTask(
            support_src=sup_src,
            support_tgt=sup_tgt,
            query_src=qry_src,
            query_tgt=qry_tgt,
            task_id=f"expert_{expert}",
        )

    def _sample_synthetic_task(self) -> MetaTask:
        """
        Campiona un task sintetico interpolando due expert in Lab.
        """
        exp_a, exp_b = self._rng.choice(self._expert_pairs)
        lam = self._rng.uniform(self.lambda_min, self.lambda_max)

        ds_a = self._datasets[exp_a]
        ds_b = self._datasets[exp_b]

        # Trova coppie con lo stesso sorgente tra i due expert
        # (in FiveK, tutti gli expert condividono gli stessi sorgenti)
        n      = min(len(ds_a), len(ds_b))
        k_tot  = self.support_size + self.query_size

        if n < k_tot:
            indices = [self._rng.randint(0, n - 1) for _ in range(k_tot)]
        else:
            indices = self._rng.sample(range(n), k_tot)

        sup_idx = indices[:self.support_size]
        qry_idx = indices[self.support_size:]

        # Carica sorgenti (identici tra A e B per FiveK)
        sup_src, sup_tgt_a = self._load_samples(ds_a, sup_idx)
        _,        sup_tgt_b = self._load_samples(ds_b, sup_idx)

        qry_src, qry_tgt_a = self._load_samples(ds_a, qry_idx)
        _,        qry_tgt_b = self._load_samples(ds_b, qry_idx)

        # Interpolazione Lab (§5.3)
        sup_tgt_synth = interpolate_styles_lab(sup_tgt_a, sup_tgt_b, lam)
        qry_tgt_synth = interpolate_styles_lab(qry_tgt_a, qry_tgt_b, lam)

        return MetaTask(
            support_src=sup_src,
            support_tgt=sup_tgt_synth,
            query_src=qry_src,
            query_tgt=qry_tgt_synth,
            task_id=f"synth_{exp_a}{exp_b}_lam{lam:.2f}",
        )

    @staticmethod
    def _load_samples(
        ds: PairedDataset,
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carica un sottoinsieme di campioni da un dataset.

        Returns:
            src_batch: (N, 3, H, W)
            tgt_batch: (N, 3, H, W)
        """
        srcs, tgts = [], []
        for idx in indices:
            sample = ds[idx]
            srcs.append(sample["src"])
            tgts.append(sample["tgt"])

        src_batch = torch.stack(srcs, dim=0)
        tgt_batch = torch.stack(tgts, dim=0)
        return src_batch, tgt_batch