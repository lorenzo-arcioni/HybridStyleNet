"""
task_sampler.py
---------------
Task (episode) sampling for Reptile meta-training (Phase 2).

A "task" is a single photographer's dataset split into:
  - support set  D_sup  (15 pairs by default) — used in the inner loop
  - query set    D_qry  (5 pairs by default)  — used to evaluate the adapted θ

TaskSampler manages a pool of real photographer tasks (from FiveK / PPR10K)
and synthetic tasks (style interpolation between real photographers).

Usage
-----
    sampler = TaskSampler(real_datasets, cfg)
    for task in sampler.sample_batch(M=2):
        support_loader = task.support_loader()
        query_loader   = task.query_loader()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .datasets import FiveKDataset, PPR10KDataset
from .augmentations import interpolate_styles, CrossPhotographerAug


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single meta-learning episode."""
    support_indices: List[int]
    query_indices:   List[int]
    dataset:         Dataset
    task_id:         str = ""

    def support_loader(self, batch_size: int = 1, **kwargs) -> DataLoader:
        sub = Subset(self.dataset, self.support_indices)
        return DataLoader(sub, batch_size=batch_size, shuffle=True, **kwargs)

    def query_loader(self, batch_size: int = 1, **kwargs) -> DataLoader:
        sub = Subset(self.dataset, self.query_indices)
        return DataLoader(sub, batch_size=batch_size, shuffle=False, **kwargs)

    @property
    def n_support(self) -> int:
        return len(self.support_indices)

    @property
    def n_query(self) -> int:
        return len(self.query_indices)


# ---------------------------------------------------------------------------
# TaskSampler
# ---------------------------------------------------------------------------

class TaskSampler:
    """
    Samples Reptile tasks from a pool of real and synthetic photographer tasks.

    Parameters
    ----------
    photographer_datasets : list of (dataset, photographer_id) — real tasks
    support_size          : |D_sup| per task
    query_size            : |D_qry| per task
    synthetic_enabled     : whether to include style-interpolated tasks
    cross_photo_enabled   : whether to include cross-photographer retrieval tasks
    seed                  : random seed
    """

    def __init__(
        self,
        photographer_datasets: List[Tuple[Dataset, str]],
        support_size: int = 15,
        query_size:   int = 5,
        synthetic_enabled:   bool = True,
        cross_photo_enabled: bool = True,
        seed: int = 0,
    ) -> None:
        self._real_ds     = photographer_datasets       # [(dataset, id), ...]
        self._support_sz  = support_size
        self._query_sz    = query_size
        self._syn_enabled = synthetic_enabled
        self._cross_enabled = cross_photo_enabled
        self._rng         = random.Random(seed)

        # Pre-build synthetic task pool
        self._synthetic_tasks: List[Task] = []
        if synthetic_enabled and len(self._real_ds) >= 2:
            self._synthetic_tasks = self._build_synthetic_tasks()

    # ------------------------------------------------------------------
    def sample_batch(self, M: int = 2) -> List[Task]:
        """
        Sample M tasks for one Reptile outer step.

        Mix: ~70 % real, ~30 % synthetic (if available).
        """
        tasks: List[Task] = []
        for _ in range(M):
            use_synthetic = (
                self._syn_enabled
                and self._synthetic_tasks
                and self._rng.random() < 0.30
            )
            if use_synthetic:
                tasks.append(self._rng.choice(self._synthetic_tasks))
            else:
                tasks.append(self._sample_real_task())
        return tasks

    # ------------------------------------------------------------------
    def _sample_real_task(self) -> Task:
        """Sample support + query indices from a random real photographer."""
        dataset, pid = self._rng.choice(self._real_ds)
        n = len(dataset)
        required = self._support_sz + self._query_sz

        if n < required:
            # sample with replacement if not enough pairs
            indices = self._rng.choices(range(n), k=required)
        else:
            indices = self._rng.sample(range(n), k=required)

        support_idx = indices[:self._support_sz]
        query_idx   = indices[self._support_sz:]

        return Task(
            support_indices=support_idx,
            query_indices=query_idx,
            dataset=dataset,
            task_id=f"real_{pid}",
        )

    # ------------------------------------------------------------------
    def _build_synthetic_tasks(self, n_lambdas: int = 10) -> List[Task]:
        """
        Build style-interpolated tasks from all pairs of real photographers.

        For photographers i and j, with interpolation factor λ ∈ (0,1):
          I_tgt_λ = Lab^{-1}(λ * Lab(I_tgt_i) + (1-λ) * Lab(I_tgt_j))
        """
        tasks: List[Task] = []
        lambdas = [k / (n_lambdas + 1) for k in range(1, n_lambdas + 1)]

        for i in range(len(self._real_ds)):
            for j in range(i + 1, len(self._real_ds)):
                ds_i, pid_i = self._real_ds[i]
                ds_j, pid_j = self._real_ds[j]

                for lam in lambdas:
                    syn_ds = SyntheticInterpolationDataset(ds_i, ds_j, lam)
                    n = len(syn_ds)
                    required = self._support_sz + self._query_sz
                    if n < required:
                        continue

                    indices = list(range(n))
                    self._rng.shuffle(indices)

                    tasks.append(Task(
                        support_indices=indices[:self._support_sz],
                        query_indices=indices[self._support_sz:self._support_sz + self._query_sz],
                        dataset=syn_ds,
                        task_id=f"synthetic_{pid_i}_{pid_j}_l{lam:.2f}",
                    ))

        return tasks

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Task]:
        """Infinite iterator yielding single tasks (for training loops)."""
        while True:
            yield self._sample_real_task()


# ---------------------------------------------------------------------------
# Synthetic interpolation dataset
# ---------------------------------------------------------------------------

class SyntheticInterpolationDataset(Dataset):
    """
    Wraps two real datasets (same sources, different experts) and
    interpolates their targets in CIE Lab space.

    Both datasets must have the same length and aligned sources.
    """

    def __init__(
        self,
        dataset_a: Dataset,
        dataset_b: Dataset,
        lam: float,
    ) -> None:
        assert len(dataset_a) == len(dataset_b), (
            "Interpolation datasets must have the same length."
        )
        self._ds_a = dataset_a
        self._ds_b = dataset_b
        self._lam  = lam

    def __len__(self) -> int:
        return len(self._ds_a)

    def __getitem__(self, idx: int) -> Dict:
        item_a = self._ds_a[idx]
        item_b = self._ds_b[idx]

        src = item_a["src"]                       # source is the same
        tgt = interpolate_styles(
            item_a["tgt"], item_b["tgt"], self._lam
        )

        return {
            "src":  src,
            "tgt":  tgt,
            "idx":  idx,
            "meta": {
                "dataset":  "synthetic_interpolation",
                "lambda":   self._lam,
                "src_file": item_a["meta"].get("src_file", ""),
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_task_sampler(
    cfg: dict,
    fivek_root:  Optional[str] = None,
    ppr10k_root: Optional[str] = None,
) -> TaskSampler:
    """
    Build a TaskSampler from the meta_training.yaml config and dataset roots.
    """
    tcfg = cfg["tasks"]
    size = (512, 384)                             # fixed for meta-training

    real_datasets: List[Tuple[Dataset, str]] = []

    # FiveK photographers (one dataset per expert)
    if fivek_root and tcfg["real"]["fivek_photographers"] > 0:
        from .datasets import FiveKDataset
        n_experts = tcfg["real"]["fivek_photographers"]
        experts   = ["A", "B", "C", "D", "E"][:n_experts]
        for expert in experts:
            ds = FiveKDataset(
                root=fivek_root,
                experts=[expert],
                target_size=size,
            )
            real_datasets.append((ds, f"fivek_{expert}"))

    # PPR10K photographers
    if ppr10k_root and tcfg["real"]["ppr10k_photographers"] > 0:
        from .datasets import PPR10KDataset
        ds = PPR10KDataset(root=ppr10k_root, target_size=size)
        real_datasets.append((ds, "ppr10k"))

    return TaskSampler(
        photographer_datasets=real_datasets,
        support_size=tcfg["real"]["support_size"],
        query_size=tcfg["real"]["query_size"],
        synthetic_enabled=tcfg["synthetic"]["enabled"],
        cross_photo_enabled=tcfg["cross_photographer"]["enabled"],
    )
