"""
photographer_dataset.py
-----------------------
Dataset for few-shot per-photographer adaptation (Phase 3).

Loads the N (src, tgt) pairs provided by the target photographer,
splits them into train / val sets, and exposes utilities for the
database preprocessing step (DINOv2 feature caching).

All pairs live in a single flat directory structure:

    pairs_dir/
      src/
        001.jpg   (or .tif / .arw / .dng)
        ...
      tgt/
        001.jpg
        ...

Filenames must match between src/ and tgt/ by stem.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset

from .raw_pipeline import load_image, load_pair


Item = Dict


# ---------------------------------------------------------------------------
# PhotographerDataset
# ---------------------------------------------------------------------------

class PhotographerDataset(Dataset):
    """
    Full dataset for a single photographer's (src, tgt) pairs.

    Parameters
    ----------
    pairs_dir   : root dir with src/ and tgt/ subdirs
    src_subdir  : name of source subdirectory (default "src")
    tgt_subdir  : name of target subdirectory (default "tgt")
    extensions  : allowed file extensions
    target_size : (H, W) resize; None = original
    keep_aspect : pad instead of stretch
    """

    def __init__(
        self,
        pairs_dir: str | Path,
        src_subdir: str = "src",
        tgt_subdir: str = "tgt",
        extensions: Optional[List[str]] = None,
        target_size: Optional[Tuple[int, int]] = (512, 384),
        keep_aspect: bool = True,
    ) -> None:
        self.pairs_dir   = Path(pairs_dir)
        self.target_size = target_size
        self.keep_aspect = keep_aspect

        self._exts = set(extensions or [
            ".jpg", ".jpeg", ".tif", ".tiff",
            ".arw", ".dng", ".cr2", ".nef",
        ])

        self._pairs = self._build_index(src_subdir, tgt_subdir)

    # ------------------------------------------------------------------
    def _build_index(
        self, src_subdir: str, tgt_subdir: str
    ) -> List[Tuple[Path, Path]]:
        src_dir = self.pairs_dir / src_subdir
        tgt_dir = self.pairs_dir / tgt_subdir

        if not src_dir.exists():
            raise FileNotFoundError(f"Source dir not found: {src_dir}")
        if not tgt_dir.exists():
            raise FileNotFoundError(f"Target dir not found: {tgt_dir}")

        pairs: List[Tuple[Path, Path]] = []
        for src_path in sorted(src_dir.iterdir()):
            if src_path.suffix.lower() not in self._exts:
                continue
            matches = [
                p for p in tgt_dir.iterdir()
                if p.stem == src_path.stem and p.suffix.lower() in self._exts
            ]
            if not matches:
                continue                          # skip unpaired files
            pairs.append((src_path, matches[0]))

        if not pairs:
            raise ValueError(
                f"No matched (src, tgt) pairs found in {self.pairs_dir}. "
                "Check that filenames share the same stem."
            )

        return pairs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Item:
        src_path, tgt_path = self._pairs[idx]

        src, tgt = load_pair(
            src_path, tgt_path,
            target_size=self.target_size,
            keep_aspect=self.keep_aspect,
        )

        return {
            "src":  src,
            "tgt":  tgt,
            "idx":  idx,
            "meta": {
                "src_file": src_path.name,
                "tgt_file": tgt_path.name,
            },
        }

    # ------------------------------------------------------------------
    # Convenience: load at original resolution (for inference / caching)
    # ------------------------------------------------------------------
    def load_original(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (src, tgt) at full resolution (no resize)."""
        src_path, tgt_path = self._pairs[idx]
        return load_pair(src_path, tgt_path, target_size=None)

    def src_path(self, idx: int) -> Path:
        return self._pairs[idx][0]

    def tgt_path(self, idx: int) -> Path:
        return self._pairs[idx][1]


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: PhotographerDataset,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Random stratified split into train and val subsets.

    Returns
    -------
    train_subset, val_subset : torch.utils.data.Subset objects
    """
    n     = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_tr  = n - n_val

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_idx = indices[:n_tr]
    val_idx   = indices[n_tr:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

def build_photographer_datasets(
    cfg: dict,
) -> Tuple[Subset, Subset, PhotographerDataset]:
    """
    Build train/val splits and the full dataset from a merged config dict
    (base.yaml merged with photographer.yaml).

    Returns
    -------
    train_subset, val_subset, full_dataset
    """
    pcfg = cfg["data"]
    size = tuple(pcfg.get("train_resolution", [512, 384]))
    keep = pcfg.get("keep_aspect", True)

    full = PhotographerDataset(
        pairs_dir=pcfg["pairs_dir"],
        src_subdir=pcfg.get("src_subdir", "src"),
        tgt_subdir=pcfg.get("tgt_subdir", "tgt"),
        extensions=pcfg.get("image_extensions"),
        target_size=size,
        keep_aspect=keep,
    )

    train_sub, val_sub = split_dataset(
        full,
        val_fraction=pcfg.get("val_split", 0.20),
        seed=pcfg.get("val_split_seed", 42),
    )

    return train_sub, val_sub, full
