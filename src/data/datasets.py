"""
datasets.py
-----------
PyTorch Dataset classes for the three pre-training sources:

  • FiveKDataset          – MIT-Adobe FiveK (5 expert retouchers × 1000 pairs)
  • PPR10KDataset         – PPR10K portrait retouching dataset
  • LightroomPresetsDataset – synthetic pairs from Lightroom presets

All datasets return dict items with keys:
  src  : (3, H, W) float32 tensor  – sRGB source image in [0, 1]
  tgt  : (3, H, W) float32 tensor  – sRGB retouched image in [0, 1]
  meta : dict with photographer/preset id and filename

A CombinedDataset merges all three with configurable sampling weights.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from .raw_pipeline import load_image, load_pair


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Item = Dict[str, torch.Tensor | dict]


# ---------------------------------------------------------------------------
# FiveK
# ---------------------------------------------------------------------------

class FiveKDataset(Dataset):
    """
    MIT-Adobe FiveK dataset.

    Expected directory layout::

        root/
          input/          # RAW or linear TIFF source images
            a0001.dng
            ...
          expertA/        # JPEG or TIFF edits by expert A
            a0001.jpg
            ...
          expertB/
            ...

    Parameters
    ----------
    root       : path to the dataset root
    experts    : list of expert names, e.g. ["A", "B"]; None = all five
    max_pairs  : cap on total pairs loaded (per expert)
    target_size: (H, W) resize resolution; None = original
    transform  : optional additional augmentation callable
    """

    EXPERTS = ["A", "B", "C", "D", "E"]

    def __init__(
        self,
        root: str | Path,
        experts: Optional[List[str]] = None,
        max_pairs: int = 1000,
        target_size: Optional[Tuple[int, int]] = (512, 384),
        keep_aspect: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root        = Path(root)
        self.experts     = experts or self.EXPERTS
        self.target_size = target_size
        self.keep_aspect = keep_aspect
        self.transform   = transform

        self._pairs: List[Tuple[Path, Path, str]] = []   # (src, tgt, expert)
        self._build_index(max_pairs)

    # ------------------------------------------------------------------
    def _build_index(self, max_pairs: int) -> None:
        src_dir = self.root / "input"
        if not src_dir.exists():
            raise FileNotFoundError(f"FiveK input dir not found: {src_dir}")

        src_files = sorted(src_dir.iterdir())

        for expert in self.experts:
            tgt_dir = self.root / f"expert{expert}"
            if not tgt_dir.exists():
                raise FileNotFoundError(f"FiveK expert dir not found: {tgt_dir}")

            count = 0
            for src_path in src_files:
                if src_path.suffix.lower() not in {
                    ".dng", ".tif", ".tiff", ".jpg", ".jpeg"
                }:
                    continue
                # find matching target by stem
                tgt_candidates = list(tgt_dir.glob(src_path.stem + ".*"))
                if not tgt_candidates:
                    continue
                self._pairs.append((src_path, tgt_candidates[0], expert))
                count += 1
                if count >= max_pairs:
                    break

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Item:
        src_path, tgt_path, expert = self._pairs[idx]

        src, tgt = load_pair(
            src_path, tgt_path,
            target_size=self.target_size,
            keep_aspect=self.keep_aspect,
        )

        item: Item = {
            "src": src,
            "tgt": tgt,
            "meta": {
                "dataset":      "fivek",
                "photographer": f"expert_{expert}",
                "src_file":     src_path.name,
                "tgt_file":     tgt_path.name,
            },
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


# ---------------------------------------------------------------------------
# PPR10K
# ---------------------------------------------------------------------------

class PPR10KDataset(Dataset):
    """
    PPR10K portrait retouching dataset.

    Expected layout::

        root/
          source/
            0001.jpg
            ...
          target_a/      # one of several retoucher subdirs
            0001.jpg
            ...
          target_b/
            ...

    Parameters
    ----------
    root         : dataset root
    target_subdirs: list of target subdirectory names to include
    use_masks    : whether to load portrait masks (if available)
    """

    def __init__(
        self,
        root: str | Path,
        target_subdirs: Optional[List[str]] = None,
        use_masks: bool = False,
        target_size: Optional[Tuple[int, int]] = (512, 384),
        keep_aspect: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root           = Path(root)
        self.use_masks      = use_masks
        self.target_size    = target_size
        self.keep_aspect    = keep_aspect
        self.transform      = transform

        self._pairs: List[Tuple[Path, Path, str]] = []
        self._build_index(target_subdirs)

    # ------------------------------------------------------------------
    def _build_index(self, target_subdirs: Optional[List[str]]) -> None:
        src_dir = self.root / "source"
        if not src_dir.exists():
            raise FileNotFoundError(f"PPR10K source dir not found: {src_dir}")

        if target_subdirs is None:
            target_subdirs = [
                d.name for d in self.root.iterdir()
                if d.is_dir() and d.name.startswith("target")
            ]

        src_files = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
        )

        for subdir in target_subdirs:
            tgt_dir = self.root / subdir
            if not tgt_dir.exists():
                continue
            for src_path in src_files:
                candidates = list(tgt_dir.glob(src_path.stem + ".*"))
                if candidates:
                    self._pairs.append((src_path, candidates[0], subdir))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Item:
        src_path, tgt_path, subdir = self._pairs[idx]

        src, tgt = load_pair(
            src_path, tgt_path,
            target_size=self.target_size,
            keep_aspect=self.keep_aspect,
        )

        item: Item = {
            "src": src,
            "tgt": tgt,
            "meta": {
                "dataset":      "ppr10k",
                "photographer": subdir,
                "src_file":     src_path.name,
                "tgt_file":     tgt_path.name,
            },
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


# ---------------------------------------------------------------------------
# Lightroom Presets (synthetic)
# ---------------------------------------------------------------------------

class LightroomPresetsDataset(Dataset):
    """
    Synthetic dataset from Lightroom presets applied to base images.

    Expected layout::

        root/
          base/           # un-edited sRGB base images
            img_0001.jpg
            ...
          preset_0001/    # each preset has its own subdir
            img_0001.jpg  # same filename as base, after preset was applied
            ...
          preset_0002/
            ...

    Each (base, preset-applied) pair is treated as one training example.
    """

    def __init__(
        self,
        root: str | Path,
        n_presets: Optional[int] = None,       # cap number of presets to load
        target_size: Optional[Tuple[int, int]] = (512, 384),
        keep_aspect: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root        = Path(root)
        self.target_size = target_size
        self.keep_aspect = keep_aspect
        self.transform   = transform

        self._pairs: List[Tuple[Path, Path, str]] = []
        self._build_index(n_presets)

    # ------------------------------------------------------------------
    def _build_index(self, n_presets: Optional[int]) -> None:
        base_dir = self.root / "base"
        if not base_dir.exists():
            raise FileNotFoundError(f"Lightroom base dir not found: {base_dir}")

        preset_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("preset")
        )
        if n_presets is not None:
            preset_dirs = preset_dirs[:n_presets]

        base_files = sorted(
            p for p in base_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
        )

        for preset_dir in preset_dirs:
            for base_path in base_files:
                candidates = list(preset_dir.glob(base_path.stem + ".*"))
                if candidates:
                    self._pairs.append((base_path, candidates[0], preset_dir.name))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Item:
        src_path, tgt_path, preset = self._pairs[idx]

        src, tgt = load_pair(
            src_path, tgt_path,
            target_size=self.target_size,
            keep_aspect=self.keep_aspect,
        )

        item: Item = {
            "src": src,
            "tgt": tgt,
            "meta": {
                "dataset":      "lightroom_presets",
                "photographer": preset,
                "src_file":     src_path.name,
                "tgt_file":     tgt_path.name,
            },
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):
    """
    Merges multiple sub-datasets with per-dataset sampling weights.

    Parameters
    ----------
    datasets : list of (dataset, weight) tuples
               weight controls relative sampling frequency
    """

    def __init__(self, datasets: List[Tuple[Dataset, float]]) -> None:
        self._datasets, weights_raw = zip(*datasets)
        self._offsets: List[int] = []
        offset = 0
        for ds in self._datasets:
            self._offsets.append(offset)
            offset += len(ds)
        self._total = offset

        # Build per-sample weights for WeightedRandomSampler
        sample_weights: List[float] = []
        for ds, w in zip(self._datasets, weights_raw):
            sample_weights.extend([w / len(ds)] * len(ds))

        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Item:
        # find which sub-dataset owns this index
        ds_idx = 0
        for i, offset in enumerate(self._offsets):
            if idx >= offset:
                ds_idx = i
        local_idx = idx - self._offsets[ds_idx]
        return self._datasets[ds_idx][local_idx]

    def make_sampler(self, num_samples: Optional[int] = None) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for use with DataLoader."""
        n = num_samples or self._total
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=n,
            replacement=True,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pretrain_dataset(cfg: dict) -> CombinedDataset:
    """
    Build the combined pre-training dataset from a config dict
    (loaded from configs/pretraining.yaml → data section).
    """
    sub: List[Tuple[Dataset, float]] = []

    dcfg = cfg["data"]
    size = tuple(dcfg.get("train_resolution", [512, 384]))
    keep = dcfg.get("keep_aspect", True)

    if dcfg["datasets"]["fivek"]["enabled"]:
        fc = dcfg["datasets"]["fivek"]
        sub.append((
            FiveKDataset(
                root=Path(cfg["paths"]["data_root"]) / "fivek",
                experts=fc.get("experts"),
                max_pairs=fc.get("pairs_per_expert", 1000),
                target_size=size,
                keep_aspect=keep,
            ),
            1.0,
        ))

    if dcfg["datasets"]["ppr10k"]["enabled"]:
        sub.append((
            PPR10KDataset(
                root=Path(cfg["paths"]["data_root"]) / "ppr10k",
                target_size=size,
                keep_aspect=keep,
            ),
            1.0,
        ))

    if dcfg["datasets"]["lightroom_presets"]["enabled"]:
        lc = dcfg["datasets"]["lightroom_presets"]
        sub.append((
            LightroomPresetsDataset(
                root=Path(cfg["paths"]["data_root"]) / "lightroom_presets",
                n_presets=lc.get("n_presets"),
                target_size=size,
                keep_aspect=keep,
            ),
            0.5,                                 # lower weight: synthetic data
        ))

    if not sub:
        raise ValueError("No datasets enabled in pretraining config.")

    return CombinedDataset(sub)
