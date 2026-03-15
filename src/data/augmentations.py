"""
augmentations.py
----------------
Data augmentation transforms for RAG-ColorNet.

Three families:

1. GeometricAug     – random flip, crop (applied to both src and tgt identically)
2. StyleAug         – style-space perturbations (Lab interpolation, hue/sat jitter)
3. CrossPhotographerAug – cross-photographer retrieval mismatch simulation

All transforms operate on Item dicts {"src": Tensor, "tgt": Tensor, ...}
and return the same dict structure.

Functional utilities (usable outside transforms):
  - interpolate_styles(tgt_a, tgt_b, lam)  — Lab-space linear mix
  - random_crop_pair(src, tgt, scale)       — consistent random crop
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils.color_utils import rgb_to_lab, lab_to_rgb   # type: ignore[import]


Item = Dict


# ---------------------------------------------------------------------------
# Geometric augmentation
# ---------------------------------------------------------------------------

class GeometricAug:
    """
    Random horizontal flip and random crop applied consistently to
    both src and tgt tensors.

    Parameters
    ----------
    flip_prob   : probability of horizontal flip
    crop        : whether to apply random crop
    crop_scale  : (min, max) fraction of original area to crop
    """

    def __init__(
        self,
        flip_prob:  float = 0.5,
        crop:       bool  = True,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
    ) -> None:
        self.flip_prob  = flip_prob
        self.crop       = crop
        self.crop_scale = crop_scale

    def __call__(self, item: Item) -> Item:
        src = item["src"]                         # (3, H, W)
        tgt = item["tgt"]

        # Horizontal flip
        if random.random() < self.flip_prob:
            src = TF.hflip(src)
            tgt = TF.hflip(tgt)

        # Consistent random crop
        if self.crop:
            src, tgt = random_crop_pair(src, tgt, self.crop_scale)

        return {**item, "src": src, "tgt": tgt}


# ---------------------------------------------------------------------------
# Style augmentation
# ---------------------------------------------------------------------------

class StyleAug:
    """
    Light colour-space perturbations on the target image only.
    The source is never modified (it represents the raw/unedited input).

    Perturbations:
    - Luminance jitter : small additive offset to L* channel
    - Hue jitter       : small rotation in a*b* plane
    - Saturation scale : scale factor on chroma radius
    """

    def __init__(
        self,
        lum_jitter:  float = 0.03,   # ±3 L* units (scale 0-1)
        hue_jitter:  float = 0.02,   # ±2° hue rotation (normalised)
        sat_jitter:  float = 0.05,   # ±5% saturation scale
        prob:        float = 0.3,    # probability of applying any perturbation
    ) -> None:
        self.lum_jitter = lum_jitter
        self.hue_jitter = hue_jitter
        self.sat_jitter = sat_jitter
        self.prob       = prob

    def __call__(self, item: Item) -> Item:
        if random.random() >= self.prob:
            return item

        tgt = item["tgt"]                         # (3, H, W) sRGB [0,1]
        tgt_lab = rgb_to_lab(tgt)                 # (3, H, W) Lab

        L, a, b = tgt_lab[0], tgt_lab[1], tgt_lab[2]

        # Luminance jitter
        if self.lum_jitter > 0:
            delta_L = (random.random() * 2 - 1) * self.lum_jitter * 100.0
            L = (L + delta_L).clamp(0, 100)

        # Saturation scale
        if self.sat_jitter > 0:
            scale = 1.0 + (random.random() * 2 - 1) * self.sat_jitter
            a = a * scale
            b = b * scale

        # Hue jitter (rotation in a*b* plane)
        if self.hue_jitter > 0:
            angle = (random.random() * 2 - 1) * self.hue_jitter * math.pi
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            a_new = a * cos_a - b * sin_a
            b_new = a * sin_a + b * cos_a
            a, b = a_new, b_new

        tgt_lab_aug = torch.stack([L, a, b], dim=0)
        tgt_aug = lab_to_rgb(tgt_lab_aug).clamp(0, 1)

        return {**item, "tgt": tgt_aug}


# ---------------------------------------------------------------------------
# Cross-photographer retrieval simulation
# ---------------------------------------------------------------------------

class CrossPhotographerAug:
    """
    Replaces the target with an edit from a different photographer.

    This teaches the MaskNet to detect low-confidence retrieval:
    when the retrieved edit is systematically wrong (wrong style),
    the confidence mask should fall back to the global grid.

    Parameters
    ----------
    foreign_targets : list of target tensors from *other* photographers
    prob            : probability of substituting the target
    """

    def __init__(
        self,
        foreign_targets: List[torch.Tensor],
        prob: float = 0.15,
    ) -> None:
        self.foreign_targets = foreign_targets
        self.prob            = prob

    def __call__(self, item: Item) -> Item:
        if not self.foreign_targets or random.random() >= self.prob:
            return item

        tgt = random.choice(self.foreign_targets)
        meta = {**item["meta"], "cross_photographer": True}
        return {**item, "tgt": tgt, "meta": meta}


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, item: Item) -> Item:
        for t in self.transforms:
            item = t(item)
        return item


# ---------------------------------------------------------------------------
# Functional utilities
# ---------------------------------------------------------------------------

def interpolate_styles(
    tgt_a: torch.Tensor,
    tgt_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Linear interpolation of two target images in CIE Lab space.

        I_tgt_λ = Lab^{-1}(λ · Lab(I_tgt_a) + (1-λ) · Lab(I_tgt_b))

    Parameters
    ----------
    tgt_a, tgt_b : (3, H, W) sRGB tensors in [0, 1]
    lam          : interpolation factor ∈ [0, 1]
                   lam=0 → pure tgt_b, lam=1 → pure tgt_a

    Returns
    -------
    (3, H, W) sRGB tensor in [0, 1]
    """
    assert 0.0 <= lam <= 1.0, f"lam must be in [0,1], got {lam}"

    lab_a = rgb_to_lab(tgt_a)
    lab_b = rgb_to_lab(tgt_b)

    lab_mix = lam * lab_a + (1.0 - lam) * lab_b
    return lab_to_rgb(lab_mix).clamp(0.0, 1.0)


def random_crop_pair(
    src: torch.Tensor,
    tgt: torch.Tensor,
    scale: Tuple[float, float] = (0.8, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the same random crop to both src and tgt.

    Parameters
    ----------
    src, tgt : (3, H, W) tensors
    scale    : (min_frac, max_frac) of area to retain

    Returns
    -------
    cropped (src, tgt) tensors of identical spatial size
    """
    _, H, W = src.shape
    area    = H * W

    for _ in range(10):                           # up to 10 attempts
        frac       = random.uniform(scale[0], scale[1])
        new_area   = int(area * frac)
        aspect     = W / H
        new_h      = int(math.sqrt(new_area / aspect))
        new_w      = int(math.sqrt(new_area * aspect))

        if new_h <= 0 or new_w <= 0 or new_h > H or new_w > W:
            continue

        top  = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)

        src_crop = src[:, top:top + new_h, left:left + new_w]
        tgt_crop = tgt[:, top:top + new_h, left:left + new_w]

        # Resize back to original dimensions
        src_out = F.interpolate(
            src_crop.unsqueeze(0), size=(H, W),
            mode="bicubic", align_corners=False, antialias=True,
        ).squeeze(0).clamp(0, 1)

        tgt_out = F.interpolate(
            tgt_crop.unsqueeze(0), size=(H, W),
            mode="bicubic", align_corners=False, antialias=True,
        ).squeeze(0).clamp(0, 1)

        return src_out, tgt_out

    # Fallback: return originals unchanged
    return src, tgt


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_train_transforms(cfg: dict) -> Compose:
    """
    Build a Compose transform from the augmentation section of pretraining.yaml.
    """
    acfg = cfg.get("augmentation", {})
    transforms = []

    if acfg.get("random_flip_h", 0) > 0 or acfg.get("random_crop", False):
        transforms.append(GeometricAug(
            flip_prob=acfg.get("random_flip_h", 0.5),
            crop=acfg.get("random_crop", True),
            crop_scale=tuple(acfg.get("crop_scale", [0.8, 1.0])),
        ))

    if not acfg.get("color_jitter", False):
        # style jitter is mild; skip if config disables color changes
        pass
    else:
        transforms.append(StyleAug())

    return Compose(transforms)
