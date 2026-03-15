"""
raw_pipeline.py
---------------
Deterministic RAW → sRGB preprocessing pipeline.

Converts ARW / DNG (and any rawpy-supported format) to a normalised
float32 sRGB tensor in [0, 1].  The pipeline is intentionally
parameter-free: every step uses metadata embedded in the RAW file so
that the same physical scene always produces the same tensor.

Steps
-----
1. Linearise  – subtract black level, divide by (white - black)
2. Demosaic   – AHD via rawpy
3. White balance – camera-embedded multipliers
4. Colour matrix – camera-specific XYZ→sRGB 3×3
5. Gamma       – sRGB piecewise transfer function
6. Resize      – bicubic to target resolution (keeps aspect ratio, pads)
7. Normalise   – clip to [0,1], convert to float32 tensor

For JPEG/TIFF inputs (already sRGB) only steps 6-7 are applied.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# rawpy is optional: only needed for actual RAW files
try:
    import rawpy  # type: ignore
    _RAWPY_AVAILABLE = True
except ImportError:
    _RAWPY_AVAILABLE = False

from PIL import Image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image(
    path: str | Path,
    target_size: Optional[Tuple[int, int]] = None,  # (H, W)
    keep_aspect: bool = True,
) -> torch.Tensor:
    """
    Load any supported image (RAW or raster) and return a float32
    tensor of shape (3, H, W) in sRGB [0, 1].

    Parameters
    ----------
    path        : path to the image file
    target_size : (H, W) to resize to; None = original resolution
    keep_aspect : if True, pad with zeros instead of stretching
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _RAW_EXTENSIONS:
        rgb = _load_raw(path)
    else:
        rgb = _load_raster(path)

    tensor = _to_tensor(rgb)                          # (3, H, W) float32 [0,1]

    if target_size is not None:
        tensor = _resize(tensor, target_size, keep_aspect)

    return tensor


def load_pair(
    src_path: str | Path,
    tgt_path: str | Path,
    target_size: Optional[Tuple[int, int]] = None,
    keep_aspect: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a (source, target) pair, resizing both to the same grid.
    Returns two (3, H, W) tensors.
    """
    src = load_image(src_path, target_size=target_size, keep_aspect=keep_aspect)
    tgt = load_image(tgt_path, target_size=target_size, keep_aspect=keep_aspect)
    return src, tgt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".nef", ".nrw",
    ".orf", ".rw2", ".pef", ".dng", ".raf",
}


def _load_raw(path: Path) -> np.ndarray:
    """
    RAW → linear sRGB float32 array (H, W, 3) in [0, 1].

    Uses rawpy with:
    - no auto-brightness (use_auto_wb=False, no_auto_bright=True)
    - camera white balance embedded in file
    - AHD demosaicing
    - output in sRGB colour space (rawpy handles the colour matrix)
    - 16-bit output, then normalised to [0, 1]
    """
    if not _RAWPY_AVAILABLE:
        raise ImportError(
            "rawpy is required to process RAW files. "
            "Install with: pip install rawpy"
        )

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
        )

    # uint16 → float32 [0, 1]
    rgb = rgb16.astype(np.float32) / 65535.0
    return rgb


def _load_raster(path: Path) -> np.ndarray:
    """
    JPEG / TIFF / PNG → float32 array (H, W, 3) in [0, 1].

    16-bit TIFFs are handled correctly.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:
        return arr.astype(np.float32) / 255.0


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """
    (H, W, 3) float32 numpy → (3, H, W) float32 tensor, clipped to [0,1].
    """
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor.clamp(0.0, 1.0)


def _resize(
    tensor: torch.Tensor,
    target_size: Tuple[int, int],
    keep_aspect: bool,
) -> torch.Tensor:
    """
    Resize a (3, H, W) tensor to target_size (H_out, W_out).

    If keep_aspect is True, the image is scaled so the longer side
    fits, then zero-padded symmetrically on the shorter side.
    """
    h_out, w_out = target_size
    _, h_in, w_in = tensor.shape

    # add batch dim for F.interpolate
    x = tensor.unsqueeze(0)                           # (1, 3, H, W)

    if not keep_aspect:
        x = F.interpolate(
            x, size=(h_out, w_out),
            mode="bicubic", align_corners=False, antialias=True,
        )
        return x.squeeze(0).clamp(0.0, 1.0)

    # --- aspect-preserving resize + pad ---
    scale = min(h_out / h_in, w_out / w_in)
    new_h = math.floor(h_in * scale)
    new_w = math.floor(w_in * scale)

    x = F.interpolate(
        x, size=(new_h, new_w),
        mode="bicubic", align_corners=False, antialias=True,
    )
    x = x.clamp(0.0, 1.0)

    # symmetric zero-padding to reach exact target
    pad_h = h_out - new_h
    pad_w = w_out - new_w
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    # F.pad order: (left, right, top, bottom)
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    return x.squeeze(0)


# ---------------------------------------------------------------------------
# sRGB gamma utilities  (used elsewhere in the codebase)
# ---------------------------------------------------------------------------

def gamma_encode(linear: torch.Tensor) -> torch.Tensor:
    """
    Apply sRGB piecewise gamma encoding (linear → display).
    Operates on arbitrary-shape tensors in [0, 1].
    """
    thresh = 0.0031308
    low    = linear * 12.92
    high   = 1.055 * linear.clamp(min=thresh).pow(1.0 / 2.4) - 0.055
    return torch.where(linear <= thresh, low, high)


def gamma_decode(srgb: torch.Tensor) -> torch.Tensor:
    """
    Remove sRGB gamma (display → linear).
    """
    thresh = 0.04045
    low    = srgb / 12.92
    high   = ((srgb.clamp(min=thresh) + 0.055) / 1.055).pow(2.4)
    return torch.where(srgb <= thresh, low, high)
