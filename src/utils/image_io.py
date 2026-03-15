"""
image_io.py
-----------
Funzioni di I/O per immagini.

Astrae i dettagli di Pillow/rawpy e fornisce un'interfaccia uniforme
per caricare e salvare immagini in tutti i formati usati nel progetto.

Contenuto:
  save_image(tensor, path, ...)   — salva (3,H,W) tensor in vari formati
  load_tensor(path, ...)          — carica qualsiasi immagine come tensor
  save_comparison_grid(...)       — griglia src | pred | tgt per debug
  tensor_to_pil(tensor)           — conversione per visualizzazione notebook
  pil_to_tensor(img)              — conversione da PIL
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_RAW_EXTENSIONS    = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf",
                      ".rw2", ".pef", ".dng", ".raf"}


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------

def save_image(
    tensor:   torch.Tensor,         # (3, H, W) float32 [0,1] o (H, W, 3)
    path:     Union[str, Path],
    fmt:      Optional[str] = None, # "tiff" | "jpeg" | "png" | None (da ext)
    quality:  int  = 95,            # JPEG quality
    bit16:    bool = False,         # salva TIFF a 16 bit
) -> Path:
    """
    Salva un tensor immagine su disco.

    Parameters
    ----------
    tensor  : (3, H, W) float32 [0,1] — CHW o HWC
    path    : percorso di output (l'estensione determina il formato se fmt=None)
    fmt     : forza il formato ("tiff", "jpeg", "png")
    quality : qualità JPEG (1–95)
    bit16   : se True e formato TIFF, salva uint16

    Returns
    -------
    path : Path dell'immagine salvata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalizza a CHW
    if tensor.dim() == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)

    arr = tensor.detach().cpu().float().clamp(0, 1).numpy()
    arr = arr.transpose(1, 2, 0)               # HWC

    ext = fmt or path.suffix.lower().lstrip(".")

    if ext in ("tif", "tiff"):
        if bit16:
            arr16 = (arr * 65535).astype(np.uint16)
            Image.fromarray(arr16, mode="RGB").save(path)
        else:
            arr8 = (arr * 255).astype(np.uint8)
            Image.fromarray(arr8, mode="RGB").save(path)

    elif ext in ("jpg", "jpeg"):
        arr8 = (arr * 255).astype(np.uint8)
        Image.fromarray(arr8, mode="RGB").save(path, quality=quality, subsampling=0)

    elif ext == "png":
        arr8 = (arr * 255).astype(np.uint8)
        Image.fromarray(arr8, mode="RGB").save(path, compress_level=6)

    else:
        # Fallback: usa Pillow con l'estensione nativa
        arr8 = (arr * 255).astype(np.uint8)
        Image.fromarray(arr8, mode="RGB").save(path)

    return path


# ---------------------------------------------------------------------------
# load_tensor
# ---------------------------------------------------------------------------

def load_tensor(
    path:        Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,   # (H, W)
    keep_aspect: bool = True,
    device:      str  = "cpu",
) -> torch.Tensor:
    """
    Carica un'immagine raster come tensor (3, H, W) float32 [0,1].

    Per file RAW delega a data.raw_pipeline.load_image.

    Parameters
    ----------
    path        : percorso dell'immagine
    target_size : (H, W) resize; None = risoluzione originale
    keep_aspect : padding invece di stretching
    device      : device del tensor output
    """
    path = Path(path)
    ext  = path.suffix.lower()

    if ext in _RAW_EXTENSIONS:
        from data.raw_pipeline import load_image     # lazy import
        return load_image(path, target_size=target_size,
                          keep_aspect=keep_aspect).to(device)

    img  = Image.open(path).convert("RGB")
    arr  = np.array(img)

    if arr.dtype == np.uint16:
        tensor = torch.from_numpy(arr.astype(np.float32) / 65535.0)
    else:
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)

    tensor = tensor.permute(2, 0, 1)             # HWC → CHW

    if target_size is not None:
        tensor = _resize_tensor(tensor, target_size, keep_aspect)

    return tensor.to(device)


# ---------------------------------------------------------------------------
# save_comparison_grid
# ---------------------------------------------------------------------------

def save_comparison_grid(
    src:    torch.Tensor,           # (3, H, W) o (B, 3, H, W)
    pred:   torch.Tensor,           # (3, H, W) o (B, 3, H, W)
    tgt:    torch.Tensor,           # (3, H, W) o (B, 3, H, W)
    path:   Union[str, Path],
    labels: Tuple[str, str, str] = ("Source", "Predicted", "Target"),
    max_images: int = 4,
) -> Path:
    """
    Salva una griglia di confronto src | pred | tgt.

    Utile per il debug visivo nei notebook e per il logging wandb/tensorboard.

    Parameters
    ----------
    src, pred, tgt : tensori (B, 3, H, W) o (3, H, W)
    path           : percorso output
    labels         : etichette per le colonne
    max_images     : max immagini del batch da includere

    Returns
    -------
    path : Path salvato
    """
    path = Path(path)

    # Normalizza a (B, 3, H, W)
    if src.dim() == 3:
        src, pred, tgt = src.unsqueeze(0), pred.unsqueeze(0), tgt.unsqueeze(0)

    B = min(src.shape[0], max_images)
    src, pred, tgt = src[:B], pred[:B], tgt[:B]

    _, _, H, W = src.shape

    # Griglia: B righe × 3 colonne
    rows = []
    for i in range(B):
        row = torch.cat([src[i], pred[i], tgt[i]], dim=-1)   # (3, H, 3W)
        rows.append(row)

    grid = torch.cat(rows, dim=-2)                           # (3, B*H, 3W)

    # Aggiungi barre di separazione bianche (2px)
    grid_np = grid.detach().cpu().float().clamp(0, 1).numpy()
    grid_np = (grid_np.transpose(1, 2, 0) * 255).astype(np.uint8)

    img = Image.fromarray(grid_np, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

    return path


# ---------------------------------------------------------------------------
# tensor_to_pil  /  pil_to_tensor
# ---------------------------------------------------------------------------

def tensor_to_pil(
    tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
) -> Image.Image:
    """Converte un tensor (3,H,W) in immagine PIL per visualizzazione notebook."""
    arr = tensor.detach().cpu().float().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Converte una PIL Image in tensor (3,H,W) float32 [0,1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Utility interna
# ---------------------------------------------------------------------------

def _resize_tensor(
    tensor:      torch.Tensor,
    target_size: Tuple[int, int],
    keep_aspect: bool,
) -> torch.Tensor:
    """Resize (3,H,W) tensor a target_size."""
    import math
    h_out, w_out = target_size
    _, h_in, w_in = tensor.shape
    x = tensor.unsqueeze(0)

    if not keep_aspect:
        x = F.interpolate(x, size=(h_out, w_out),
                          mode="bicubic", align_corners=False, antialias=True)
        return x.squeeze(0).clamp(0, 1)

    scale = min(h_out / h_in, w_out / w_in)
    new_h = math.floor(h_in * scale)
    new_w = math.floor(w_in * scale)
    x = F.interpolate(x, size=(new_h, new_w),
                      mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0, 1)

    pad_h     = h_out - new_h
    pad_w     = w_out - new_w
    pad_top   = pad_h // 2
    pad_left  = pad_w // 2
    x = F.pad(x, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
    return x.squeeze(0)
