"""
color_utils.py
--------------
Funzioni pure per operazioni cromatiche.

Usate da modelli, loss, dataset e evaluation — nessuno stato,
nessuna dipendenza circolare.

Contenuto:
  rgb_to_lab(img)          — sRGB [0,1] → CIE Lab  (differenziabile)
  lab_to_rgb(lab)          — CIE Lab → sRGB [0,1]  (differenziabile)
  delta_e_2000(lab1, lab2) — ΔE₀₀ scalare medio su un batch
  soft_histogram(channel)  — istogramma differenziabile 1D
  lab_channel_stats(img)   — media e std per canale Lab

Tutte le funzioni operano su tensori PyTorch (B, 3, H, W) o (3, H, W)
e sono differenziabili rispetto all'input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Matrice sRGB→XYZ (D65, IEC 61966-2-1)
_RGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

# Matrice XYZ→sRGB
_XYZ_TO_RGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

# Illuminante D65 normalizzato
_D65 = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)

# Soglie per la funzione f di CIE Lab
_LAB_DELTA  = 6.0 / 29.0
_LAB_DELTA3 = _LAB_DELTA ** 3          # ≈ 0.008856
_LAB_COEFF  = 1.0 / (3.0 * _LAB_DELTA ** 2)  # ≈ 7.787


# ---------------------------------------------------------------------------
# sRGB ↔ lineare (gamma)
# ---------------------------------------------------------------------------

def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] → lineare [0,1]  (gamma decode)."""
    thresh = 0.04045
    low    = img / 12.92
    high   = ((img.clamp(min=thresh) + 0.055) / 1.055).pow(2.4)
    return torch.where(img <= thresh, low, high)


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    """Lineare [0,1] → sRGB [0,1]  (gamma encode)."""
    thresh = 0.0031308
    low    = img * 12.92
    high   = 1.055 * img.clamp(min=thresh).pow(1.0 / 2.4) - 0.055
    return torch.where(img <= thresh, low, high)


# ---------------------------------------------------------------------------
# CIE Lab  f  e  f⁻¹
# ---------------------------------------------------------------------------

def _lab_f(t: torch.Tensor) -> torch.Tensor:
    """Funzione di trasferimento CIE Lab f(t)."""
    return torch.where(
        t > _LAB_DELTA3,
        t.clamp(min=_LAB_DELTA3).pow(1.0 / 3.0),
        _LAB_COEFF * t + 4.0 / 29.0,
    )


def _lab_f_inv(t: torch.Tensor) -> torch.Tensor:
    """Inversa di f: f⁻¹(t)."""
    thresh = _LAB_DELTA + 4.0 / 29.0   # f(_LAB_DELTA3) = delta + 4/29
    return torch.where(
        t > thresh,
        t.pow(3.0),
        (t - 4.0 / 29.0) / _LAB_COEFF,
    )


# ---------------------------------------------------------------------------
# rgb_to_lab
# ---------------------------------------------------------------------------

def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """
    Converte sRGB [0,1] in CIE Lab.

    Differenziabile rispetto a img — usabile nelle loss.

    Parameters
    ----------
    img : (..., 3, H, W)  sRGB float32 [0, 1]

    Returns
    -------
    lab : (..., 3, H, W)  [L* ∈ [0,100], a* ∈ [-128,127], b* ∈ [-128,127]]
    """
    shape = img.shape
    dev   = img.device

    # Porta le matrici sul device corretto
    M = _RGB_TO_XYZ.to(dev)       # (3, 3)
    d = _D65.to(dev)               # (3,)

    # Flatten spaziale: (..., 3, H*W)
    flat = img.reshape(*shape[:-3], 3, -1)   # (..., 3, N)

    # gamma decode
    lin = srgb_to_linear(flat)

    # sRGB → XYZ: matrice (3,3) × canali (3, N)
    xyz = torch.einsum("ij,...jn->...in", M, lin)   # (..., 3, N)

    # Normalizzazione per D65
    xyz = xyz / d.view(1, 3, 1) if flat.dim() == 3 else xyz / d.view(*([1] * (xyz.dim() - 2)), 3, 1)

    # f applicata per canale
    f_xyz = _lab_f(xyz)            # (..., 3, N)

    # L*, a*, b*
    L = 116.0 * f_xyz[..., 1:2, :] - 16.0
    a = 500.0 * (f_xyz[..., 0:1, :] - f_xyz[..., 1:2, :])
    b = 200.0 * (f_xyz[..., 1:2, :] - f_xyz[..., 2:3, :])

    lab_flat = torch.cat([L, a, b], dim=-2)   # (..., 3, N)
    return lab_flat.reshape(shape)


# ---------------------------------------------------------------------------
# lab_to_rgb
# ---------------------------------------------------------------------------

def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    Converte CIE Lab in sRGB [0,1]. Differenziabile.

    Parameters
    ----------
    lab : (..., 3, H, W)  [L*, a*, b*]

    Returns
    -------
    img : (..., 3, H, W)  sRGB [0,1]
    """
    shape = lab.shape
    dev   = lab.device

    M_inv = _XYZ_TO_RGB.to(dev)
    d     = _D65.to(dev)

    flat = lab.reshape(*shape[:-3], 3, -1)   # (..., 3, N)

    L = flat[..., 0:1, :]
    a = flat[..., 1:2, :]
    b = flat[..., 2:3, :]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    f_xyz = torch.cat([fx, fy, fz], dim=-2)   # (..., 3, N)

    # f⁻¹ → XYZ normalizzato
    xyz_norm = _lab_f_inv(f_xyz)

    # De-normalizzazione D65
    d_view = d.view(*([1] * (xyz_norm.dim() - 2)), 3, 1)
    xyz = xyz_norm * d_view

    # XYZ → lineare sRGB
    lin = torch.einsum("ij,...jn->...in", M_inv, xyz)

    # gamma encode + clip
    rgb = linear_to_srgb(lin.clamp(0.0, 1.0))
    return rgb.reshape(shape)


# ---------------------------------------------------------------------------
# delta_e_2000  (versione scalare — per metriche, non per training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def delta_e_2000_mean(
    img_pred: torch.Tensor,   # (B, 3, H, W) sRGB
    img_tgt:  torch.Tensor,   # (B, 3, H, W) sRGB
    eps: float = 1e-7,
) -> float:
    """
    Calcola il ΔE₀₀ medio su un batch (scalar float).

    Versione senza gradiente per le metriche di evaluation.
    Per la versione differenziabile usata nella loss vedere
    losses/color_losses.py::DeltaELoss.
    """
    from losses.color_losses import DeltaELoss   # lazy import
    loss_fn = DeltaELoss(eps=eps)
    return loss_fn(img_pred, img_tgt).item()


# ---------------------------------------------------------------------------
# soft_histogram
# ---------------------------------------------------------------------------

def soft_histogram(
    channel: torch.Tensor,   # (B, H, W) oppure (H, W) — entrambi accettati
    n_bins:  int   = 64,
    vmin:    float = 0.0,
    vmax:    float = 1.0,
) -> torch.Tensor:
    # Normalizza sempre a (B, H*W)
    if channel.dim() == 2:
        channel = channel.unsqueeze(0)   # (1, H, W)
    B      = channel.shape[0]
    delta  = (vmax - vmin) / n_bins
    centers = torch.linspace(
        vmin + delta / 2, vmax - delta / 2,
        n_bins, device=channel.device,
    )
    sigma  = delta * 0.5
    pixels = channel.reshape(B, -1)
    diff   = pixels.unsqueeze(-1) - centers.view(1, 1, -1)
    w      = torch.exp(-diff.pow(2) / (2 * sigma ** 2))
    w_norm = w / (w.sum(-1, keepdim=True) + 1e-8)
    return w_norm.mean(dim=1)            # (B, n_bins)

# ---------------------------------------------------------------------------
# lab_channel_stats
# ---------------------------------------------------------------------------

@torch.no_grad()
def lab_channel_stats(img: torch.Tensor) -> dict:
    """
    Calcola media e std per canale Lab su un batch.

    Parameters
    ----------
    img : (B, 3, H, W) sRGB [0,1]

    Returns
    -------
    dict con:
      "L_mean", "L_std"  — luminanza
      "a_mean", "a_std"  — canale a*
      "b_mean", "b_std"  — canale b*
      "chroma_mean"      — saturazione media √(a²+b²)
    """
    lab = rgb_to_lab(img)              # (B, 3, H, W)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    chroma = (a.pow(2) + b.pow(2)).sqrt()

    return {
        "L_mean":      L.mean().item(),
        "L_std":       L.std().item(),
        "a_mean":      a.mean().item(),
        "a_std":       a.std().item(),
        "b_mean":      b.mean().item(),
        "b_std":       b.std().item(),
        "chroma_mean": chroma.mean().item(),
    }
