"""
utils/color_space.py

Conversioni di spazio colore differenziabili in PyTorch.
Tutte le funzioni operano su tensori float32 in [0,1] con
shape (..., 3) dove l'ultimo asse è il canale colore.
Le operazioni sono batch-safe e resolution-agnostic.

Pipeline supportata:
    sRGB ↔ sRGB lineare ↔ CIE XYZ D65 ↔ CIE L*a*b*
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# ── Costanti ─────────────────────────────────────────────────────────────────

# Bianco di riferimento D65 (CIE standard)
_D65_WHITE = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)

# Matrice sRGB primarie → CIE XYZ D65  (IEC 61966-2-1)
_M_SRGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

# Matrice inversa: CIE XYZ D65 → sRGB lineare
_M_XYZ_TO_SRGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

# Soglie per la funzione di trasferimento sRGB
_SRGB_LINEAR_THRESH   = 0.0031308   # soglia linearizzazione
_SRGB_ENCODED_THRESH  = 0.04045     # soglia gamma encoding
_SRGB_ALPHA           = 0.055
_SRGB_GAMMA           = 2.4
_SRGB_LINEAR_SCALE    = 12.92

# Soglia funzione f di CIE Lab
_LAB_DELTA      = 6.0 / 29.0
_LAB_DELTA_3    = _LAB_DELTA ** 3           # ≈ 0.008856
_LAB_COEFF_A    = 1.0 / (3.0 * _LAB_DELTA ** 2)  # ≈ 7.787
_LAB_COEFF_B    = 4.0 / 29.0               # ≈ 0.13793

EPS = 1e-8


# ── sRGB ↔ sRGB lineare ──────────────────────────────────────────────────────

def rgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """
    sRGB [0,1] → sRGB lineare [0,1]  (inversa EOTF / OETF).

    Args:
        img: Tensore float32 shape (..., 3), valori in [0,1].

    Returns:
        Tensore float32 shape (..., 3), spazio lineare.
    """
    img = img.clamp(0.0, 1.0)
    low  = img / _SRGB_LINEAR_SCALE
    high = ((img + _SRGB_ALPHA) / (1.0 + _SRGB_ALPHA)) ** _SRGB_GAMMA
    return torch.where(img <= _SRGB_ENCODED_THRESH, low, high)


def linear_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """
    sRGB lineare [0,1] → sRGB [0,1]  (EOTF / gamma encoding).

    Args:
        img: Tensore float32 shape (..., 3), valori in [0,1].

    Returns:
        Tensore float32 shape (..., 3), spazio sRGB.
    """
    img = img.clamp(0.0, 1.0)
    low  = img * _SRGB_LINEAR_SCALE
    high = (1.0 + _SRGB_ALPHA) * (img ** (1.0 / _SRGB_GAMMA)) - _SRGB_ALPHA
    return torch.where(img <= _SRGB_LINEAR_THRESH, low, high)


# ── sRGB lineare ↔ CIE XYZ D65 ──────────────────────────────────────────────

def linear_rgb_to_xyz(img: torch.Tensor) -> torch.Tensor:
    """
    sRGB lineare [0,1] → CIE XYZ D65.

    Args:
        img: Tensore float32 shape (..., 3).

    Returns:
        Tensore float32 shape (..., 3).
    """
    M = _M_SRGB_TO_XYZ.to(img.device)
    return torch.matmul(img, M.T)


def xyz_to_linear_rgb(xyz: torch.Tensor) -> torch.Tensor:
    """
    CIE XYZ D65 → sRGB lineare [0,1], con clipping.

    Args:
        xyz: Tensore float32 shape (..., 3).

    Returns:
        Tensore float32 shape (..., 3) in [0,1].
    """
    M = _M_XYZ_TO_SRGB.to(xyz.device)
    return torch.matmul(xyz, M.T).clamp(0.0, 1.0)


# ── CIE XYZ ↔ CIE L*a*b* ────────────────────────────────────────────────────

def _f_lab(t: torch.Tensor) -> torch.Tensor:
    """Funzione ausiliaria f per conversione XYZ→Lab (vettorizzata)."""
    return torch.where(
        t > _LAB_DELTA_3,
        t.clamp(min=EPS).pow(1.0 / 3.0),
        _LAB_COEFF_A * t + _LAB_COEFF_B,
    )


def _f_lab_inv(t: torch.Tensor) -> torch.Tensor:
    """Funzione ausiliaria f^{-1} per conversione Lab→XYZ (vettorizzata)."""
    return torch.where(
        t > _LAB_DELTA,
        t ** 3,
        (t - _LAB_COEFF_B) / _LAB_COEFF_A,
    )


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    """
    CIE XYZ D65 → CIE L*a*b* (D65).

    Args:
        xyz: Tensore float32 shape (..., 3), canali [X, Y, Z].

    Returns:
        Tensore float32 shape (..., 3), canali [L*, a*, b*].
        L* ∈ [0, 100], a* ∈ [-128, 127], b* ∈ [-128, 127] approssimativamente.
    """
    white = _D65_WHITE.to(xyz.device)
    xyz_n = xyz / white.clamp(min=EPS)          # normalizza per bianco D65

    f = _f_lab(xyz_n)                           # (..., 3)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return torch.stack([L, a, b], dim=-1)


def lab_to_xyz(lab: torch.Tensor) -> torch.Tensor:
    """
    CIE L*a*b* (D65) → CIE XYZ D65.

    Args:
        lab: Tensore float32 shape (..., 3), canali [L*, a*, b*].

    Returns:
        Tensore float32 shape (..., 3), canali [X, Y, Z].
    """
    white = _D65_WHITE.to(lab.device)

    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    xyz_n = torch.stack([
        _f_lab_inv(fx),
        _f_lab_inv(fy),
        _f_lab_inv(fz),
    ], dim=-1)

    return xyz_n * white


# ── Shortcut sRGB ↔ CIE L*a*b* ──────────────────────────────────────────────

def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """
    sRGB [0,1] → CIE L*a*b* (D65).

    Pipeline: sRGB → sRGB lineare → XYZ → Lab.

    Args:
        img: Tensore float32 shape (..., 3), valori in [0,1].

    Returns:
        Tensore float32 shape (..., 3), canali [L*, a*, b*].
    """
    lin = rgb_to_linear(img)
    xyz = linear_rgb_to_xyz(lin)
    return xyz_to_lab(xyz)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    CIE L*a*b* (D65) → sRGB [0,1].

    Pipeline: Lab → XYZ → sRGB lineare → sRGB (con clipping).

    Args:
        lab: Tensore float32 shape (..., 3), canali [L*, a*, b*].

    Returns:
        Tensore float32 shape (..., 3), valori in [0,1].
    """
    xyz  = lab_to_xyz(lab)
    lin  = xyz_to_linear_rgb(xyz)
    return linear_to_rgb(lin)


# ── Utilità cromatiche ───────────────────────────────────────────────────────

def lab_chroma(lab: torch.Tensor) -> torch.Tensor:
    """
    Calcola la chroma C* = sqrt(a*^2 + b*^2 + eps) da un tensore Lab.

    Args:
        lab: Tensore float32 shape (..., 3).

    Returns:
        Tensore float32 shape (...,).
    """
    a, b = lab[..., 1], lab[..., 2]
    return torch.sqrt(a ** 2 + b ** 2 + EPS)


def lab_hue(lab: torch.Tensor) -> torch.Tensor:
    """
    Calcola l'angolo di hue h* = atan2(b*, a*) in radianti ∈ (-π, π].

    Args:
        lab: Tensore float32 shape (..., 3).

    Returns:
        Tensore float32 shape (...,).
    """
    return torch.atan2(lab[..., 2], lab[..., 1])


def circular_distance(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """
    Distanza geodetica circolare tra angoli in radianti.

    d = |atan2(sin(h1-h2), cos(h1-h2))|  ∈ [0, π]

    Args:
        h1, h2: Tensori float32 di qualsiasi shape, valori in radianti.

    Returns:
        Tensore float32 con distanze in [0, π].
    """
    diff = h1 - h2
    return torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))


def luminance(img: torch.Tensor) -> torch.Tensor:
    """
    Calcola la luminanza BT.601 da un'immagine sRGB.

    g = 0.299 R + 0.587 G + 0.114 B

    Args:
        img: Tensore float32 shape (..., H, W, 3) oppure (..., 3).

    Returns:
        Tensore float32 shape (..., H, W) oppure (...,).
    """
    weights = torch.tensor([0.299, 0.587, 0.114],
                           dtype=img.dtype, device=img.device)
    return (img * weights).sum(dim=-1)