"""
data/augmentation.py

Data augmentation per coppie (src, tgt) — §7.4.

Regola fondamentale: le trasformazioni geometriche sono applicate
IDENTICAMENTE a src e tgt. Le perturbazioni di acquisizione
(esposizione, rumore) sono applicate SOLO a src.
"""

import logging
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PairAugmentation:
    """
    Augmentation geometrica + perturbazioni di acquisizione per coppie.

    Args:
        horizontal_flip_prob: Probabilità di flip orizzontale.
        random_crop_scale:    Range (min, max) della frazione di area
                              per il random crop.
        rotation_degrees:     Range angolo di rotazione (±gradi).
        exposure_perturb_prob: Probabilità di perturbazione esposizione.
        exposure_range:       Range moltiplicatore esposizione (min, max).
        noise_prob:           Probabilità di aggiungere rumore gaussiano.
        noise_sigma:          Deviazione standard del rumore.
        seed:                 Seme per riproducibilità (None = random).
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        random_crop_scale: Tuple[float, float] = (0.7, 1.0),
        rotation_degrees: float = 5.0,
        exposure_perturb_prob: float = 0.3,
        exposure_range: Tuple[float, float] = (0.9, 1.1),
        noise_prob: float = 0.3,
        noise_sigma: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.hflip_prob    = horizontal_flip_prob
        self.crop_scale    = random_crop_scale
        self.rot_degrees   = rotation_degrees
        self.exp_prob      = exposure_perturb_prob
        self.exp_range     = exposure_range
        self.noise_prob    = noise_prob
        self.noise_sigma   = noise_sigma

        self._rng = random.Random(seed)
        torch.manual_seed(seed if seed is not None else 0)

    # ── API pubblica ──────────────────────────────────────────────────────────

    def __call__(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applica augmentation alla coppia (src, tgt).

        Args:
            src: Tensore float32 (3, H, W) — immagine sorgente.
            tgt: Tensore float32 (3, H, W) — immagine target.

        Returns:
            Coppia aumentata (src_aug, tgt_aug).
        """
        # ── 1. Flip orizzontale (identico su entrambi) ────────────────────────
        if self._rng.random() < self.hflip_prob:
            src = src.flip(dims=[2])
            tgt = tgt.flip(dims=[2])

        # ── 2. Random crop (identico su entrambi) ─────────────────────────────
        src, tgt = self._random_crop_pair(src, tgt)

        # ── 3. Rotazione (identica su entrambi) ──────────────────────────────
        if self.rot_degrees > 0:
            src, tgt = self._rotate_pair(src, tgt)

        # ── 4. Perturbazione esposizione (solo src) ───────────────────────────
        if self._rng.random() < self.exp_prob:
            gamma = self._rng.uniform(*self.exp_range)
            src = (src * gamma).clamp(0.0, 1.0)

        # ── 5. Rumore gaussiano (solo src) ────────────────────────────────────
        if self._rng.random() < self.noise_prob:
            noise = torch.randn_like(src) * self.noise_sigma
            src = (src + noise).clamp(0.0, 1.0)

        return src, tgt

    # ── Trasformazioni geometriche ────────────────────────────────────────────

    def _random_crop_pair(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ritaglia la stessa regione casuale da src e tgt.

        La frazione di area viene campionata da [crop_scale_min, 1.0].
        L'aspect ratio viene preservato.
        """
        _, h, w = src.shape
        scale = self._rng.uniform(self.crop_scale[0], self.crop_scale[1])

        crop_h = round(h * scale)
        crop_w = round(w * scale)
        crop_h = max(crop_h, 1)
        crop_w = max(crop_w, 1)

        top  = self._rng.randint(0, max(h - crop_h, 0))
        left = self._rng.randint(0, max(w - crop_w, 0))

        src = src[:, top:top + crop_h, left:left + crop_w]
        tgt = tgt[:, top:top + crop_h, left:left + crop_w]

        return src, tgt

    def _rotate_pair(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ruota src e tgt dello stesso angolo casuale in [-deg, +deg].

        Usa grid_sample con align_corners=False per interpolazione bilineare.
        I bordi introdotti dalla rotazione vengono riempiti col valore medio.
        """
        angle_deg = self._rng.uniform(-self.rot_degrees, self.rot_degrees)
        angle_rad = angle_deg * (3.14159265358979 / 180.0)

        cos_a = torch.tensor(angle_rad).cos().item()
        sin_a = torch.tensor(angle_rad).sin().item()

        # Matrice affine di rotazione attorno al centro
        # shape richiesta da affine_grid: (1, 2, 3)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)

        _, h, w = src.shape
        grid = F.affine_grid(theta, size=(1, 3, h, w), align_corners=False)

        src_rot = F.grid_sample(
            src.unsqueeze(0), grid,
            mode="bilinear", padding_mode="reflection", align_corners=False,
        ).squeeze(0)

        tgt_rot = F.grid_sample(
            tgt.unsqueeze(0), grid,
            mode="bilinear", padding_mode="reflection", align_corners=False,
        ).squeeze(0)

        return src_rot, tgt_rot