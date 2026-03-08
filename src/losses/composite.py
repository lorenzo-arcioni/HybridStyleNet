"""
losses/composite.py

Color-Aesthetic Loss composita  (§6.4).

L = λ_ΔE · L_ΔE
  + λ_hist · L_hist
  + λ_perc · L_perc
  + λ_style · L_style
  + λ_cos  · L_cos
  + λ_chroma · L_chroma
  + λ_id · L_id   (con prob. p_id)

I pesi seguono il curriculum della sezione 6.6.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from .delta_e    import DeltaELoss
from .histogram  import ColorHistogramLoss
from .perceptual import PerceptualLoss, StyleLoss
from .chroma     import CosineLoss, ChromaConsistencyLoss
from .identity   import IdentityLoss

logger = logging.getLogger(__name__)


@dataclass
class LossWeights:
    """
    Pesi dei singoli termini della loss composita.
    Corrisponde alle colonne della tabella §6.6.
    """
    lambda_delta_e: float = 0.5
    lambda_hist:    float = 0.3
    lambda_perc:    float = 0.4
    lambda_style:   float = 0.2
    lambda_cos:     float = 0.15
    lambda_chroma:  float = 0.2
    lambda_id:      float = 0.5

    @classmethod
    def from_dict(cls, d: Dict) -> "LossWeights":
        return cls(
            lambda_delta_e = d.get("lambda_delta_e", 0.5),
            lambda_hist    = d.get("lambda_hist",    0.3),
            lambda_perc    = d.get("lambda_perc",    0.4),
            lambda_style   = d.get("lambda_style",   0.2),
            lambda_cos     = d.get("lambda_cos",     0.15),
            lambda_chroma  = d.get("lambda_chroma",  0.2),
            lambda_id      = d.get("lambda_id",      0.5),
        )

    # Curriculum presets (§6.6)
    @classmethod
    def curriculum_1_5(cls) -> "LossWeights":
        return cls(
            lambda_delta_e=0.6, lambda_hist=0.4,
            lambda_perc=0.0,    lambda_style=0.0,
            lambda_cos=0.0,     lambda_chroma=0.0,
            lambda_id=0.5,
        )

    @classmethod
    def curriculum_6_10(cls) -> "LossWeights":
        return cls(
            lambda_delta_e=0.5, lambda_hist=0.3,
            lambda_perc=0.2,    lambda_style=0.1,
            lambda_cos=0.0,     lambda_chroma=0.1,
            lambda_id=0.5,
        )

    @classmethod
    def curriculum_11_plus(cls) -> "LossWeights":
        return cls(
            lambda_delta_e=0.5, lambda_hist=0.3,
            lambda_perc=0.4,    lambda_style=0.2,
            lambda_cos=0.15,    lambda_chroma=0.2,
            lambda_id=0.5,
        )


class ColorAestheticLoss(nn.Module):
    """
    Loss composita per photographer-specific color grading  (§6.4).

    Combina sette termini complementari:
      - L_ΔE    : accuratezza cromatica percettiva (CIEDE2000)
      - L_hist  : distribuzione spettrale globale (EMD)
      - L_perc  : similarità semantica multi-scala (VGG19)
      - L_style : correlazioni cromatico-texturali (Gram)
      - L_cos   : direzione cromatica (hue)
      - L_chroma: saturazione e hue circolare
      - L_id    : prevenzione overediting (identity)

    Args:
        weights:        Istanza LossWeights con i λ correnti.
        hist_bins:      Bin per l'istogramma.
        vgg_layers:     Layer VGG19 per perceptual/style loss.
        vgg_weights:    Pesi per layer perceptual.
        identity_prob:  Probabilità di applicare L_id.
        hue_weight:     Peso di L_hue in L_chroma.
        delta_e_eps:    Epsilon per CIEDE2000.
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        hist_bins: int = 64,
        vgg_layers=None,
        vgg_weights=None,
        identity_prob: float = 0.2,
        hue_weight: float = 0.5,
        delta_e_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.weights = weights or LossWeights()

        # Istanzia tutti i termini
        self.l_delta_e = DeltaELoss(eps=delta_e_eps)
        self.l_hist    = ColorHistogramLoss(bins=hist_bins)
        self.l_perc    = PerceptualLoss(
            layers=vgg_layers, weights=vgg_weights, normalize=True
        )
        self.l_style   = StyleLoss(layers=vgg_layers, normalize=True)
        self.l_cos     = CosineLoss()
        self.l_chroma  = ChromaConsistencyLoss(hue_weight=hue_weight)
        self.l_id      = IdentityLoss(p_id=identity_prob)

    def update_weights(self, weights: LossWeights) -> None:
        """Aggiorna i pesi (chiamato dal curriculum scheduler)."""
        self.weights = weights
        logger.debug(f"Loss weights aggiornati: {weights}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        force_identity: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcola la loss composita.

        Args:
            pred:           (B, 3, H, W) in [0,1] — predizione.
            target:         (B, 3, H, W) in [0,1] — ground truth.
            src:            (B, 3, H, W) in [0,1] — sorgente originale.
                            Necessario per L_id.
            force_identity: Se True, forza il calcolo di L_id
                            (bypassa il campionamento probabilistico).

        Returns:
            Dizionario con:
                "total":   Scalare — loss totale pesata.
                "delta_e": Scalare — contributo L_ΔE.
                "hist":    Scalare — contributo L_hist.
                "perc":    Scalare — contributo L_perc.
                "style":   Scalare — contributo L_style.
                "cos":     Scalare — contributo L_cos.
                "chroma":  Scalare — contributo L_chroma.
                "id":      Scalare — contributo L_id (0 se non applicato).
        """
        w = self.weights
        losses: Dict[str, torch.Tensor] = {}
        zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # ── L_ΔE ─────────────────────────────────────────────────────────────
        l_de = self.l_delta_e(pred, target)
        losses["delta_e"] = l_de

        # ── L_hist ───────────────────────────────────────────────────────────
        l_h = self.l_hist(pred, target)
        losses["hist"] = l_h

        # ── L_perc ───────────────────────────────────────────────────────────
        if w.lambda_perc > 0.0:
            l_p = self.l_perc(pred, target)
        else:
            l_p = zero
        losses["perc"] = l_p

        # ── L_style ──────────────────────────────────────────────────────────
        if w.lambda_style > 0.0:
            l_s = self.l_style(pred, target)
        else:
            l_s = zero
        losses["style"] = l_s

        # ── L_cos ────────────────────────────────────────────────────────────
        if w.lambda_cos > 0.0:
            l_c = self.l_cos(pred, target)
        else:
            l_c = zero
        losses["cos"] = l_c

        # ── L_chroma ─────────────────────────────────────────────────────────
        if w.lambda_chroma > 0.0:
            l_ch = self.l_chroma(pred, target)
        else:
            l_ch = zero
        losses["chroma"] = l_ch

        # ── L_id ─────────────────────────────────────────────────────────────
        apply_id = (
            src is not None
            and w.lambda_id > 0.0
            and (force_identity or self.l_id.should_apply())
        )
        if apply_id:
            l_id = self.l_id(pred, src)
        else:
            l_id = zero
        losses["id"] = l_id

        # ── Totale ────────────────────────────────────────────────────────────
        total = (
            w.lambda_delta_e * l_de
            + w.lambda_hist   * l_h
            + w.lambda_perc   * l_p
            + w.lambda_style  * l_s
            + w.lambda_cos    * l_c
            + w.lambda_chroma * l_ch
            + w.lambda_id     * l_id
        )
        losses["total"] = total

        return losses

    @classmethod
    def from_config(cls, cfg: Dict) -> "ColorAestheticLoss":
        """
        Costruisce la loss da un dizionario di configurazione
        (come quello prodotto da OmegaConf / yaml).

        Args:
            cfg: Dizionario con chiavi corrispondenti ai parametri __init__.

        Returns:
            Istanza di ColorAestheticLoss.
        """
        loss_cfg = cfg.get("loss", cfg)
        weights  = LossWeights.from_dict(loss_cfg)

        return cls(
            weights=weights,
            hist_bins=loss_cfg.get("hist_bins", 64),
            vgg_layers=loss_cfg.get("vgg_layers", None),
            vgg_weights=loss_cfg.get("vgg_weights", None),
            identity_prob=loss_cfg.get("identity_prob", 0.2),
            hue_weight=loss_cfg.get("chroma_hue_weight", 0.5),
            delta_e_eps=loss_cfg.get("delta_e_eps", 1e-8),
        )