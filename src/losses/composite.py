"""
losses/composite.py

Color-Aesthetic Loss composita per photographer-specific color grading.

Combina tutti i termini di loss con pesi variabili secondo il curriculum
di training (§6.6 della tesi). Il curriculum ha quattro fasi:

    Epoche 1–5:   Warm-up con L1Lab dominante, ΔE disattivato
    Epoche 6–10:  Transizione graduale L1Lab → ΔE, introduce percezione
    Epoche 11+:   Regime stabile con ΔE + perceptual come termini principali

Formula completa (§6.5):
    L = λ_ΔE·L_ΔE + λ_L1Lab·L_L1Lab + λ_hist·L_hist + λ_perc·L_perc
      + λ_chroma·L_chroma + λ_id·L_id + λ_TV·L_TV + λ_lum·L_lum
      + λ_e·L_entropy

Tabella curriculum (§6.6):
    Epoca  | ΔE   L1Lab  hist  perc  chroma  id    TV     lum   entropy
    -------|------------------------------------------------------------
    1–5    | 0.0  0.8    0.4   0.0   0.0     0.5   0.01   0.3   0.01
    6–10   | 0.3  0.4    0.3   0.3   0.1     0.5   0.01   0.3   0.01
    11–20  | 0.5  0.0    0.3   0.6   0.2     0.5   0.01   0.3   0.01
    21+    | 0.5  0.0    0.3   0.6   0.2     0.5   0.01   0.3   0.01

NOTA sull'identity loss:
    L_id viene calcolata solo sui mini-batch identità (p_id=0.2).
    Il training loop (adapt.py) si occupa di: (a) scegliere casualmente
    il 20% dei batch come "identità" sostituendo target=src, (b) passare
    il flag is_identity=True a questo modulo.

NOTA sulle griglie per L_TV:
    Il training loop deve passare entrambe le bilateral grids (globale e
    locale) nel dict `grids` con chiavi "global" e "local".
    Se `grids` è None, L_TV viene saltata silenziosamente.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional

from losses.delta_e    import DeltaELoss
from losses.l1_lab     import L1LabLoss
from losses.histogram  import ColorHistogramLoss
from losses.perceptual import PerceptualLoss
from losses.chroma     import ChromaConsistencyLoss
from losses.identity   import IdentityLoss
from losses.tv         import TotalVariationLoss
from losses.luminance  import LuminancePreservationLoss
from losses.entropy    import EntropyLoss


# ── Tabella dei pesi del curriculum ──────────────────────────────────────────

@dataclass
class LossWeights:
    """
    Pesi per tutti i termini della Color-Aesthetic Loss.
    Corrisponde a una riga del curriculum §6.6.
    """
    delta_e:  float = 0.0
    l1_lab:   float = 0.8
    hist:     float = 0.4
    perc:     float = 0.0
    chroma:   float = 0.0
    identity: float = 0.5
    tv:       float = 0.01
    lum:      float = 0.3
    entropy:  float = 0.01


# Curriculum: mappa (epoca_inizio, epoca_fine_esclusa) → LossWeights
# Gli intervalli sono [start, end) — l'ultimo è aperto (21+)
_CURRICULUM: list = [
    # (epoch_start, epoch_end_exclusive, weights)
    (1,  6,  LossWeights(delta_e=0.0, l1_lab=0.8, hist=0.4, perc=0.0,
                          chroma=0.0, identity=0.5, tv=0.01, lum=0.3, entropy=0.01)),
    (6,  11, LossWeights(delta_e=0.3, l1_lab=0.4, hist=0.3, perc=0.3,
                          chroma=0.1, identity=0.5, tv=0.01, lum=0.3, entropy=0.01)),
    (11, None, LossWeights(delta_e=0.5, l1_lab=0.0, hist=0.3, perc=0.6,
                            chroma=0.2, identity=0.5, tv=0.01, lum=0.3, entropy=0.01)),
]


def get_weights_for_epoch(epoch: int) -> LossWeights:
    """
    Restituisce i pesi del curriculum per l'epoca corrente.

    Args:
        epoch: Numero di epoca (1-indexed).

    Returns:
        LossWeights per l'intervallo corrispondente.
    """
    for start, end, weights in _CURRICULUM:
        if end is None or epoch < end:
            if epoch >= start:
                return weights
    # Fallback all'ultimo intervallo
    return _CURRICULUM[-1][2]


# ── Dataclass per il dizionario di output ────────────────────────────────────

@dataclass
class LossOutput:
    """
    Output strutturato di CompositeLoss.forward(), con loss totale e
    i singoli termini per il logging.
    """
    total:    torch.Tensor
    delta_e:  torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    l1_lab:   torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    hist:     torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    perc:     torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    chroma:   torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    identity: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    tv:       torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    lum:      torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    entropy:  torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def as_dict(self) -> Dict[str, float]:
        """Per logging: restituisce scalari Python."""
        return {
            "loss/total":   self.total.item(),
            "loss/delta_e": self.delta_e.item(),
            "loss/l1_lab":  self.l1_lab.item(),
            "loss/hist":    self.hist.item(),
            "loss/perc":    self.perc.item(),
            "loss/chroma":  self.chroma.item(),
            "loss/identity":self.identity.item(),
            "loss/tv":      self.tv.item(),
            "loss/lum":     self.lum.item(),
            "loss/entropy": self.entropy.item(),
        }


# ── Modulo principale ─────────────────────────────────────────────────────────

class ColorAestheticLoss(nn.Module):
    """
    Loss composita per photographer-specific color grading.

    Combina 9 termini di loss con pesi variabili secondo il curriculum §6.6.
    I pesi vengono aggiornati chiamando `set_epoch(epoch)` all'inizio di
    ogni epoca nel training loop.

    Tutti i calcoli interni sono in float32 (i tensori in fp16 vengono
    promossi automaticamente da ogni singolo modulo di loss).

    Args:
        device: Device su cui inizializzare i moduli. Default 'cuda' se
                disponibile, altrimenti 'cpu'.

    Usage nel training loop:
        ```python
        criterion = ColorAestheticLoss()
        criterion.set_epoch(epoch)

        for src, tgt in loader:
            is_identity = (random.random() < 0.2)
            if is_identity:
                tgt = src.clone()

            pred, alpha, grid_global, grid_local = model(src)

            out = criterion(
                pred=pred,
                target=tgt,
                source=src,
                alpha=alpha,
                grids={"global": grid_global, "local": grid_local},
                is_identity=is_identity,
            )
            out.total.backward()
        ```

    Example:
        >>> criterion = ColorAestheticLoss()
        >>> criterion.set_epoch(1)   # warm-up
        >>> pred   = torch.rand(2, 3, 384, 512)
        >>> target = torch.rand(2, 3, 384, 512)
        >>> source = torch.rand(2, 3, 384, 512)
        >>> alpha  = torch.sigmoid(torch.randn(2, 1, 384, 512))
        >>> out = criterion(pred, target, source, alpha)
        >>> print(out.as_dict())
    """

    def __init__(self):
        super().__init__()

        # Istanzia tutti i moduli di loss
        self.loss_delta_e  = DeltaELoss()
        self.loss_l1_lab   = L1LabLoss()
        self.loss_hist     = ColorHistogramLoss()
        self.loss_perc     = PerceptualLoss()
        self.loss_chroma   = ChromaConsistencyLoss()
        self.loss_identity = IdentityLoss()
        self.loss_tv       = TotalVariationLoss()
        self.loss_lum      = LuminancePreservationLoss()
        self.loss_entropy  = EntropyLoss()

        # Pesi correnti (aggiornati da set_epoch)
        self._weights: LossWeights = get_weights_for_epoch(1)
        self._epoch: int = 1

    def set_epoch(self, epoch: int) -> None:
        """
        Aggiorna i pesi del curriculum per l'epoca corrente.

        Deve essere chiamato all'inizio di ogni epoca nel training loop,
        PRIMA del primo forward della epoch.

        Args:
            epoch: Numero di epoca (1-indexed).
        """
        self._epoch   = epoch
        self._weights = get_weights_for_epoch(epoch)

    @property
    def current_weights(self) -> LossWeights:
        """Pesi attivi per l'epoca corrente."""
        return self._weights

    def forward(
        self,
        pred:        torch.Tensor,
        target:      torch.Tensor,
        source:      torch.Tensor,
        alpha:       torch.Tensor,
        grids:       Optional[Dict[str, torch.Tensor]] = None,
        is_identity: bool = False,
    ) -> LossOutput:
        """
        Calcola la Color-Aesthetic Loss composita.

        Args:
            pred:        Immagine predetta (B, 3, H, W) sRGB [0,1].
            target:      Immagine target (B, 3, H, W) sRGB [0,1].
                         In un mini-batch identità coincide con source.
            source:      Immagine sorgente (B, 3, H, W) sRGB [0,1].
                         Usata da L_lum (NON il target) e da L_id.
            alpha:       Confidence mask (B, 1, H, W) valori in [0,1].
                         Output di σ(MaskNet(...)).
            grids:       Dict con le bilateral grids per L_TV:
                           {"global": tensor (B,12,8,8,8),
                            "local":  tensor (B,12,16,16,8)}
                         Se None, L_TV viene saltata.
            is_identity: Se True, il batch è un "batch identità" (target=source).
                         In questo caso viene calcolata L_id, altrimenti no.
                         Il peso λ_id si applica solo quando is_identity=True.

        Returns:
            LossOutput con loss totale e singoli termini.
        """
        w = self._weights
        device = pred.device

        # Accumula con tensori zero per il logging anche quando λ=0
        _z = torch.zeros(1, device=device, dtype=torch.float32)

        # ── ΔE Loss ──────────────────────────────────────────────────────────
        if w.delta_e > 0.0:
            l_de = self.loss_delta_e(pred, target)
        else:
            l_de = _z.clone()

        # ── L1 Lab Loss (warm-up) ─────────────────────────────────────────────
        if w.l1_lab > 0.0:
            l_l1 = self.loss_l1_lab(pred, target)
        else:
            l_l1 = _z.clone()

        # ── Histogram Loss ────────────────────────────────────────────────────
        if w.hist > 0.0:
            l_hist = self.loss_hist(pred, target)
        else:
            l_hist = _z.clone()

        # ── Perceptual Loss ───────────────────────────────────────────────────
        if w.perc > 0.0:
            l_perc = self.loss_perc(pred, target)
        else:
            l_perc = _z.clone()

        # ── Chroma Loss ───────────────────────────────────────────────────────
        if w.chroma > 0.0:
            l_chroma = self.loss_chroma(pred, target)
        else:
            l_chroma = _z.clone()

        # ── Identity Loss (solo nei batch identità) ───────────────────────────
        if is_identity and w.identity > 0.0:
            l_id = self.loss_identity(pred, source)
        else:
            l_id = _z.clone()

        # ── Total Variation Loss ──────────────────────────────────────────────
        if grids is not None and w.tv > 0.0:
            tv_terms = []
            for grid_name, grid_tensor in grids.items():
                tv_terms.append(self.loss_tv(grid_tensor))
            l_tv = torch.stack(tv_terms).mean() if tv_terms else _z.clone()
        else:
            l_tv = _z.clone()

        # ── Luminance Preservation Loss ───────────────────────────────────────
        if w.lum > 0.0:
            l_lum = self.loss_lum(pred, source)   # confronto con SOURCE
        else:
            l_lum = _z.clone()

        # ── Entropy Loss (confidence mask) ────────────────────────────────────
        if w.entropy > 0.0:
            l_entropy = self.loss_entropy(alpha)
        else:
            l_entropy = _z.clone()

        # ── Loss totale ───────────────────────────────────────────────────────
        total = (w.delta_e  * l_de
               + w.l1_lab   * l_l1
               + w.hist     * l_hist
               + w.perc     * l_perc
               + w.chroma   * l_chroma
               + w.identity * l_id
               + w.tv       * l_tv
               + w.lum      * l_lum
               + w.entropy  * l_entropy)

        return LossOutput(
            total    = total,
            delta_e  = l_de,
            l1_lab   = l_l1,
            hist     = l_hist,
            perc     = l_perc,
            chroma   = l_chroma,
            identity = l_id,
            tv       = l_tv,
            lum      = l_lum,
            entropy  = l_entropy,
        )
