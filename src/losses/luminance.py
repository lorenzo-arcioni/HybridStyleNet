"""
losses/luminance.py

Luminance Preservation Loss — preservazione della struttura luminosa.

Vincola la rete a non allontanarsi troppo dalla struttura di luminanza
dell'immagine sorgente (non del target). Questo previene che la bilateral
grid apprenda trasformazioni che alterano la luminanza per minimizzare ΔE
in zone cromatiche difficili, producendo perdita di dettagli e aloni colorati.

Riferimento tesi: §6.5.7
Formula: L_lum = (1/HW) Σ_{i,j} |L*_pred(i,j) - L*_src(i,j)|

NOTA CRUCIALE: Il confronto è con I^src (sorgente), NON con I^tgt (target).
    Questo è intenzionale: il fotografo potrebbe aver modificato leggermente
    la luminanza (correzioni di esposizione, dodge & burn), e vincolare pred
    a replicare esattamente L*_tgt sarebbe ridondante con L_ΔE.
    Il vincolo rispetto a src impone che la rete non si allontani troppo
    dalla struttura luminosa originale, lasciando libertà per le correzioni
    intenzionali del fotografo.

Perché attiva fin dall'epoca 1 con λ_lum = 0.3:
    È critico che la struttura luminosa sia preservata fin dal primo
    aggiornamento dei pesi, prima che la bilateral grid possa apprendere
    trasformazioni strutturalmente distruttive.

Differenza da L_id:
    L_id previene overediting in ampiezza (trasformazione totale troppo grande).
    L_lum vincola specificamente la dimensione luminosa, indipendentemente
    dall'ampiezza delle modifiche cromatiche. Si può avere editing cromatico
    aggressivo (hue shift totale) con L_lum bassa, ma non aloni strutturali.
"""

import torch
import torch.nn as nn

from utils.color_space import rgb_to_lab

class LuminancePreservationLoss(nn.Module):
    """
    Loss L1 tra il canale L* dell'immagine predetta e dell'immagine sorgente.

    Il confronto è SEMPRE con la sorgente (non il target): questo preserva
    la struttura luminosa originale lasciando libertà di modifica cromatica.

    Formula:
        L_lum = (1/HW) Σ |L*_pred(i,j) - L*_src(i,j)|

    Dove L* è il canale di luminanza CIE Lab (scala [0, 100]).

    Example:
        >>> loss_fn = LuminancePreservationLoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> src  = torch.rand(2, 3, 384, 512)   # immagine sorgente NON il target
        >>> loss = loss_fn(pred, src)
    """

    def __init__(self):
        super().__init__()

    def _extract_luminance(self, img: torch.Tensor) -> torch.Tensor:
        """
        Estrae il canale L* da un'immagine sRGB.

        Args:
            img: (B, 3, H, W) float32, sRGB in [0,1].

        Returns:
            (B, H, W) float32, canale L* in [0, 100].
        """
        lab = rgb_to_lab(img.float().permute(0, 2, 3, 1))  # (B, H, W, 3)
        return lab[..., 0]   # (B, H, W)

    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   Immagine predetta (B, 3, H, W) in [0,1].
            source: Immagine SORGENTE (B, 3, H, W) in [0,1].
                    ATTENZIONE: NON il target — cfr. nota in §6.5.7.

        Returns:
            Scalare: MAE tra i canali L* di pred e source.
        """
        L_pred = self._extract_luminance(pred)     # (B, H, W)
        L_src  = self._extract_luminance(source)

        return torch.abs(L_pred - L_src).mean()
