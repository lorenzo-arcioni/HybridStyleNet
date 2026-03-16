"""
color_losses.py
---------------
Loss cromatiche di RAG-ColorNet.

Cinque funzioni di loss che misurano la fedeltà cromatica tra
l'output predetto e il target del fotografo:

  DeltaELoss        — CIEDE2000 perceptually-uniform colour error
  L1LabLoss         — L1 in spazio CIE Lab (warm-up)
  HistogramEMDLoss  — Earth Mover's Distance sugli istogrammi Lab
  PerceptualLoss    — distanza feature DINOv2 (semanticamente più ricca di VGG)
  ChromaConsistencyLoss — coerenza saturazione + hue tra pred e target

Tutte le loss operano su batch (B, 3, H, W) sRGB float32 [0, 1].
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.color_utils import rgb_to_lab          # type: ignore[import]


# ---------------------------------------------------------------------------
# CIEDE2000 — ΔE Loss
# ---------------------------------------------------------------------------

class DeltaELoss(nn.Module):
    """
    CIEDE2000 colour difference loss.

    Implementazione differenziabile di ΔE₀₀ secondo lo standard CIE.
    È la metrica perceptual più accurata per la differenza cromatica
    percepita dall'occhio umano.

    L_ΔE = (1 / H·W) · Σᵢⱼ ΔE₀₀(pred_Lab(i,j), tgt_Lab(i,j))

    Nota: usa ε-smoothing su radici quadrate e divisioni per
    garantire la differenziabilità ovunque.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        tgt:  torch.Tensor,
    ) -> torch.Tensor:
        # Forza fp32 — CIEDE2000 ha radici quadrate instabili in fp16
        pred_lab = rgb_to_lab(pred.float().clamp(0, 1))
        tgt_lab  = rgb_to_lab(tgt.float().clamp(0, 1))
        return self._delta_e_2000(pred_lab, tgt_lab).mean()

    # ------------------------------------------------------------------
    def _delta_e_2000(
        self,
        lab1: torch.Tensor,   # (B, 3, H, W)
        lab2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola ΔE₀₀ pixel per pixel.
        Returns: (B, H, W)
        """
        eps = self.eps

        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

        # --- Passo 1: C* e h* ---
        C1 = (a1.pow(2) + b1.pow(2) + eps).sqrt()
        C2 = (a2.pow(2) + b2.pow(2) + eps).sqrt()

        C_avg_pow7 = ((C1 + C2) / 2).pow(7)
        denom      = (C_avg_pow7 + 25.0 ** 7 + eps).sqrt()
        G          = 0.5 * (1.0 - (C_avg_pow7 / (denom + eps)).sqrt())

        a1p = a1 * (1.0 + G)
        a2p = a2 * (1.0 + G)
        C1p = (a1p.pow(2) + b1.pow(2) + eps).sqrt()
        C2p = (a2p.pow(2) + b2.pow(2) + eps).sqrt()

        h1p = torch.atan2(b1, a1p)               # rad
        h2p = torch.atan2(b2, a2p)

        # --- Passo 2: ΔL', ΔC', ΔH' ---
        dLp = L2 - L1
        dCp = C2p - C1p

        # Δhp con gestione del wraparound
        dhp = h2p - h1p
        dhp = torch.where(
            dhp > torch.pi,  dhp - 2 * torch.pi, dhp
        )
        dhp = torch.where(
            dhp < -torch.pi, dhp + 2 * torch.pi, dhp
        )
        # Δhp = 0 se uno dei due C' è zero
        dhp = torch.where((C1p * C2p) < eps, torch.zeros_like(dhp), dhp)

        dHp = 2.0 * (C1p * C2p + eps).sqrt() * torch.sin(dhp / 2.0)

        # --- Passo 3: medi e pesi ---
        Lp_avg  = (L1 + L2) / 2.0
        Cp_avg  = (C1p + C2p) / 2.0

        # hp_avg con gestione wraparound
        hp_avg = (h1p + h2p) / 2.0
        hp_avg = torch.where(
            (h1p - h2p).abs() > torch.pi,
            torch.where(
                (h1p + h2p) < 2 * torch.pi,
                hp_avg + torch.pi,
                hp_avg - torch.pi,
            ),
            hp_avg,
        )
        hp_avg = torch.where((C1p * C2p) < eps, h1p + h2p, hp_avg)

        # T
        T = (1.0
             - 0.17 * torch.cos(hp_avg - torch.deg2rad(torch.tensor(30.0, device=hp_avg.device)))
             + 0.24 * torch.cos(2.0 * hp_avg)
             + 0.32 * torch.cos(3.0 * hp_avg + torch.deg2rad(torch.tensor(6.0,  device=hp_avg.device)))
             - 0.20 * torch.cos(4.0 * hp_avg - torch.deg2rad(torch.tensor(63.0, device=hp_avg.device))))

        # Fattori di pesatura SL, SC, SH
        SL = 1.0 + 0.015 * (Lp_avg - 50.0).pow(2) / (
            20.0 + (Lp_avg - 50.0).pow(2) + eps
        ).sqrt()
        SC = 1.0 + 0.045 * Cp_avg
        SH = 1.0 + 0.015 * Cp_avg * T

        # RC e RT
        Cp_avg_pow7 = Cp_avg.pow(7)
        RC = 2.0 * (Cp_avg_pow7 / (Cp_avg_pow7 + 25.0 ** 7 + eps)).sqrt()
        d_theta = torch.deg2rad(
            torch.tensor(30.0, device=hp_avg.device)
        ) * torch.exp(
            -((hp_avg - torch.deg2rad(torch.tensor(275.0, device=hp_avg.device)))
              / torch.deg2rad(torch.tensor(25.0, device=hp_avg.device))).pow(2)
        )
        RT = -torch.sin(2.0 * d_theta) * RC

        # --- Passo 4: ΔE₀₀ ---
        kL, kC, kH = 1.0, 1.0, 1.0
        term_L = (dLp  / (kL * SL + eps)).pow(2)
        term_C = (dCp  / (kC * SC + eps)).pow(2)
        term_H = (dHp  / (kH * SH + eps)).pow(2)
        term_R = RT * (dCp / (kC * SC + eps)) * (dHp / (kH * SH + eps))

        delta_e = (term_L + term_C + term_H + term_R + eps).clamp(min=0).sqrt()
        return delta_e                            # (B, H, W)


# ---------------------------------------------------------------------------
# L1 Lab Loss
# ---------------------------------------------------------------------------

class L1LabLoss(nn.Module):
    """
    L1 in spazio CIE Lab — loss di warm-up (epoche 1-10).

    Più stabile di ΔE₀₀ nelle prime fasi del training perché
    non ha le singolarità della formula CIEDE2000 a C*=0.

    L_L1Lab = (1/H·W) · Σᵢⱼ ‖pred_Lab(i,j) - tgt_Lab(i,j)‖₁
    """

    def forward(
        self,
        pred: torch.Tensor,
        tgt:  torch.Tensor,
    ) -> torch.Tensor:
        pred_lab = rgb_to_lab(pred)
        tgt_lab  = rgb_to_lab(tgt)
        return F.l1_loss(pred_lab, tgt_lab)


# ---------------------------------------------------------------------------
# Histogram EMD Loss
# ---------------------------------------------------------------------------

class HistogramEMDLoss(nn.Module):
    """
    Earth Mover's Distance (Wasserstein-1) sugli istogrammi Lab.

    Misura quanto le distribuzioni cromatiche globali di pred e target
    sono distanti. È invariante alla posizione spaziale — cattura lo
    "stile cromatico" complessivo dell'immagine.

    Approssimazione efficiente: CDF distance (equivalente all'EMD 1D).

    L_hist = (1/3) · Σ_c Σ_k |CDF_c^pred(k) - CDF_c^tgt(k)|

    Parameters
    ----------
    n_bins : numero di bin per canale (deve corrispondere a ColorHistogram)
    """

    def __init__(self, n_bins: int = 64) -> None:
        super().__init__()
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    def _soft_histogram(self, channel: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        """
        Istogramma soft differenziabile per un singolo canale.

        channel : (B, H, W)
        Returns : (B, n_bins)
        """
        B = channel.shape[0]
        pixels = channel.reshape(B, -1)           # (B, H*W)
        delta  = (vmax - vmin) / self.n_bins
        centers = torch.linspace(
            vmin + delta / 2, vmax - delta / 2,
            self.n_bins, device=channel.device,
        )
        sigma = delta * 0.5

        diff    = pixels.unsqueeze(-1) - centers.view(1, 1, -1)
        weights = torch.exp(-diff.pow(2) / (2 * sigma ** 2))
        w_norm  = weights / (weights.sum(-1, keepdim=True) + 1e-8)
        return w_norm.mean(dim=1)                 # (B, n_bins)

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        tgt:  torch.Tensor,
    ) -> torch.Tensor:
        pred_lab = rgb_to_lab(pred)               # (B, 3, H, W)
        tgt_lab  = rgb_to_lab(tgt)

        ranges = [(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)]
        emd_total = torch.tensor(0.0, device=pred.device)

        for c, (vmin, vmax) in enumerate(ranges):
            h_pred = self._soft_histogram(pred_lab[:, c], vmin, vmax)
            h_tgt  = self._soft_histogram(tgt_lab[:, c],  vmin, vmax)

            cdf_pred = h_pred.cumsum(dim=-1)
            cdf_tgt  = h_tgt.cumsum(dim=-1)

            emd_total = emd_total + (cdf_pred - cdf_tgt).abs().mean()

        return emd_total / 3.0


# ---------------------------------------------------------------------------
# Perceptual Loss (DINOv2)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Perceptual loss basata su feature DINOv2.

    Più ricca semanticamente di VGG16 — capisce la struttura semantica
    dell'immagine (pelle, cielo, vegetazione) invece di semplici texture.

    Poiché DINOv2 è già stato eseguito nel forward pass del modello,
    è possibile riusare F_sem invece di ricalcolarlo.

    L_perc = (1 / N·d) · ‖F_sem(pred) - F_sem(tgt)‖²_F

    Parameters
    ----------
    backbone : DINOv2 frozen (passato dall'esterno per evitare doppia istanza)
               Se None, viene caricato internamente (solo per test isolati)
    """

    def __init__(self, backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            self.backbone = None

    # ------------------------------------------------------------------
    def forward(
        self,
        pred:      torch.Tensor,
        tgt:       torch.Tensor,
        F_sem_pred: Optional[torch.Tensor] = None,
        F_sem_tgt:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if F_sem_pred is not None and F_sem_tgt is not None:
            return F.mse_loss(F_sem_pred.float(), F_sem_tgt.float())

        if self.backbone is None:
            raise ValueError(
                "PerceptualLoss richiede un backbone DINOv2 o feature pre-calcolate."
            )

        # Forza fp32 e clamp [0,1] prima di passare a DINOv2
        pred_f = pred.float().clamp(0, 1)
        tgt_f  = tgt.float().clamp(0, 1)

        with torch.no_grad():
            out_pred = self.backbone.forward_features(pred_f)
            out_tgt  = self.backbone.forward_features(tgt_f)

        f_pred = out_pred["x_norm_patchtokens"] if isinstance(out_pred, dict) else out_pred[:, 1:]
        f_tgt  = out_tgt["x_norm_patchtokens"]  if isinstance(out_tgt,  dict) else out_tgt[:, 1:]

        return F.mse_loss(f_pred.float(), f_tgt.float())


# ---------------------------------------------------------------------------
# Chroma Consistency Loss
# ---------------------------------------------------------------------------

class ChromaConsistencyLoss(nn.Module):
    """
    Penalizza differenze di saturazione e hue tra pred e target.

    L_chroma = L_sat + hue_weight · L_hue

    L_sat : differenza di saturazione (raggio nel piano a*b*)
    L_hue : differenza di angolo hue (distanza angolare in a*b*)

    Entrambe sono computate in CIE Lab per una percezione uniforme.

    Parameters
    ----------
    hue_weight : peso della componente hue (default 0.5)
    eps        : stabilità numerica per divisioni
    """

    def __init__(self, hue_weight: float = 0.5, eps: float = 1e-7) -> None:
        super().__init__()
        self.hue_weight = hue_weight
        self.eps        = eps

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        tgt:  torch.Tensor,
    ) -> torch.Tensor:
        pred_lab = rgb_to_lab(pred)               # (B, 3, H, W)
        tgt_lab  = rgb_to_lab(tgt)

        a_pred, b_pred = pred_lab[:, 1], pred_lab[:, 2]
        a_tgt,  b_tgt  = tgt_lab[:, 1],  tgt_lab[:, 2]

        # Saturazione = raggio nel piano a*b*
        sat_pred = (a_pred.pow(2) + b_pred.pow(2) + self.eps).sqrt()
        sat_tgt  = (a_tgt.pow(2)  + b_tgt.pow(2)  + self.eps).sqrt()
        L_sat    = F.l1_loss(sat_pred, sat_tgt)

        # Hue = angolo; la differenza angolare è in [-π, π]
        hue_pred = torch.atan2(b_pred, a_pred)
        hue_tgt  = torch.atan2(b_tgt,  a_tgt)
        dhue     = hue_pred - hue_tgt
        # Wrap a [-π, π]
        dhue = torch.remainder(dhue + torch.pi, 2 * torch.pi) - torch.pi
        L_hue = dhue.abs().mean()

        return L_sat + self.hue_weight * L_hue
