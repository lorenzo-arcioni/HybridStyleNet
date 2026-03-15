"""
composite_loss.py
-----------------
Loss composita con curriculum dei pesi per RAG-ColorNet.

Assembla tutte le loss in un'unica funzione scalare e gestisce
il curriculum dei pesi in base alla fase e all'epoca corrente:

  Phase 1 (pretrain) / Phase 2 (meta):
    L = λ_ΔE·L_ΔE + λ_perc·L_perc + λ_ret·L_ret

  Phase 3 — Epoche 1-5 (early):
    L = 0.8·L_L1Lab + 0.4·L_hist + 0.3·L_cluster + 0.5·L_ret
      + 0.01·L_TV + 0.3·L_lum + 0.01·L_entropy

  Phase 3 — Epoche 6-10 (mid):
    L = 0.3·L_ΔE + 0.4·L_L1Lab + 0.3·L_hist + 0.3·L_perc
      + 0.1·L_chroma + 0.3·L_ret + 0.01·L_TV + 0.3·L_lum + 0.01·L_entropy

  Phase 3 — Epoche 11+ (late):
    L = 0.5·L_ΔE + 0.3·L_hist + 0.6·L_perc + 0.2·L_chroma
      + 0.2·L_ret + 0.01·L_TV + 0.3·L_lum + 0.01·L_entropy

L'interfaccia principale è CompositeLoss.forward(model_output, batch)
che riceve l'intero output del modello e il batch di training e
calcola automaticamente tutte le loss necessarie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .color_losses      import (
    DeltaELoss, L1LabLoss, HistogramEMDLoss,
    PerceptualLoss, ChromaConsistencyLoss,
)
from .structural_losses import (
    TotalVariationLoss, EntropyMaskLoss, LuminancePreservationLoss,
)
from .retrieval_loss    import RetrievalQualityLoss, ClusterAssignmentLoss


# ---------------------------------------------------------------------------
# LossWeights  (dataclass per i pesi del curriculum)
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    delta_e:   float = 0.0
    l1_lab:    float = 0.0
    histogram: float = 0.0
    perceptual:float = 0.0
    chroma:    float = 0.0
    cluster:   float = 0.0
    retrieval: float = 0.0
    tv:        float = 0.0
    luminance: float = 0.0
    entropy:   float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "LossWeights":
        return cls(**{k: float(v) for k, v in d.items()})

    def as_dict(self) -> Dict[str, float]:
        return {
            "delta_e":   self.delta_e,
            "l1_lab":    self.l1_lab,
            "histogram": self.histogram,
            "perceptual":self.perceptual,
            "chroma":    self.chroma,
            "cluster":   self.cluster,
            "retrieval": self.retrieval,
            "tv":        self.tv,
            "luminance": self.luminance,
            "entropy":   self.entropy,
        }


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Restituisce i pesi corretti in base alla fase e all'epoca.

    Parameters
    ----------
    cfg              : config dict completo (base.yaml merged)
    early_end_epoch  : ultima epoca dello stage "early" (default 5)
    mid_end_epoch    : ultima epoca dello stage "mid"   (default 10)
    """

    def __init__(
        self,
        cfg:             dict,
        early_end_epoch: int = 5,
        mid_end_epoch:   int = 10,
    ) -> None:
        lcfg = cfg["loss"]

        self._pretrain = LossWeights.from_dict(lcfg["pretrain"])
        self._early    = LossWeights.from_dict(lcfg["curriculum"]["early"])
        self._mid      = LossWeights.from_dict(lcfg["curriculum"]["mid"])
        self._late     = LossWeights.from_dict(lcfg["curriculum"]["late"])

        self.early_end = early_end_epoch
        self.mid_end   = mid_end_epoch

    # ------------------------------------------------------------------
    def get_weights(
        self,
        phase: str,           # "pretrain" | "meta" | "adapt"
        epoch: int = 1,       # 1-based
    ) -> LossWeights:
        """
        Restituisce i LossWeights appropriati.

        Parameters
        ----------
        phase : fase di training corrente
        epoch : epoca corrente (rilevante solo per phase="adapt")
        """
        if phase in ("pretrain", "meta"):
            return self._pretrain

        # phase == "adapt"
        if epoch <= self.early_end:
            return self._early
        elif epoch <= self.mid_end:
            return self._mid
        else:
            return self._late


# ---------------------------------------------------------------------------
# LossBreakdown  (risultato dettagliato del forward)
# ---------------------------------------------------------------------------

@dataclass
class LossBreakdown:
    total:      torch.Tensor
    delta_e:    Optional[torch.Tensor] = None
    l1_lab:     Optional[torch.Tensor] = None
    histogram:  Optional[torch.Tensor] = None
    perceptual: Optional[torch.Tensor] = None
    chroma:     Optional[torch.Tensor] = None
    cluster:    Optional[torch.Tensor] = None
    retrieval:  Optional[torch.Tensor] = None
    tv:         Optional[torch.Tensor] = None
    luminance:  Optional[torch.Tensor] = None
    entropy:    Optional[torch.Tensor] = None

    def as_loggable(self) -> Dict[str, float]:
        """Dizionario {nome: valore float} per logging."""
        out = {"loss/total": self.total.item()}
        for name in [
            "delta_e", "l1_lab", "histogram", "perceptual", "chroma",
            "cluster", "retrieval", "tv", "luminance", "entropy",
        ]:
            val = getattr(self, name)
            if val is not None:
                out[f"loss/{name}"] = val.item()
        return out


# ---------------------------------------------------------------------------
# CompositeLoss
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Loss composita con curriculum.

    Istanzia tutte le loss componenti e le combina secondo i pesi
    del curriculum corrente. Il metodo forward riceve l'output completo
    del modello e il batch di training.

    Parameters
    ----------
    cfg        : config dict completo
    backbone   : DINOv2 backbone per PerceptualLoss (può essere None
                 se si passano F_sem pre-calcolati)
    """

    def __init__(
        self,
        cfg:      dict,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.scheduler = CurriculumScheduler(
            cfg,
            early_end_epoch = cfg.get("adaptation", {}).get("curriculum_early_end", 5),
            mid_end_epoch   = cfg.get("adaptation", {}).get("curriculum_mid_end",   10),
        )

        # Istanzia tutte le loss componenti
        eps = cfg["loss"]["eps"]
        chroma_hue_w = cfg["loss"].get("chroma_hue_weight", 0.5)

        self.delta_e_loss    = DeltaELoss(eps=eps)
        self.l1_lab_loss     = L1LabLoss()
        self.hist_loss       = HistogramEMDLoss(
            n_bins=cfg["histogram"]["n_bins"]
        )
        self.perceptual_loss = PerceptualLoss(backbone=backbone)
        self.chroma_loss     = ChromaConsistencyLoss(hue_weight=chroma_hue_w, eps=eps)
        self.tv_loss         = TotalVariationLoss()
        self.entropy_loss    = EntropyMaskLoss(eps=eps)
        self.lum_loss        = LuminancePreservationLoss()
        self.retrieval_loss  = RetrievalQualityLoss()
        self.cluster_loss    = ClusterAssignmentLoss(eps=eps)

        # Stato corrente del curriculum (aggiornato da update_curriculum)
        self._phase  = "pretrain"
        self._epoch  = 1
        self._weights: LossWeights = self.scheduler.get_weights("pretrain")

    # ------------------------------------------------------------------
    def update_curriculum(self, phase: str, epoch: int) -> LossWeights:
        """
        Aggiorna la fase e l'epoca correnti e restituisce i nuovi pesi.
        Da chiamare all'inizio di ogni epoca nel training loop.
        """
        self._phase   = phase
        self._epoch   = epoch
        self._weights = self.scheduler.get_weights(phase, epoch)
        return self._weights

    # ------------------------------------------------------------------
    def forward(
        self,
        model_output: dict,
        batch:        dict,
        cluster_labels: Optional[torch.Tensor] = None,  # (B,) per ClusterAssignmentLoss
        edit_target:    Optional[torch.Tensor] = None,  # per RetrievalQualityLoss
    ) -> LossBreakdown:
        """
        Calcola la loss composita dal model_output e dal batch.

        Parameters
        ----------
        model_output    : dict restituito da RAGColorNet.forward()
                          chiavi: I_out, I_pred, I_global, I_local,
                                  alpha, p, G_global, G_local, F_sem, h, guide
        batch           : dict del DataLoader
                          chiavi: src, tgt
        cluster_labels  : (B,) hard assignment K-Means per ClusterAssignmentLoss
        edit_target     : (B, d_r, n_h, n_w) per RetrievalQualityLoss

        Returns
        -------
        LossBreakdown con total e tutti i termini individuali
        """
        w      = self._weights
        pred   = model_output["I_out"]            # (B, 3, H, W) — post-gamma
        src    = batch["src"].to(pred.device)
        tgt    = batch["tgt"].to(pred.device)

        total = torch.tensor(0.0, device=pred.device)
        terms: dict = {}

        # ── ΔE₀₀ ────────────────────────────────────────────────────────
        if w.delta_e > 0:
            v = self.delta_e_loss(pred, tgt)
            total = total + w.delta_e * v
            terms["delta_e"] = v

        # ── L1 Lab ──────────────────────────────────────────────────────
        if w.l1_lab > 0:
            v = self.l1_lab_loss(pred, tgt)
            total = total + w.l1_lab * v
            terms["l1_lab"] = v

        # ── Histogram EMD ────────────────────────────────────────────────
        if w.histogram > 0:
            v = self.hist_loss(pred, tgt)
            total = total + w.histogram * v
            terms["histogram"] = v

        # ── Perceptual ───────────────────────────────────────────────────
        if w.perceptual > 0:
            # Riusa F_sem di I_out se già calcolato (zero costo aggiuntivo)
            # In pratica il modello calcola F_sem su I_src — qui usiamo
            # il backbone per I_pred vs I_tgt
            v = self.perceptual_loss(pred, tgt)
            total = total + w.perceptual * v
            terms["perceptual"] = v

        # ── Chroma consistency ───────────────────────────────────────────
        if w.chroma > 0:
            v = self.chroma_loss(pred, tgt)
            total = total + w.chroma * v
            terms["chroma"] = v

        # ── Cluster assignment ────────────────────────────────────────────
        if w.cluster > 0 and cluster_labels is not None:
            v = self.cluster_loss(model_output["p"], cluster_labels)
            total = total + w.cluster * v
            terms["cluster"] = v

        # ── Retrieval quality ─────────────────────────────────────────────
        if w.retrieval > 0 and edit_target is not None:
            # R_spatial viene reshapato da (B, d_r, n_h, n_w) per il confronto
            v = self.retrieval_loss(
                model_output.get("R_spatial", edit_target),   # fallback
                edit_target,
            )
            total = total + w.retrieval * v
            terms["retrieval"] = v

        # ── Total Variation ───────────────────────────────────────────────
        if w.tv > 0:
            v = self.tv_loss(model_output["G_global"], model_output["G_local"])
            total = total + w.tv * v
            terms["tv"] = v

        # ── Luminance preservation ────────────────────────────────────────
        if w.luminance > 0:
            v = self.lum_loss(pred, src)
            total = total + w.luminance * v
            terms["luminance"] = v

        # ── Entropy mask ──────────────────────────────────────────────────
        if w.entropy > 0:
            v = self.entropy_loss(model_output["alpha"])
            total = total + w.entropy * v
            terms["entropy"] = v

        return LossBreakdown(total=total, **terms)

    # ------------------------------------------------------------------
    @property
    def current_weights(self) -> LossWeights:
        return self._weights

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg:      dict,
        backbone: Optional[nn.Module] = None,
    ) -> "CompositeLoss":
        return cls(cfg=cfg, backbone=backbone)
