"""
rag_colornet.py
---------------
RAG-ColorNet — Modello completo.

Orchestra i 5 componenti dell'architettura nel forward pass end-to-end:

  1. SceneEncoder         → F_sem, h, Q
  2. ClusterNet           → p (soft assignment)
  3. RetrievalModule      → R_spatial (retrieved edit come mappa spaziale)
  4. BilateralGridRenderer→ I_global, I_local, G_global, G_local, guide
  5. ConfidenceMaskBlender→ alpha, I_pred, I_out

Il database del fotografo (cluster_db) è passato come argomento al forward
per mantenere il modello stateless rispetto ai dati del fotografo.
Questo permette aggiornamenti incrementali del database senza retraining.

Parametri trainable totali (escl. DINOv2):
  ClusterNet:     ~100K
  Retrieval:      ~320K  (W_Q, W_K, W_V)
  GridNet:        ~500K
  MaskNet:        ~100K
  guide MLP:       ~25K
  ─────────────────────
  Totale:        ~1.05M
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .scene_encoder    import SceneEncoder
from .cluster_net      import ClusterNet
from .retrieval_module import RetrievalModule
from .bilateral_grid   import BilateralGridRenderer
from .confidence_mask  import ConfidenceMaskBlender


# ---------------------------------------------------------------------------
# RAGColorNet
# ---------------------------------------------------------------------------

class RAGColorNet(nn.Module):
    """
    RAG-ColorNet: Retrieval-Augmented Grading Network.

    Parameters
    ----------
    scene_encoder    : Componente 1 — estrazione feature
    cluster_net      : Componente 2 — soft cluster assignment
    retrieval_module : Componente 3 — cross-image local attention
    grid_renderer    : Componente 4 — bilateral grid rendering
    mask_blender     : Componente 5 — confidence mask + blending
    """

    def __init__(
        self,
        scene_encoder:    SceneEncoder,
        cluster_net:      ClusterNet,
        retrieval_module: RetrievalModule,
        grid_renderer:    BilateralGridRenderer,
        mask_blender:     ConfidenceMaskBlender,
    ) -> None:
        super().__init__()
        self.scene_encoder    = scene_encoder
        self.cluster_net      = cluster_net
        self.retrieval_module = retrieval_module
        self.grid_renderer    = grid_renderer
        self.mask_blender     = mask_blender

    # ------------------------------------------------------------------
    def forward(
        self,
        img:        torch.Tensor,           # (B, 3, H, W) sRGB [0,1]
        cluster_db: Dict[int, Dict],        # {k: {"keys": Tensor, "values": Tensor}}
    ) -> dict:
        """
        Forward pass completo.

        Parameters
        ----------
        img        : batch di immagini sorgente
        cluster_db : database del fotografo pre-calcolato
                     {k: {"keys":   (M_k, N_i, 416),
                           "values": (M_k, N_i, 384)}}

        Returns
        -------
        dict con tutte le uscite intermedie e finali:
          "I_out"   : (B, 3, H, W)  output finale sRGB ∈ [0,1]
          "I_pred"  : (B, 3, H, W)  pre-gamma blended
          "I_global": (B, 3, H, W)  grid globale applicata
          "I_local" : (B, 3, H, W)  grid locale applicata
          "alpha"   : (B, 1, H, W)  confidence mask
          "p"       : (B, K)        soft cluster assignment
          "G_global": (B, 12, 8,  8,  8)
          "G_local" : (B, 12, 16, 16, 8)
          "F_sem"   : (B, N, 384)   patch features DINOv2
          "h"       : (B, 192)      color histogram
          "guide"   : (B, H, W)     guida bilateral slicing
        """
        # ── 1. Scene Encoder ────────────────────────────────────────────
        enc_out = self.scene_encoder(img)
        F_sem   = enc_out["F_sem"]          # (B, N, 384)
        h       = enc_out["h"]              # (B, 192)
        Q       = enc_out["Q"]              # (B, N, 416)
        n_h     = enc_out["n_h"]
        n_w     = enc_out["n_w"]

        # ── 2. Cluster Assignment ────────────────────────────────────────
        p = self.cluster_net(h)             # (B, K)

        # ── 3. Local Retrieval ───────────────────────────────────────────
        R_spatial = self.retrieval_module(
            Q=Q, cluster_db=cluster_db, p=p, n_h=n_h, n_w=n_w
        )                                   # (B, d_r, n_h, n_w)

        # ── 4. Bilateral Grid Rendering ──────────────────────────────────
        render_out = self.grid_renderer(
            R_spatial=R_spatial, F_sem=F_sem, img=img, n_h=n_h, n_w=n_w
        )
        I_global = render_out["I_global"]   # (B, 3, H, W)
        I_local  = render_out["I_local"]    # (B, 3, H, W)
        G_global = render_out["G_global"]
        G_local  = render_out["G_local"]
        guide    = render_out["guide"]

        # ── 5. Confidence Mask + Blending ────────────────────────────────
        blend_out = self.mask_blender(
            F_sem=F_sem, I_local=I_local, I_global=I_global, n_h=n_h, n_w=n_w
        )
        alpha  = blend_out["alpha"]         # (B, 1, H, W)
        I_pred = blend_out["I_pred"]        # (B, 3, H, W)  pre-gamma
        I_out  = blend_out["I_out"]         # (B, 3, H, W)  finale

        return {
            # Output principale
            "I_out":    I_out,
            "I_pred":   I_pred,
            # Intermedi per la loss
            "I_global": I_global,
            "I_local":  I_local,
            "alpha":    alpha,
            "p":        p,
            "G_global": G_global,
            "G_local":  G_local,
            # Per debug / visualizzazione
            "F_sem":    F_sem,
            "h":        h,
            "guide":    guide,
        }

    # ------------------------------------------------------------------
    def encode_only(self, img: torch.Tensor) -> dict:
        """
        Esegue solo lo Scene Encoder (utile per il preprocessing del database).
        Non richiede cluster_db.

        Returns F_sem, h, Q, n_h, n_w.
        """
        return self.scene_encoder(img)

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """
        Generatore dei soli parametri trainable (esclude DINOv2 frozen).
        Da usare nell'ottimizzatore.
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def trainable_named_parameters(self):
        """Come trainable_parameters ma restituisce (nome, parametro)."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def count_all_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def set_adaptation_mode(self, step: int) -> None:
        """
        Imposta i moduli trainable in base allo step dell'adaptation.

        step=1 : congela i layer early del GridNet (partial fine-tune)
        step=2 : sblocca tutto il trainable
        """
        if step == 1:
            # Congela i primi layer del grid_renderer
            for name, param in self.grid_renderer.named_parameters():
                if "dino_proj" in name or "fusion_conv" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # Tutti gli altri trainable
            for module in [
                self.cluster_net,
                self.retrieval_module,
                self.mask_blender,
            ]:
                for param in module.parameters():
                    param.requires_grad = True

        elif step == 2:
            # Sblocca tutti i parametri trainable (DINOv2 resta frozen)
            for name, param in self.named_parameters():
                if "scene_encoder.backbone" not in name:
                    param.requires_grad = True

    # ------------------------------------------------------------------
    def replace_cluster_net(self, new_cluster_net: ClusterNet) -> None:
        """
        Sostituisce il ClusterNet dopo un re-clustering con K* diverso.
        Utile per la memoria incrementale quando K* cambia.
        """
        self.cluster_net = new_cluster_net

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, n_clusters: int) -> "RAGColorNet":
        """
        Factory method: costruisce il modello completo dal config dict.

        Parameters
        ----------
        cfg        : config dict (base.yaml + eventuali override)
        n_clusters : K* determinato da K-Means sul fotografo corrente
        """
        scene_encoder    = SceneEncoder.from_config(cfg)
        cluster_net      = ClusterNet.from_config(cfg, n_clusters)
        retrieval_module = RetrievalModule.from_config(cfg)
        grid_renderer    = BilateralGridRenderer.from_config(cfg)
        mask_blender     = ConfidenceMaskBlender.from_config(cfg)

        return cls(
            scene_encoder    = scene_encoder,
            cluster_net      = cluster_net,
            retrieval_module = retrieval_module,
            grid_renderer    = grid_renderer,
            mask_blender     = mask_blender,
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Stringa di riepilogo parametri per ogni sottomodulo."""
        lines = ["RAGColorNet — parameter count", "─" * 44]
        modules = {
            "scene_encoder (DINOv2 frozen)": self.scene_encoder.backbone,
            "scene_encoder (trainable)":     [
                self.scene_encoder.histogram,
                self.scene_encoder.chroma_feat,
                self.scene_encoder.layer_norm,
            ],
            "cluster_net":                   self.cluster_net,
            "retrieval_module":              self.retrieval_module,
            "grid_renderer":                 self.grid_renderer,
            "mask_blender":                  self.mask_blender,
        }
        for label, mod in modules.items():
            if isinstance(mod, list):
                n = sum(
                    sum(p.numel() for p in m.parameters() if p.requires_grad)
                    for m in mod
                )
            else:
                n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            lines.append(f"  {label:<40} {n:>10,}")

        lines.append("─" * 44)
        lines.append(f"  {'Total trainable':<40} {self.count_trainable_params():>10,}")
        lines.append(f"  {'Total (incl. frozen DINOv2)':<40} {self.count_all_params():>10,}")
        return "\n".join(lines)
