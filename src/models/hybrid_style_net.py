"""
models/hybrid_style_net.py

HybridStyleNet — modello completo  (§5.1).

Forward pass end-to-end:
    I_src (B,3,H,W) → I_pred (B,3,H,W)

Pipeline (§6.3 Forward Pass Completo):
  1. CNN stem: P1, P2, P3
  2. Swin stage 4-5: P4, P5
  3. Style prototype: s (da cache o calcolato live)
  4. Cross-attention: C3, C4
  5. AdaIN conditioning su P5 → Global Branch → G_global
  6. SPADE conditioning su P4, P3 → Local Branch → G_local
  7. Bilateral slicing a piena risoluzione
  8. Confidence Mask → blending
  9. Clipping [0,1]
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EfficientNetStem, StyleEncoder
from .swin import SwinStages
from .bilateral_grid import BilateralGrid, GlobalGridPredictor, LocalGridPredictor
from .adain import AdaIN, SPADEResBlock
from .set_transformer import SetTransformer, StylePrototypeExtractor
from .cross_attention import ContextualStyleConditioner
from .confidence_mask import ConfidenceMask

logger = logging.getLogger(__name__)


class HybridStyleNet(nn.Module):
    """
    Modello principale per photographer-specific color grading.

    Args:
        pretrained_encoder:  Se True, usa pesi ImageNet per EfficientNet-B4.
        frozen_stages:       Stage CNN da congelare nell'adattamento.
        style_embed_dim:     Dimensione embedding StyleEncoder.
        prototype_dim:       Dimensione del style prototype s.
        swin_window:         Dimensione finestra Swin.
        swin_depths:         [depth_stage4, depth_stage5].
        swin_heads:          [heads_stage4, heads_stage5].
        use_rope:            Se True, usa RoPE nel Swin.
        bil_global_h/w:      Risoluzione grid globale (default 8×8).
        bil_local_h/w:       Risoluzione grid locale (default 32×32).
        bil_luma_bins:       Bin di luminanza (default 8).
        cross_attn_out_dim:  Dimensione output cross-attention (default 128).
        cross_attn_heads:    Teste cross-attention.
        set_transformer_layers: Layer Set Transformer.
        set_transformer_heads:  Teste Set Transformer.
    """

    def __init__(
        self,
        pretrained_encoder: bool = True,
        frozen_stages: Optional[List[int]] = None,
        style_embed_dim: int = 512,
        prototype_dim: int = 128,
        swin_window: int = 7,
        swin_depths: Tuple[int, int] = (2, 2),
        swin_heads: Tuple[int, int] = (8, 16),
        use_rope: bool = True,
        bil_global_h: int = 8,
        bil_global_w: int = 8,
        bil_local_h: int = 32,
        bil_local_w: int = 32,
        bil_luma_bins: int = 8,
        cross_attn_out_dim: int = 128,
        cross_attn_heads: int = 4,
        set_transformer_layers: int = 2,
        set_transformer_heads: int = 8,
    ) -> None:
        super().__init__()

        # ── 1. CNN Stem ───────────────────────────────────────────────────────
        self.cnn_stem = EfficientNetStem(
            pretrained=pretrained_encoder,
            frozen_stages=frozen_stages or [],
        )
        p3_ch = self.cnn_stem.out_channels["P3"]   # 128
        p4_ch = 256
        p5_ch = 512

        # ── 2. Swin Transformer stage 4-5 ────────────────────────────────────
        self.swin = SwinStages(
            in_channels=p3_ch,
            stage4_channels=p4_ch,
            stage5_channels=p5_ch,
            stage4_heads=swin_heads[0],
            stage5_heads=swin_heads[1],
            stage4_depth=swin_depths[0],
            stage5_depth=swin_depths[1],
            window_size=swin_window,
            use_rope=use_rope,
        )

        # ── 3. Style Encoder + Set Transformer ───────────────────────────────
        self.style_encoder = StyleEncoder(
            embed_dim=style_embed_dim,
            pretrained=pretrained_encoder,
            frozen_stages=frozen_stages or [],
        )

        self.set_transformer = SetTransformer(
            input_dim=style_embed_dim,
            hidden_dim=prototype_dim,
            output_dim=prototype_dim,
            num_heads=set_transformer_heads,
            num_layers=set_transformer_layers,
        )

        self.prototype_extractor = StylePrototypeExtractor(
            style_encoder=self.style_encoder,
            set_transformer=self.set_transformer,
        )

        # ── 4. Cross-Attention ────────────────────────────────────────────────
        self.context_conditioner = ContextualStyleConditioner(
            query_dim=p5_ch,
            embed_dim=style_embed_dim,
            out_dim=cross_attn_out_dim,
            num_heads=cross_attn_heads,
        )

        # ── 5. Global Branch (AdaIN + MLP → G_global) ────────────────────────
        self.adain_global = AdaIN(
            num_features=p5_ch,
            style_dim=prototype_dim,
        )
        self.global_grid_predictor = GlobalGridPredictor(
            in_features=p5_ch,
            grid_h=bil_global_h,
            grid_w=bil_global_w,
            luma_bins=bil_luma_bins,
        )

        # ── 6. Local Branch (SPADE ResBlocks → G_local) ──────────────────────
        self.spade_block_p4 = SPADEResBlock(
            in_channels=p4_ch,
            out_channels=p4_ch,
            cond_channels=cross_attn_out_dim,
        )
        self.spade_block_p3 = SPADEResBlock(
            in_channels=p3_ch + p4_ch,   # skip connection da P3
            out_channels=p4_ch,
            cond_channels=cross_attn_out_dim,
        )
        self.local_grid_predictor = LocalGridPredictor(
            in_channels=p4_ch,
            grid_h=bil_local_h,
            grid_w=bil_local_w,
            luma_bins=bil_luma_bins,
        )

        # ── 7. Bilateral Grid (slicing) ───────────────────────────────────────
        self.bil_grid = BilateralGrid(
            grid_h=bil_global_h,
            grid_w=bil_global_w,
            luma_bins=bil_luma_bins,
        )
        self.bil_grid_local = BilateralGrid(
            grid_h=bil_local_h,
            grid_w=bil_local_w,
            luma_bins=bil_luma_bins,
        )

        # ── 8. Confidence Mask ────────────────────────────────────────────────
        self.confidence_mask = ConfidenceMask(
            p3_channels=p3_ch,
            p4_channels=p4_ch,
        )

        # Prototype cachato (None finché non calcolato)
        self._cached_prototype:    Optional[torch.Tensor] = None
        self._cached_train_keys:   Optional[torch.Tensor] = None
        self._cached_train_values: Optional[torch.Tensor] = None

        logger.info("HybridStyleNet inizializzato.")

    # ── Cache management ──────────────────────────────────────────────────────

    def set_style_cache(
        self,
        src_list: List[torch.Tensor],
        tgt_list: List[torch.Tensor],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Calcola e casha il style prototype + chiavi/valori del training set.

        Chiamare UNA VOLTA dopo aver caricato il training set del fotografo.

        Args:
            src_list: Lista di (3, H_i, W_i) — sorgenti normalizzati ImageNet.
            tgt_list: Lista di (3, H_i, W_i) — target normalizzati ImageNet.
            batch_size: Batch per il calcolo.

        Returns:
            s: (prototype_dim,) — style prototype.
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            # Calcola embedding
            keys, values = [], []
            for i in range(0, len(src_list), batch_size):
                b_src = torch.stack(src_list[i:i+batch_size]).to(device)
                b_tgt = torch.stack(tgt_list[i:i+batch_size]).to(device)
                e_src = self.style_encoder(b_src)
                e_tgt = self.style_encoder(b_tgt)
                keys.append(e_src.cpu())
                values.append((e_tgt - e_src).cpu())

            all_keys   = torch.cat(keys,   dim=0)   # (N, embed_dim)
            all_values = torch.cat(values, dim=0)   # (N, embed_dim)

            # Style prototype
            s = self.set_transformer(all_values.to(device))

        self._cached_prototype    = s.detach()
        self._cached_train_keys   = all_keys.to(device).detach()
        self._cached_train_values = all_values.to(device).detach()

        self.context_conditioner.set_train_cache(
            self._cached_train_keys,
            self._cached_train_values,
        )

        logger.info(
            f"Style cache impostata: N={len(src_list)} coppie, "
            f"prototype shape={s.shape}"
        )
        return s

    def clear_cache(self) -> None:
        """Cancella la cache dello stile."""
        self._cached_prototype    = None
        self._cached_train_keys   = None
        self._cached_train_values = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        src: torch.Tensor,
        prototype: Optional[torch.Tensor] = None,
        train_keys: Optional[torch.Tensor] = None,
        train_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo.

        Args:
            src:          (B, 3, H, W) — immagine sorgente normalizzata ImageNet.
            prototype:    (B, prototype_dim) o (prototype_dim,) — style prototype s.
                          Se None, usa la cache interna.
            train_keys:   (N, embed_dim) — override cache.
            train_values: (N, embed_dim) — override cache.

        Returns:
            Dizionario con:
                "pred":    (B, 3, H, W)  immagine graded in [0,1]
                "alpha":   (B, 1, H, W)  confidence mask
                "G_global": (B, 12, L, H_g, W_g)  bilateral grid globale
                "G_local":  (B, 12, L, H_l, W_l)  bilateral grid locale
        """
        B, _, H, W = src.shape

        # Denormalizziamo ImageNet → [0,1] per il bilateral slicing
        # (il CNN stem opera su valori normalizzati)
        src_01 = self._denormalize(src)  # (B, 3, H, W) in [0,1]

        # ── Step 1: CNN stem ──────────────────────────────────────────────────
        cnn_feats = self.cnn_stem(src)
        p3 = cnn_feats["P3"]   # (B, 128, H/8, W/8)

        # ── Step 2: Swin ──────────────────────────────────────────────────────
        swin_feats = self.swin(p3)
        p4 = swin_feats["P4"]  # (B, 256, H/16, W/16)
        p5 = swin_feats["P5"]  # (B, 512, H/32, W/32)

        # ── Step 3: Style prototype ───────────────────────────────────────────
        s = prototype
        if s is None:
            assert self._cached_prototype is not None, \
                "Prototype non disponibile. Chiama set_style_cache() prima."
            s = self._cached_prototype

        # Broadcast a batch se necessario
        if s.dim() == 1:
            s = s.unsqueeze(0).expand(B, -1)  # (B, prototype_dim)

        # ── Step 4: Cross-attention ───────────────────────────────────────────
        keys   = train_keys   if train_keys   is not None else self._cached_train_keys
        values = train_values if train_values is not None else self._cached_train_values

        C3, C4 = self.context_conditioner(
            p5=p5,
            p3_size=(p3.shape[2], p3.shape[3]),
            p4_size=(p4.shape[2], p4.shape[3]),
            train_keys=keys,
            train_values=values,
        )   # C3: (B,256,H/8,W/8), C4: (B,256,H/16,W/16)

        # ── Step 5: Global Branch ─────────────────────────────────────────────
        p5_adain = self.adain_global(p5, s)                 # (B,512,H5,W5)
        f_global = p5_adain.mean(dim=[2, 3])                 # (B,512) GAP
        G_global = self.global_grid_predictor(f_global)
        # G_global: (B, 12, luma_bins, grid_h_g, grid_w_g)

        # ── Step 6: Local Branch ──────────────────────────────────────────────
        x4 = self.spade_block_p4(p4, C4)                    # (B,256,H4,W4)

        # Upsampling x4 alla risoluzione di P3 + skip connection
        x4_up = F.interpolate(x4, size=(p3.shape[2], p3.shape[3]),
                              mode="bilinear", align_corners=False)
        x3_in = torch.cat([x4_up, p3], dim=1)               # (B,256+128,H3,W3)

        x3 = self.spade_block_p3(x3_in, C3)                 # (B,256,H3,W3)
        G_local = self.local_grid_predictor(x3)
        # G_local: (B, 12, luma_bins, grid_h_l, grid_w_l)

        # ── Step 7: Bilateral slicing a risoluzione piena ────────────────────
        I_global = self.bil_grid.apply(G_global, src_01)     # (B,3,H,W)
        I_local  = self.bil_grid_local.apply(G_local, I_global)  # (B,3,H,W)

        # ── Step 8: Confidence mask + blending ────────────────────────────────
        alpha = self.confidence_mask(p3, p4, full_size=(H, W))  # (B,1,H,W)
        I_out = alpha * I_local + (1.0 - alpha) * I_global      # (B,3,H,W)

        # ── Step 9: Clipping ──────────────────────────────────────────────────
        pred = I_out.clamp(0.0, 1.0)

        return {
            "pred":     pred,
            "alpha":    alpha,
            "G_global": G_global,
            "G_local":  G_local,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _denormalize(x: torch.Tensor) -> torch.Tensor:
        """ImageNet normalization → [0,1]."""
        mean = torch.tensor([0.485, 0.456, 0.406],
                             dtype=x.dtype, device=x.device)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225],
                             dtype=x.dtype, device=x.device)[:, None, None]
        return (x * std + mean).clamp(0.0, 1.0)

    def freeze_for_adaptation(self) -> None:
        """
        Congela i parametri per la Fase 3A (§5.4):
        CNN stage 1-2 + Swin stage 4 → solo Theta_slow e Theta_adapt aggiornati.
        """
        self.cnn_stem.freeze_stages([1, 2])
        for p in self.swin.stage4.parameters():
            p.requires_grad = False
        logger.info("HybridStyleNet: Fase 3A — frozen CNN[1-2] + Swin stage4.")

    def unfreeze_all(self) -> None:
        """
        Scongela tutti i parametri per la Fase 3B (§5.4).
        """
        for p in self.parameters():
            p.requires_grad = True
        self.cnn_stem.unfreeze_all()
        logger.info("HybridStyleNet: Fase 3B — tutti i parametri scongelati.")

    def count_parameters(self) -> Dict[str, int]:
        """Conta i parametri per componente."""
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "cnn_stem":           _count(self.cnn_stem),
            "swin":               _count(self.swin),
            "style_encoder":      _count(self.style_encoder),
            "set_transformer":    _count(self.set_transformer),
            "cross_attention":    _count(self.context_conditioner),
            "global_branch":      _count(self.adain_global) +
                                  _count(self.global_grid_predictor),
            "local_branch":       _count(self.spade_block_p4) +
                                  _count(self.spade_block_p3) +
                                  _count(self.local_grid_predictor),
            "confidence_mask":    _count(self.confidence_mask),
            "total":              _count(self),
        }