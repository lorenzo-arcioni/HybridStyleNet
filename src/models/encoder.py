"""
models/encoder.py

CNN Stem: EfficientNet-B0 stage 1-3  (§5.1 — CNN Stem).

Estrae feature maps a tre scale:
  P3: (B, 128, H/8,  W/8)   ← feature locali: texture, bordi, skin tone
  P2: (B,  24, H/4,  W/4)   ← scala intermedia
  P1: (B,  16, H/2,  W/2)   ← bassa astrazione

Usa pesi pre-addestrati ImageNet via timm.
Supporta congelamento selettivo degli stage per il few-shot adaptation.

Dipendenze: timm >= 0.9
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canali di output per stage di EfficientNet-B0
# Stage 1 → 16ch, Stage 2 → 24ch, Stage 3 → 40ch
# Usiamo un projection layer per portare stage3 a 128ch come da tesi
_EFFICIENTNET_B0_CHANNELS = {1: 16, 2: 24, 3: 40}
_P3_OUT_CHANNELS = 128   # canali target per P3 dopo proiezione


class EfficientNetStem(nn.Module):
    """
    Encoder CNN basato su EfficientNet-B0 stage 1-3.

    Restituisce un dizionario di feature maps:
        {
            "P1": (B,  16, H/2,  W/2),
            "P2": (B,  24, H/4,  W/4),
            "P3": (B, 128, H/8,  W/8),
        }

    EfficientNet-B0 è più leggero di B4 (5.3M vs 19M parametri) e
    sufficiente per il task di color grading, dove la semantica
    richiesta è cromatica piuttosto che ad alta astrazione.

    Args:
        pretrained:      Se True, carica pesi ImageNet da timm.
        frozen_stages:   Lista di stage da congelare (es. [1, 2]).
        out_channels_p3: Canali di output per P3 dopo proiezione
                         (default 128, come da §5.1).
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: Optional[List[int]] = None,
        out_channels_p3: int = _P3_OUT_CHANNELS,
    ) -> None:
        super().__init__()

        self.frozen_stages = frozen_stages or []
        self.out_channels  = {
            "P1": _EFFICIENTNET_B0_CHANNELS[1],
            "P2": _EFFICIENTNET_B0_CHANNELS[2],
            "P3": out_channels_p3,
        }

        # ── Carica EfficientNet-B0 da timm ───────────────────────────────────
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm è richiesto per EfficientNetStem. "
                "Installa con: pip install timm"
            )

        # features_only=True → restituisce feature maps intermedie
        # out_indices=(1,2,3) → stage 1, 2, 3 (0-indexed in timm)
        out_idx = (1, 2, 3)
        self._backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_idx,
        )

        # Canali reali rilevati da timm feature_info (robusto a variazioni di versione)
        feature_info = self._backbone.feature_info
        real_ch = [feature_info.info[i]["num_chs"] for i in out_idx]
        # real_ch[0] = stage1_ch (~16), real_ch[1] = stage2_ch (~24), real_ch[2] = stage3_ch (~40)

        # Aggiorna i canali reali per P1 e P2 da feature_info
        self.out_channels["P1"] = real_ch[0]
        self.out_channels["P2"] = real_ch[1]

        # ── Proiezione P3: stage3_ch (40) → out_channels_p3 (128) ────────────
        stage3_ch = real_ch[2]
        if stage3_ch != out_channels_p3:
            self._proj_p3 = nn.Sequential(
                nn.Conv2d(stage3_ch, out_channels_p3, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels_p3),
                nn.SiLU(inplace=True),
            )
            logger.debug(
                f"EfficientNetStem: proiezione P3 {stage3_ch}ch → {out_channels_p3}ch"
            )
        else:
            self._proj_p3 = nn.Identity()

        # ── Congelamento stage ────────────────────────────────────────────────
        self._apply_frozen_stages()

        logger.info(
            f"EfficientNetStem (B0) inizializzato — pretrained={pretrained}, "
            f"frozen_stages={self.frozen_stages}, "
            f"out_channels={self.out_channels}"
        )

    # ── Congelamento ─────────────────────────────────────────────────────────

    def _apply_frozen_stages(self) -> None:
        """
        Congela i parametri degli stage specificati.

        In timm features_only per EfficientNet-B0:
          stage 1 → blocks[0]  (MBConv1, 16ch)
          stage 2 → blocks[1]  (MBConv6, 24ch)
          stage 3 → blocks[2]  (MBConv6, 40ch)
        """
        if not self.frozen_stages:
            return

        stage_to_block = {1: 0, 2: 1, 3: 2}

        for stage in self.frozen_stages:
            block_idx = stage_to_block.get(stage)
            if block_idx is None:
                logger.warning(f"Stage {stage} non trovato per il congelamento.")
                continue

            # Congela anche conv_stem + bn1 se si congela lo stage 1
            if stage == 1:
                for module_name in ["conv_stem", "bn1"]:
                    m = getattr(self._backbone, module_name, None)
                    if m is not None:
                        for p in m.parameters():
                            p.requires_grad = False

            block_list = getattr(self._backbone, "blocks", None)
            if block_list is not None and block_idx < len(block_list):
                for p in block_list[block_idx].parameters():
                    p.requires_grad = False
                logger.debug(f"Stage {stage} (block {block_idx}) congelato.")

    def freeze_stages(self, stages: List[int]) -> None:
        """Congela dinamicamente una lista di stage durante il training."""
        self.frozen_stages = stages
        self._apply_frozen_stages()

    def unfreeze_all(self) -> None:
        """Scongela tutti i parametri."""
        for p in self.parameters():
            p.requires_grad = True
        self.frozen_stages = []
        logger.info("EfficientNetStem: tutti i parametri scongelati.")

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass del CNN stem.

        Args:
            x: Tensore float32 (B, 3, H, W) normalizzato ImageNet.

        Returns:
            Dizionario con chiavi "P1", "P2", "P3":
                P1: (B,  16, H/2,  W/2)
                P2: (B,  24, H/4,  W/4)
                P3: (B, 128, H/8,  W/8)
        """
        feats = self._backbone(x)   # lista [feat_stage1, feat_stage2, feat_stage3]

        p1 = feats[0]                      # (B, 16, H/2, W/2)
        p2 = feats[1]                      # (B, 24, H/4, W/4)
        p3 = self._proj_p3(feats[2])       # (B, 128, H/8, W/8)

        return {"P1": p1, "P2": p2, "P3": p3}


# ── Style Encoder (CNN stem + GAP) ───────────────────────────────────────────

class StyleEncoder(nn.Module):
    """
    Encoder per estrarre un vettore di stile globale da un'immagine.

    Pipeline: EfficientNet-B0 stage 1-3 → P3 → Global Average Pool → MLP

    Usato nel Set Transformer per calcolare le edit delta:
        δ_i = StyleEncoder(tgt_i) - StyleEncoder(src_i)

    Il GAP rende l'encoder resolution-agnostic: l'output è sempre
    un vettore fisso di dimensione `embed_dim` indipendentemente
    dalla risoluzione dell'input.

    Args:
        embed_dim:     Dimensione del vettore di embedding output (default 512).
        pretrained:    Se True, usa pesi ImageNet.
        frozen_stages: Stage da congelare.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        pretrained: bool = True,
        frozen_stages: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self._stem = EfficientNetStem(
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )

        p3_ch = self._stem.out_channels["P3"]   # 128 dopo proiezione

        # MLP di proiezione: P3_gap (128) → embed_dim (512)
        self._proj = nn.Sequential(
            nn.Linear(p3_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.embed_dim = embed_dim

        logger.info(
            f"StyleEncoder (B0) — embed_dim={embed_dim}, "
            f"p3_ch={p3_ch}, pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalizzato ImageNet.

        Returns:
            (B, embed_dim) vettore di stile globale.
        """
        feats = self._stem(x)
        p3    = feats["P3"]           # (B, 128, H/8, W/8)
        gap   = p3.mean(dim=[2, 3])   # (B, 128) — Global Average Pool
        return self._proj(gap)         # (B, embed_dim)