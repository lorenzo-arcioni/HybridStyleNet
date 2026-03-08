"""
losses/perceptual.py

Perceptual Loss  (§6.4.3)  e  Style Loss con Gram Matrix  (§6.4.4).

Perceptual Loss:
    L_perc = Σ_l w_l · (1/C_l H_l W_l) ‖φ_l(pred) - φ_l(tgt)‖²_F

Style Loss:
    L_style = (1/4) Σ_l ‖G_l(pred) - G_l(tgt)‖²_F

dove φ_l sono le feature maps di VGG19 frozen ai layer:
    relu1_2, relu2_2, relu3_4, relu4_4
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Layer VGG19 e pesi perceptual (§6.4.3, tabella)
_VGG_LAYERS  = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]
_VGG_WEIGHTS = [1.0, 0.75, 0.5, 0.25]

# Normalizzazione ImageNet (VGG19 è addestrato su immagini normalizzate)
_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VGG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGG19Features(nn.Module):
    """
    Estrattore di feature da VGG19 pre-addestrato (pesi congelati).

    Restituisce feature maps ai layer specificati.
    I pesi sono SEMPRE congelati (require_grad=False).

    Args:
        layers:    Lista nomi layer da estrarre
                   (sottoinsieme di _VGG_LAYERS).
        normalize: Se True, normalizza l'input con statistiche ImageNet
                   prima di passarlo a VGG19. Usare True se l'input
                   è in [0,1] sRGB, False se è già normalizzato.
    """

    # Mappa nome layer → indice nel modello VGG19 sequenziale
    _LAYER_MAP = {
        "relu1_1":  1,
        "relu1_2":  3,
        "relu2_1":  6,
        "relu2_2":  8,
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,
        "relu3_4": 17,
        "relu4_1": 20,
        "relu4_2": 22,
        "relu4_3": 24,
        "relu4_4": 26,
    }

    def __init__(
        self,
        layers: List[str] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        layers = layers or _VGG_LAYERS

        # Verifica che tutti i layer richiesti siano validi
        for l in layers:
            if l not in self._LAYER_MAP:
                raise ValueError(
                    f"Layer VGG19 '{l}' non riconosciuto. "
                    f"Validi: {list(self._LAYER_MAP.keys())}"
                )

        self.layers    = layers
        self.normalize = normalize

        # Carica VGG19 pre-addestrato
        try:
            import torchvision.models as tvm
            vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                import torchvision.models as tvm
                vgg = tvm.vgg19(pretrained=True)
            except Exception as e:
                raise ImportError(
                    f"torchvision è richiesto per PerceptualLoss: {e}"
                )

        # Tronca il modello al layer massimo richiesto
        max_idx = max(self._LAYER_MAP[l] for l in layers)
        self._features = nn.Sequential(
            *list(vgg.features.children())[:max_idx + 1]
        )

        # Congela TUTTI i parametri
        for p in self._features.parameters():
            p.requires_grad = False

        self._max_idx = max_idx

        # Registra le statistiche ImageNet come buffer (segue il device del modello)
        self.register_buffer("_mean", _VGG_MEAN)
        self.register_buffer("_std",  _VGG_STD)

        logger.info(
            f"VGG19Features inizializzato — layers={layers}, "
            f"normalize={normalize}, frozen=True"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estrae feature maps ai layer specificati.

        Args:
            x: (B, 3, H, W) in [0,1] se normalize=True,
               o già normalizzato ImageNet se normalize=False.

        Returns:
            Dizionario {nome_layer: feature_map}.
        """
        if self.normalize:
            x = (x - self._mean) / self._std

        features: Dict[str, torch.Tensor] = {}
        layer_indices = {self._LAYER_MAP[l]: l for l in self.layers}

        out = x
        for i, module in enumerate(self._features):
            out = module(out)
            if i in layer_indices:
                features[layer_indices[i]] = out

        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss basata su feature VGG19  (§6.4.3).

    L_perc = Σ_l w_l · (1/C_l H_l W_l) ‖φ_l(pred) - φ_l(tgt)‖²_F

    Args:
        layers:  Layer VGG19 da usare.
        weights: Pesi per layer (default [1.0, 0.75, 0.5, 0.25]).
        normalize: Normalizza input con ImageNet stats.
    """

    def __init__(
        self,
        layers: List[str] = None,
        weights: List[float] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.layers  = layers  or _VGG_LAYERS
        self.weights = weights or _VGG_WEIGHTS

        assert len(self.layers) == len(self.weights), \
            "layers e weights devono avere la stessa lunghezza"

        self.vgg = VGG19Features(layers=self.layers, normalize=normalize)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [0,1].
            target: (B, 3, H, W) in [0,1].

        Returns:
            Scalare — perceptual loss pesata.
        """
        feats_pred   = self.vgg(pred)
        feats_target = self.vgg(target)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for layer, w in zip(self.layers, self.weights):
            fp = feats_pred[layer]
            ft = feats_target[layer]

            # Norma di Frobenius normalizzata per dimensione
            B, C, H, W = fp.shape
            norm_factor = C * H * W

            layer_loss = (fp - ft).pow(2).sum() / (B * norm_factor)
            loss = loss + w * layer_loss

        return loss


class StyleLoss(nn.Module):
    """
    Style Loss con Gram Matrix  (§6.4.4).

    L_style = (1/4) Σ_l ‖G_l(pred) - G_l(tgt)‖²_F

    G_l(I)_{c1,c2} = (1/C_l H_l W_l) Σ_p F_l^{c1}(p) · F_l^{c2}(p)

    Args:
        layers:    Layer VGG19.
        normalize: Normalizza input.
    """

    def __init__(
        self,
        layers: List[str] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.layers = layers or _VGG_LAYERS
        self.vgg    = VGG19Features(layers=self.layers, normalize=normalize)

    @staticmethod
    def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        """
        Calcola la Gram matrix normalizzata.

        Args:
            feat: (B, C, H, W)

        Returns:
            G: (B, C, C)
        """
        B, C, H, W = feat.shape
        # Reshape: (B, C, H*W)
        f = feat.view(B, C, H * W)
        # G = (1/CHW) · F · F^T
        G = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)
        return G

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [0,1].
            target: (B, 3, H, W) in [0,1].

        Returns:
            Scalare — style loss.
        """
        feats_pred   = self.vgg(pred)
        feats_target = self.vgg(target)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for layer in self.layers:
            G_pred = self.gram_matrix(feats_pred[layer])    # (B, C, C)
            G_tgt  = self.gram_matrix(feats_target[layer])  # (B, C, C)
            loss   = loss + (G_pred - G_tgt).pow(2).sum() / G_pred.shape[0]

        return loss / 4.0