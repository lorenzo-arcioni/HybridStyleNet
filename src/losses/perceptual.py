"""
losses/perceptual.py

Perceptual Loss — similarità semantica multi-scala via VGG16.

Usa le feature maps di VGG16 pre-addestrata su ImageNet (frozen) a due
layer specifici per calcolare la distanza L2 normalizzata tra immagine
predetta e target nella rappresentazione spazio delle feature.

Riferimento tesi: §6.5.3
Formula: L_perc = Σ_{l∈{2,3}} w_l · (1/C_l H_l W_l) · ‖φ_l(pred) - φ_l(tgt)‖²_F

Layer usati:
    relu2_2  → C=128, H/2 × W/2  — texture, pattern semplici
    relu3_3  → C=256, H/4 × W/4  — strutture, parti semantiche
Pesi: w = [1.0, 0.75]

NOTA: I pesi VGG16 sono sempre frozen (requires_grad=False).
      Il calcolo avviene in float32, indipendentemente dalla precisione
      del forward pass del modello principale.

NOTA sul peso λ_perc = 0.6:
    Aumentato da 0.4 rispetto alla versione con style loss, per compensare
    la rimozione di L_style (ridondante con L_perc su stesse feature VGG16,
    correlazione ρ ≈ 1.0 sul ranking delle trasformazioni — §6.5.3).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from typing import List, Dict

EPS = 1e-8

# Nomi dei layer VGG16 e pesi corrispondenti (§6.5.3)
_VGG_LAYER_NAMES  = ["relu2_2", "relu3_3"]
_VGG_LAYER_WEIGHTS = {"relu2_2": 1.0, "relu3_3": 0.75}

# Indici nei features di torchvision VGG16 corrispondenti ai layer target
# VGG16 features sequenziali:
#   0  conv1_1  1  relu1_1  2  conv1_2  3  relu1_2  4  maxpool
#   5  conv2_1  6  relu2_1  7  conv2_2  8  relu2_2  ← indice 8
#   9  maxpool
#  10  conv3_1 11  relu3_1 12  conv3_2 13  relu3_2 14  conv3_3 15  relu3_3  ← indice 15
_VGG_LAYER_INDICES: Dict[str, int] = {
    "relu2_2": 8,
    "relu3_3": 15,
}

# Statistiche ImageNet per la normalizzazione dell'input a VGG16
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


class VGG16FeatureExtractor(nn.Module):
    """
    Estrae feature maps da VGG16 ai layer relu2_2 e relu3_3.

    Il modello VGG16 è caricato con pesi ImageNet, troncato all'ultimo
    layer necessario e completamente frozen (no gradient flow).

    Il forward ritorna un dizionario layer_name → feature_map.
    """

    def __init__(self):
        super().__init__()

        # Carica VGG16 pre-addestrata, prendi solo i features
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Tronca al massimo indice necessario (relu3_3 = indice 15 → +1)
        max_idx = max(_VGG_LAYER_INDICES.values()) + 1
        self.features = nn.Sequential(*list(features.children())[:max_idx])

        # Congela tutti i parametri
        for param in self.parameters():
            param.requires_grad_(False)

        # Registra i layer di interesse per hook
        self._layer_outputs: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        # Buffer per normalizzazione ImageNet
        self.register_buffer(
            "imagenet_mean",
            _IMAGENET_MEAN.view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            _IMAGENET_STD.view(1, 3, 1, 1),
        )

    def _register_hooks(self):
        """Registra forward hook per catturare output ai layer target."""
        children = list(self.features.children())
        for name, idx in _VGG_LAYER_INDICES.items():
            # Closure per catturare `name`
            def make_hook(layer_name):
                def hook(module, input, output):
                    self._layer_outputs[layer_name] = output
                return hook
            children[idx].register_forward_hook(make_hook(name))

    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalizza un'immagine sRGB [0,1] con le statistiche ImageNet.

        Args:
            img: (B, 3, H, W) float32, valori in [0,1].

        Returns:
            (B, 3, H, W) float32 normalizzato.
        """
        mean = self.imagenet_mean.to(img.device, img.dtype)
        std  = self.imagenet_std.to(img.device, img.dtype)
        return (img - mean) / (std + EPS)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estrae feature maps da VGG16.

        Args:
            img: (B, 3, H, W) float32, sRGB in [0,1].

        Returns:
            Dict layer_name → feature_map (B, C_l, H_l, W_l) float32.
        """
        self._layer_outputs.clear()
        x = self.normalize(img.float())
        self.features(x)  # forward — gli hook popolano _layer_outputs
        # Copia per evitare che il dizionario venga sovrascritto al prossimo call
        return {k: v for k, v in self._layer_outputs.items()}


class PerceptualLoss(nn.Module):
    """
    Perceptual loss multi-scala tramite feature VGG16.

    Calcola la norma di Frobenius normalizzata tra feature maps
    dell'immagine predetta e del target a relu2_2 e relu3_3 di VGG16.
    Nessuna gram matrix (la style loss è stata rimossa per ridondanza).

    Formula:
        L_perc = Σ_{l} w_l · (1/C_l H_l W_l) · ‖φ_l(pred) - φ_l(tgt)‖²_F

    Example:
        >>> loss_fn = PerceptualLoss()
        >>> pred = torch.rand(2, 3, 384, 512)
        >>> tgt  = torch.rand(2, 3, 384, 512)
        >>> loss = loss_fn(pred, tgt)   # scalar tensor
    """

    def __init__(self):
        super().__init__()
        self.extractor = VGG16FeatureExtractor()
        # Pesi per layer (§6.5.3): relu2_2=1.0, relu3_3=0.75
        self.layer_weights = _VGG_LAYER_WEIGHTS

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   sRGB (B, 3, H, W) in [0,1].
            target: sRGB (B, 3, H, W) in [0,1].

        Returns:
            Scalare: somma pesata delle distanze L2 nelle feature spaces.
        """
        # VGG forward su pred e target — in float32
        feats_pred   = self.extractor(pred)
        feats_target = self.extractor(target)

        loss = torch.zeros(1, device=pred.device, dtype=torch.float32)

        for layer_name, w in self.layer_weights.items():
            fp = feats_pred[layer_name]    # (B, C_l, H_l, W_l)
            ft = feats_target[layer_name]

            # Norma di Frobenius normalizzata per numero di elementi
            B, C, H, W = fp.shape
            n_elements = C * H * W

            diff_sq = (fp - ft.detach()) ** 2   # target non contribuisce al grad
            loss = loss + w * diff_sq.sum() / (B * n_elements)

        return loss
