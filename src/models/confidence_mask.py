"""
models/confidence_mask.py

Confidence Mask α(x,y) ∈ [0,1]  (§5.1.7).

Pesa pixel per pixel quanto usare il ramo locale vs il ramo globale:
    I_out = α ⊙ I_local + (1-α) ⊙ I_global

α è predetta da una rete leggera su feature di media risoluzione
(P3 + P4 interpolato), poi portata a piena risoluzione con
interpolazione bilineare (smooth transition).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceMask(nn.Module):
    """
    Predice la mappa di confidenza α da feature a media risoluzione.

    Architettura:
        [P4_upsampled ; P3]  →  Conv3×3 → ReLU → Conv1×1 → Sigmoid
        Upsampling bilineare a (H, W) per mascheratura finale.

    Args:
        p3_channels:  Canali di P3 (default 128).
        p4_channels:  Canali di P4 (default 256).
        hidden_dim:   Canali intermedi (default 64).
    """

    def __init__(
        self,
        p3_channels: int = 128,
        p4_channels: int = 256,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        in_ch = p3_channels + p4_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3,
                      padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

        # Inizializza l'ultimo layer con bias = 0
        # → sigmoid(0) = 0.5, bilancia equamente locale e globale
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        full_size: tuple,
    ) -> torch.Tensor:
        """
        Args:
            p3:        (B, C3, H3, W3)
            p4:        (B, C4, H4, W4)
            full_size: (H, W) — risoluzione piena dell'immagine

        Returns:
            alpha: (B, 1, H, W) ∈ [0,1]
                   1 = usa completamente il ramo locale
                   0 = usa completamente il ramo globale
        """
        H3, W3 = p3.shape[2], p3.shape[3]

        # Interpola P4 alla risoluzione di P3
        p4_up = F.interpolate(p4, size=(H3, W3),
                              mode="bilinear", align_corners=False)

        # Concatena lungo i canali
        x = torch.cat([p4_up, p3], dim=1)  # (B, C4+C3, H3, W3)

        # Predici logit della maschera
        alpha_low = torch.sigmoid(self.net(x))  # (B, 1, H3, W3)

        # Upsampling bilineare a risoluzione piena (smooth)
        alpha = F.interpolate(alpha_low, size=full_size,
                              mode="bilinear", align_corners=False)

        return alpha  # (B, 1, H, W)