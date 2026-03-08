"""
models/adain.py

AdaIN (Adaptive Instance Normalization) — §5.1.5
SPADE (Spatially Adaptive Normalization)  — §5.1.6
SPADEResBlock                              — §5.1.6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ── AdaIN ─────────────────────────────────────────────────────────────────────

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (Huang et al., ICCV 2017).

    AdaIN(h, s)_c = γ_c(s) · (h_c - μ_c(h)) / σ_c(h) + β_c(s)

    γ e β sono proiettati linearmente dallo stile s.

    Args:
        num_features:  Numero di canali C della feature map h.
        style_dim:     Dimensione del vettore di stile s.
    """

    def __init__(self, num_features: int, style_dim: int = 256) -> None:
        super().__init__()
        self.num_features = num_features

        # Proiezioni lineari: s → (γ, β) per ogni canale
        self.fc_gamma = nn.Linear(style_dim, num_features)
        self.fc_beta  = nn.Linear(style_dim, num_features)

        # Inizializza γ=1, β=0 → identità iniziale
        nn.init.ones_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, C, H, W)   feature map.
            s: (B, style_dim) vettore di stile.

        Returns:
            (B, C, H, W)  feature map normalizzata con stile.
        """
        B, C, H, W = h.shape

        # Statistiche per canale (Instance Norm)
        mu    = h.mean(dim=[2, 3], keepdim=True)        # (B, C, 1, 1)
        sigma = h.var(dim=[2, 3], keepdim=True).sqrt()  # (B, C, 1, 1)

        h_norm = (h - mu) / (sigma + EPS)               # (B, C, H, W)

        # Parametri dipendenti dallo stile
        gamma = self.fc_gamma(s).view(B, C, 1, 1)       # (B, C, 1, 1)
        beta  = self.fc_beta(s).view(B, C, 1, 1)        # (B, C, 1, 1)

        return gamma * h_norm + beta


# ── SPADE ─────────────────────────────────────────────────────────────────────

class SPADE(nn.Module):
    """
    Spatially Adaptive (De)normalization (Park et al., CVPR 2019).

    SPADE(h, m)_{c,x,y} = γ_c(x,y) · (h_{c,x,y} - μ_c) / σ_c + β_c(x,y)

    γ e β sono mappe spaziali derivate dalla mappa di conditioning m
    tramite convoluzioni 3×3.

    Args:
        num_features:  Canali C della feature map h.
        cond_channels: Canali C_m della mappa di conditioning m.
        hidden_dim:    Canali intermedi nelle conv di γ e β.
    """

    def __init__(
        self,
        num_features: int,
        cond_channels: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_features = num_features

        # Condizionamento: m → (γ, β)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_dim, kernel_size=3,
                      padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv_gamma = nn.Conv2d(hidden_dim, num_features,
                                    kernel_size=3, padding=1, bias=True)
        self.conv_beta  = nn.Conv2d(hidden_dim, num_features,
                                    kernel_size=3, padding=1, bias=True)

        # Inizializzazione identità
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, C, H, W)        feature map da normalizzare.
            m: (B, C_m, H_m, W_m)  mappa di conditioning (context).
               Viene interpolata bilinearmente a (H, W) se necessario.

        Returns:
            (B, C, H, W)  feature map normalizzata con conditioning spaziale.
        """
        B, C, H, W = h.shape

        # Interpolazione della mappa di conditioning alla risoluzione di h
        if m.shape[2] != H or m.shape[3] != W:
            m = F.interpolate(m, size=(H, W), mode="bilinear",
                              align_corners=False)

        # Statistiche per canale (normalizzazione)
        mu    = h.mean(dim=[2, 3], keepdim=True)        # (B, C, 1, 1)
        sigma = h.var(dim=[2, 3], keepdim=True).sqrt()  # (B, C, 1, 1)
        h_norm = (h - mu) / (sigma + EPS)               # (B, C, H, W)

        # Mappe spaziali γ e β
        shared = self.shared(m)                          # (B, hidden, H, W)
        gamma  = self.conv_gamma(shared)                 # (B, C, H, W)
        beta   = self.conv_beta(shared)                  # (B, C, H, W)

        return gamma * h_norm + beta


# ── SPADEResBlock ─────────────────────────────────────────────────────────────

class SPADEResBlock(nn.Module):
    """
    Residual block con SPADE normalization.

    Architecture (§5.1.6):
        Residual path:
            SPADE(h, m) → ReLU → Conv3×3 →
            SPADE(·, m) → ReLU → Conv3×3
        Shortcut (se C_in ≠ C_out):
            Conv1×1

    Args:
        in_channels:   Canali input C_in.
        out_channels:  Canali output C_out.
        cond_channels: Canali della mappa di conditioning m.
        hidden_dim:    Canali intermedi in SPADE.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        # Normalizzazioni SPADE per le due conv del residual path
        self.spade1 = SPADE(in_channels,  cond_channels, hidden_dim)
        self.spade2 = SPADE(out_channels, cond_channels, hidden_dim)

        # Convoluzioni residual
        self.conv1 = nn.Conv2d(in_channels,  out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)

        # Shortcut: necessaria se i canali cambiano
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, C_in, H, W)
            m: (B, C_m, H_m, W_m)  mappa di conditioning

        Returns:
            (B, C_out, H, W)
        """
        # Shortcut path
        h_sc = self.shortcut(h)

        # Residual path
        h1 = self.conv1(self.act(self.spade1(h, m)))
        h2 = self.conv2(self.act(self.spade2(h1, m)))

        return h2 + h_sc