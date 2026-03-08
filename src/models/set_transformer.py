"""
models/set_transformer.py

Set Transformer per il calcolo del Style Prototype  (§5.1.3).

Pipeline:
  1. Calcola edit delta: δ_i = Enc(tgt_i) - Enc(src_i)
  2. Set Transformer (self-attention su {δ_i}) → pesa le coppie tipiche
  3. Mean pool → s ∈ R^256  (style prototype)

Proprietà: invariante alle permutazioni dell'input  (Teorema 1).
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention standard (senza positional encoding).
    Invariante alle permutazioni per costruzione.

    Args:
        dim:       Dimensione del token.
        num_heads: Numero di teste.
        dropout:   Dropout sull'attenzione.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim ({dim}) deve essere divisibile per num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)

        Returns:
            (B, N, dim)
        """
        B, N, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)          # ciascuno (B, H, N, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class SetTransformerLayer(nn.Module):
    """
    Singolo layer del Set Transformer:
        LayerNorm → MHSA → residual
        LayerNorm → FFN  → residual

    Args:
        dim:       Dimensione token.
        num_heads: Teste attenzione.
        mlp_ratio: Ratio espansione FFN.
        dropout:   Dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)  oppure  (N, dim)  (unbatched)

        Returns:
            stessa shape
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SetTransformer(nn.Module):
    """
    Set Transformer per aggregazione robusta delle edit delta.

    Fasi:
      1. Proiezione input: δ_i ∈ R^512 → R^128
      2. L_ST layer di self-attention (permutation-invariant)
      3. Mean pool → s ∈ R^128

    Il self-attention pesa le delta "tipiche" più degli outlier
    (coppie anomale), producendo un prototype robusto.

    Args:
        input_dim:    Dimensione delle edit delta (es. 512).
        hidden_dim:   Dimensione interna (default 256).
        output_dim:   Dimensione del prototype s (default 256).
        num_heads:    Teste di attenzione.
        num_layers:   Numero di layer Set Transformer (L_ST).
        mlp_ratio:    Ratio FFN.
        dropout:      Dropout.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Proiezione input
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack di layer Set Transformer
        self.layers = nn.ModuleList([
            SetTransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Proiezione output
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Aggrega le edit delta in un unico style prototype.

        Args:
            deltas: (B, N, input_dim)  oppure  (N, input_dim)
                    dove N = numero di coppie nel training set.

        Returns:
            (B, output_dim)  oppure  (output_dim,)  — style prototype s.
        """
        unbatched = deltas.dim() == 2
        if unbatched:
            deltas = deltas.unsqueeze(0)   # (1, N, input_dim)

        B, N, _ = deltas.shape

        # Proiezione input
        x = self.input_proj(deltas)       # (B, N, hidden_dim)

        # Self-attention layers (permutation-invariant)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Mean pool su N: (B, hidden_dim)
        x = x.mean(dim=1)

        # Proiezione output
        s = self.output_proj(x)           # (B, output_dim)

        if unbatched:
            s = s.squeeze(0)              # (output_dim,)

        return s


class StylePrototypeExtractor(nn.Module):
    """
    Calcola il style prototype s da un training set di coppie.

    Pipeline:
      δ_i = StyleEncoder(tgt_i) - StyleEncoder(src_i)   per i=1..N
      s   = SetTransformer({δ_i})

    Il prototype viene calcolato UNA SOLA VOLTA e cachato.
    Alla test time viene passato direttamente senza ricalcolo.

    Args:
        style_encoder: Modulo che produce embedding (B, embed_dim).
        set_transformer: Modulo SetTransformer.
    """

    def __init__(
        self,
        style_encoder: nn.Module,
        set_transformer: SetTransformer,
    ) -> None:
        super().__init__()
        self.style_encoder   = style_encoder
        self.set_transformer = set_transformer

    @torch.no_grad()
    def compute_prototype(
        self,
        src_list: List[torch.Tensor],
        tgt_list: List[torch.Tensor],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Calcola e restituisce il style prototype s.

        Args:
            src_list: Lista di tensori (3, H_i, W_i) normalizzati ImageNet.
            tgt_list: Lista di tensori (3, H_i, W_i) normalizzati ImageNet.
            batch_size: Dimensione batch per il calcolo (gestisce la memoria).

        Returns:
            s: (output_dim,)  — style prototype del fotografo.
        """
        assert len(src_list) == len(tgt_list), \
            "src_list e tgt_list devono avere la stessa lunghezza"

        device = next(self.style_encoder.parameters()).device
        deltas: List[torch.Tensor] = []

        # Calcolo delta in mini-batch per efficienza
        for i in range(0, len(src_list), batch_size):
            batch_src = torch.stack(src_list[i:i + batch_size]).to(device)
            batch_tgt = torch.stack(tgt_list[i:i + batch_size]).to(device)

            enc_src = self.style_encoder(batch_src)  # (b, embed_dim)
            enc_tgt = self.style_encoder(batch_tgt)  # (b, embed_dim)

            delta = enc_tgt - enc_src                # (b, embed_dim)
            deltas.append(delta.cpu())

        all_deltas = torch.cat(deltas, dim=0)        # (N, embed_dim)

        # Set Transformer su tutti i delta
        s = self.set_transformer(all_deltas.to(device))   # (output_dim,)
        return s

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola il prototype da tensori già in batch.

        Usato durante il training (gradient flow abilitato).

        Args:
            src: (N, 3, H, W)
            tgt: (N, 3, H, W)

        Returns:
            s: (output_dim,)
        """
        enc_src = self.style_encoder(src)   # (N, embed_dim)
        enc_tgt = self.style_encoder(tgt)   # (N, embed_dim)
        deltas  = enc_tgt - enc_src         # (N, embed_dim)
        return self.set_transformer(deltas) # (output_dim,)