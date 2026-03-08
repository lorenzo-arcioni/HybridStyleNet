"""
models/cross_attention.py

Cross-Attention per in-context style conditioning  (§5.1.4).

Data la feature map dell'immagine di test P5 e le coppie del
training set, seleziona dinamicamente le edit delta più rilevanti
per ogni token/regione dell'immagine di test.

Risultato: context ∈ R^(T, d_c) — conditioning spazialmente variabile.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """
    Cross-Attention tra feature dell'immagine di test e
    chiavi/valori del training set.

    Query  = P5 features della nuova immagine  (T tokens)
    Keys   = Enc(src_i)  del training set       (N tokens)
    Values = δ_i = Enc(tgt_i) - Enc(src_i)     (N tokens)

    A_{t,n} ≥ 0 rappresenta il peso che il token t assegna
    alla coppia di training n → mixture of edits contestuale.

    Args:
        query_dim:  Dimensione dei token di P5 (es. 512).
        key_dim:    Dimensione dei token del training set (es. 512).
        value_dim:  Dimensione delle edit delta (es. 512).
        out_dim:    Dimensione del context output (d_c, default 256).
        num_heads:  Numero di teste di attenzione.
        dropout:    Dropout sull'attenzione.
    """

    def __init__(
        self,
        query_dim: int = 512,
        key_dim: int = 512,
        value_dim: int = 512,
        out_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert out_dim % num_heads == 0, \
            f"out_dim ({out_dim}) deve essere divisibile per num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.out_dim   = out_dim

        self.proj_q = nn.Linear(query_dim, out_dim, bias=False)
        self.proj_k = nn.Linear(key_dim,   out_dim, bias=False)
        self.proj_v = nn.Linear(value_dim, out_dim, bias=False)
        self.proj_o = nn.Linear(out_dim,   out_dim, bias=True)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:  (B, T, query_dim)   T = H5*W5 token di P5
            keys:   (N, key_dim)   oppure  (B, N, key_dim)
            values: (N, value_dim) oppure  (B, N, value_dim)
                    N = numero coppie nel training set

        Returns:
            context: (B, T, out_dim)  — conditioning contestuale
        """
        B, T, _ = query.shape
        H = self.num_heads
        Dh = self.head_dim

        # Gestione keys/values unbatched (N, D) → (1, N, D) → (B, N, D)
        if keys.dim() == 2:
            keys   = keys.unsqueeze(0).expand(B, -1, -1)
        if values.dim() == 2:
            values = values.unsqueeze(0).expand(B, -1, -1)

        N = keys.shape[1]

        # Proiezioni
        q = self.proj_q(query)   # (B, T, out_dim)
        k = self.proj_k(keys)    # (B, N, out_dim)
        v = self.proj_v(values)  # (B, N, out_dim)

        # Multi-head reshape
        q = q.reshape(B, T, H, Dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)
        k = k.reshape(B, N, H, Dh).permute(0, 2, 1, 3)  # (B, H, N, Dh)
        v = v.reshape(B, N, H, Dh).permute(0, 2, 1, 3)  # (B, H, N, Dh)

        # Attenzione
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, H, T, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Aggregazione
        out = attn @ v                                    # (B, H, T, Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, out_dim)

        # Proiezione finale + LayerNorm
        context = self.norm(self.proj_o(out))             # (B, T, out_dim)
        return context

    def forward_spatial(
        self,
        p5: torch.Tensor,
        train_keys: torch.Tensor,
        train_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wrapper che gestisce P5 come feature map spaziale.

        Args:
            p5:           (B, 512, H5, W5)
            train_keys:   (N, 512)
            train_values: (N, 512)

        Returns:
            context_map: (B, out_dim, H5, W5)
        """
        B, C, H5, W5 = p5.shape
        T = H5 * W5

        # Flatten spaziale
        query = p5.flatten(2).permute(0, 2, 1)  # (B, T, C)

        context = self.forward(query, train_keys, train_values)  # (B, T, out_dim)

        # Riporta a mappa spaziale
        return context.permute(0, 2, 1).reshape(B, self.out_dim, H5, W5)


class ContextualStyleConditioner(nn.Module):
    """
    Componente completo per il conditioning contestuale:
      1. Mantiene cache delle chiavi/valori del training set
      2. Chiama CrossAttention al test time
      3. Interpola il context alle risoluzioni di P3 e P4

    Args:
        query_dim:  Canali di P5 (default 512).
        embed_dim:  Dimensione embedding encoder (default 512).
        out_dim:    Dimensione context output (default 128).
        num_heads:  Teste attenzione (default 4).
    """

    def __init__(
        self,
        query_dim: int = 512,
        embed_dim: int = 512,
        out_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.cross_attn = CrossAttention(
            query_dim=query_dim,
            key_dim=embed_dim,
            value_dim=embed_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.out_dim = out_dim

        # Cache (non parametri del modello)
        self._train_keys:   Optional[torch.Tensor] = None  # (N, embed_dim)
        self._train_values: Optional[torch.Tensor] = None  # (N, embed_dim)

    def set_train_cache(
        self,
        train_keys: torch.Tensor,
        train_values: torch.Tensor,
    ) -> None:
        """
        Imposta la cache delle feature del training set.
        Chiamare una volta dopo aver calcolato le feature del dataset.

        Args:
            train_keys:   (N, embed_dim)  — Enc(src_i)
            train_values: (N, embed_dim)  — δ_i = Enc(tgt_i) - Enc(src_i)
        """
        self._train_keys   = train_keys
        self._train_values = train_values

    def forward(
        self,
        p5: torch.Tensor,
        p3_size: Tuple[int, int],
        p4_size: Tuple[int, int],
        train_keys: Optional[torch.Tensor] = None,
        train_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola il context e lo interpola alle risoluzioni di P3 e P4.

        Args:
            p5:           (B, 512, H5, W5)
            p3_size:      (H3, W3)
            p4_size:      (H4, W4)
            train_keys:   (N, embed_dim) — opzionale, override della cache
            train_values: (N, embed_dim) — opzionale, override della cache

        Returns:
            C3: (B, out_dim, H3, W3)
            C4: (B, out_dim, H4, W4)
        """
        # Usa cache se non fornite esplicitamente
        keys   = train_keys   if train_keys   is not None else self._train_keys
        values = train_values if train_values is not None else self._train_values

        assert keys   is not None, "train_keys non disponibili. Chiama set_train_cache()."
        assert values is not None, "train_values non disponibili."

        # Cross-attention a risoluzione P5
        context_p5 = self.cross_attn.forward_spatial(
            p5, keys, values
        )   # (B, out_dim, H5, W5)

        # Interpolazione bilineare alle risoluzioni di P4 e P3
        C4 = F.interpolate(context_p5, size=p4_size,
                           mode="bilinear", align_corners=False)
        C3 = F.interpolate(context_p5, size=p3_size,
                           mode="bilinear", align_corners=False)

        return C3, C4