"""
models/swin.py

Swin Transformer stage 4-5 con Rotary Position Embedding (RoPE).  §5.1.2

Input:  P3 (B, 128, H/8, W/8)  dal CNN stem
Output: P4 (B, 256, H/16, W/16)
        P5 (B, 512, H/32, W/32)

Complessità: O(T · M² · d)  con T = H/32 · W/32 token (lineare in HW)
RoPE garantisce generalizzazione a risoluzioni non viste al training.

Dipendenze: timm >= 0.9  (SwinTransformer nativo)
            oppure implementazione custom se timm non disponibile
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Rotary Position Embedding (RoPE 2-D) ─────────────────────────────────────

class RotaryEmbedding2D(nn.Module):
    """
    Rotary Position Embedding 2-D per Vision Transformers.

    Per una posizione (row, col), genera le frequenze di rotazione
    nel piano complesso. Il prodotto scalare q^T k dipende solo
    dalla differenza di posizione (Δrow, Δcol) → invarianza a risoluzione.

    Args:
        dim:        Dimensione di ogni testa di attenzione (d/h).
                    Deve essere divisibile per 4 (2 componenti spaziali × 2).
        max_seq_len: Lunghezza massima della sequenza per pre-calcolo.
        base:       Base delle frequenze (default 10000).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ) -> None:
        super().__init__()
        assert dim % 4 == 0, \
            f"RoPE 2D richiede dim divisibile per 4, got {dim}"

        self.dim     = dim
        self.base    = base
        half_dim     = dim // 2       # metà per righe, metà per colonne

        # Frequenze θ_j = 1 / base^(2j/dim)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, half_dim, 2).float() / half_dim)
        )
        self.register_buffer("inv_freq", inv_freq)  # (half_dim/2,)

    def _build_freqs(
        self, seq_len_row: int, seq_len_col: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-calcola le frequenze per una griglia (seq_len_row, seq_len_col).

        Returns:
            freqs_row: (seq_len_row, half_dim/2)
            freqs_col: (seq_len_col, half_dim/2)
        """
        t_row = torch.arange(seq_len_row, device=device, dtype=torch.float32)
        t_col = torch.arange(seq_len_col, device=device, dtype=torch.float32)
        freqs_row = torch.outer(t_row, self.inv_freq)  # (R, D/4)
        freqs_col = torch.outer(t_col, self.inv_freq)  # (C, D/4)
        return freqs_row, freqs_col

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Ruota i vettori nel piano complesso: [a, b] → [-b, a]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applica RoPE 2-D a query e key.

        Args:
            q: (B, num_heads, T, head_dim)  T = h*w
            k: (B, num_heads, T, head_dim)
            h: numero di token in altezza
            w: numero di token in larghezza

        Returns:
            q_rot, k_rot con stesse shape.
        """
        device    = q.device
        head_dim  = q.shape[-1]
        half_dim  = head_dim // 2
        quarter   = half_dim // 2

        freqs_row, freqs_col = self._build_freqs(h, w, device)
        # freqs_row: (h, quarter)  freqs_col: (w, quarter)

        # Espandi a griglia (h, w, quarter)
        fr = freqs_row[:, None, :].expand(h, w, quarter)   # (h, w, quarter)
        fc = freqs_col[None, :, :].expand(h, w, quarter)   # (h, w, quarter)

        # Concatena e flatten: (T, half_dim)
        freqs = torch.cat([fr, fc], dim=-1).reshape(h * w, half_dim)

        # Duplica per [cos, sin]: (T, head_dim)
        cos = freqs.cos()   # (T, half_dim)
        sin = freqs.sin()   # (T, half_dim)

        # Padding se head_dim > half_dim * 2 (non dovrebbe accadere)
        if head_dim > half_dim * 2:
            pad = torch.zeros(h * w, head_dim - half_dim * 2, device=device)
            cos = torch.cat([cos, pad], dim=-1)
            sin = torch.cat([sin, pad], dim=-1)

        # Broadcast: (1, 1, T, head_dim)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot


# ── Window Attention ──────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA / SW-MSA).

    Implementa l'attenzione con finestre di dimensione M×M
    e bias posizionale relativo appreso (§5.1.2).

    Args:
        dim:         Dimensione del token.
        num_heads:   Numero di teste di attenzione.
        window_size: Dimensione della finestra M (quadrata).
        use_rope:    Se True, sostituisce il bias relativo con RoPE.
        qkv_bias:    Se True, aggiunge bias alle proiezioni QKV.
        attn_drop:   Dropout sull'attenzione.
        proj_drop:   Dropout sulla proiezione finale.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        use_rope: bool = True,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.use_rope    = use_rope
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj  = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rope:
            head_dim = dim // num_heads
            self.rope = RotaryEmbedding2D(dim=head_dim)
        else:
            # Bias posizionale relativo classico (Swin originale)
            self.rope = None
            self._build_relative_bias(window_size)

    def _build_relative_bias(self, ws: int) -> None:
        """Crea la tabella di bias posizionale relativo appresa."""
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w,
                                               indexing="ij"))   # (2, ws, ws)
        coords_flat = coords.flatten(1)                          # (2, ws²)
        rel_coords  = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords   = rel_coords.permute(1, 2, 0).contiguous() # (ws², ws², 2)
        rel_coords[:, :, 0] += ws - 1
        rel_coords[:, :, 1] += ws - 1
        rel_coords[:, :, 0] *= 2 * ws - 1
        self.register_buffer(
            "relative_position_index",
            rel_coords.sum(-1),  # (ws², ws²)
        )

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B*num_windows, ws², dim)
            h, w: altezza e larghezza in token della finestra
            mask: maschera di attenzione per SW-MSA (opzionale)

        Returns:
            (B*num_windows, ws², dim)
        """
        Bw, N, C = x.shape
        H = self.num_heads
        head_dim = C // H

        qkv = self.qkv(x).reshape(Bw, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # ciascuno (Bw, H, N, head_dim)

        q = q * self.scale

        if self.use_rope and self.rope is not None:
            ws = self.window_size
            q, k = self.rope.apply_rope(q, k, ws, ws)

        attn = q @ k.transpose(-2, -1)   # (Bw, H, N, N)

        if not self.use_rope:
            # Aggiungi bias posizionale relativo
            bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
            attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, H, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, H, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ── Swin Block ────────────────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    """
    Singolo blocco Swin Transformer con W-MSA o SW-MSA.

    Args:
        dim:         Dimensione token.
        num_heads:   Numero teste.
        window_size: M.
        shift:       Se True, usa Shifted Window (SW-MSA).
        use_rope:    Se True, usa RoPE invece del bias relativo.
        mlp_ratio:   Ratio espansione MLP.
        drop:        Dropout generale.
        attn_drop:   Dropout attenzione.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift: bool = False,
        use_rope: bool = True,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.shift_size  = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            use_rope=use_rope,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    @staticmethod
    def _window_partition(
        x: torch.Tensor, window_size: int
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Partiziona (B, H, W, C) in finestre (B*nW, ws, ws, C).

        Returns:
            windows: (B*nW, ws², C)
            h_pad:   altezza dopo padding
            w_pad:   larghezza dopo padding
        """
        B, H, W, C = x.shape
        ws = window_size

        # Padding a multiplo di ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, ws * ws, C)
        return windows, Hp, Wp

    @staticmethod
    def _window_reverse(
        windows: torch.Tensor,
        window_size: int,
        Hp: int,
        Wp: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Ricompone le finestre in (B, H, W, C), rimuove il padding.
        """
        ws = window_size
        B  = int(windows.shape[0] / (Hp * Wp / ws / ws))
        x  = windows.view(B, Hp // ws, Wp // ws, ws, ws, -1)
        x  = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        return x[:, :H, :W, :].contiguous()

    def _compute_attn_mask(
        self, H: int, W: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Calcola la maschera per SW-MSA."""
        if self.shift_size == 0:
            return None

        ws = self.window_size
        ss = self.shift_size

        # Padding a multiplo di ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        Hp = H + pad_h
        Wp = W + pad_w

        img_mask = torch.zeros(1, Hp, Wp, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))

        cnt = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, hs, ws_, :] = cnt
                cnt += 1

        mask_windows, _, _ = self._window_partition(img_mask, ws)
        mask_windows = mask_windows.squeeze(-1)  # (nW, ws²)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask  # (nW, ws², ws²)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Converti in (B, H, W, C) per Swin
        x_hw = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Cyclic shift per SW-MSA
        if self.shift_size > 0:
            x_shifted = torch.roll(x_hw,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            x_shifted = x_hw

        # Window partition
        x_windows, Hp, Wp = self._window_partition(x_shifted, self.window_size)

        # Attenzione con maschera
        attn_mask = self._compute_attn_mask(H, W, x.device)
        attn_out  = self.attn(
            self.norm1(x_windows), self.window_size, self.window_size,
            mask=attn_mask,
        )

        # Residual 1
        x_windows = x_windows + attn_out

        # Window reverse
        x_hw = self._window_reverse(x_windows, self.window_size, Hp, Wp, H, W)

        # Cyclic shift inverso
        if self.shift_size > 0:
            x_hw = torch.roll(x_hw,
                              shifts=(self.shift_size, self.shift_size),
                              dims=(1, 2))

        # MLP + Residual 2
        x_hw = x_hw + self.mlp(self.norm2(x_hw))

        return x_hw.permute(0, 3, 1, 2)  # (B, C, H, W)


# ── Patch Merging (downsampling 2×) ──────────────────────────────────────────

class PatchMerging(nn.Module):
    """
    Downsampling 2× con patch merging (concatena 2×2 vicini → proiezione).

    Input:  (B, C_in, H, W)
    Output: (B, C_out, H/2, W/2)

    Args:
        in_channels:  Canali input.
        out_channels: Canali output (tipicamente 2×in_channels).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_channels)
        self.proj = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Padding se H o W sono dispari
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))

        # Reshape per patch merging
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]   # (B, H/2, W/2, C) — top-left
        x1 = x[:, 1::2, 0::2, :]   # bottom-left
        x2 = x[:, 0::2, 1::2, :]   # top-right
        x3 = x[:, 1::2, 1::2, :]   # bottom-right

        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x_cat = self.norm(x_cat)
        x_cat = self.proj(x_cat)                       # (B, H/2, W/2, C_out)

        return x_cat.permute(0, 3, 1, 2)               # (B, C_out, H/2, W/2)


# ── Swin Stage ────────────────────────────────────────────────────────────────

class SwinStage(nn.Module):
    """
    Stage Swin Transformer: PatchMerging + N×SwinBlock.

    Args:
        in_channels:  Canali input.
        out_channels: Canali output (dopo patch merging).
        depth:        Numero di SwinBlock nel stage.
        num_heads:    Teste di attenzione.
        window_size:  M.
        use_rope:     Se True, usa RoPE.
        mlp_ratio:    Ratio MLP.
        drop:         Dropout.
        attn_drop:    Dropout attenzione.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        use_rope: bool = True,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.downsample = PatchMerging(in_channels, out_channels)

        # Alterna W-MSA e SW-MSA
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=out_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift=(i % 2 == 1),
                use_rope=use_rope,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)

        Returns:
            (B, C_out, H/2, W/2)
        """
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ── SwinStages (stage 4 + stage 5) ───────────────────────────────────────────

class SwinStages(nn.Module):
    """
    Swin Transformer stage 4 e 5 che ricevono P3 dal CNN stem.

    Input:  P3 (B, 128, H/8, W/8)
    Output:
        P4: (B, 256, H/16, W/16)
        P5: (B, 512, H/32, W/32)

    Args:
        in_channels:     Canali di P3 (default 128).
        stage4_channels: Canali di P4 (default 256).
        stage5_channels: Canali di P5 (default 512).
        stage4_heads:    Teste per stage 4 (default 8).
        stage5_heads:    Teste per stage 5 (default 16).
        stage4_depth:    Blocchi Swin nello stage 4 (default 2).
        stage5_depth:    Blocchi Swin nello stage 5 (default 2).
        window_size:     M (default 7).
        use_rope:        Se True usa RoPE (default True).
    """

    def __init__(
        self,
        in_channels: int = 128,
        stage4_channels: int = 256,
        stage5_channels: int = 512,
        stage4_heads: int = 8,
        stage5_heads: int = 16,
        stage4_depth: int = 2,
        stage5_depth: int = 2,
        window_size: int = 7,
        use_rope: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.stage4 = SwinStage(
            in_channels=in_channels,
            out_channels=stage4_channels,
            depth=stage4_depth,
            num_heads=stage4_heads,
            window_size=window_size,
            use_rope=use_rope,
            drop=drop,
            attn_drop=attn_drop,
        )

        self.stage5 = SwinStage(
            in_channels=stage4_channels,
            out_channels=stage5_channels,
            depth=stage5_depth,
            num_heads=stage5_heads,
            window_size=window_size,
            use_rope=use_rope,
            drop=drop,
            attn_drop=attn_drop,
        )

        self.out_channels = {
            "P4": stage4_channels,
            "P5": stage5_channels,
        }

        logger.info(
            f"SwinStages: P3({in_channels}) → "
            f"P4({stage4_channels}) → P5({stage5_channels}), "
            f"window={window_size}, RoPE={use_rope}"
        )

    def forward(
        self, p3: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            p3: (B, 128, H/8, W/8)

        Returns:
            {"P4": (B,256,H/16,W/16), "P5": (B,512,H/32,W/32)}
        """
        p4 = self.stage4(p3)   # (B, 256, H/16, W/16)
        p5 = self.stage5(p4)   # (B, 512, H/32, W/32)
        return {"P4": p4, "P5": p5}