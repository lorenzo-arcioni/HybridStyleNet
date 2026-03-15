"""
visualization.py
----------------
Funzioni di visualizzazione per debug e analisi nei notebook.

Tutte le funzioni restituiscono tensori (3,H,W) o figure matplotlib —
non salvano su disco (quello lo fa image_io.save_image).

Contenuto:
  make_comparison_grid(src, pred, tgt)  — griglia side-by-side
  make_mask_overlay(img, alpha)         — maschera di confidenza sovrapposta
  make_attention_heatmap(attn, img)     — heatmap attention su immagine
  make_cluster_histogram(h, p)          — visualizza assignment cluster
  make_grid_coeffs_viz(G)               — visualizza bilateral grid coefficients
  plot_loss_curves(history)             — curva di loss multi-termine
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# make_comparison_grid
# ---------------------------------------------------------------------------

def make_comparison_grid(
    src:    torch.Tensor,          # (3,H,W) o (B,3,H,W)
    pred:   torch.Tensor,
    tgt:    torch.Tensor,
    max_b:  int = 4,
    gap_px: int = 4,
) -> torch.Tensor:
    """
    Crea una griglia side-by-side  src | pred | tgt.

    Returns
    -------
    grid : (3, B*H + gap*(B-1), 3*W + gap*2)  tensor [0,1]
    """
    if src.dim() == 3:
        src, pred, tgt = src.unsqueeze(0), pred.unsqueeze(0), tgt.unsqueeze(0)

    B = min(src.shape[0], max_b)
    H, W = src.shape[-2], src.shape[-1]

    gap  = torch.ones(3, H, gap_px)
    rows = []
    for i in range(B):
        row = torch.cat([
            src[i].clamp(0,1), gap,
            pred[i].clamp(0,1), gap,
            tgt[i].clamp(0,1),
        ], dim=-1)                              # (3, H, 3W+2*gap)
        rows.append(row)

    if len(rows) == 1:
        return rows[0]

    h_gap = torch.ones(3, gap_px, rows[0].shape[-1])
    grid_rows = []
    for i, r in enumerate(rows):
        grid_rows.append(r)
        if i < len(rows) - 1:
            grid_rows.append(h_gap)

    return torch.cat(grid_rows, dim=-2)         # (3, tot_H, tot_W)


# ---------------------------------------------------------------------------
# make_mask_overlay
# ---------------------------------------------------------------------------

def make_mask_overlay(
    img:   torch.Tensor,   # (3,H,W) sRGB [0,1]
    alpha: torch.Tensor,   # (1,H,W) o (H,W) ∈ [0,1]
    colormap: str = "jet", # "jet" | "viridis" | "bwr"
) -> torch.Tensor:
    """
    Sovrappone la confidence mask sull'immagine come heatmap colorata.

    Returns
    -------
    overlay : (3,H,W) [0,1]
    """
    if alpha.dim() == 3:
        alpha = alpha.squeeze(0)               # (H,W)

    alpha_np = alpha.detach().cpu().float().numpy()
    heatmap  = _apply_colormap(alpha_np, colormap)   # (H,W,3) float [0,1]
    heatmap_t = torch.from_numpy(heatmap).permute(2,0,1).float()

    # Blending: 60% immagine + 40% heatmap
    return (0.6 * img.clamp(0,1) + 0.4 * heatmap_t).clamp(0,1)


# ---------------------------------------------------------------------------
# make_attention_heatmap
# ---------------------------------------------------------------------------

def make_attention_heatmap(
    attn:        torch.Tensor,   # (N_new, N_ref) attention weights
    ref_img:     torch.Tensor,   # (3,H,W) immagine di riferimento
    query_patch: int = 0,        # indice della patch query da visualizzare
    n_h:         int = 27,
    n_w:         int = 36,
) -> torch.Tensor:
    """
    Visualizza i pesi di attention di una patch query sulle patch
    dell'immagine di riferimento.

    Returns
    -------
    overlay : (3,H,W)  heatmap sovrapposta a ref_img
    """
    H, W = ref_img.shape[-2], ref_img.shape[-1]

    # Attenzione della patch query_patch su tutte le patch ref
    attn_row = attn[query_patch].detach().cpu().float()   # (N_ref,)

    # Reshape a mappa spaziale (n_h_ref, n_w_ref)
    n_ref = attn_row.shape[0]
    n_h_ref = int(n_ref ** 0.5)
    n_w_ref = n_ref // n_h_ref

    attn_map = attn_row[:n_h_ref * n_w_ref].reshape(1, 1, n_h_ref, n_w_ref)
    attn_up  = F.interpolate(attn_map, size=(H, W), mode="bilinear",
                             align_corners=False).squeeze()   # (H,W)

    # Normalizza
    attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    heatmap   = _apply_colormap(attn_up.numpy(), "jet")
    heatmap_t = torch.from_numpy(heatmap).permute(2,0,1).float()

    return (0.5 * ref_img.clamp(0,1) + 0.5 * heatmap_t).clamp(0,1)


# ---------------------------------------------------------------------------
# make_cluster_histogram
# ---------------------------------------------------------------------------

def make_cluster_histogram(
    p:        torch.Tensor,   # (K,) o (B,K) soft assignment
    k_labels: Optional[List[str]] = None,
) -> "matplotlib.figure.Figure":   # type: ignore[name-defined]
    """
    Barchart delle probabilità di cluster per un'immagine.

    Returns
    -------
    fig : matplotlib Figure (da mostrare con plt.show() o logger)
    """
    import matplotlib.pyplot as plt

    if p.dim() == 2:
        p = p[0]                               # prima immagine del batch

    K     = p.shape[0]
    probs = p.detach().cpu().float().numpy()
    labels = k_labels or [f"k={i}" for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 3))
    bars = ax.bar(range(K), probs, color="steelblue", alpha=0.85)
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("p(cluster)")
    ax.set_title("Soft cluster assignment")
    ax.set_ylim(0, 1)

    # Etichetta i bar con il valore
    for bar, val in zip(bars, probs):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# make_grid_coeffs_viz
# ---------------------------------------------------------------------------

def make_grid_coeffs_viz(
    G: torch.Tensor,    # (B, 12, S, S, L) bilateral grid
    batch_idx: int = 0,
    coeff_idx: int = 0, # quale dei 12 coefficienti visualizzare
) -> torch.Tensor:
    """
    Visualizza un piano (coeff_idx, :, :) della bilateral grid
    come immagine a falsi colori.

    Returns
    -------
    viz : (3, S*L, S)  tensor [0,1]  — colonne = luminance bins
    """
    G_single = G[batch_idx, coeff_idx]          # (S, S, L)
    S1, S2, L = G_single.shape
    G_np = G_single.detach().cpu().float().numpy()

    # Normalizza
    vmin, vmax = G_np.min(), G_np.max()
    G_norm = (G_np - vmin) / (vmax - vmin + 1e-8)

    # Affianca i bin di luminanza orizzontalmente: (S, S*L)
    slices = [G_norm[:, :, l] for l in range(L)]
    grid   = np.concatenate(slices, axis=1)     # (S, S*L)

    heatmap = _apply_colormap(grid, "viridis")  # (S, S*L, 3)
    return torch.from_numpy(heatmap).permute(2, 0, 1).float()


# ---------------------------------------------------------------------------
# plot_loss_curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history: Dict[str, List[float]],
    title:   str = "Training loss",
    log_scale: bool = False,
) -> "matplotlib.figure.Figure":   # type: ignore[name-defined]
    """
    Curva di loss multi-termine.

    Parameters
    ----------
    history : {"loss/total": [ep1, ep2, ...], "loss/delta_e": [...], ...}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    for key, values in history.items():
        label = key.split("/")[-1]
        alpha = 1.0 if "total" in key else 0.6
        lw    = 2.0 if "total" in key else 1.0
        ax.plot(values, label=label, alpha=alpha, linewidth=lw)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# _apply_colormap  (helper interno)
# ---------------------------------------------------------------------------

def _apply_colormap(
    arr:      np.ndarray,   # (H, W) float [0,1]
    colormap: str = "jet",
) -> np.ndarray:
    """Applica una colormap matplotlib a un array 2D. Returns (H,W,3) float [0,1]."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        return cmap(arr)[:, :, :3].astype(np.float32)
    except ImportError:
        # Fallback: grayscale
        gray = np.stack([arr, arr, arr], axis=-1)
        return gray.astype(np.float32)
