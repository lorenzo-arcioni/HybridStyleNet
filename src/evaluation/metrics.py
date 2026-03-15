"""
metrics.py
----------
Wrapper per le metriche di valutazione di RAG-ColorNet.

Tutte le funzioni accettano batch di tensori (B,3,H,W) float32 [0,1]
e restituiscono scalari float — pronti per il logging e il confronto
con lo stato dell'arte.

Metriche:
  compute_delta_e     — ΔE₀₀ medio (metrica principale)
  compute_ssim_L      — SSIM sulla luminanza L* (struttura tonale)
  compute_lpips       — LPIPS perceptual distance
  compute_nima_delta  — Δ punteggio estetico NIMA (pred vs src)
  compute_all         — calcola tutte in un colpo solo

Dipendenze opzionali:
  - pytorch_msssim (per SSIM differenziabile)
  - lpips           (per LPIPS)
  - NIMA            (modello estetico — scaricato da torch.hub)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from utils.color_utils import rgb_to_lab, delta_e_2000_mean   # type: ignore[import]


# ---------------------------------------------------------------------------
# ΔE₀₀
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_delta_e(
    pred: torch.Tensor,   # (B,3,H,W) [0,1]
    tgt:  torch.Tensor,
) -> float:
    """ΔE₀₀ medio sul batch. Metrica principale di RAG-ColorNet."""
    return delta_e_2000_mean(pred, tgt)


# ---------------------------------------------------------------------------
# SSIM sulla luminanza L*
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ssim_L(
    pred: torch.Tensor,
    tgt:  torch.Tensor,
) -> float:
    """
    SSIM sulla luminanza L* normalizzata in [0,1].

    Misura la fedeltà strutturale tonale indipendentemente dal colore.
    Target: > 0.96.
    """
    try:
        from pytorch_msssim import ssim as _ssim
        L_pred = (rgb_to_lab(pred)[:, 0:1] / 100.0).clamp(0, 1)
        L_tgt  = (rgb_to_lab(tgt)[:, 0:1]  / 100.0).clamp(0, 1)
        return _ssim(L_pred, L_tgt, data_range=1.0, size_average=True).item()
    except ImportError:
        # Fallback: SSIM manuale su luminanza
        return _ssim_manual(
            (rgb_to_lab(pred)[:, 0] / 100.0).clamp(0, 1),
            (rgb_to_lab(tgt)[:,  0] / 100.0).clamp(0, 1),
        )


def _ssim_manual(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> float:
    """SSIM minimale in PyTorch per il caso senza pytorch_msssim."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x = x.unsqueeze(1)   # (B,1,H,W)
    y = y.unsqueeze(1)

    # Gaussian kernel
    k = window
    sigma = 1.5
    coords = torch.arange(k, dtype=torch.float32) - k // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = (g1d.unsqueeze(0) * g1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(x.device)

    pad = k // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)

    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sig_x  = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sig_y  = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sig_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
    return (num / den.clamp(min=1e-8)).mean().item()


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_lpips(
    pred:    torch.Tensor,
    tgt:     torch.Tensor,
    net:     str = "alex",
    model:   Optional[object] = None,   # istanza LPIPS pre-caricata
) -> float:
    """
    LPIPS (Learned Perceptual Image Patch Similarity).

    Misura la distanza percettiva — più basso è meglio.
    Target: < 0.08.

    Parameters
    ----------
    pred, tgt : (B,3,H,W) float32 [0,1]
    net       : "alex" | "vgg" | "squeeze"
    model     : istanza lpips.LPIPS pre-caricata (per non ricaricarla ogni volta)
    """
    try:
        import lpips
        if model is None:
            model = lpips.LPIPS(net=net).to(pred.device)
            model.eval()

        # LPIPS si aspetta input in [-1, 1]
        pred_n = pred * 2.0 - 1.0
        tgt_n  = tgt  * 2.0 - 1.0
        dist   = model(pred_n, tgt_n)
        return dist.mean().item()

    except ImportError:
        # Fallback: MSE nello spazio RGB come proxy
        return F.mse_loss(pred, tgt).item()


# ---------------------------------------------------------------------------
# NIMA Δ score
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_nima_delta(
    pred: torch.Tensor,
    src:  torch.Tensor,
    model: Optional[object] = None,
) -> float:
    """
    Δ punteggio estetico NIMA: quanto migliora l'estetica rispetto alla sorgente.

    NIMA (Neural Image Assessment) predice una distribuzione di punteggi
    estetici [1–10] — si calcola il punteggio medio atteso.

    Δ_NIMA = E[score(pred)] - E[score(src)]
    Target: > 0.5 punti.

    Richiede il modello NIMA (MobileNet fine-tuned su AVA dataset).
    Se non disponibile, restituisce 0.0.
    """
    try:
        nima = _get_nima_model(model, pred.device)
        if nima is None:
            return 0.0

        score_pred = _nima_score(nima, pred)
        score_src  = _nima_score(nima, src)
        return score_pred - score_src

    except Exception:
        return 0.0


def _get_nima_model(model, device):
    """Carica il modello NIMA se non già caricato."""
    if model is not None:
        return model
    try:
        nima = torch.hub.load(
            "YijunMaverick/NIMA", "nima",
            pretrained=True, verbose=False,
        ).to(device).eval()
        return nima
    except Exception:
        return None


def _nima_score(model, img: torch.Tensor) -> float:
    """Calcola il punteggio NIMA medio su un batch."""
    # NIMA si aspetta (B,3,224,224) normalizzato con ImageNet stats
    img_resized = F.interpolate(img, size=(224, 224), mode="bilinear",
                                align_corners=False, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1,3,1,1)
    img_norm = (img_resized - mean) / std

    out = model(img_norm)                       # (B, 10) distribuzione di score
    scores = torch.arange(1, 11, dtype=torch.float32, device=img.device)
    return (out * scores).sum(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_all(
    pred:         torch.Tensor,
    tgt:          torch.Tensor,
    src:          Optional[torch.Tensor] = None,
    lpips_model:  Optional[object] = None,
    nima_model:   Optional[object] = None,
    compute_nima: bool = False,
) -> Dict[str, float]:
    """
    Calcola tutte le metriche in un colpo solo.

    Parameters
    ----------
    pred, tgt     : (B,3,H,W) sRGB [0,1]
    src           : immagine sorgente (per NIMA Δ)
    lpips_model   : istanza lpips.LPIPS pre-caricata
    nima_model    : istanza NIMA pre-caricata
    compute_nima  : se False salta NIMA (lento)

    Returns
    -------
    dict con: delta_e, ssim_L, lpips, nima_delta (opzionale)
    """
    results: Dict[str, float] = {}

    results["delta_e"] = compute_delta_e(pred, tgt)
    results["ssim_L"]  = compute_ssim_L(pred, tgt)
    results["lpips"]   = compute_lpips(pred, tgt, model=lpips_model)

    if compute_nima and src is not None:
        results["nima_delta"] = compute_nima_delta(pred, src, model=nima_model)

    return results
