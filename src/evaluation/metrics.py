"""
evaluation/metrics.py

Metriche quantitative per la valutazione  (§9.1):
  - CIEDE2000 (ΔE₀₀)     §9.1.1
  - SSIM su canale L*      §9.1.2
  - LPIPS                  §9.1.3
  - Delta NIMA             §9.1.4
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses.delta_e import DeltaELoss, ciede2000
from utils.color_space import rgb_to_lab

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Risultato aggregato della valutazione."""
    delta_e:       float = 0.0   # CIEDE2000 medio
    ssim:          float = 0.0   # SSIM su L* medio
    lpips:         float = 0.0   # LPIPS medio (0 se non disponibile)
    delta_nima:    float = 0.0   # Δ NIMA score medio (0 se non disponibile)
    n_samples:     int   = 0

    extras: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ΔE₀₀={self.delta_e:.4f} | "
            f"SSIM={self.ssim:.4f} | "
            f"LPIPS={self.lpips:.4f} | "
            f"ΔNIMA={self.delta_nima:.4f} | "
            f"N={self.n_samples}"
        )


# ── SSIM ─────────────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Genera un kernel gaussiano 2D per SSIM."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g1d    = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d    = g1d / g1d.sum()
    g2d    = torch.outer(g1d, g1d)
    return g2d.unsqueeze(0).unsqueeze(0)   # (1, 1, size, size)


def compute_ssim_lstar(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Calcola SSIM sul canale L* (luminanza) in CIE Lab  (§9.1.2).

    Confronta struttura della luminanza, non dei colori
    (i colori vengono modificati intenzionalmente dal grading).

    Args:
        pred:   (B, 3, H, W) in [0,1].
        target: (B, 3, H, W) in [0,1].

    Returns:
        (B,) — SSIM per elemento del batch, ∈ [-1, 1].
    """
    import torch.nn.functional as F

    # Estrai L* (canale 0 in Lab)
    pred_lab   = rgb_to_lab(pred.permute(0, 2, 3, 1))
    target_lab = rgb_to_lab(target.permute(0, 2, 3, 1))

    # L* ∈ [0, 100] → normalizza a [0, 1]
    L_pred = (pred_lab[..., 0] / 100.0).clamp(0, 1).unsqueeze(1)    # (B,1,H,W)
    L_tgt  = (target_lab[..., 0] / 100.0).clamp(0, 1).unsqueeze(1)  # (B,1,H,W)

    # Kernel gaussiano
    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
    pad    = window_size // 2

    # Medie locali
    mu_x = F.conv2d(L_pred, kernel, padding=pad, groups=1)
    mu_y = F.conv2d(L_tgt,  kernel, padding=pad, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Varianze locali
    sigma_x2 = F.conv2d(L_pred * L_pred, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(L_tgt  * L_tgt,  kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(L_pred * L_tgt,  kernel, padding=pad) - mu_xy

    L  = 1.0     # data range
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    ssim_map = (
        (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    return ssim_map.mean(dim=[1, 2, 3])   # (B,)


# ── LPIPS ─────────────────────────────────────────────────────────────────────

class LPIPSMetric:
    """
    Wrapper per LPIPS (Learned Perceptual Image Patch Similarity)  (§9.1.3).

    Usa il pacchetto `lpips` se disponibile, altrimenti fallback su
    una perceptual distance semplificata (VGG features L2).
    """

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        self._available = False
        self._fn        = None
        self.device     = torch.device(device)

        try:
            import lpips
            self._fn        = lpips.LPIPS(net=net).to(self.device)
            self._available = True
            logger.info("LPIPS: usando pacchetto lpips.")
        except ImportError:
            logger.warning(
                "lpips non installato. LPIPS calcolato con VGG features L2."
            )
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Fallback: distanza L2 su feature VGG relu2_2."""
        try:
            from losses.perceptual import VGG19Features
            self._vgg = VGG19Features(
                layers=["relu2_2"], normalize=True
            ).to(self.device)
            self._available = True
        except Exception:
            self._available = False

    @torch.no_grad()
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [0,1].
            target: (B, 3, H, W) in [0,1].

        Returns:
            (B,) — LPIPS per elemento.
        """
        if not self._available:
            return torch.zeros(pred.shape[0], device=pred.device)

        pred   = pred.to(self.device)
        target = target.to(self.device)

        if self._fn is not None:
            # Rescala a [-1, 1] come richiesto da lpips
            return self._fn(pred * 2 - 1, target * 2 - 1).squeeze()
        else:
            # Fallback VGG
            fp = self._vgg(pred)["relu2_2"]
            ft = self._vgg(target)["relu2_2"]
            return (fp - ft).pow(2).mean(dim=[1, 2, 3])


# ── NIMA ─────────────────────────────────────────────────────────────────────

class NIMAMetric:
    """
    Wrapper per NIMA (Neural Image Assessment)  (§9.1.4).

    Calcola il delta NIMA: μ_NIMA(pred) - μ_NIMA(src).
    Valori positivi indicano miglioramento estetico rispetto all'originale.

    Usa il pacchetto `nima-pytorch` se disponibile.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._available = False
        self.device     = torch.device(device)

        try:
            # Tenta import di nima_pytorch o simili
            import nima_pytorch
            self._model = nima_pytorch.NIMA().to(self.device)
            self._available = True
            logger.info("NIMA: modello caricato.")
        except ImportError:
            logger.warning(
                "nima_pytorch non disponibile. "
                "Metrica NIMA non calcolata (restituisce 0)."
            )

    @torch.no_grad()
    def score(self, img: torch.Tensor) -> torch.Tensor:
        """
        Calcola il mean opinion score NIMA.

        Args:
            img: (B, 3, H, W) in [0,1].

        Returns:
            (B,) — score ∈ [1, 10].
        """
        if not self._available:
            return torch.zeros(img.shape[0])

        img = img.to(self.device)
        probs = self._model(img)    # (B, 10)
        bins  = torch.arange(1, 11, dtype=probs.dtype, device=probs.device)
        return (probs * bins).sum(dim=1)

    def delta_score(
        self,
        pred: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola Δ_NIMA = μ_NIMA(pred) - μ_NIMA(src).

        Returns:
            (B,)
        """
        return self.score(pred) - self.score(src)


# ── Funzioni di valutazione ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    src: Optional[torch.Tensor] = None,
    lpips_metric: Optional[LPIPSMetric] = None,
    nima_metric:  Optional[NIMAMetric]  = None,
) -> Dict[str, torch.Tensor]:
    """
    Calcola tutte le metriche su un singolo batch.

    Args:
        pred:   (B, 3, H, W) in [0,1].
        target: (B, 3, H, W) in [0,1].
        src:    (B, 3, H, W) in [0,1] — necessario per Δ_NIMA.

    Returns:
        Dizionario con tensori per elemento del batch.
    """
    device = pred.device

    # ΔE₀₀
    pred_lab   = rgb_to_lab(pred.permute(0, 2, 3, 1))
    target_lab = rgb_to_lab(target.permute(0, 2, 3, 1))
    de = ciede2000(pred_lab, target_lab).mean(dim=[1, 2])  # (B,)

    # SSIM su L*
    ssim = compute_ssim_lstar(pred, target)   # (B,)

    results = {
        "delta_e": de,
        "ssim":    ssim,
    }

    # LPIPS
    if lpips_metric is not None:
        lp = lpips_metric(pred, target)
        results["lpips"] = lp.to(device)

    # Δ NIMA
    if nima_metric is not None and src is not None:
        dn = nima_metric.delta_score(pred, src)
        results["delta_nima"] = dn.to(device)

    return results


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    compute_lpips: bool = True,
    compute_nima: bool = False,
) -> MetricsResult:
    """
    Valuta il modello su un intero dataset.

    Args:
        model:         Modello HybridStyleNet in eval mode.
        loader:        DataLoader con sample {src, tgt}.
        device:        Device.
        compute_lpips: Se True, calcola LPIPS.
        compute_nima:  Se True, calcola Δ NIMA.

    Returns:
        MetricsResult aggregato.
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    lpips_metric = LPIPSMetric(device=device) if compute_lpips else None
    nima_metric  = NIMAMetric(device=device)  if compute_nima  else None

    total: Dict[str, float] = {
        "delta_e": 0.0, "ssim": 0.0,
        "lpips": 0.0, "delta_nima": 0.0,
    }
    n_samples = 0

    for batch in loader:
        src = batch["src"].to(dev)
        tgt = batch["tgt"].to(dev)
        B   = src.shape[0]

        out  = model(src)
        pred = out["pred"]

        batch_metrics = evaluate_batch(
            pred, tgt, src=src,
            lpips_metric=lpips_metric,
            nima_metric=nima_metric,
        )

        for k, v in batch_metrics.items():
            total[k] = total.get(k, 0.0) + v.sum().item()

        n_samples += B

    n = max(n_samples, 1)
    result = MetricsResult(
        delta_e    = total["delta_e"]    / n,
        ssim       = total["ssim"]       / n,
        lpips      = total.get("lpips", 0.0)      / n,
        delta_nima = total.get("delta_nima", 0.0) / n,
        n_samples  = n_samples,
    )

    logger.info(f"Evaluation: {result}")
    return result