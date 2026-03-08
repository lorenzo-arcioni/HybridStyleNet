from .delta_e import DeltaELoss
from .histogram import ColorHistogramLoss
from .perceptual import PerceptualLoss, StyleLoss
from .chroma import CosineLoss, ChromaConsistencyLoss
from .identity import IdentityLoss
from .composite import ColorAestheticLoss

__all__ = [
    "DeltaELoss",
    "ColorHistogramLoss",
    "PerceptualLoss",
    "StyleLoss",
    "CosineLoss",
    "ChromaConsistencyLoss",
    "IdentityLoss",
    "ColorAestheticLoss",
]