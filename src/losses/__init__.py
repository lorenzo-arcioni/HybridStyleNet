"""
losses/__init__.py

Package delle loss functions per HybridStyleNet.

Esporta i moduli principali per comodità di import:

    from losses import ColorAestheticLoss, LossWeights, get_weights_for_epoch
    from losses import DeltaELoss, ciede2000
    from losses import ColorHistogramLoss
    from losses import PerceptualLoss
    from losses import ChromaConsistencyLoss
    from losses import IdentityLoss
    from losses import TotalVariationLoss
    from losses import LuminancePreservationLoss
    from losses import EntropyLoss
    from losses import L1LabLoss
"""

from losses.composite  import ColorAestheticLoss, LossWeights, LossOutput, get_weights_for_epoch
from losses.delta_e    import DeltaELoss, ciede2000
from losses.l1_lab     import L1LabLoss
from losses.histogram  import ColorHistogramLoss, soft_histogram, histogram_emd
from losses.perceptual import PerceptualLoss, VGG16FeatureExtractor
from losses.chroma     import ChromaConsistencyLoss
from losses.identity   import IdentityLoss
from losses.tv         import TotalVariationLoss
from losses.luminance  import LuminancePreservationLoss
from losses.entropy    import EntropyLoss

__all__ = [
    # Composite
    "ColorAestheticLoss",
    "LossWeights",
    "LossOutput",
    "get_weights_for_epoch",
    # Singoli termini
    "DeltaELoss",
    "ciede2000",
    "L1LabLoss",
    "ColorHistogramLoss",
    "soft_histogram",
    "histogram_emd",
    "PerceptualLoss",
    "VGG16FeatureExtractor",
    "ChromaConsistencyLoss",
    "IdentityLoss",
    "TotalVariationLoss",
    "LuminancePreservationLoss",
    "EntropyLoss",
]
