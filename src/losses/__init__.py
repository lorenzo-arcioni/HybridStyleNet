"""
losses/
-------
Funzioni di loss per RAG-ColorNet, con curriculum dei pesi.
"""

from .color_losses import (
    DeltaELoss,
    L1LabLoss,
    HistogramEMDLoss,
    PerceptualLoss,
    ChromaConsistencyLoss,
)
from .structural_losses import (
    TotalVariationLoss,
    EntropyMaskLoss,
    LuminancePreservationLoss,
)
from .retrieval_loss import (
    RetrievalQualityLoss,
    ClusterAssignmentLoss,
)
from .composite_loss import (
    LossWeights,
    LossBreakdown,
    CurriculumScheduler,
    CompositeLoss,
)

__all__ = [
    # Color
    "DeltaELoss",
    "L1LabLoss",
    "HistogramEMDLoss",
    "PerceptualLoss",
    "ChromaConsistencyLoss",
    # Structural
    "TotalVariationLoss",
    "EntropyMaskLoss",
    "LuminancePreservationLoss",
    # Retrieval
    "RetrievalQualityLoss",
    "ClusterAssignmentLoss",
    # Composite
    "LossWeights",
    "LossBreakdown",
    "CurriculumScheduler",
    "CompositeLoss",
]
