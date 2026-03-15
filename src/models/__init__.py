"""
models/
-------
Architettura RAG-ColorNet — i 5 componenti + il modello completo.
"""

from .scene_encoder    import SceneEncoder, ColorHistogram, ChromaticPatchFeatures
from .cluster_net      import ClusterNet
from .retrieval_module import RetrievalModule
from .bilateral_grid   import (
    BilateralGridRenderer,
    GridNet,
    SemanticGuide,
    bilateral_slice,
)
from .confidence_mask  import ConfidenceMaskBlender, MaskNet
from .rag_colornet     import RAGColorNet

__all__ = [
    # Componente 1
    "SceneEncoder",
    "ColorHistogram",
    "ChromaticPatchFeatures",
    # Componente 2
    "ClusterNet",
    # Componente 3
    "RetrievalModule",
    # Componente 4
    "BilateralGridRenderer",
    "GridNet",
    "SemanticGuide",
    "bilateral_slice",
    # Componente 5
    "ConfidenceMaskBlender",
    "MaskNet",
    # Modello completo
    "RAGColorNet",
]
