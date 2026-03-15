"""
evaluation/
-----------
Metriche di valutazione, ablation studies e benchmark comparativi.
"""

from .metrics        import (
    compute_delta_e,
    compute_ssim_L,
    compute_lpips,
    compute_nima_delta,
    compute_all,
)
from .ablation_runner import AblationRunner, ABLATION_CONFIGS
from .benchmark       import BenchmarkRunner, BaselineModel, RAGColorNetWrapper

__all__ = [
    # Metriche
    "compute_delta_e",
    "compute_ssim_L",
    "compute_lpips",
    "compute_nima_delta",
    "compute_all",
    # Ablation
    "AblationRunner",
    "ABLATION_CONFIGS",
    # Benchmark
    "BenchmarkRunner",
    "BaselineModel",
    "RAGColorNetWrapper",
]
