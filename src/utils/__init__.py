"""
utils/
------
Funzioni pure di supporto: colore, I/O, clustering, logging, visualizzazione.
Nessuno stato, nessuna dipendenza circolare.
"""

from .color_utils    import (
    rgb_to_lab, lab_to_rgb,
    srgb_to_linear, linear_to_srgb,
    delta_e_2000_mean,
    soft_histogram,
    lab_channel_stats,
)
from .image_io       import (
    save_image, load_tensor,
    save_comparison_grid,
    tensor_to_pil, pil_to_tensor,
)
from .kmeans_init    import (
    elbow_kmeans, run_kmeans,
    assign_to_clusters, elbow_k,
)
from .logging_utils  import Logger
from .visualization  import (
    make_comparison_grid,
    make_mask_overlay,
    make_attention_heatmap,
    make_cluster_histogram,
    make_grid_coeffs_viz,
    plot_loss_curves,
)

__all__ = [
    # color
    "rgb_to_lab", "lab_to_rgb",
    "srgb_to_linear", "linear_to_srgb",
    "delta_e_2000_mean", "soft_histogram", "lab_channel_stats",
    # io
    "save_image", "load_tensor",
    "save_comparison_grid", "tensor_to_pil", "pil_to_tensor",
    # kmeans
    "elbow_kmeans", "run_kmeans", "assign_to_clusters", "elbow_k",
    # logging
    "Logger",
    # visualization
    "make_comparison_grid", "make_mask_overlay",
    "make_attention_heatmap", "make_cluster_histogram",
    "make_grid_coeffs_viz", "plot_loss_curves",
]
