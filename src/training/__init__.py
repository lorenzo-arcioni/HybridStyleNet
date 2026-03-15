"""
training/
---------
Funzioni di training modulari per RAG-ColorNet.
Progettate per essere chiamate da notebook Jupyter.

Fase 1 — Pre-training:    pretrain.run_epoch, pretrain.build_optimizer
Fase 2 — Meta-training:   meta_train.meta_train_step, reptile.reptile_step
Fase 3 — Adaptation:      adapt.setup_adaptation, adapt.adaptation_step1_epoch
                           adapt.adaptation_step2_epoch, adapt.validate
"""

from .reptile       import reptile_step, inner_loop, outer_update
from .lr_scheduler  import WarmupCosineScheduler, build_scheduler
from .early_stopping import EarlyStopping
from .pretrain      import (
    train_step,
    val_step,
    run_epoch,
    build_optimizer,
    build_empty_cluster_db,
    build_cluster_db_for_batch,
)
from .meta_train    import meta_train_step, evaluate_on_task, build_meta_optimizer
from .adapt         import (
    setup_adaptation,
    switch_to_step2,
    adaptation_step1_epoch,
    adaptation_step2_epoch,
    validate,
)

__all__ = [
    # Reptile
    "reptile_step", "inner_loop", "outer_update",
    # Schedulers
    "WarmupCosineScheduler", "build_scheduler",
    # Early stopping
    "EarlyStopping",
    # Step functions (usate in tutti i notebook)
    "train_step", "val_step", "run_epoch",
    "build_optimizer", "build_empty_cluster_db", "build_cluster_db_for_batch",
    # Meta-training
    "meta_train_step", "evaluate_on_task", "build_meta_optimizer",
    # Adaptation
    "setup_adaptation", "switch_to_step2",
    "adaptation_step1_epoch", "adaptation_step2_epoch", "validate",
]
