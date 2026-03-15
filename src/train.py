"""
train.py
--------
Entry point CLI per le tre fasi di training di HybridStyleNet.

Uso:
    # Fase 1 — pre-training su FiveK + PPR10K + Lightroom presets
    python train.py --phase pretrain --config configs/pretraining.yaml

    # Fase 2 — meta-training Reptile
    python train.py --phase meta --config configs/meta_training.yaml

    # Fase 3 — few-shot adaptation su un fotografo specifico
    python train.py --phase adapt \
        --config configs/base.yaml \
        --photographer-config configs/photographer.yaml \
        --photographer-id photographer_01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HybridStyleNet training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["pretrain", "meta", "adapt"],
        help="Fase di training da eseguire",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path al config base (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--photographer-config",
        default=None,
        help="Path al config del fotografo (richiesto per --phase adapt)",
    )
    parser.add_argument(
        "--photographer-id",
        default=None,
        help="ID del fotografo target (richiesto per --phase adapt)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda se disponibile)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path a un checkpoint da cui riprendere il training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Esegui solo un batch per verificare che tutto funzioni",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(base_path: str, override_path: str | None = None) -> dict:
    """
    Carica il config base e opzionalmente sovrascrive con un config secondario.
    I valori del config secondario hanno precedenza su quelli del base.
    """
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge ricorsivo: override ha precedenza su base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pretrain(cfg: dict, args: argparse.Namespace) -> None:
    """Fase 1: pre-training su dataset multipli."""
    from data.datasets          import build_pretrain_dataset
    from models.rag_colornet    import RAGColorNet
    from losses.composite_loss  import CompositeLoss
    from training.pretrain      import (
        run_epoch, build_optimizer, build_empty_cluster_db
    )
    from training.lr_scheduler  import build_scheduler
    from training.early_stopping import EarlyStopping
    from torch.utils.data        import DataLoader

    print(f"\n{'='*60}")
    print("  FASE 1 — Pre-training")
    print(f"{'='*60}")

    device = args.device
    n_epochs = cfg["training"]["n_epochs"]

    # Dataset
    dataset    = build_pretrain_dataset(cfg)
    sampler    = dataset.make_sampler()
    train_loader = DataLoader(
        dataset,
        batch_size  = cfg["training"]["batch_size"],
        sampler     = sampler,
        num_workers = cfg["hardware"].get("num_workers", 4),
        pin_memory  = True,
    )

    # Modello — K=1 generico in pre-training (nessun fotografo specifico)
    model     = RAGColorNet.from_config(cfg, n_clusters=1).to(device)
    loss_fn   = CompositeLoss.from_config(cfg, backbone=model.scene_encoder.backbone)
    optimizer = build_optimizer(model, cfg, phase="pretrain",
                                lr_override=cfg["optimizer"].get("lr", 1e-4))
    scheduler = build_scheduler(optimizer, cfg, phase="pretrain", n_epochs=n_epochs)
    cluster_db = build_empty_cluster_db(n_clusters=1)

    ckpt_path = Path(cfg["checkpointing"].get(
        "output_path", "checkpoints/pretrain_best.pth"
    ))
    stopper = EarlyStopping(
        patience        = 5,
        mode            = "min",
        checkpoint_path = ckpt_path,
    )

    print(f"  Dataset     : {len(dataset):,} coppie")
    print(f"  Parametri   : {model.count_trainable_params():,} trainable")
    print(f"  Epoche      : {n_epochs}")
    print(f"  Device      : {device}")

    if args.dry_run:
        print("\n  [DRY RUN] Eseguo solo 1 batch...")
        n_epochs = 1

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        metrics = run_epoch(
            model      = model,
            loader     = train_loader,
            loss_fn    = loss_fn,
            cluster_db = cluster_db,
            device     = device,
            optimizer  = optimizer,
            phase      = "pretrain",
            epoch      = epoch,
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Ep {epoch:3d}/{n_epochs}  "
            f"loss={metrics['loss/total']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"{elapsed:.1f}s"
        )

        if stopper.step(metrics["loss/total"], model=model, epoch=epoch):
            pass
        if stopper.should_stop:
            print(f"  Early stopping @ epoch {epoch}")
            break

        if args.dry_run:
            break

    print(f"\n  ✓ Pre-training completato. Checkpoint: {ckpt_path}")


def run_meta(cfg: dict, args: argparse.Namespace) -> None:
    """Fase 2: meta-training Reptile."""
    from models.rag_colornet    import RAGColorNet
    from losses.composite_loss  import CompositeLoss
    from training.meta_train    import meta_train_step, build_meta_optimizer
    from data.task_sampler      import build_task_sampler

    print(f"\n{'='*60}")
    print("  FASE 2 — Meta-training (Reptile)")
    print(f"{'='*60}")

    device      = args.device
    n_iters     = cfg["training"]["n_iterations"]

    # Carica da checkpoint pre-training se disponibile
    model = RAGColorNet.from_config(cfg, n_clusters=1).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"  Caricato checkpoint: {args.resume}")

    loss_fn = CompositeLoss.from_config(cfg, backbone=model.scene_encoder.backbone)
    task_sampler = build_task_sampler(
        cfg,
        fivek_root  = str(Path(cfg["paths"]["data_root"]) / "fivek"),
        ppr10k_root = str(Path(cfg["paths"]["data_root"]) / "ppr10k"),
    )

    print(f"  Iterazioni : {n_iters}")
    print(f"  Task/batch : {cfg['reptile']['n_tasks_per_batch']}")
    print(f"  Inner steps: {cfg['reptile']['n_inner_steps']}")

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for i in range(1, n_iters + 1):
        result = meta_train_step(
            model        = model,
            task_sampler = task_sampler,
            loss_fn      = loss_fn,
            cfg          = cfg,
            device       = device,
            iteration    = i,
        )

        if i % cfg["logging"].get("log_every_n_steps", 50) == 0:
            print(f"  Iter {i:5d}/{n_iters}  meta_loss={result['meta_loss']:.4f}")

        if result["meta_loss"] < best_loss:
            best_loss = result["meta_loss"]
            torch.save(
                {"iteration": i, "model_state": model.state_dict(),
                 "meta_loss": best_loss},
                ckpt_dir / "meta_best.pth",
            )

        if args.dry_run:
            break

    print(f"\n  ✓ Meta-training completato. Best loss: {best_loss:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'meta_best.pth'}")


def run_adapt(cfg: dict, args: argparse.Namespace) -> None:
    """Fase 3: few-shot adaptation su un fotografo specifico."""
    from models.rag_colornet       import RAGColorNet
    from losses.composite_loss     import CompositeLoss
    from data.photographer_dataset import build_photographer_datasets
    from memory.database           import PhotographerDatabase
    from memory.faiss_index        import FAISSIndexManager
    from memory.incremental_update import IncrementalUpdater
    from training.adapt            import (
        setup_adaptation, switch_to_step2,
        adaptation_step1_epoch, adaptation_step2_epoch, validate,
    )
    from training.early_stopping   import EarlyStopping
    from utils.kmeans_init         import elbow_kmeans
    from torch.utils.data          import DataLoader
    import numpy as np

    photographer_id = args.photographer_id or cfg.get("photographer", {}).get("id", "unknown")
    print(f"\n{'='*60}")
    print(f"  FASE 3 — Few-shot Adaptation: {photographer_id}")
    print(f"{'='*60}")

    device = args.device

    # Dataset
    train_sub, val_sub, full_ds = build_photographer_datasets(cfg)
    train_loader = DataLoader(
        train_sub, batch_size=cfg["adaptation"]["batch_size"],
        shuffle=True, num_workers=cfg["hardware"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_sub, batch_size=cfg["adaptation"]["batch_size"],
        shuffle=False, num_workers=cfg["hardware"].get("num_workers", 4),
        pin_memory=True,
    )

    print(f"  Coppie train: {len(train_sub)}")
    print(f"  Coppie val  : {len(val_sub)}")

    # K-Means init — calcola histograms su tutte le immagini di training
    from utils.color_utils import rgb_to_lab, soft_histogram
    hists = []
    for i in range(len(train_sub)):
        item = train_sub[i]
        img  = item["src"].unsqueeze(0)
        lab  = rgb_to_lab(img)
        parts = []
        for c, (vmin, vmax) in enumerate([(0.,100.),(-128.,127.),(-128.,127.)]):
            h = soft_histogram(lab[0,c], n_bins=cfg["histogram"]["n_bins"],
                               vmin=vmin, vmax=vmax)
            parts.append(h.squeeze(0).numpy())
        hists.append(np.concatenate(parts))
    histograms = np.stack(hists)

    k_star, centroids, assignments = elbow_kmeans(
        histograms,
        k_max = cfg["cluster"]["k_max"],
        tau   = cfg["cluster"]["elbow_tau"],
    )
    print(f"  K* cluster  : {k_star}")

    # Modello
    model = RAGColorNet.from_config(cfg, n_clusters=k_star).to(device)
    model.cluster_net.reinitialise_from_centroids(
        torch.from_numpy(centroids).float()
    )

    # Database e FAISS
    database  = PhotographerDatabase.from_config(cfg, k_star, photographer_id)
    database.centroids = centroids
    faiss_mgr = FAISSIndexManager.from_config(cfg, k_star)
    updater   = IncrementalUpdater(model, database, faiss_mgr, cfg, device)

    cluster_labels_map = {
        idx: int(assignments[i]) for i, idx in enumerate(train_sub.indices)
    }
    updater.preprocess_all(full_ds, assignments, show_progress=True)

    # Setup adaptation
    loss_fn           = CompositeLoss.from_config(cfg, backbone=model.scene_encoder.backbone)
    optimizer, scaler = setup_adaptation(model, cfg, device)

    ckpt_path = Path(cfg["output"]["adapted_checkpoint"])
    stopper   = EarlyStopping.from_config(cfg, checkpoint_path=ckpt_path)

    n_step1 = cfg["adaptation"]["step1_epochs"]
    n_step2 = cfg["adaptation"]["step2_epochs"]

    # Step 1
    print(f"\n  Step 1 ({n_step1} epoche, freeze parziale)...")
    for epoch in range(1, n_step1 + 1):
        train_m = adaptation_step1_epoch(
            model, train_loader, optimizer, loss_fn, scaler,
            database, faiss_mgr, cfg, epoch, device, cluster_labels_map,
        )
        val_m = validate(model, val_loader, loss_fn, database, faiss_mgr, cfg, epoch, device)
        print(f"    Ep {epoch:2d}  train={train_m['loss/total']:.4f}  val={val_m['loss/total']:.4f}")
        stopper.step(val_m["loss/total"], model=model, epoch=epoch)
        if args.dry_run:
            break

    # Step 2
    optimizer = switch_to_step2(model, cfg, device)
    stopper.reset()
    print(f"\n  Step 2 ({n_step2} epoche, full fine-tune)...")
    for epoch in range(n_step1 + 1, n_step1 + n_step2 + 1):
        train_m = adaptation_step2_epoch(
            model, train_loader, optimizer, loss_fn, scaler,
            database, faiss_mgr, cfg, epoch, device, cluster_labels_map,
        )
        val_m = validate(model, val_loader, loss_fn, database, faiss_mgr, cfg, epoch, device)
        print(f"    Ep {epoch:2d}  train={train_m['loss/total']:.4f}  val={val_m['loss/total']:.4f}")
        stopper.step(val_m["loss/total"], model=model, epoch=epoch)
        if stopper.should_stop:
            print(f"    Early stopping @ epoch {epoch}")
            break
        if args.dry_run:
            break

    # Salva database
    db_path = Path(cfg["output"]["photographer_db_path"])
    database.save(db_path)
    print(f"\n  ✓ Adaptation completata")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Database   : {db_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Carica config
    cfg = load_config(
        base_path     = args.config,
        override_path = args.photographer_config,
    )

    # Override device dal CLI
    cfg.setdefault("hardware", {})["device"] = args.device

    if args.phase == "pretrain":
        run_pretrain(cfg, args)
    elif args.phase == "meta":
        run_meta(cfg, args)
    elif args.phase == "adapt":
        if not args.photographer_config:
            print("ERRORE: --phase adapt richiede --photographer-config", file=sys.stderr)
            sys.exit(1)
        run_adapt(cfg, args)


if __name__ == "__main__":
    main()
