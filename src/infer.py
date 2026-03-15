"""
infer.py
--------
Entry point CLI per l'inferenza di HybridStyleNet.

Uso:
    # Singola immagine
    python infer.py \
        --input  photo.jpg \
        --output graded.tiff \
        --checkpoint checkpoints/photographer_01_adapted.pth \
        --db      memory/photographer_01/

    # Cartella intera (batch)
    python infer.py \
        --input  /path/to/folder/ \
        --output /path/to/output/ \
        --checkpoint checkpoints/photographer_01_adapted.pth \
        --db      memory/photographer_01/ \
        --format  tiff

    # Dry run — stampa info senza salvare
    python infer.py --input photo.jpg --checkpoint ... --db ... --dry-run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HybridStyleNet inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",  "-i", required=True,
        help="Immagine sorgente (.jpg/.png/.tif) o cartella",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output (.tiff/.jpg) o cartella. Default: ./output/",
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path al checkpoint adattato (.pth)",
    )
    parser.add_argument(
        "--db", required=True,
        help="Path alla directory del database del fotografo",
    )
    parser.add_argument(
        "--config", default="configs/base.yaml",
        help="Path al config base",
    )
    parser.add_argument(
        "--format", default="tiff",
        choices=["tiff", "jpeg", "png"],
        help="Formato output (default: tiff)",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="Qualità JPEG (1-95, default: 95)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Usa fp16 per l'inferenza (default: True)",
    )
    parser.add_argument(
        "--top-m", type=int, default=10,
        help="Top-M immagini per cluster nel retrieval (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostra info senza salvare",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def load_model_and_db(args: argparse.Namespace) -> tuple:
    """Carica modello, database e FAISS index dal checkpoint."""
    import yaml
    from models.rag_colornet  import RAGColorNet
    from memory.database      import PhotographerDatabase
    from memory.faiss_index   import FAISSIndexManager

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Checkpoint
    print(f"  Caricamento checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)

    k_star = ckpt.get("k_star", 8)
    model  = RAGColorNet.from_config(cfg, n_clusters=k_star)
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(args.device).eval()

    n_trainable = model.count_trainable_params()
    print(f"  Modello caricato  — K*={k_star}, params trainable: {n_trainable:,}")

    # Database
    print(f"  Caricamento database: {args.db}")
    database  = PhotographerDatabase.load(args.db)
    faiss_mgr = FAISSIndexManager.from_config(cfg, n_clusters=database.n_clusters)
    faiss_mgr.build_from_database(database)

    print(f"  Database caricato — {len(database)} coppie, "
          f"{database.memory_usage_mb():.1f} MB RAM")

    return model, database, faiss_mgr, cfg


@torch.no_grad()
def grade_image(
    model:     torch.nn.Module,
    database,
    faiss_mgr,
    img_tensor: torch.Tensor,    # (3, H, W) sRGB [0,1]
    device:    str,
    top_m:     int  = 10,
    fp16:      bool = True,
) -> tuple:
    """
    Applica il grading a un singolo tensor immagine.

    Returns
    -------
    pred  : (3, H, W) tensor sRGB [0,1]
    alpha : (1, H, W) confidence mask
    p     : (K,) soft cluster assignment
    """
    from training.pretrain import build_cluster_db_for_batch

    src = img_tensor.unsqueeze(0).to(device)

    # Costruisce cluster_db
    h = model.scene_encoder.histogram(src)
    cluster_db = build_cluster_db_for_batch(
        query_hist = h,
        database   = database,
        faiss_mgr  = faiss_mgr,
        top_m      = top_m,
        device     = device,
    )

    with autocast(enabled=fp16):
        out = model(src, cluster_db)

    pred  = out["I_out"][0].float().cpu()
    alpha = out["alpha"][0].float().cpu()
    p     = out["p"][0].float().cpu()

    return pred, alpha, p


def collect_input_paths(input_path: str) -> list:
    """Raccoglie tutti i file immagine da processare."""
    p   = Path(input_path)
    ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff",
           ".arw", ".dng", ".cr2", ".nef"}

    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in ext)
        if not files:
            raise ValueError(f"Nessuna immagine trovata in: {p}")
        return files
    else:
        raise FileNotFoundError(f"Input non trovato: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from utils.image_io import load_tensor, save_image

    # Raccogli file di input
    input_files = collect_input_paths(args.input)
    print(f"\n{'='*55}")
    print(f"  HybridStyleNet — Inferenza")
    print(f"{'='*55}")
    print(f"  File da processare : {len(input_files)}")
    print(f"  Device             : {args.device}")
    print(f"  FP16               : {args.fp16}")
    print(f"  Formato output     : {args.format}")
    print(f"  Top-M retrieval    : {args.top_m}")

    if args.dry_run:
        print("\n  [DRY RUN] Nessun file verrà salvato.")

    # Carica modello e database
    model, database, faiss_mgr, cfg = load_model_and_db(args)

    # Directory di output
    if args.output:
        out_base = Path(args.output)
    else:
        out_base = Path("output")

    if len(input_files) > 1 or Path(args.input).is_dir():
        out_base.mkdir(parents=True, exist_ok=True)

    # Inferenza
    total_t = 0.0
    print(f"\n  Processing...\n")

    for i, src_path in enumerate(input_files, 1):
        t0 = time.perf_counter()

        # Carica immagine a piena risoluzione
        img = load_tensor(src_path, device="cpu")

        # Grade
        pred, alpha, p = grade_image(
            model     = model,
            database  = database,
            faiss_mgr = faiss_mgr,
            img_tensor = img,
            device    = args.device,
            top_m     = args.top_m,
            fp16      = args.fp16,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        total_t += elapsed

        # Cluster dominante
        dominant_k = int(p.argmax().item())
        dominant_p = float(p.max().item())

        print(
            f"  [{i:4d}/{len(input_files)}] {src_path.name:<30}  "
            f"{img.shape[1]}×{img.shape[2]}  "
            f"k={dominant_k}({dominant_p:.2f})  "
            f"{elapsed:.0f}ms"
        )

        if args.dry_run:
            continue

        # Determina path di output
        if len(input_files) == 1 and args.output and not Path(args.output).is_dir():
            out_path = Path(args.output)
        else:
            suffix   = "." + args.format.replace("jpeg", "jpg")
            out_path = out_base / (src_path.stem + "_graded" + suffix)

        save_image(
            tensor  = pred,
            path    = out_path,
            quality = args.quality,
            bit16   = (args.format == "tiff"),
        )

    # Riepilogo
    avg_ms = total_t / len(input_files)
    print(f"\n  {'─'*45}")
    print(f"  Immagini processate : {len(input_files)}")
    print(f"  Tempo medio         : {avg_ms:.0f} ms/img")
    print(f"  Tempo totale        : {total_t/1000:.1f}s")
    if not args.dry_run:
        print(f"  Output salvato in   : {out_base}")
    print(f"  {'─'*45}\n")


if __name__ == "__main__":
    main()
