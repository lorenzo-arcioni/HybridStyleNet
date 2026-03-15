"""
batch_grade.py
--------------
Batch processing di una cartella di immagini.

Espone BatchGrader che processa un'intera cartella sorgente
e salva i risultati in una cartella di output, con progress bar
e riepilogo delle metriche di inferenza.

Uso tipico:
    from inference.batch_grade import BatchGrader

    bg = BatchGrader(
        checkpoint = "checkpoints/photographer_01_adapted.pth",
        db_path    = "memory/photographer_01/",
        output_dir = "output/graded/",
    )
    summary = bg.run("input/photos/")
    print(summary)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .grade import Grader, GradingResult


# ---------------------------------------------------------------------------
# ImageFolderDataset  (caricamento lazy delle immagini)
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """
    Dataset minimale che carica immagini da una cartella flat.
    Usato dal DataLoader per il prefetching parallelo.
    """

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff",
                   ".arw", ".dng", ".cr2", ".nef"}

    def __init__(self, folder: Path) -> None:
        self.paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in self._EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"Nessuna immagine trovata in: {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        from utils.image_io import load_tensor
        path = self.paths[idx]
        return {
            "tensor":   load_tensor(path, device="cpu"),
            "filename": path.name,
            "stem":     path.stem,
        }


# ---------------------------------------------------------------------------
# BatchSummary
# ---------------------------------------------------------------------------

@dataclass
class BatchSummary:
    """Riepilogo di un'esecuzione batch."""
    n_processed:    int   = 0
    n_errors:       int   = 0
    total_time_s:   float = 0.0
    avg_ms:         float = 0.0
    min_ms:         float = float("inf")
    max_ms:         float = 0.0
    output_dir:     str   = ""
    errors:         List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "─" * 48,
            f"  Immagini processate : {self.n_processed}",
            f"  Errori              : {self.n_errors}",
            f"  Tempo totale        : {self.total_time_s:.1f}s",
            f"  Tempo medio         : {self.avg_ms:.0f} ms/img",
            f"  Tempo min/max       : {self.min_ms:.0f} / {self.max_ms:.0f} ms",
            f"  Output              : {self.output_dir}",
            "─" * 48,
        ]
        if self.errors:
            lines.append(f"  Errori dettagliati:")
            for e in self.errors[:5]:
                lines.append(f"    • {e}")
            if len(self.errors) > 5:
                lines.append(f"    ... e altri {len(self.errors)-5}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BatchGrader
# ---------------------------------------------------------------------------

class BatchGrader:
    """
    Processa una cartella di immagini con progress bar.

    Internamente usa un DataLoader con num_workers per il prefetching
    delle immagini mentre la GPU gradua in parallelo.

    Parameters
    ----------
    checkpoint  : path al checkpoint adattato
    db_path     : path al database del fotografo
    output_dir  : cartella di output
    config      : path al config base
    device      : "cuda" | "cpu"
    fp16        : usa fp16
    top_m       : top-M retrieval
    output_fmt  : "tiff" | "jpeg" | "png"
    jpeg_quality: qualità JPEG (1-95)
    num_workers : worker per il DataLoader (prefetching CPU)
    suffix      : suffisso aggiunto al nome file ("_graded" → foto_graded.tiff)
    """

    def __init__(
        self,
        checkpoint:   Union[str, Path],
        db_path:      Union[str, Path],
        output_dir:   Union[str, Path],
        config:       Union[str, Path] = "configs/base.yaml",
        device:       str  = "cuda" if torch.cuda.is_available() else "cpu",
        fp16:         bool = True,
        top_m:        int  = 10,
        output_fmt:   str  = "tiff",
        jpeg_quality: int  = 95,
        num_workers:  int  = 4,
        suffix:       str  = "_graded",
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.output_fmt   = output_fmt
        self.jpeg_quality = jpeg_quality
        self.num_workers  = num_workers
        self.suffix       = suffix

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Carica modello e database una volta sola
        self._grader = Grader(
            checkpoint = checkpoint,
            db_path    = db_path,
            config     = config,
            device     = device,
            fp16       = fp16,
            top_m      = top_m,
        )

        print(f"✓ BatchGrader pronto")
        print(f"  Fotografo  : {self._grader.photographer_id}")
        print(f"  Database   : {self._grader.n_pairs} coppie, K={self._grader.k_star}")
        print(f"  Output fmt : {output_fmt}")
        print(f"  Output dir : {self.output_dir}")

    # ------------------------------------------------------------------
    def run(
        self,
        input_dir:  Union[str, Path],
        show_progress: bool = True,
    ) -> BatchSummary:
        """
        Processa tutti i file immagine in input_dir.

        Parameters
        ----------
        input_dir      : cartella con le immagini sorgente
        show_progress  : mostra tqdm progress bar

        Returns
        -------
        BatchSummary con statistiche di inferenza
        """
        try:
            from tqdm import tqdm
            _tqdm_available = True
        except ImportError:
            _tqdm_available = False

        from utils.image_io import save_image

        dataset = ImageFolderDataset(Path(input_dir))
        loader  = DataLoader(
            dataset,
            batch_size  = 1,              # batch=1 per piena risoluzione
            num_workers = self.num_workers,
            pin_memory  = (self._grader.device == "cuda"),
            prefetch_factor = 2 if self.num_workers > 0 else None,
        )

        summary   = BatchSummary(output_dir=str(self.output_dir))
        times_ms: List[float] = []

        iterator = loader
        if show_progress and _tqdm_available:
            iterator = tqdm(loader, desc="Batch grading", unit="img",
                            total=len(dataset))

        print(f"\nProcessing {len(dataset)} immagini da: {input_dir}")

        for batch in iterator:
            filename = batch["filename"][0]
            stem     = batch["stem"][0]
            tensor   = batch["tensor"][0]   # (3, H, W) — rimuove batch dim

            try:
                result = self._grader.grade(tensor)

                # Salva output
                ext      = "." + self.output_fmt.replace("jpeg", "jpg")
                out_path = self.output_dir / (stem + self.suffix + ext)
                save_image(
                    tensor  = result.pred,
                    path    = out_path,
                    quality = self.jpeg_quality,
                    bit16   = (self.output_fmt == "tiff"),
                )

                times_ms.append(result.elapsed_ms)
                summary.n_processed += 1

                if show_progress and _tqdm_available:
                    iterator.set_postfix({
                        "file": filename[:20],
                        "k":    result.dominant_k,
                        "ms":   f"{result.elapsed_ms:.0f}",
                    })

            except Exception as e:
                summary.n_errors += 1
                summary.errors.append(f"{filename}: {e}")
                if show_progress and _tqdm_available:
                    iterator.set_postfix({"ERROR": filename[:20]})

        # Calcola statistiche
        if times_ms:
            summary.total_time_s = sum(times_ms) / 1000
            summary.avg_ms       = sum(times_ms) / len(times_ms)
            summary.min_ms       = min(times_ms)
            summary.max_ms       = max(times_ms)

        print(f"\n{summary}")
        return summary

    # ------------------------------------------------------------------
    @property
    def grader(self) -> Grader:
        """Accesso diretto al Grader sottostante."""
        return self._grader
