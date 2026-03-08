"""
data/dataset.py

Dataset PyTorch per:
  - FiveKDataset:   MIT-Adobe FiveK (5 expert photographers)
  - CustomDataset:  Dataset del fotografo target (few-shot)
  - PairedDataset:  Wrapper generico per qualsiasi lista di coppie
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .raw_pipeline import RawPipeline
from .augmentation import PairAugmentation
from .utils import (
    get_fivek_pairs,
    scan_pairs,
    filter_valid_pairs,
)

logger = logging.getLogger(__name__)


# ── Dataset generico per coppie ───────────────────────────────────────────────

class PairedDataset(Dataset):
    """
    Dataset generico per coppie (sorgente, target).

    Ogni elemento restituisce:
        {
            "src":      (3, H, W) float32 normalizzato ImageNet,
            "tgt":      (3, H, W) float32 in [0,1],
            "src_path": str,
            "tgt_path": str,
        }

    Args:
        pairs:        Lista di tuple (src_path, tgt_path).
        pipeline:     Istanza di RawPipeline per il sorgente.
        augmentation: Istanza di PairAugmentation (None = nessuna aug.).
        return_meta:  Se True include meta nei sample.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        pipeline: RawPipeline,
        augmentation: Optional[PairAugmentation] = None,
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        self.pairs        = pairs
        self.pipeline     = pipeline
        self.augmentation = augmentation
        self.return_meta  = return_meta

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        src_path, tgt_path = self.pairs[idx]

        # ── Carica sorgente ───────────────────────────────────────────────────
        try:
            src_tensor, meta = self.pipeline(src_path)
        except Exception as e:
            logger.error(f"Errore caricamento sorgente {src_path}: {e}")
            # Restituisce tensore zero per non bloccare il DataLoader
            return self._empty_sample(src_path, tgt_path)

        # ── Carica target ─────────────────────────────────────────────────────
        try:
            scale = meta.get("scale_factor", None)
            tgt_tensor = self.pipeline.load_target(tgt_path, scale=scale)
        except Exception as e:
            logger.error(f"Errore caricamento target {tgt_path}: {e}")
            return self._empty_sample(src_path, tgt_path)

        # ── De-normalizza src per l'augmentation (opera in [0,1]) ─────────────
        # L'augmentation lavora in [0,1]; re-normalizziamo dopo
        if self.augmentation is not None:
            src_denorm = self._denormalize_imagenet(src_tensor)
            src_denorm, tgt_tensor = self.augmentation(src_denorm, tgt_tensor)
            src_tensor = self._normalize_imagenet(src_denorm)

        sample: Dict = {
            "src":      src_tensor,
            "tgt":      tgt_tensor,
            "src_path": src_path,
            "tgt_path": tgt_path,
        }

        if self.return_meta:
            sample["meta"] = meta

        return sample

    # ── Utilità ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_imagenet(img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406],
                             dtype=img.dtype, device=img.device)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225],
                             dtype=img.dtype, device=img.device)[:, None, None]
        return (img - mean) / std

    @staticmethod
    def _denormalize_imagenet(img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406],
                             dtype=img.dtype, device=img.device)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225],
                             dtype=img.dtype, device=img.device)[:, None, None]
        return (img * std + mean).clamp(0.0, 1.0)

    def _empty_sample(self, src_path: str, tgt_path: str) -> Dict:
        dummy = torch.zeros(3, 64, 64, dtype=torch.float32)
        return {
            "src":      dummy,
            "tgt":      dummy,
            "src_path": src_path,
            "tgt_path": tgt_path,
        }


# ── FiveK Dataset ─────────────────────────────────────────────────────────────

class FiveKDataset(PairedDataset):
    """
    Dataset MIT-Adobe FiveK per un singolo expert photographer.

    Args:
        fivek_root:   Radice del dataset FiveK.
        expert:       Lettera expert: 'A' | 'B' | 'C' | 'D' | 'E'.
        split:        'train' | 'val' | 'test' | None (tutto).
        pipeline:     Istanza RawPipeline.
        augmentation: Istanza PairAugmentation (solo per split='train').
        val_ratio:    Frazione validazione.
        test_ratio:   Frazione test.
        seed:         Seme shuffle.
        max_pairs:    Numero massimo di coppie da usare (None = tutto).
    """

    def __init__(
        self,
        fivek_root: str,
        expert: str,
        split: Optional[str] = "train",
        pipeline: Optional[RawPipeline] = None,
        augmentation: Optional[PairAugmentation] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        max_pairs: Optional[int] = None,
    ) -> None:
        if pipeline is None:
            pipeline = RawPipeline(target_long_side=768)

        pairs = get_fivek_pairs(
            fivek_root=fivek_root,
            expert=expert,
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        logger.info(
            f"FiveKDataset — expert={expert}, split={split}, "
            f"n_pairs={len(pairs)}"
        )

        # Augmentation solo in training
        aug = augmentation if split == "train" else None

        super().__init__(pairs=pairs, pipeline=pipeline, augmentation=aug)

        self.expert     = expert
        self.split      = split
        self.fivek_root = fivek_root


# ── Custom Dataset (fotografo target) ────────────────────────────────────────

class CustomDataset(PairedDataset):
    """
    Dataset per il fotografo target nella fase di few-shot adaptation.

    Struttura attesa di custom_root:
        custom_root/
          src/   ← file RAW (.arw, .dng) o JPEG originali
          tgt/   ← file JPEG/TIFF editati

    In alternativa, src e tgt possono essere nella stessa directory
    con prefissi o suffissi distinti (non ancora supportato — usa
    src_dir/tgt_dir separati).

    Args:
        custom_root:   Radice con sottodirectory src/ e tgt/.
        split:         'train' | 'val' | None (tutto).
        val_split:     Frazione di coppie da usare come validazione.
        pipeline:      Istanza RawPipeline.
        augmentation:  Istanza PairAugmentation.
        max_pairs:     Numero massimo di coppie.
        seed:          Seme per lo split riproducibile.
        src_extensions: Estensioni accettate per sorgente.
        tgt_extensions: Estensioni accettate per target.
    """

    def __init__(
        self,
        custom_root: str,
        split: Optional[str] = "train",
        val_split: float = 0.2,
        pipeline: Optional[RawPipeline] = None,
        augmentation: Optional[PairAugmentation] = None,
        max_pairs: Optional[int] = None,
        seed: int = 42,
        src_extensions: Optional[List[str]] = None,
        tgt_extensions: Optional[List[str]] = None,
    ) -> None:
        import random

        if pipeline is None:
            pipeline = RawPipeline(target_long_side=768)

        root     = Path(custom_root)
        src_dir  = root / "src"
        tgt_dir  = root / "tgt"

        if not src_dir.is_dir() or not tgt_dir.is_dir():
            raise FileNotFoundError(
                f"CustomDataset richiede le sottodirectory 'src/' e 'tgt/' "
                f"in {custom_root}. Trovato: {list(root.iterdir())}"
            )

        all_pairs = scan_pairs(
            str(src_dir),
            str(tgt_dir),
            src_extensions=src_extensions,
            tgt_extensions=tgt_extensions,
        )
        all_pairs = filter_valid_pairs(all_pairs)

        if max_pairs is not None:
            all_pairs = all_pairs[:max_pairs]

        # Split train/val riproducibile
        rng = random.Random(seed)
        shuffled = all_pairs[:]
        rng.shuffle(shuffled)

        n_val = max(1, round(len(shuffled) * val_split))
        n_train = len(shuffled) - n_val

        if split == "train":
            pairs = shuffled[:n_train]
        elif split == "val":
            pairs = shuffled[n_train:]
        else:
            pairs = shuffled

        logger.info(
            f"CustomDataset — root={custom_root}, split={split}, "
            f"n_pairs={len(pairs)} (tot={len(all_pairs)})"
        )

        if len(pairs) == 0:
            logger.warning(
                "CustomDataset: nessuna coppia trovata. "
                "Verifica la struttura src/tgt della directory."
            )

        aug = augmentation if split == "train" else None

        super().__init__(pairs=pairs, pipeline=pipeline, augmentation=aug)

        self.custom_root = custom_root
        self.split       = split
        self.all_pairs   = all_pairs


# ── Collate function ──────────────────────────────────────────────────────────

def collate_paired(batch: List[Dict]) -> Dict:
    """
    Collate function personalizzata per il DataLoader.

    Gestisce batch di immagini a dimensioni variabili facendo
    padding al massimo all'interno del batch. In alternativa,
    si può usare batch_size=1 a piena risoluzione.

    Args:
        batch: Lista di sample dal Dataset.

    Returns:
        Dizionario con tensori (B, 3, H_max, W_max).
    """
    # Dimensioni massime nel batch
    max_h = max(s["src"].shape[1] for s in batch)
    max_w = max(s["src"].shape[2] for s in batch)

    srcs, tgts = [], []
    for s in batch:
        src = s["src"]
        tgt = s["tgt"]
        _, h, w = src.shape

        # Padding a destra e in basso (reflect per il sorgente, zero per il target)
        pad_h = max_h - h
        pad_w = max_w - w

        if pad_h > 0 or pad_w > 0:
            src = torch.nn.functional.pad(src, [0, pad_w, 0, pad_h],
                                           mode="reflect")
            tgt = torch.nn.functional.pad(tgt, [0, pad_w, 0, pad_h],
                                           mode="constant", value=0.0)

        srcs.append(src)
        tgts.append(tgt)

    return {
        "src":       torch.stack(srcs, dim=0),
        "tgt":       torch.stack(tgts, dim=0),
        "src_paths": [s["src_path"] for s in batch],
        "tgt_paths": [s["tgt_path"] for s in batch],
    }