"""
data/utils.py

Funzioni di utilità per la gestione delle coppie di immagini:
  - Scansione directory
  - Matching sorgente ↔ target per nome/stem
  - Verifica integrità coppia
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Estensioni supportate ─────────────────────────────────────────────────────

RAW_EXTENSIONS = {".arw", ".dng", ".ARW", ".DNG",
                  ".nef", ".NEF", ".cr2", ".CR2",
                  ".cr3", ".CR3", ".raf", ".RAF"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG",
                    ".tif", ".tiff", ".TIF", ".TIFF",
                    ".png", ".PNG"}


# ── Scansione ────────────────────────────────────────────────────────────────

def scan_pairs(
    src_dir: str,
    tgt_dir: str,
    src_extensions: Optional[List[str]] = None,
    tgt_extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Tuple[str, str]]:
    """
    Individua le coppie (sorgente, target) abbinando i file per stem del nome.

    La funzione tollera estensioni diverse tra sorgente e target
    (es. "IMG_001.arw" ↔ "IMG_001.jpg").

    Args:
        src_dir:        Directory delle immagini sorgente (RAW).
        tgt_dir:        Directory delle immagini target (editato).
        src_extensions: Lista estensioni ammesse per sorgente.
                        Default: tutte le estensioni RAW.
        tgt_extensions: Lista estensioni ammesse per target.
                        Default: tutte le estensioni immagine.
        recursive:      Se True, ricerca ricorsiva nelle sottodirectory.

    Returns:
        Lista ordinata di tuple (percorso_sorgente, percorso_target).
        Solo le coppie con entrambi i file presenti vengono incluse.
    """
    src_exts = set(src_extensions) if src_extensions \
        else RAW_EXTENSIONS | IMAGE_EXTENSIONS
    tgt_exts = set(tgt_extensions) if tgt_extensions else IMAGE_EXTENSIONS

    src_path = Path(src_dir)
    tgt_path = Path(tgt_dir)

    if not src_path.is_dir():
        raise FileNotFoundError(f"Directory sorgente non trovata: {src_dir}")
    if not tgt_path.is_dir():
        raise FileNotFoundError(f"Directory target non trovata: {tgt_dir}")

    # Costruisci mappa stem → Path per sorgente
    glob_fn = src_path.rglob if recursive else src_path.glob
    src_map: Dict[str, Path] = {}
    for f in glob_fn("*"):
        if f.is_file() and f.suffix in src_exts:
            stem = f.stem.lower()
            if stem in src_map:
                logger.warning(
                    f"Stem duplicato in src: '{stem}' — mantenuto {src_map[stem]}"
                )
            else:
                src_map[stem] = f

    # Costruisci mappa stem → Path per target
    tgt_map: Dict[str, Path] = {}
    glob_fn_t = tgt_path.rglob if recursive else tgt_path.glob
    for f in glob_fn_t("*"):
        if f.is_file() and f.suffix in tgt_exts:
            stem = f.stem.lower()
            if stem not in tgt_map:
                tgt_map[stem] = f

    # Abbina per stem
    pairs: List[Tuple[str, str]] = []
    for stem, src_file in sorted(src_map.items()):
        if stem in tgt_map:
            pairs.append((str(src_file), str(tgt_map[stem])))
        else:
            logger.debug(f"Nessun target per sorgente: {src_file.name}")

    # Report
    n_src = len(src_map)
    n_tgt = len(tgt_map)
    n_pairs = len(pairs)
    logger.info(
        f"scan_pairs: {n_src} sorgenti, {n_tgt} target → {n_pairs} coppie trovate"
    )

    if n_pairs == 0:
        logger.warning(
            "Nessuna coppia trovata. "
            "Verifica che i nomi file (senza estensione) corrispondano."
        )

    return pairs


def match_pairs(
    file_list_src: List[str],
    file_list_tgt: List[str],
) -> List[Tuple[str, str]]:
    """
    Abbina due liste di file per stem, in modo ordinato.

    Utile quando i file sono già noti (es. da un file di testo indice).

    Args:
        file_list_src: Lista percorsi file sorgente.
        file_list_tgt: Lista percorsi file target.

    Returns:
        Lista di coppie (src, tgt) abbinate per stem.
    """
    tgt_map = {Path(f).stem.lower(): f for f in file_list_tgt}
    pairs: List[Tuple[str, str]] = []
    for src in sorted(file_list_src):
        stem = Path(src).stem.lower()
        if stem in tgt_map:
            pairs.append((src, tgt_map[stem]))
        else:
            logger.debug(f"Nessun match per: {Path(src).name}")
    return pairs


# ── Verifica ─────────────────────────────────────────────────────────────────

def verify_pair(src_path: str, tgt_path: str) -> bool:
    """
    Verifica che una coppia sia integra:
      - Entrambi i file esistono e sono leggibili
      - La dimensione di entrambi i file è > 0
      - Il target è un'immagine apribile (PIL)

    Args:
        src_path: Percorso file sorgente.
        tgt_path: Percorso file target.

    Returns:
        True se la coppia è valida, False altrimenti.
    """
    src = Path(src_path)
    tgt = Path(tgt_path)

    if not src.exists():
        logger.warning(f"Sorgente mancante: {src_path}")
        return False

    if not tgt.exists():
        logger.warning(f"Target mancante: {tgt_path}")
        return False

    if src.stat().st_size == 0:
        logger.warning(f"Sorgente vuota (0 byte): {src_path}")
        return False

    if tgt.stat().st_size == 0:
        logger.warning(f"Target vuoto (0 byte): {tgt_path}")
        return False

    # Verifica apertura del target come immagine
    try:
        from PIL import Image
        with Image.open(tgt_path) as img:
            img.verify()  # controlla integrità senza caricare
    except Exception as e:
        logger.warning(f"Target non leggibile come immagine ({tgt_path}): {e}")
        return False

    return True


def filter_valid_pairs(
    pairs: List[Tuple[str, str]],
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Filtra una lista di coppie tenendo solo quelle valide.

    Args:
        pairs:   Lista di coppie (src, tgt).
        verbose: Se True, logga statistiche.

    Returns:
        Lista di coppie valide.
    """
    valid = [p for p in pairs if verify_pair(p[0], p[1])]
    n_removed = len(pairs) - len(valid)

    if verbose:
        logger.info(
            f"filter_valid_pairs: {len(valid)}/{len(pairs)} coppie valide "
            f"({n_removed} rimosse)"
        )

    return valid


# ── Utilità FiveK ─────────────────────────────────────────────────────────────

def get_fivek_pairs(
    fivek_root: str,
    expert: str,
    split: Optional[str] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Restituisce le coppie (RAW, editato) per un expert del dataset FiveK.

    La struttura attesa di fivek_root è:
        fivek_root/
          raw/          ← file DNG (o TIFF pre-processati)
          expert_A/     ← editati dall'expert A
          expert_B/
          ...
          expert_E/

    Args:
        fivek_root: Radice del dataset FiveK.
        expert:     Lettera expert ('A', 'B', 'C', 'D', 'E').
        split:      'train', 'val', 'test' oppure None (tutto).
        val_ratio:  Frazione di validazione.
        test_ratio: Frazione di test.
        seed:       Seme per lo shuffle riproducibile.

    Returns:
        Lista di coppie (src_path, tgt_path).
    """
    import random

    root  = Path(fivek_root)
    src_dir = root / "raw"
    tgt_dir = root / f"expert_{expert.upper()}"

    if not src_dir.is_dir():
        # Fallback: cerca TIFF pre-processati
        src_dir = root / "tiff"
        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"FiveK: directory sorgente non trovata in {fivek_root}"
            )

    if not tgt_dir.is_dir():
        raise FileNotFoundError(
            f"FiveK: directory expert_{expert.upper()} non trovata in {fivek_root}"
        )

    pairs = scan_pairs(str(src_dir), str(tgt_dir))
    pairs = filter_valid_pairs(pairs, verbose=False)

    if not pairs:
        logger.warning(f"FiveK expert {expert}: nessuna coppia valida trovata.")
        return pairs

    if split is None:
        return pairs

    # Split riproducibile
    rng = random.Random(seed)
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)

    n = len(pairs_shuffled)
    n_test = max(1, round(n * test_ratio))
    n_val  = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test

    if split == "train":
        return pairs_shuffled[:n_train]
    elif split == "val":
        return pairs_shuffled[n_train:n_train + n_val]
    elif split == "test":
        return pairs_shuffled[n_train + n_val:]
    else:
        raise ValueError(f"split deve essere 'train', 'val', 'test' o None. Got: {split}")