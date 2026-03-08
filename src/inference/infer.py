"""
inference/infer.py

Inferenza a risoluzione piena  (§5.6, §6.3 Forward Pass).

Pipeline:
  1. Carica RAW (ARW/DNG) → tensore via RawPipeline
  2. Forward pass HybridStyleNet a risoluzione piena (s=1)
  3. Salva JPG/TIFF graded

Supporta:
  - Singola immagine
  - Batch (directory)
  - Formati output: JPEG, TIFF 16-bit, PNG
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

from data.raw_pipeline import RawPipeline
from models.hybrid_style_net import HybridStyleNet
from utils.checkpoint import load_checkpoint
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class Inferencer:
    """
    Inferenza end-to-end RAW → graded JPEG/TIFF.

    Args:
        model:           HybridStyleNet già caricato con i pesi del fotografo.
        device:          Device di inferenza ('cuda', 'cpu', 'mps').
        output_format:   Formato output: 'jpeg', 'tiff', 'png'.
        jpeg_quality:    Qualità JPEG (1-100).
        use_amp:         Se True, usa fp16 per l'inferenza (richiede CUDA).
        tile_size:       Se non None, elabora l'immagine a tile per limitare
                         la VRAM (es. 2048). None = piena risoluzione.
        tile_overlap:    Sovrapposizione tra tile in pixel (default 64).
    """

    def __init__(
        self,
        model: HybridStyleNet,
        device: str = "cuda",
        output_format: str = "jpeg",
        jpeg_quality: int = 95,
        use_amp: bool = True,
        tile_size: Optional[int] = None,
        tile_overlap: int = 64,
    ) -> None:
        self.model         = model
        self.device        = torch.device(device)
        self.output_format = output_format.lower()
        self.jpeg_quality  = jpeg_quality
        self.use_amp       = use_amp and (device == "cuda")
        self.tile_size     = tile_size
        self.tile_overlap  = tile_overlap

        self.model.to(self.device)
        self.model.eval()

        # Pipeline RAW senza downsampling (s=1)
        self._pipeline = RawPipeline(
            target_long_side=None,   # nessun ridimensionamento
            normalize_imagenet=True,
        )

        logger.info(
            f"Inferencer: device={device}, format={output_format}, "
            f"amp={self.use_amp}, tile_size={tile_size}"
        )

    # ── API pubblica ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_cfg: Optional[Dict] = None,
        device: str = "cuda",
        **kwargs,
    ) -> "Inferencer":
        """
        Costruisce un Inferencer caricando il modello da checkpoint.

        Args:
            checkpoint_path: Percorso al .pth del fotografo.
            model_cfg:       Configurazione del modello (default.yaml model section).
            device:          Device.

        Returns:
            Istanza Inferencer pronta all'uso.
        """
        cfg = model_cfg or {}
        model = HybridStyleNet(
            pretrained_encoder=False,   # pesi caricati dal checkpoint
            swin_window=cfg.get("swin_window_size", 7),
            swin_depths=tuple(cfg.get("swin_depths", [2, 2])),
            swin_heads=tuple(cfg.get("swin_num_heads", [8, 16])),
            use_rope=cfg.get("use_rope", True),
            bil_global_h=cfg.get("bil_grid_global_h", 8),
            bil_global_w=cfg.get("bil_grid_global_w", 8),
            bil_local_h=cfg.get("bil_grid_local_h",  32),
            bil_local_w=cfg.get("bil_grid_local_w",  32),
            bil_luma_bins=cfg.get("bil_grid_depth",  8),
            prototype_dim=cfg.get("prototype_dim",   256),
        )

        state = load_checkpoint(checkpoint_path, model, device=device)

        # Ripristina la style cache se presente nel checkpoint
        if "style_cache" in state:
            cache = state["style_cache"]
            model._cached_prototype    = cache["prototype"].to(device)
            model._cached_train_keys   = cache["train_keys"].to(device)
            model._cached_train_values = cache["train_values"].to(device)
            model.context_conditioner.set_train_cache(
                model._cached_train_keys,
                model._cached_train_values,
            )
            logger.info("Style cache ripristinata dal checkpoint.")

        return cls(model=model, device=device, **kwargs)

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        return_tensor: bool = False,
    ) -> Union[torch.Tensor, str]:
        """
        Processa un singolo file RAW.

        Args:
            input_path:    Percorso al file RAW (.arw, .dng) o immagine.
            output_path:   Percorso output. Se None, costruisce automaticamente.
            return_tensor: Se True, restituisce il tensore invece di salvare.

        Returns:
            Tensore (3, H, W) se return_tensor=True, altrimenti percorso file.
        """
        input_path = str(input_path)

        # ── Carica e pre-processa ─────────────────────────────────────────────
        t0 = time.time()
        src_tensor, meta = self._pipeline(input_path)
        src_tensor = src_tensor.unsqueeze(0).to(self.device)   # (1,3,H,W)
        t_load = time.time() - t0

        # ── Inferenza ─────────────────────────────────────────────────────────
        t1 = time.time()
        with torch.no_grad():
            if self.tile_size is not None:
                pred = self._tiled_forward(src_tensor)
            else:
                pred = self._full_forward(src_tensor)
        t_infer = time.time() - t1

        logger.info(
            f"Inferenza {Path(input_path).name}: "
            f"load={t_load:.2f}s, infer={t_infer:.2f}s, "
            f"size={pred.shape[2]}×{pred.shape[3]}"
        )

        if return_tensor:
            return pred.squeeze(0).cpu()

        # ── Salva ─────────────────────────────────────────────────────────────
        if output_path is None:
            output_path = self._build_output_path(input_path)

        self._save(pred.squeeze(0).cpu(), output_path)
        return output_path

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> List[str]:
        """
        Processa tutti i file RAW in una directory.

        Args:
            input_dir:  Directory sorgente.
            output_dir: Directory output.
            extensions: Estensioni da processare. Default: ARW + DNG.
            recursive:  Se True, ricerca ricorsiva.

        Returns:
            Lista di percorsi output.
        """
        exts = extensions or [".arw", ".ARW", ".dng", ".DNG"]
        src_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        glob_fn  = src_dir.rglob if recursive else src_dir.glob
        files    = [f for f in glob_fn("*") if f.suffix in exts]

        if not files:
            logger.warning(
                f"Nessun file RAW trovato in {input_dir} "
                f"con estensioni {exts}"
            )
            return []

        logger.info(f"Processando {len(files)} file...")
        output_paths = []

        for i, fpath in enumerate(sorted(files), 1):
            # Mantieni struttura subdirectory
            rel_path   = fpath.relative_to(src_dir)
            out_stem   = rel_path.with_suffix(self._extension())
            out_path   = out_dir / out_stem
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                out = self.process_file(str(fpath), str(out_path))
                output_paths.append(out)
                logger.info(f"[{i}/{len(files)}] {fpath.name} → {out_path.name}")
            except Exception as e:
                logger.error(f"Errore su {fpath.name}: {e}")

        logger.info(
            f"Completato: {len(output_paths)}/{len(files)} file processati."
        )
        return output_paths

    # ── Forward ───────────────────────────────────────────────────────────────

    def _full_forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass a piena risoluzione."""
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(src)
        return out["pred"]

    def _tiled_forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass a tile per immagini molto grandi.

        Divide l'immagine in tile sovrapposti, elabora ciascuno
        separatamente e ricompone con blending ai bordi (feathering).

        Args:
            src: (1, 3, H, W)

        Returns:
            pred: (1, 3, H, W)
        """
        _, _, H, W = src.shape
        ts  = self.tile_size
        ov  = self.tile_overlap

        # Accumulator e weight map
        pred_accum  = torch.zeros_like(src)
        weight_map  = torch.zeros(1, 1, H, W, device=src.device)

        # Costruisce maschera di feathering (cosine window)
        def _feather(h: int, w: int, ov: int) -> torch.Tensor:
            fy = torch.ones(h, dtype=src.dtype, device=src.device)
            fx = torch.ones(w, dtype=src.dtype, device=src.device)
            if ov > 0:
                ramp = torch.linspace(0, 1, ov, device=src.device)
                fy[:ov]  = ramp
                fy[-ov:] = ramp.flip(0)
                fx[:ov]  = ramp
                fx[-ov:] = ramp.flip(0)
            return torch.outer(fy, fx).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)

        # Genera tile
        step = max(ts - ov, 1)
        ys   = list(range(0, H - ts, step)) + [max(H - ts, 0)]
        xs   = list(range(0, W - ts, step)) + [max(W - ts, 0)]

        for y in ys:
            for x in xs:
                y2 = min(y + ts, H)
                x2 = min(x + ts, W)
                y1, x1 = y2 - ts, x2 - ts
                y1 = max(y1, 0)
                x1 = max(x1, 0)

                tile = src[:, :, y1:y2, x1:x2]
                h_t, w_t = tile.shape[2], tile.shape[3]

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    with torch.no_grad():
                        out_tile = self.model(tile)["pred"]

                mask = _feather(h_t, w_t, ov)

                pred_accum[:, :, y1:y2, x1:x2]  += out_tile * mask
                weight_map[:, :, y1:y2, x1:x2]  += mask

        # Normalizza per i pesi
        pred = pred_accum / (weight_map + 1e-8)
        return pred.clamp(0.0, 1.0)

    # ── Salvataggio ───────────────────────────────────────────────────────────

    def _save(self, tensor: torch.Tensor, path: str) -> None:
        """
        Salva un tensore (3, H, W) in [0,1] come immagine.

        Formati supportati: jpeg, tiff (16-bit), png.
        """
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Tensore → numpy uint8 o uint16
        arr = tensor.permute(1, 2, 0).numpy()   # (H, W, 3)

        if self.output_format == "tiff":
            # TIFF 16-bit per massima qualità
            arr16 = (arr * 65535.0).clip(0, 65535).astype(np.uint16)
            img   = Image.fromarray(arr16, mode="RGB")
            img.save(str(path), format="TIFF",
                     compression="tiff_lzw")

        elif self.output_format == "jpeg":
            arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            img  = Image.fromarray(arr8)
            img.save(str(path), format="JPEG",
                     quality=self.jpeg_quality,
                     subsampling=0)   # 4:4:4 per massima qualità

        elif self.output_format == "png":
            arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            img  = Image.fromarray(arr8)
            img.save(str(path), format="PNG")

        else:
            raise ValueError(
                f"Formato output non supportato: {self.output_format}. "
                f"Usa 'jpeg', 'tiff' o 'png'."
            )

        logger.debug(f"Salvato: {path}")

    def _build_output_path(self, input_path: str) -> str:
        """Costruisce il percorso di output dallo stesso stem dell'input."""
        p    = Path(input_path)
        stem = p.stem
        ext  = self._extension()
        return str(p.parent / f"{stem}_graded{ext}")

    def _extension(self) -> str:
        return {
            "jpeg": ".jpg",
            "tiff": ".tiff",
            "png":  ".png",
        }.get(self.output_format, ".jpg")

    # ── Utility: aggiorna la style cache da un training set ──────────────────

    def update_style_cache(
        self,
        src_paths: List[str],
        tgt_paths: List[str],
        batch_size: int = 8,
    ) -> None:
        """
        Aggiorna la style cache del modello da una lista di coppie.

        Args:
            src_paths: Lista percorsi sorgenti.
            tgt_paths: Lista percorsi target (editati).
            batch_size: Batch per il calcolo degli embedding.
        """
        assert len(src_paths) == len(tgt_paths), \
            "src_paths e tgt_paths devono avere la stessa lunghezza."

        logger.info(
            f"Aggiornamento style cache: {len(src_paths)} coppie..."
        )

        # Pipeline per i target (senza downsampling aggressivo)
        tgt_pipeline = RawPipeline(
            target_long_side=512,
            normalize_imagenet=True,
        )

        src_tensors, tgt_tensors = [], []
        for sp, tp in zip(src_paths, tgt_paths):
            try:
                s, _ = self._pipeline(sp)
                t    = tgt_pipeline.load_target(tp)
                # Normalizza il target come se fosse un input
                t_norm = tgt_pipeline._normalize_imagenet(t)
                src_tensors.append(s)
                tgt_tensors.append(t_norm)
            except Exception as e:
                logger.warning(f"Coppia saltata ({Path(sp).name}): {e}")

        if not src_tensors:
            raise RuntimeError("Nessuna coppia valida per la style cache.")

        self.model.set_style_cache(
            src_tensors, tgt_tensors, batch_size=batch_size
        )
        logger.info("Style cache aggiornata con successo.")