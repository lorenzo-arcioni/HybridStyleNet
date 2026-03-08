"""
data/raw_pipeline.py

Pipeline deterministica RAW → tensore sRGB normalizzato (§6.2).

Supporta:
  - Sony ARW  (.arw, .ARW)
  - Adobe DNG (.dng, .DNG)

Dipendenze esterne:
  - rawpy   : lettura RAW + demosaicatura AHD
  - lensfunpy (opzionale): correzione vignettatura + aberrazione cromatica
  - numpy, torch
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Costanti ──────────────────────────────────────────────────────────────────

# Normalizzazione ImageNet (canali RGB)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# Lanczos kernel cache
_LANCZOS_CACHE: Dict[Tuple[int, int, float], torch.Tensor] = {}


# ── Utilità kernel Lanczos ────────────────────────────────────────────────────

def _lanczos_kernel_1d(size: int, a: int = 3) -> torch.Tensor:
    """
    Genera un kernel Lanczos 1-D normalizzato di lunghezza `size`.

    Args:
        size: Numero di tap del filtro (dispari consigliato).
        a:    Parametro Lanczos (default 3).

    Returns:
        Tensore float32 shape (size,) normalizzato a somma 1.
    """
    half = size // 2
    x = torch.linspace(-half, half, size, dtype=torch.float32)
    # sinc(x) * sinc(x/a), con sinc(0) = 1
    x_safe = x.abs().clamp(min=1e-8)
    sinc_x  = torch.sin(torch.pi * x) / (torch.pi * x_safe)
    sinc_xa = torch.sin(torch.pi * x / a) / (torch.pi * x_safe / a)
    sinc_x[x == 0]  = 1.0
    sinc_xa[x == 0] = 1.0
    kernel = sinc_x * sinc_xa
    kernel[x.abs() >= a] = 0.0
    return kernel / kernel.sum()


def _build_lanczos_2d(scale: float, a: int = 3) -> torch.Tensor:
    """
    Costruisce un kernel 2-D Lanczos separabile per un dato fattore di scala.

    Args:
        scale: Fattore di downsampling (0 < scale <= 1).
        a:     Parametro Lanczos.

    Returns:
        Tensore float32 shape (1, 1, K, K) per uso con F.conv2d.
    """
    # Numero di tap: 2*a*ceil(1/scale) arrotondato al dispari
    taps = max(int(2 * a * np.ceil(1.0 / scale)), 2 * a + 1)
    if taps % 2 == 0:
        taps += 1
    k1d = _lanczos_kernel_1d(taps, a)
    k2d = torch.outer(k1d, k1d)
    return k2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)


# ── Classe principale ─────────────────────────────────────────────────────────

class RawPipeline:
    """
    Pipeline deterministica RAW (ARW / DNG) → tensore sRGB normalizzato.

    Fasi (§6.2):
      1. Lettura e linearizzazione (§6.2.1)
      2. Rumore Poisson-Gaussiano opzionale (§6.2.2)
      3. Demosaicatura AHD via rawpy (§6.2.3)
      4. Correzione lens opzionale via lensfunpy (§6.2.4)
      5. White balance (§6.2.5) — applicato da rawpy
      6. Camera matrix → XYZ → sRGB lineare (§6.2.6-7) — rawpy
      7. Gamma encoding sRGB (§6.2.8)
      8. Downsampling Lanczos adattivo (§6.2.9)
      9. Normalizzazione ImageNet (§6.2.10)

    Args:
        target_long_side: Lato lungo target per il training (L_train).
                          None = nessun ridimensionamento (inferenza).
        lanczos_a:        Parametro filtro Lanczos.
        use_lens_correction: Se True tenta la correzione lens via lensfunpy.
        normalize_imagenet:  Se True applica la normalizzazione ImageNet.
    """

    def __init__(
        self,
        target_long_side: Optional[int] = 768,
        lanczos_a: int = 3,
        use_lens_correction: bool = False,
        normalize_imagenet: bool = True,
    ) -> None:
        self.target_long_side   = target_long_side
        self.lanczos_a          = lanczos_a
        self.use_lens_correction = use_lens_correction
        self.normalize_imagenet  = normalize_imagenet

        # Verifica dipendenze
        try:
            import rawpy  # noqa: F401
            self._rawpy_ok = True
        except ImportError:
            self._rawpy_ok = False
            logger.warning("rawpy non trovato. Usa _fallback_load per TIFF/PNG.")

        self._lensfun_ok = False
        if use_lens_correction:
            try:
                import lensfunpy  # noqa: F401
                self._lensfun_ok = True
            except ImportError:
                logger.warning(
                    "lensfunpy non trovato. Correzione lens disabilitata."
                )

    # ── API pubblica ──────────────────────────────────────────────────────────

    def __call__(self, path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Processa un file RAW e restituisce il tensore pronto per il modello.

        Args:
            path: Percorso al file RAW (.arw, .dng) o immagine comune.

        Returns:
            tensor: float32 (3, H, W) in [0,1] (o normalizzato ImageNet).
            meta:   Dizionario con metadati EXIF rilevanti.
        """
        ext = Path(path).suffix.lower()
        if ext in {".arw", ".dng", ".nef", ".cr2", ".cr3", ".raf"} \
                and self._rawpy_ok:
            return self._process_raw(path)
        else:
            return self._process_image(path)

    # ── Elaborazione RAW ──────────────────────────────────────────────────────

    def _process_raw(self, path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Elaborazione completa di un file RAW tramite rawpy.

        rawpy gestisce internamente:
          - §6.2.1  linearizzazione (black/saturation level)
          - §6.2.3  demosaicatura AHD
          - §6.2.5  white balance (dai metadati camera)
          - §6.2.6  camera matrix → XYZ
          - §6.2.7  XYZ → sRGB lineare
          - §6.2.8  gamma encoding sRGB (output_gamma=(2.222, 4.5) ≈ sRGB)
        """
        import rawpy

        with rawpy.imread(path) as raw:
            meta = self._extract_meta(raw)

            # Demosaicatura + conversione colore via rawpy
            # use_camera_wb=True  → WB dai metadati EXIF (§6.2.5)
            # output_color=rawpy.ColorSpace.sRGB → matrice cam→sRGB (§6.2.6-7)
            # output_bps=16      → 16 bit per canale, più precisione
            # no_auto_bright     → nessuna correzione automatica dell'esposizione
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,
                no_auto_bright=True,
                gamma=(2.222, 4.5),   # approssimazione sRGB standard
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )  # uint16 (H, W, 3)

        # uint16 → float32 [0,1]
        img = rgb16.astype(np.float32) / 65535.0

        # Correzione lens opzionale (§6.2.4)
        if self._lensfun_ok and self.use_lens_correction:
            img = self._apply_lens_correction(img, meta)

        return self._finalize(img, meta)

    def _extract_meta(self, raw) -> Dict:
        """
        Estrae metadati rilevanti dall'oggetto rawpy.RawPy.

        Returns:
            Dizionario con: black_level, saturation_level, wb_coeffs,
            camera_make, camera_model, iso, exposure_time, bayer_pattern.
        """
        meta: Dict = {}
        try:
            meta["black_level"]      = int(raw.black_level_per_channel[0])
            meta["saturation_level"] = int(raw.white_level)
            meta["wb_coeffs"]        = raw.camera_whitebalance    # [R,G,B,G2]
            meta["camera_make"]      = getattr(raw, "camera_make",  "unknown")
            meta["camera_model"]     = getattr(raw, "camera_model", "unknown")
            meta["bayer_pattern"]    = raw.raw_pattern.tolist()
            meta["raw_h"]            = raw.raw_image.shape[0]
            meta["raw_w"]            = raw.raw_image.shape[1]
        except Exception as e:
            logger.debug(f"Metadati parziali: {e}")
        return meta

    # ── Correzione lens (§6.2.4) ─────────────────────────────────────────────

    def _apply_lens_correction(
        self, img: np.ndarray, meta: Dict
    ) -> np.ndarray:
        """
        Applica correzione vignettatura + aberrazione cromatica via lensfunpy.

        Fallback silenzioso se il profilo non viene trovato.
        """
        try:
            import lensfunpy
            db = lensfunpy.Database()
            cam = db.find_cameras(meta.get("camera_make", ""),
                                  meta.get("camera_model", ""))
            if not cam:
                return img

            cam    = cam[0]
            # Tenta di trovare un obiettivo "standard" se non disponibile
            lens_list = db.find_lenses(cam)
            if not lens_list:
                return img
            lens = lens_list[0]

            h, w  = img.shape[:2]
            mod   = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
            mod.initialize(focal_length=50, aperture=5.6, distance=10)

            # Correzione distorsione + aberrazione cromatica
            undist = mod.apply_geometry_distortion()
            img    = lensfunpy.remap(img, undist)

            # Vignettatura
            vgn_scale = mod.apply_color_modification(img)
            if vgn_scale is not None:
                img = img * vgn_scale[..., np.newaxis]

        except Exception as e:
            logger.debug(f"Lens correction fallback: {e}")

        return img

    # ── Fallback per immagini standard (JPEG, TIFF, PNG) ─────────────────────

    def _process_image(self, path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Carica una immagine standard (JPEG/TIFF/PNG) come sRGB float32.
        Usato per i target editati o quando rawpy non è disponibile.
        """
        from PIL import Image
        import numpy as np

        pil = Image.open(path).convert("RGB")
        img = np.array(pil, dtype=np.float32) / 255.0
        meta: Dict = {"source": "pil", "path": path}
        return self._finalize(img, meta)

    # ── Finalizzazione comune ─────────────────────────────────────────────────

    def _finalize(
        self, img: np.ndarray, meta: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Passi finali comuni a RAW e immagini standard:
          - numpy → torch (H,W,3) → (3,H,W)
          - Downsampling Lanczos adattivo (§6.2.9)
          - Normalizzazione ImageNet (§6.2.10)

        Args:
            img:  numpy float32 (H, W, 3) in [0,1].
            meta: Dizionario metadati.

        Returns:
            tensor: float32 (3, Hs, Ws).
            meta:   Dizionario aggiornato con scale_factor, H_s, W_s.
        """
        h0, w0 = img.shape[:2]

        # numpy (H,W,3) → torch (1,3,H,W)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        # Downsampling Lanczos (§6.2.9)
        if self.target_long_side is not None:
            long_side = max(h0, w0)
            scale = self.target_long_side / long_side
            if scale < 1.0:
                t = self._lanczos_downsample(t, scale)
            # aggiorna meta
            meta["scale_factor"] = scale
        else:
            meta["scale_factor"] = 1.0

        t = t.squeeze(0)  # (3, Hs, Ws)
        meta["H_s"] = t.shape[1]
        meta["W_s"] = t.shape[2]

        # Normalizzazione ImageNet (§6.2.10)
        if self.normalize_imagenet:
            mean = _IMAGENET_MEAN.to(t.device)[:, None, None]
            std  = _IMAGENET_STD.to(t.device)[:, None, None]
            t = (t - mean) / std
        else:
            t = t.clamp(0.0, 1.0)

        return t, meta

    # ── Downsampling Lanczos ──────────────────────────────────────────────────

    def _lanczos_downsample(
        self, img: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """
        Downsampling anti-aliasing con filtro Lanczos 2-D separabile.

        Implementazione:
          1. Convoluzione separata per canale con kernel Lanczos 2-D
          2. Ricampionamento bilineare alle dimensioni target
             (bilinear è preciso al subpixel dopo il pre-filtraggio)

        Args:
            img:   Tensore float32 (1, 3, H, W).
            scale: Fattore di scala (0 < scale < 1).

        Returns:
            Tensore float32 (1, 3, Hs, Ws).
        """
        _, _, h, w = img.shape
        h_new = round(h * scale)
        w_new = round(w * scale)
        h_new = max(h_new, 1)
        w_new = max(w_new, 1)

        # Kernel Lanczos (1, 1, K, K)
        kernel = _build_lanczos_2d(scale, self.lanczos_a).to(img.device)
        k_size = kernel.shape[-1]
        pad    = k_size // 2

        # Applica separatamente per ogni canale (depthwise)
        # Raggruppa i 3 canali come batch di dimensione 3
        img_pad = F.pad(img, [pad, pad, pad, pad], mode="reflect")
        # img_pad: (1, 3, H+2p, W+2p)
        # Usa grouped conv con groups=3
        kernel3 = kernel.expand(3, 1, k_size, k_size)  # (3,1,K,K)
        smoothed = F.conv2d(img_pad, kernel3, padding=0, groups=3)

        # Ricampiona alle dimensioni esatte
        out = F.interpolate(
            smoothed,
            size=(h_new, w_new),
            mode="bilinear",
            align_corners=False,
        )
        return out

    # ── Utility: carica target JPEG/TIFF ─────────────────────────────────────

    def load_target(
        self, path: str, scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Carica un'immagine target (JPEG/TIFF editato) come tensore sRGB [0,1].

        Non normalizza con ImageNet (il target deve restare in [0,1]).
        Applica il downsampling allo stesso scale_factor del sorgente.

        Args:
            path:  Percorso immagine target.
            scale: Se fornito, downsampling con questo fattore; altrimenti
                   usa self.target_long_side per calcolare lo scale.

        Returns:
            Tensore float32 (3, Hs, Ws) in [0,1].
        """
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        img = np.array(pil, dtype=np.float32) / 255.0
        h0, w0 = img.shape[:2]

        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        if scale is None and self.target_long_side is not None:
            scale = self.target_long_side / max(h0, w0)

        if scale is not None and scale < 1.0:
            t = self._lanczos_downsample(t, scale)

        return t.squeeze(0).clamp(0.0, 1.0)