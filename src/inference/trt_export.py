"""
trt_export.py
-------------
Export del modello in formato ottimizzato per inferenza veloce.

Supporta due modalità:
  TorchScript  — export universale, ~2× speedup rispetto a eager mode
  TensorRT     — export NVIDIA, ~3-4× speedup su GPU, richiede torch2trt

In entrambi i casi il database del fotografo non viene esportato
(rimane in RAM) — solo la parte neurale del modello viene ottimizzata.

Uso:
    from inference.trt_export import export_torchscript, export_tensorrt

    # TorchScript (consigliato, nessuna dipendenza aggiuntiva)
    export_torchscript(
        checkpoint = "checkpoints/photographer_01_adapted.pth",
        output     = "checkpoints/photographer_01_scripted.pt",
    )

    # TensorRT (richiede torch2trt e TensorRT installati)
    export_tensorrt(
        checkpoint  = "checkpoints/photographer_01_adapted.pth",
        output      = "checkpoints/photographer_01_trt.pt",
        input_shape = (1, 3, 512, 384),
        int8        = False,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# export_torchscript
# ---------------------------------------------------------------------------

def export_torchscript(
    checkpoint:  Union[str, Path],
    output:      Union[str, Path],
    config:      Union[str, Path] = "configs/base.yaml",
    input_shape: Tuple[int, ...] = (1, 3, 512, 384),
    device:      str = "cuda",
    optimize:    bool = True,
) -> Path:
    """
    Esporta il modello in TorchScript (torch.jit.trace).

    TorchScript è il metodo più semplice e portabile — funziona
    su qualsiasi hardware senza dipendenze aggiuntive.
    Speedup tipico: 1.5–2× rispetto a eager mode.

    Nota: poiché il modello usa strutture dati dinamiche (cluster_db dict),
    si usa torch.jit.trace su un forward pass con database vuoto.
    In produzione il database viene passato a runtime come prima.

    Parameters
    ----------
    checkpoint   : checkpoint adattato al fotografo
    output       : path dove salvare il .pt scriptato
    config       : path al config base
    input_shape  : (B, C, H, W) shape dell'input di esempio per il trace
    device       : device per il trace
    optimize     : applica ottimizzazioni torch.jit.optimize_for_inference

    Returns
    -------
    output path
    """
    import yaml
    from models.rag_colornet import RAGColorNet

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Carica modello
    ckpt   = torch.load(checkpoint, map_location=device)
    k_star = ckpt.get("k_star", 8)
    model  = RAGColorNet.from_config(cfg, n_clusters=k_star)
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(device).eval()

    print(f"  Modello caricato — K*={k_star}")

    # Input di esempio e cluster_db vuoto per il trace
    example_input = torch.randn(*input_shape, device=device)
    empty_db      = {k: None for k in range(k_star)}

    print(f"  Tracing con input shape {input_shape}...")
    with torch.no_grad():
        try:
            scripted = torch.jit.trace(
                model,
                (example_input, empty_db),
                strict=False,
            )
        except Exception as e:
            warnings.warn(
                f"torch.jit.trace fallito ({e}). "
                "Il modello usa strutture dinamiche — "
                "considera torch.jit.script per i sottomoduli."
            )
            raise

    if optimize:
        scripted = torch.jit.optimize_for_inference(scripted)
        print("  ✓ Ottimizzazioni JIT applicate")

    scripted.save(str(output))
    size_mb = output.stat().st_size / (1024 ** 2)
    print(f"  ✓ TorchScript salvato: {output}  ({size_mb:.1f} MB)")

    # Benchmark rapido
    _benchmark(model, scripted, example_input, empty_db, device)

    return output


# ---------------------------------------------------------------------------
# export_tensorrt
# ---------------------------------------------------------------------------

def export_tensorrt(
    checkpoint:  Union[str, Path],
    output:      Union[str, Path],
    config:      Union[str, Path] = "configs/base.yaml",
    input_shape: Tuple[int, ...] = (1, 3, 512, 384),
    device:      str = "cuda",
    int8:        bool = False,
    fp16:        bool = True,
    workspace_mb: int = 4096,
) -> Path:
    """
    Esporta il modello in TensorRT tramite torch2trt.

    Richiede:
        pip install torch2trt
        TensorRT installato (disponibile su NVIDIA NGC containers)

    Speedup tipico rispetto a eager: 3–4× su GPU NVIDIA.
    int8 richiede calibrazione — usa fp16 per la maggior parte dei casi.

    Parameters
    ----------
    checkpoint   : checkpoint adattato
    output       : path output .pt
    config       : config base
    input_shape  : (B, C, H, W)
    device       : deve essere "cuda"
    int8         : quantizzazione int8 (richiede dataset di calibrazione)
    fp16         : quantizzazione fp16 (consigliato)
    workspace_mb : memoria workspace TensorRT in MB

    Returns
    -------
    output path
    """
    try:
        from torch2trt import torch2trt   # type: ignore
    except ImportError:
        raise ImportError(
            "torch2trt non trovato. Installa con:\n"
            "  git clone https://github.com/NVIDIA-AI-IOT/torch2trt\n"
            "  cd torch2trt && pip install ."
        )

    import yaml
    from models.rag_colornet import RAGColorNet

    assert device == "cuda", "TensorRT richiede device='cuda'"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    ckpt   = torch.load(checkpoint, map_location=device)
    k_star = ckpt.get("k_star", 8)
    model  = RAGColorNet.from_config(cfg, n_clusters=k_star)
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(device).eval()

    print(f"  Modello caricato — K*={k_star}")
    print(f"  FP16={fp16}, INT8={int8}, workspace={workspace_mb}MB")

    example_input = torch.randn(*input_shape, device=device)

    # TensorRT ottimizza solo la parte neurale senza il database
    # Usiamo un wrapper che fissa il cluster_db vuoto
    wrapper = _TRTWrapper(model, k_star).to(device).eval()

    print("  Compilazione TensorRT in corso (può richiedere qualche minuto)...")
    with torch.no_grad():
        model_trt = torch2trt(
            wrapper,
            [example_input],
            fp16_mode           = fp16,
            int8_mode           = int8,
            max_workspace_size  = workspace_mb * (1024 ** 2),
        )

    torch.save(model_trt.state_dict(), str(output))
    size_mb = output.stat().st_size / (1024 ** 2)
    print(f"  ✓ TensorRT engine salvato: {output}  ({size_mb:.1f} MB)")

    return output


# ---------------------------------------------------------------------------
# _TRTWrapper  (wrapper per TensorRT export)
# ---------------------------------------------------------------------------

class _TRTWrapper(nn.Module):
    """
    Wrapper che espone solo l'output I_out per il trace TensorRT.
    Il cluster_db è fissato a vuoto — usato solo per l'export,
    non per l'inferenza reale.
    """

    def __init__(self, model: nn.Module, k_star: int) -> None:
        super().__init__()
        self.model  = model
        self.k_star = k_star

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        empty_db = {k: None for k in range(self.k_star)}
        out      = self.model(x, empty_db)
        return out["I_out"]


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def _benchmark(
    model_eager:    nn.Module,
    model_scripted: nn.Module,
    example_input:  torch.Tensor,
    empty_db:       dict,
    device:         str,
    n_warmup: int = 5,
    n_runs:   int = 20,
) -> None:
    """Confronto latenza eager vs scripted."""
    import time

    def measure(fn, *args):
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000

    ms_eager    = measure(model_eager,    example_input, empty_db)
    ms_scripted = measure(model_scripted, example_input, empty_db)
    speedup     = ms_eager / ms_scripted

    print(f"\n  Benchmark ({n_runs} run, shape {tuple(example_input.shape)}):")
    print(f"  Eager      : {ms_eager:.1f} ms")
    print(f"  TorchScript: {ms_scripted:.1f} ms")
    print(f"  Speedup    : {speedup:.2f}×")
