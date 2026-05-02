# %% [markdown]
# # RAG-ColorNet — Fase 3: Few-Shot Adaptation
#
# Questo notebook esegue l'adattamento del modello alle foto di un
# fotografo specifico. Non richiede pre-training né meta-training:
# parte da pesi random e adatta direttamente sulle coppie fornite.
#
# **Struttura del dataset attesa:**
# ```
# data/photographers/photographer_01/
#   src/   ← immagini sorgente (.png)
#   tgt/   ← immagini editate dal fotografo (.png)
# ```
# I filename devono corrispondere per stem (es. `001.png` in entrambe).
#
# **Indice:**
# 1. Setup e configurazione
# 2. Verifica del dataset
# 3. Pre-elaborazione: cache DINOv2 (one-shot, ~20-30 min)
# 4. K-Means e inizializzazione cluster
# 5. Step 1 — Adaptation parziale (epoche 1-10)
# 6. Step 2 — Full fine-tuning (epoche 11-30)
# 7. Visualizzazione risultati
# 8. Salvataggio checkpoint e database

# %% [markdown]
# ## 1. Setup e configurazione

# %%
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
import time, json, yaml, copy
from tqdm.notebook import tqdm

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ─── CONFIGURA QUI ────────────────────────────────────────────────────────────
PHOTOGRAPHER_ID  = "photographer_01"
PAIRS_DIR        = Path(f"../data/photographers/{PHOTOGRAPHER_ID}")
SRC_SUBDIR       = "src"
TGT_SUBDIR       = "tgt"

TRAIN_RESOLUTION = (448, 448)   # (H, W) — multiplo di 14 per DINOv2
MAX_IMG_SIZE     = 448

# Checkpoints
CHECKPOINT_DIR   = Path("../checkpoints")
DB_DIR           = Path(f"../memory/{PHOTOGRAPHER_ID}")
CACHE_DIR        = Path(f"../cache/{PHOTOGRAPHER_ID}")

# Training
STEP1_EPOCHS     = 10
STEP2_EPOCHS     = 20
BATCH_SIZE       = 1           # su CPU: sempre 1
GRAD_CLIP        = 1.0

# Mostra una visualizzazione ogni N epoche
VIS_EVERY        = 5
# ──────────────────────────────────────────────────────────────────────────────

# Crea le directory necessarie
for d in [CHECKPOINT_DIR, DB_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\nPhotographer : {PHOTOGRAPHER_ID}")
print(f"Pairs dir    : {PAIRS_DIR}")
print(f"Device       : {DEVICE}")

# %% [markdown]
# ## 1b. Caricamento config e fix automatici

# %%
def load_and_fix_config(base_path: str, override_path: str = None) -> dict:
    """
    Carica base.yaml + photographer.yaml con merge ricorsivo.
    Applica fix automatici per discrepanze note tra YAML e codice.
    """
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override)

    # Fix 1: loss.epsilon → loss.eps (il codice legge "eps")
    if "epsilon" in cfg.get("loss", {}):
        cfg["loss"]["eps"] = cfg["loss"].pop("epsilon")

    # Fix 2: loss.pretrain deve avere tutte le chiavi di LossWeights
    # (alcune mancano nel YAML base — vanno a zero di default)
    all_loss_keys = [
        "delta_e", "l1_lab", "histogram", "perceptual",
        "chroma", "cluster", "retrieval", "tv", "luminance", "entropy"
    ]
    for stage in ["pretrain", "curriculum"]:
        if stage in cfg.get("loss", {}):
            if stage == "pretrain":
                for k in all_loss_keys:
                    cfg["loss"]["pretrain"].setdefault(k, 0.0)
            else:
                for substage in ["early", "mid", "late"]:
                    if substage in cfg["loss"]["curriculum"]:
                        for k in all_loss_keys:
                            cfg["loss"]["curriculum"][substage].setdefault(k, 0.0)

    # Fix 3: hardware.fp16 = False su CPU (autocast su CPU richiede PyTorch >= 2.0)
    if DEVICE == "cpu":
        cfg.setdefault("hardware", {})["fp16"] = False

    # Fix 4: percorsi fotografo
    cfg.setdefault("data", {})["pairs_dir"] = str(PAIRS_DIR)
    cfg["data"]["src_subdir"] = SRC_SUBDIR
    cfg["data"]["tgt_subdir"] = TGT_SUBDIR
    cfg["data"]["train_resolution"] = list(TRAIN_RESOLUTION)
    cfg["data"]["keep_aspect"] = True
    cfg["data"]["val_split"] = 0.20
    cfg["data"]["val_split_seed"] = 42

    # Fix 5: output paths
    cfg.setdefault("output", {})["photographer_db_path"] = str(DB_DIR)
    cfg["output"]["adapted_checkpoint"] = str(CHECKPOINT_DIR / f"{PHOTOGRAPHER_ID}_adapted.pth")

    # Fix 6: adaptation defaults
    cfg.setdefault("adaptation", {}).setdefault("batch_size", BATCH_SIZE)
    cfg["adaptation"].setdefault("step1_epochs", STEP1_EPOCHS)
    cfg["adaptation"].setdefault("step2_epochs", STEP2_EPOCHS)
    cfg["adaptation"].setdefault("lr_step1", 5e-5)
    cfg["adaptation"].setdefault("lr_step2", 2.5e-5)
    cfg["adaptation"].setdefault("curriculum_early_end", 5)
    cfg["adaptation"].setdefault("curriculum_mid_end", 10)

    cfg.setdefault("early_stopping", {}).setdefault("patience", 5)
    cfg["early_stopping"].setdefault("min_delta", 0.01)
    cfg["early_stopping"].setdefault("mode", "min")

    cfg.setdefault("incremental", {}).setdefault("recluster_every", 50)
    cfg.setdefault("scheduler", {}).setdefault("warmup_epochs", 2)
    cfg["scheduler"].setdefault("eta_min", 1e-7)

    return cfg

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# Carica configurazione
BASE_CFG         = "../configs/base.yaml"
PHOTOGRAPHER_CFG = "../configs/photographer.yaml"

cfg = load_and_fix_config(BASE_CFG, PHOTOGRAPHER_CFG)
print("Config caricata e corretta.")
print(f"  FP16 attivo: {cfg['hardware']['fp16']}")
print(f"  Curriculum early end: {cfg['adaptation']['curriculum_early_end']}")
print(f"  Curriculum mid end:   {cfg['adaptation']['curriculum_mid_end']}")

# %% [markdown]
# ## 2. Verifica del dataset

# %%
from data.photographer_dataset import PhotographerDataset, split_dataset
from data.raw_pipeline import load_image

def load_tensor_from_path(path: Path, max_size: int = MAX_IMG_SIZE) -> torch.Tensor:
    """Carica un'immagine PNG come tensor (3, H, W) float32 [0,1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

# Verifica che la struttura delle directory sia corretta
src_dir = PAIRS_DIR / SRC_SUBDIR
tgt_dir = PAIRS_DIR / TGT_SUBDIR
assert src_dir.exists(), f"Directory src non trovata: {src_dir}"
assert tgt_dir.exists(), f"Directory tgt non trovata: {tgt_dir}"

src_files = sorted(src_dir.glob("*.png"))
tgt_files = sorted(tgt_dir.glob("*.png"))
assert len(src_files) > 0, f"Nessun file .png in {src_dir}"

# Verifica corrispondenza per stem
src_stems = {p.stem for p in src_files}
tgt_stems = {p.stem for p in tgt_dir.glob("*.png")}
matched   = src_stems & tgt_stems
unmatched_src = src_stems - tgt_stems
unmatched_tgt = tgt_stems - src_stems

print(f"Coppie trovate   : {len(matched)}")
if unmatched_src:
    print(f"  ⚠ src senza tgt : {sorted(unmatched_src)[:5]}")
if unmatched_tgt:
    print(f"  ⚠ tgt senza src : {sorted(unmatched_tgt)[:5]}")

assert len(matched) >= 10, \
    f"Servono almeno 10 coppie, trovate {len(matched)}"

# Dataset completo e split train/val
full_dataset = PhotographerDataset(
    pairs_dir   = PAIRS_DIR,
    src_subdir  = SRC_SUBDIR,
    tgt_subdir  = TGT_SUBDIR,
    extensions  = [".png"],
    target_size = TRAIN_RESOLUTION,
    keep_aspect = True,
)
train_sub, val_sub = split_dataset(full_dataset, val_fraction=0.20, seed=42)

print(f"\nDataset totale   : {len(full_dataset)} coppie")
print(f"  Train          : {len(train_sub)}")
print(f"  Val            : {len(val_sub)}")

# Visualizza 3 coppie di esempio
n_show = min(3, len(full_dataset))
fig, axes = plt.subplots(n_show, 2, figsize=(10, 4 * n_show))
if n_show == 1:
    axes = [axes]
for i in range(n_show):
    item = full_dataset[i]
    for j, (img_t, label) in enumerate([(item["src"], "src"), (item["tgt"], "tgt")]):
        axes[i][j].imshow(img_t.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i][j].set_title(f"{label} [{item['meta']['src_file']}]", fontsize=9)
        axes[i][j].axis("off")
plt.suptitle("Coppie di esempio", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Pre-elaborazione: cache DINOv2
#
# **Questo step è il più lungo (~20-30 min su CPU per 500 immagini)
# ma viene eseguito una sola volta.**
# Calcola e salva su disco:
# - `F_sem` — patch features DINOv2 di src e tgt
# - `Q`     — query descriptor di src
# - `h`     — color histogram Lab di src
#
# Se la cache esiste già, viene saltato automaticamente.

# %%
from models.scene_encoder import SceneEncoder

# Crea lo SceneEncoder (solo per il preprocessing — DINOv2 frozen)
print("Caricamento SceneEncoder (DINOv2)...")
scene_enc = SceneEncoder(
    embed_dim   = cfg["encoder"]["embed_dim"],
    patch_size  = cfg["encoder"]["patch_size"],
    n_bins      = cfg["histogram"]["n_bins"],
    sigma_scale = cfg["histogram"]["sigma_scale"],
    chroma_dim  = cfg["descriptor"]["chroma_dim"],
).to(DEVICE)
scene_enc.eval()
print(f"  DINOv2 caricato — parametri frozen: "
      f"{sum(p.numel() for p in scene_enc.backbone.parameters()):,}")

CACHE_FILE = CACHE_DIR / "dino_cache.pt"

if CACHE_FILE.exists():
    print(f"\nCache trovata: {CACHE_FILE}  — skip preprocessing")
    cache = torch.load(CACHE_FILE, map_location="cpu")
    print(f"  Coppie in cache: {len(cache)}")
else:
    print(f"\nCache non trovata — avvio preprocessing ({len(full_dataset)} coppie)...")
    cache = {}
    t0 = time.time()

    for i in tqdm(range(len(full_dataset)), desc="Preprocessing DINOv2"):
        src_path, tgt_path = full_dataset._pairs[i]

        # Carica a risoluzione di training
        src_t = load_image(src_path, target_size=TRAIN_RESOLUTION, keep_aspect=True)
        tgt_t = load_image(tgt_path, target_size=TRAIN_RESOLUTION, keep_aspect=True)

        src_t = src_t.unsqueeze(0).to(DEVICE)
        tgt_t = tgt_t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            enc_src = scene_enc(src_t)
            # Per tgt serve solo F_sem (per calcolare l'edit signature)
            F_sem_tgt, n_h_tgt, n_w_tgt = scene_enc.extract_patch_features(tgt_t)

        cache[i] = {
            # Src
            "F_sem_src": enc_src["F_sem"].squeeze(0).cpu().half(),  # (N, 384) fp16
            "Q_src":     enc_src["Q"].squeeze(0).cpu().half(),      # (N, 416) fp16
            "h_src":     enc_src["h"].squeeze(0).cpu(),              # (192,)   fp32
            "n_h":       enc_src["n_h"],
            "n_w":       enc_src["n_w"],
            # Tgt
            "F_sem_tgt": F_sem_tgt.squeeze(0).cpu().half(),         # (N, 384) fp16
            # Paths (per debug)
            "src_file":  src_path.name,
            "tgt_file":  tgt_path.name,
        }

    torch.save(cache, CACHE_FILE)
    elapsed = time.time() - t0
    cache_mb = CACHE_FILE.stat().st_size / (1024**2)
    print(f"\nPreprocessing completato in {elapsed/60:.1f} min")
    print(f"Cache salvata: {CACHE_FILE}  ({cache_mb:.1f} MB)")

# Stima occupazione RAM
n_patches_avg = cache[0]["F_sem_src"].shape[0]
ram_mb = len(cache) * n_patches_avg * (384 + 416 + 384) * 2 / (1024**2)
print(f"\nPatch medie per immagine : {n_patches_avg}")
print(f"Occupazione RAM stimata  : {ram_mb:.0f} MB")

# %% [markdown]
# ## 4. K-Means e inizializzazione cluster

# %%
from utils.kmeans_init import elbow_kmeans
from models.cluster_net import ClusterNet

print("Calcolo K-Means sui color histograms...")

# Raccoglie tutti gli istogrammi dalla cache
all_histograms = np.stack([
    cache[i]["h_src"].numpy() for i in range(len(full_dataset))
], axis=0)  # (N, 192)

print(f"  Histograms shape: {all_histograms.shape}")

k_max = cfg["cluster"]["k_max"]
tau   = cfg["cluster"]["elbow_tau"]

k_star, centroids, assignments = elbow_kmeans(
    all_histograms,
    k_max = min(k_max, len(full_dataset) // 3),  # K max = N/3 per stabilità
    tau   = tau,
    seed  = 42,
)

print(f"\nK* ottimale: {k_star}")
print(f"Distribuzione cluster:")
for k in range(k_star):
    n_k = (assignments == k).sum()
    print(f"  Cluster {k}: {n_k} immagini  ({n_k/len(assignments)*100:.0f}%)")

# Inizializza ClusterNet con i centroidi K-Means
cluster_net_init = ClusterNet(
    input_dim  = 192,
    hidden_dim = cfg["cluster"]["hidden_dim"],
    n_clusters = k_star,
)
cluster_net_init.reinitialise_from_centroids(
    torch.from_numpy(centroids).float()
)

# Mappa pair_idx → cluster_id (per ClusterAssignmentLoss)
cluster_labels_map = {i: int(assignments[i]) for i in range(len(full_dataset))}

# Visualizza distribuzione cluster
fig, ax = plt.subplots(figsize=(max(6, k_star), 3))
counts = [(assignments == k).sum() for k in range(k_star)]
ax.bar(range(k_star), counts, color="steelblue", alpha=0.8)
ax.set_xlabel("Cluster")
ax.set_ylabel("N immagini")
ax.set_title(f"Distribuzione cluster K*={k_star}")
ax.set_xticks(range(k_star))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4b. Costruzione database e FAISS

# %%
from memory.database    import PhotographerDatabase
from memory.faiss_index import FAISSIndexManager

print("Costruzione database del fotografo...")

database = PhotographerDatabase(
    n_clusters      = k_star,
    desc_dim        = cfg["descriptor"]["output_dim"],
    edit_dim        = cfg["encoder"]["embed_dim"],
    photographer_id = PHOTOGRAPHER_ID,
)
database.centroids = centroids.astype(np.float32)

# Popola il database con le rappresentazioni dalla cache
for i in range(len(full_dataset)):
    key   = cache[i]["Q_src"].float()       # (N, 416)
    value = (cache[i]["F_sem_tgt"].float()  # edit signature: Δ F_sem
           - cache[i]["F_sem_src"].float())  # (N, 384)
    hist  = cache[i]["h_src"]               # (192,)
    cluster_id = int(assignments[i])

    database.clusters[cluster_id].add(
        key       = key,
        value     = value,
        meta      = {"pair_idx": i, "src_file": cache[i]["src_file"]},
        histogram = hist,
    )

print(f"Database costruito: {len(database)} coppie")
print(f"  Cluster sizes: {database.cluster_sizes()}")
print(f"  RAM stimata: {database.memory_usage_mb():.1f} MB")

# Indice FAISS
faiss_mgr = FAISSIndexManager(
    n_clusters = k_star,
    desc_dim   = cfg["descriptor"]["output_dim"],
    nlist      = cfg["retrieval"]["faiss_nlist"],
    nprobe     = cfg["retrieval"]["faiss_nprobe"],
    pq_m       = cfg["retrieval"]["faiss_pq_m"],
)
faiss_mgr.build_from_database(database)
faiss_status = faiss_mgr.status()
print(f"\nFAISS status:")
for k, s in faiss_status.items():
    mode = "FAISS IVF-PQ" if s["is_faiss"] else "brute-force"
    print(f"  Cluster {k}: {s['n_images']} img  ({mode})")

# %% [markdown]
# ## 5. Costruzione del modello

# %%
from models.rag_colornet import RAGColorNet
from losses.composite_loss import CompositeLoss
from training.early_stopping import EarlyStopping
from training.lr_scheduler import build_scheduler, WarmupCosineScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

print("Costruzione modello RAGColorNet...")

model = RAGColorNet.from_config(cfg, n_clusters=k_star).to(DEVICE)

# Sostituisci il ClusterNet con quello inizializzato da K-Means
model.replace_cluster_net(cluster_net_init.to(DEVICE))

print(model.summary())

# Loss composita
loss_fn = CompositeLoss.from_config(cfg, backbone=model.scene_encoder.backbone)

# DataLoader train e val
train_loader = DataLoader(
    train_sub,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 0,        # 0 su CPU per evitare overhead
    pin_memory  = False,
)
val_loader = DataLoader(
    val_sub,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 0,
    pin_memory  = False,
)
print(f"\nDataLoader pronto")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")

# %% [markdown]
# ## 5b. Funzioni helper per il training loop
#
# Versione ottimizzata che usa la **cache DINOv2** invece di
# ricalcolare il backbone ad ogni batch — riduce i tempi da ~3s/img a ~0.1s/img.

# %%
def build_cluster_db_from_cache(
    query_hist: torch.Tensor,   # (B, 192)
    top_m: int = 10,
) -> dict:
    """Costruisce cluster_db usando il database e FAISS già costruiti."""
    return database.get_cluster_db(
        query_hist = query_hist,
        top_m      = top_m,
        device     = DEVICE,
    )

def train_epoch_cached(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    loss_fn:     CompositeLoss,
    scaler:      GradScaler,
    phase:       str,
    epoch:       int,
    top_m:       int = 10,
) -> dict:
    """
    Epoca di training che usa la cache DINOv2.
    Per ogni batch: costruisce cluster_db dal database pre-calcolato,
    poi esegue forward/backward solo sui moduli trainable.
    """
    model.train()
    model.scene_encoder.backbone.eval()  # DINOv2 sempre frozen

    weights = loss_fn.update_curriculum(phase, epoch)
    accum = {}
    n_batches = 0
    fp16_enabled = cfg["hardware"]["fp16"]

    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)
        idxs = batch.get("idx")

        optimizer.zero_grad(set_to_none=True)

        # Costruisce cluster_db
        with torch.no_grad():
            h = model.scene_encoder.histogram(src)  # (B, 192)
        cluster_db = build_cluster_db_from_cache(h, top_m=top_m)

        # Cluster labels per ClusterAssignmentLoss
        cluster_labels = None
        if idxs is not None and weights.cluster > 0:
            cluster_labels = torch.tensor(
                [cluster_labels_map.get(int(i), 0) for i in idxs],
                dtype=torch.long, device=DEVICE,
            )

        from torch.cuda.amp import autocast
        with autocast(enabled=fp16_enabled):
            out = model(src, cluster_db)
            breakdown = loss_fn(
                model_output   = out,
                batch          = {"src": src, "tgt": tgt},
                cluster_labels = cluster_labels,
            )

        if fp16_enabled:
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            breakdown.total.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
            optimizer.step()

        for k, v in breakdown.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}
    metrics["n_batches"] = n_batches
    return metrics

@torch.no_grad()
def val_epoch_cached(
    model:    nn.Module,
    loader:   DataLoader,
    loss_fn:  CompositeLoss,
    phase:    str,
    epoch:    int,
    top_m:    int = 10,
) -> dict:
    """Epoca di validazione con cache."""
    model.eval()
    fp16_enabled = cfg["hardware"]["fp16"]
    loss_fn.update_curriculum(phase, epoch)
    accum = {}
    n_batches = 0

    from torch.cuda.amp import autocast
    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        with torch.no_grad():
            h = model.scene_encoder.histogram(src)
        cluster_db = build_cluster_db_from_cache(h, top_m=top_m)

        with autocast(enabled=fp16_enabled):
            out = model(src, cluster_db)
            breakdown = loss_fn(
                model_output = out,
                batch        = {"src": src, "tgt": tgt},
            )

        for k, v in breakdown.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}
    metrics["n_batches"] = n_batches
    return metrics

def show_progress(model, epoch, title=""):
    """Visualizza src → pred vs tgt su 2 immagini di validazione."""
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    n_show = min(2, len(val_sub))

    with torch.no_grad():
        for row in range(n_show):
            item = val_sub[row]
            src_t = item["src"].unsqueeze(0).to(DEVICE)
            tgt_t = item["tgt"]

            h = model.scene_encoder.histogram(src_t)
            cluster_db = build_cluster_db_from_cache(h)
            out = model(src_t, cluster_db)

            pred = out["I_out"][0].cpu().clamp(0, 1)
            alpha = out["alpha"][0, 0].cpu()
            p = out["p"][0].cpu()

            for col, (img, lbl) in enumerate([
                (item["src"], "src"),
                (pred,        f"pred (ep{epoch})"),
                (tgt_t,       "tgt"),
            ]):
                axes[row][col].imshow(img.permute(1,2,0).clamp(0,1).numpy())
                axes[row][col].set_title(lbl, fontsize=9)
                axes[row][col].axis("off")

    plt.suptitle(title or f"Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Step 1 — Adaptation parziale (epoche 1–10)
#
# Parametri frozen: DINOv2 + layer early del GridNet (`dino_proj`, `fusion_conv`).
# Parametri trainable: ClusterNet, W_Q/W_K/W_V, ultimi layer GridNet, MaskNet.

# %%
# Imposta freeze parziale
model.set_adaptation_mode(step=1)
n_trainable_step1 = model.count_trainable_params()
print(f"Step 1 — parametri trainable: {n_trainable_step1:,}")

lr_step1   = cfg["adaptation"]["lr_step1"]
optimizer1 = torch.optim.AdamW(
    list(model.trainable_parameters()),
    lr           = lr_step1,
    betas        = tuple(cfg["optimizer"]["betas"]),
    eps          = cfg["optimizer"]["eps"],
    weight_decay = cfg["optimizer"]["weight_decay"],
)
scheduler1 = WarmupCosineScheduler(
    optimizer1,
    warmup_epochs = 1,
    max_epochs    = STEP1_EPOCHS,
    eta_min       = cfg["scheduler"]["eta_min"],
)
scaler1 = GradScaler(enabled=cfg["hardware"]["fp16"])

stopper1 = EarlyStopping(
    patience        = cfg["early_stopping"]["patience"],
    min_delta       = cfg["early_stopping"]["min_delta"],
    mode            = "min",
    checkpoint_path = CHECKPOINT_DIR / f"{PHOTOGRAPHER_ID}_step1_best.pth",
    verbose         = True,
)

# Storico delle metriche per il plotting finale
history = {
    "train_loss": [], "val_loss": [],
    "train_l1":   [], "val_l1":   [],
}

print(f"\n{'='*60}")
print(f"STEP 1 — {STEP1_EPOCHS} epoche, lr={lr_step1:.1e}")
print(f"{'='*60}\n")

for epoch in range(1, STEP1_EPOCHS + 1):
    t0 = time.time()

    train_m = train_epoch_cached(
        model, train_loader, optimizer1, loss_fn, scaler1,
        phase="adapt", epoch=epoch,
    )
    val_m = val_epoch_cached(
        model, val_loader, loss_fn,
        phase="adapt", epoch=epoch,
    )
    scheduler1.step()

    elapsed = time.time() - t0

    # Log
    tr_loss = train_m.get("loss/total", 0)
    va_loss = val_m.get("loss/total", 0)
    tr_l1   = train_m.get("loss/l1_lab", 0)
    va_l1   = val_m.get("loss/l1_lab", 0)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_l1"].append(tr_l1)
    history["val_l1"].append(va_l1)

    print(
        f"  Ep {epoch:2d}/{STEP1_EPOCHS}  "
        f"train={tr_loss:.4f}  val={va_loss:.4f}  "
        f"l1={tr_l1:.4f}  "
        f"lr={optimizer1.param_groups[0]['lr']:.2e}  "
        f"{elapsed:.1f}s"
    )

    # Salva miglior checkpoint step1
    stopper1.step(va_loss, model=model, epoch=epoch, extras={"k_star": k_star})
    if stopper1.should_stop:
        print(f"  ⏹ Early stopping step 1 @ epoch {epoch}")
        break

    # Visualizzazione periodica
    if epoch % VIS_EVERY == 0 or epoch == STEP1_EPOCHS:
        show_progress(model, epoch, title=f"Step 1 — Epoch {epoch}")

print("\n✓ Step 1 completato")

# %% [markdown]
# ## 7. Step 2 — Full fine-tuning (epoche 11–30)
#
# Tutti i parametri trainable sbloccati (DINOv2 sempre frozen).
# LR ridotto, cosine annealing.

# %%
# Sblocca tutti i parametri trainable
model.set_adaptation_mode(step=2)
n_trainable_step2 = model.count_trainable_params()
print(f"Step 2 — parametri trainable: {n_trainable_step2:,}")

lr_step2   = cfg["adaptation"]["lr_step2"]
optimizer2 = torch.optim.AdamW(
    list(model.trainable_parameters()),
    lr           = lr_step2,
    betas        = tuple(cfg["optimizer"]["betas"]),
    eps          = cfg["optimizer"]["eps"],
    weight_decay = cfg["optimizer"]["weight_decay"],
)
scheduler2 = WarmupCosineScheduler(
    optimizer2,
    warmup_epochs = 1,
    max_epochs    = STEP2_EPOCHS,
    eta_min       = cfg["scheduler"]["eta_min"],
)
scaler2 = GradScaler(enabled=cfg["hardware"]["fp16"])

stopper2 = EarlyStopping(
    patience        = cfg["early_stopping"]["patience"],
    min_delta       = cfg["early_stopping"]["min_delta"],
    mode            = "min",
    checkpoint_path = CHECKPOINT_DIR / f"{PHOTOGRAPHER_ID}_adapted.pth",
    verbose         = True,
)

print(f"\n{'='*60}")
print(f"STEP 2 — {STEP2_EPOCHS} epoche, lr={lr_step2:.1e}")
print(f"{'='*60}\n")

for epoch in range(STEP1_EPOCHS + 1, STEP1_EPOCHS + STEP2_EPOCHS + 1):
    t0 = time.time()

    train_m = train_epoch_cached(
        model, train_loader, optimizer2, loss_fn, scaler2,
        phase="adapt", epoch=epoch,
    )
    val_m = val_epoch_cached(
        model, val_loader, loss_fn,
        phase="adapt", epoch=epoch,
    )
    scheduler2.step()

    elapsed = time.time() - t0

    tr_loss = train_m.get("loss/total", 0)
    va_loss = val_m.get("loss/total", 0)
    tr_de   = train_m.get("loss/delta_e", 0)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_l1"].append(train_m.get("loss/l1_lab", 0))
    history["val_l1"].append(val_m.get("loss/l1_lab", 0))

    print(
        f"  Ep {epoch:2d}/{STEP1_EPOCHS+STEP2_EPOCHS}  "
        f"train={tr_loss:.4f}  val={va_loss:.4f}  "
        f"ΔE={tr_de:.4f}  "
        f"lr={optimizer2.param_groups[0]['lr']:.2e}  "
        f"{elapsed:.1f}s"
    )

    stopper2.step(va_loss, model=model, epoch=epoch, extras={"k_star": k_star})
    if stopper2.should_stop:
        print(f"  ⏹ Early stopping step 2 @ epoch {epoch}")
        break

    if epoch % VIS_EVERY == 0 or epoch == STEP1_EPOCHS + STEP2_EPOCHS:
        show_progress(model, epoch, title=f"Step 2 — Epoch {epoch}")

print("\n✓ Step 2 completato")

# %% [markdown]
# ## 8. Curva di loss e visualizzazione finale

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

epochs_all = list(range(1, len(history["train_loss"]) + 1))

axes[0].plot(epochs_all, history["train_loss"], label="train total", linewidth=2)
axes[0].plot(epochs_all, history["val_loss"],   label="val total",   linewidth=2, linestyle="--")
axes[0].axvline(x=STEP1_EPOCHS, color="gray", linestyle=":", alpha=0.7, label="step1→step2")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss totale")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_all, history["train_l1"], label="train L1Lab", linewidth=2)
axes[1].plot(epochs_all, history["val_l1"],   label="val L1Lab",   linewidth=2, linestyle="--")
axes[1].axvline(x=STEP1_EPOCHS, color="gray", linestyle=":", alpha=0.7)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("L1 Lab")
axes[1].set_title("L1 Lab (warm-up)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Training — {PHOTOGRAPHER_ID}", fontsize=13)
plt.tight_layout()
plt.show()

# Visualizzazione finale su tutte le immagini di validazione
print("Visualizzazione risultati finali sul val set:")
model.eval()

# Carica il miglior checkpoint
best_ckpt = CHECKPOINT_DIR / f"{PHOTOGRAPHER_ID}_adapted.pth"
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Caricato best checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val loss {ckpt.get('metric', '?'):.4f}")

n_val_show = min(4, len(val_sub))
fig, axes = plt.subplots(n_val_show, 3, figsize=(13, 4 * n_val_show))
if n_val_show == 1:
    axes = [axes]

with torch.no_grad():
    for row in range(n_val_show):
        item = val_sub[row]
        src_t = item["src"].unsqueeze(0).to(DEVICE)
        tgt_t = item["tgt"]

        h = model.scene_encoder.histogram(src_t)
        cluster_db = build_cluster_db_from_cache(h)
        out = model(src_t, cluster_db)

        pred = out["I_out"][0].cpu().clamp(0, 1)
        p    = out["p"][0].cpu()
        dom_k = int(p.argmax().item())

        # MAE per questa immagine
        mae = (pred - tgt_t).abs().mean().item()

        for col, (img, lbl) in enumerate([
            (item["src"], "src"),
            (pred,        f"pred  k={dom_k}  MAE={mae:.3f}"),
            (tgt_t,       "tgt (ground truth)"),
        ]):
            axes[row][col].imshow(img.permute(1,2,0).clamp(0,1).numpy())
            axes[row][col].set_title(lbl, fontsize=9)
            axes[row][col].axis("off")

plt.suptitle(f"Risultati finali — {PHOTOGRAPHER_ID}", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Salvataggio checkpoint e database

# %%
print("Salvataggio database del fotografo...")
database.save(DB_DIR)
print(f"  Database salvato: {DB_DIR}")
print(f"  Coppie: {len(database)}")
print(f"  Cluster: {database.cluster_sizes()}")

# Salva anche il K* nel checkpoint finale (necessario per il caricamento)
final_ckpt_path = CHECKPOINT_DIR / f"{PHOTOGRAPHER_ID}_adapted.pth"
if final_ckpt_path.exists():
    ckpt = torch.load(final_ckpt_path, map_location="cpu")
    ckpt["k_star"]           = k_star
    ckpt["photographer_id"]  = PHOTOGRAPHER_ID
    ckpt["n_pairs"]          = len(database)
    ckpt["train_resolution"] = list(TRAIN_RESOLUTION)
    torch.save(ckpt, final_ckpt_path)

print(f"\nCheckpoint finale: {final_ckpt_path}")

# Riepilogo
print(f"\n{'='*60}")
print(f"  ADAPTATION COMPLETATA — {PHOTOGRAPHER_ID}")
print(f"{'='*60}")
print(f"  Coppie di training   : {len(train_sub)}")
print(f"  Coppie di val        : {len(val_sub)}")
print(f"  K* cluster           : {k_star}")
print(f"  Parametri trainable  : {model.count_trainable_params():,}")
print(f"  Checkpoint           : {final_ckpt_path}")
print(f"  Database             : {DB_DIR}")
print(f"\nPer l'inferenza usa il notebook 02_inference.ipynb")
print(f"o direttamente:")
print(f"  from inference.grade import Grader")
print(f"  grader = Grader(checkpoint='{final_ckpt_path}', db_path='{DB_DIR}')")
print(f"  result = grader.grade('nuova_foto.png')")
