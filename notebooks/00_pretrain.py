# %% [markdown]
# # RAG-ColorNet — 00: Pre-Training (Fase 1)
#
# Addestra i moduli trainable (W_Q, W_K, W_V, GridNet, MaskNet, guide MLP)
# su un dataset di coppie (src, tgt) PNG di un singolo fotografo.
#
# **In produzione** questa fase userebbe FiveK + PPR10K + preset Lightroom.
# **In questo test** usiamo le coppie di Lorenzo per verificare che
# l'intera pipeline funzioni end-to-end prima di scalare.
#
# **Output:** `checkpoints/pretrain_best.pth`
#
# **Durata stimata su CPU con cache DINOv2:**
# - Preprocessing (one-shot): ~2-3s/coppia
# - Training 50 epoche: ~1-2 min/epoca

# %% [markdown]
# ## 0. Imports e configurazione

# %%
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time, yaml
from tqdm.notebook import tqdm

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")

# ─── CONFIGURA QUI ────────────────────────────────────────────────────────────
PHOTOGRAPHER_ID  = "lorenzo"
PAIRS_DIR        = Path(f"../data/{PHOTOGRAPHER_ID}")
SRC_SUBDIR       = "src"
TGT_SUBDIR       = "tgt"

TRAIN_RESOLUTION = (448, 448)   # (H, W) — multiplo di 14
VAL_SPLIT        = 0.15         # frazione di validazione

N_EPOCHS         = 50
BATCH_SIZE       = 1            # 1 su CPU; aumentare se si ha GPU
GRAD_CLIP        = 1.0
VIS_EVERY        = 10           # mostra immagini ogni N epoche

CHECKPOINT_DIR   = Path("../checkpoints")
CACHE_DIR        = Path(f"../cache/{PHOTOGRAPHER_ID}")
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Caricamento config

# %%
def load_config(base_path: str, override_path: str = None) -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if override_path and Path(override_path).exists():
        with open(override_path) as f:
            ov = yaml.safe_load(f)
        cfg = _deep_merge(cfg, ov)
    _fix_config(cfg)
    return cfg

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def _fix_config(cfg: dict) -> None:
    """Applica fix per discrepanze note tra YAML e codice."""
    # loss.epsilon → loss.eps
    if "epsilon" in cfg.get("loss", {}):
        cfg["loss"]["eps"] = cfg["loss"].pop("epsilon")
    # Assicura tutte le chiavi di LossWeights in ogni stage
    all_keys = ["delta_e","l1_lab","histogram","perceptual","chroma",
                "cluster","retrieval","tv","luminance","entropy"]
    for stage in ["pretrain"]:
        for k in all_keys:
            cfg["loss"].setdefault(stage, {}).setdefault(k, 0.0)
    for stage in ["early","mid","late"]:
        for k in all_keys:
            cfg["loss"].setdefault("curriculum", {}).setdefault(stage, {}).setdefault(k, 0.0)
    # FP16 disabilitato su CPU
    if DEVICE == "cpu":
        cfg.setdefault("hardware", {})["fp16"] = False

cfg = load_config("../configs/base.yaml")
print("Config caricata.")
print(f"  FP16    : {cfg['hardware']['fp16']}")
print(f"  Encoder : {cfg['encoder']['backbone']}")

# %% [markdown]
# ## 2. Verifica dataset e split train/val

# %%
from data.photographer_dataset import PhotographerDataset, split_dataset

src_dir = PAIRS_DIR / SRC_SUBDIR
tgt_dir = PAIRS_DIR / TGT_SUBDIR
assert src_dir.exists(), f"Directory non trovata: {src_dir}"
assert tgt_dir.exists(), f"Directory non trovata: {tgt_dir}"

src_stems = {p.stem for p in src_dir.glob("*.png")}
tgt_stems = {p.stem for p in tgt_dir.glob("*.png")}
matched   = src_stems & tgt_stems
assert len(matched) >= 5, f"Servono almeno 5 coppie, trovate {len(matched)}"

unmatched = (src_stems - tgt_stems) | (tgt_stems - src_stems)
if unmatched:
    print(f"⚠  File senza corrispondenza: {sorted(unmatched)[:5]}")

full_dataset = PhotographerDataset(
    pairs_dir   = PAIRS_DIR,
    src_subdir  = SRC_SUBDIR,
    tgt_subdir  = TGT_SUBDIR,
    extensions  = [".png"],
    target_size = TRAIN_RESOLUTION,
    keep_aspect = True,
)
train_sub, val_sub = split_dataset(full_dataset, val_fraction=VAL_SPLIT, seed=42)

print(f"\nCoppie totali : {len(full_dataset)}")
print(f"  Train       : {len(train_sub)}")
print(f"  Val         : {len(val_sub)}")

# Visualizza 2 coppie di esempio
n_show = min(2, len(full_dataset))
fig, axes = plt.subplots(n_show, 2, figsize=(9, 4 * n_show))
if n_show == 1:
    axes = [axes]
for i in range(n_show):
    item = full_dataset[i]
    for j, (t, lbl) in enumerate([(item["src"],"src"),(item["tgt"],"tgt")]):
        axes[i][j].imshow(t.permute(1,2,0).clamp(0,1).numpy())
        axes[i][j].set_title(f"{lbl}  {item['meta']['src_file']}", fontsize=9)
        axes[i][j].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Cache DINOv2 (one-shot)
#
# DINOv2 è frozen — ha senso calcolarlo una volta sola e salvarlo.
# Se la cache esiste già viene saltato automaticamente.

# %%
from data.raw_pipeline import load_image
from models.scene_encoder import SceneEncoder

CACHE_FILE = CACHE_DIR / "dino_cache.pt"

scene_enc = SceneEncoder(
    embed_dim   = cfg["encoder"]["embed_dim"],
    patch_size  = cfg["encoder"]["patch_size"],
    n_bins      = cfg["histogram"]["n_bins"],
    sigma_scale = cfg["histogram"]["sigma_scale"],
    chroma_dim  = cfg["descriptor"]["chroma_dim"],
).to(DEVICE)
scene_enc.eval()

if CACHE_FILE.exists():
    print(f"Cache trovata: {CACHE_FILE} — skip preprocessing")
    cache = torch.load(CACHE_FILE, map_location="cpu")
    print(f"  Coppie in cache: {len(cache)}")
else:
    print(f"Avvio preprocessing DINOv2 su {len(full_dataset)} coppie...")
    cache = {}
    t0 = time.time()
    for i in tqdm(range(len(full_dataset)), desc="DINOv2 cache"):
        src_path, tgt_path = full_dataset._pairs[i]
        src_t = load_image(src_path, target_size=TRAIN_RESOLUTION, keep_aspect=True).unsqueeze(0).to(DEVICE)
        tgt_t = load_image(tgt_path, target_size=TRAIN_RESOLUTION, keep_aspect=True).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            enc_src = scene_enc(src_t)
            F_tgt, _, _ = scene_enc.extract_patch_features(tgt_t)
        cache[i] = {
            "F_sem_src": enc_src["F_sem"].squeeze(0).cpu().half(),
            "Q_src"    : enc_src["Q"].squeeze(0).cpu().half(),
            "h_src"    : enc_src["h"].squeeze(0).cpu(),
            "F_sem_tgt": F_tgt.squeeze(0).cpu().half(),
            "n_h"      : enc_src["n_h"],
            "n_w"      : enc_src["n_w"],
            "src_file" : src_path.name,
        }
    torch.save(cache, CACHE_FILE)
    print(f"Cache salvata in {time.time()-t0:.0f}s — {CACHE_FILE.stat().st_size/1e6:.1f} MB")

# %% [markdown]
# ## 4. K-Means e inizializzazione cluster

# %%
from utils.kmeans_init import elbow_kmeans
from models.cluster_net import ClusterNet

histograms = np.stack([cache[i]["h_src"].numpy() for i in range(len(full_dataset))], axis=0)

k_star, centroids, assignments = elbow_kmeans(
    histograms,
    k_max = min(cfg["cluster"]["k_max"], max(1, len(full_dataset) // 3)),
    tau   = cfg["cluster"]["elbow_tau"],
    seed  = 42,
)
print(f"K* = {k_star}")
for k in range(k_star):
    n_k = (assignments == k).sum()
    print(f"  Cluster {k}: {n_k} immagini ({n_k/len(assignments)*100:.0f}%)")

cluster_net = ClusterNet(
    input_dim  = 192,
    hidden_dim = cfg["cluster"]["hidden_dim"],
    n_clusters = k_star,
)
cluster_net.reinitialise_from_centroids(torch.from_numpy(centroids).float())

cluster_labels_map = {i: int(assignments[i]) for i in range(len(full_dataset))}

fig, ax = plt.subplots(figsize=(max(5, k_star), 3))
ax.bar(range(k_star), [(assignments==k).sum() for k in range(k_star)], color="steelblue", alpha=0.8)
ax.set_xlabel("Cluster"); ax.set_ylabel("N immagini"); ax.set_title(f"K*={k_star}")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Costruzione database

# %%
from memory.database    import PhotographerDatabase
from memory.faiss_index import FAISSIndexManager

database = PhotographerDatabase(
    n_clusters      = k_star,
    desc_dim        = cfg["descriptor"]["output_dim"],
    edit_dim        = cfg["encoder"]["embed_dim"],
    photographer_id = PHOTOGRAPHER_ID,
)
database.centroids = centroids.astype(np.float32)

for i in range(len(full_dataset)):
    key   = cache[i]["Q_src"].float()
    value = cache[i]["F_sem_tgt"].float() - cache[i]["F_sem_src"].float()
    database.clusters[int(assignments[i])].add(
        key       = key,
        value     = value,
        meta      = {"pair_idx": i, "src_file": cache[i]["src_file"]},
        histogram = cache[i]["h_src"],
    )

faiss_mgr = FAISSIndexManager(
    n_clusters = k_star,
    desc_dim   = cfg["descriptor"]["output_dim"],
    nlist      = cfg["retrieval"]["faiss_nlist"],
    nprobe     = cfg["retrieval"]["faiss_nprobe"],
    pq_m       = cfg["retrieval"]["faiss_pq_m"],
)
faiss_mgr.build_from_database(database)

print(f"Database: {len(database)} coppie")
print(f"  Cluster sizes: {database.cluster_sizes()}")
print(f"  RAM stimata  : {database.memory_usage_mb():.1f} MB")

# %% [markdown]
# ## 6. Modello e loss

# %%
from models.rag_colornet   import RAGColorNet
from losses.composite_loss import CompositeLoss
from training.lr_scheduler import WarmupCosineScheduler
from training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

model = RAGColorNet.from_config(cfg, n_clusters=k_star).to(DEVICE)
model.replace_cluster_net(cluster_net.to(DEVICE))
print(model.summary())

loss_fn = CompositeLoss.from_config(cfg, backbone=model.scene_encoder.backbone)

train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% [markdown]
# ## 7. Funzioni di training

# %%
def get_cluster_db(model, src_batch):
    """Costruisce cluster_db dal database pre-calcolato."""
    with torch.no_grad():
        h = model.scene_encoder.histogram(src_batch)
    return database.get_cluster_db(h, top_m=cfg["retrieval"]["top_m"], device=DEVICE)

def train_epoch(model, loader, optimizer, loss_fn, scaler, epoch):
    model.train(); model.scene_encoder.backbone.eval()
    weights = loss_fn.update_curriculum("pretrain", epoch)
    accum, n = {}, 0
    fp16 = cfg["hardware"]["fp16"]

    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)
        idxs = batch.get("idx")
        cluster_db = get_cluster_db(model, src)

        cluster_labels = None
        if idxs is not None and weights.cluster > 0:
            cluster_labels = torch.tensor(
                [cluster_labels_map.get(int(i), 0) for i in idxs],
                dtype=torch.long, device=DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=fp16):
            out = model(src, cluster_db)
            bd  = loss_fn(out, {"src": src, "tgt": tgt}, cluster_labels=cluster_labels)

        if fp16:
            scaler.scale(bd.total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            bd.total.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
            optimizer.step()

        for k, v in bd.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1

    return {k: v/max(n,1) for k, v in accum.items()}

@torch.no_grad()
def val_epoch(model, loader, loss_fn, epoch):
    model.eval()
    loss_fn.update_curriculum("pretrain", epoch)
    accum, n = {}, 0
    fp16 = cfg["hardware"]["fp16"]

    for batch in loader:
        src = batch["src"].to(DEVICE); tgt = batch["tgt"].to(DEVICE)
        cluster_db = get_cluster_db(model, src)
        with autocast(enabled=fp16):
            out = model(src, cluster_db)
            bd  = loss_fn(out, {"src": src, "tgt": tgt})
        for k, v in bd.as_loggable().items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1

    return {k: v/max(n,1) for k, v in accum.items()}

def visualize(model, epoch, n_show=2):
    model.eval()
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4*n_show))
    if n_show == 1: axes = [axes]
    with torch.no_grad():
        for row in range(n_show):
            item = val_sub[row]
            src_t = item["src"].unsqueeze(0).to(DEVICE)
            db    = get_cluster_db(model, src_t)
            out   = model(src_t, db)
            pred  = out["I_out"][0].cpu().clamp(0,1)
            mae   = (pred - item["tgt"]).abs().mean().item()
            for col, (img, lbl) in enumerate([
                (item["src"], "src"),
                (pred,        f"pred  MAE={mae:.3f}"),
                (item["tgt"], "tgt"),
            ]):
                axes[row][col].imshow(img.permute(1,2,0).clamp(0,1).numpy())
                axes[row][col].set_title(lbl, fontsize=9); axes[row][col].axis("off")
    plt.suptitle(f"Pretrain epoch {epoch}", fontsize=11)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Training loop

# %%
optimizer = torch.optim.AdamW(
    list(model.trainable_parameters()),
    lr           = 1e-4,
    betas        = tuple(cfg["optimizer"]["betas"]),
    eps          = cfg["optimizer"]["eps"],
    weight_decay = cfg["optimizer"]["weight_decay"],
)
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, max_epochs=N_EPOCHS, eta_min=1e-7)
scaler    = GradScaler(enabled=cfg["hardware"]["fp16"])
stopper   = EarlyStopping(
    patience        = 7,
    min_delta       = 0.005,
    mode            = "min",
    checkpoint_path = CHECKPOINT_DIR / "pretrain_best.pth",
    verbose         = True,
)

history = {"train": [], "val": []}

print(f"\n{'='*55}")
print(f"  PRE-TRAINING — {N_EPOCHS} epoche")
print(f"  Parametri trainable: {model.count_trainable_params():,}")
print(f"{'='*55}\n")

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch)
    va = val_epoch(model, val_loader, loss_fn, epoch)
    scheduler.step()

    tr_loss = tr.get("loss/total", 0)
    va_loss = va.get("loss/total", 0)
    history["train"].append(tr_loss)
    history["val"].append(va_loss)

    print(f"  Ep {epoch:3d}/{N_EPOCHS}  "
          f"train={tr_loss:.4f}  val={va_loss:.4f}  "
          f"lr={optimizer.param_groups[0]['lr']:.2e}  "
          f"{time.time()-t0:.1f}s")

    stopper.step(va_loss, model=model, epoch=epoch, extras={"k_star": k_star})
    if stopper.should_stop:
        print(f"\n  ⏹ Early stopping @ epoch {epoch}")
        break

    if epoch % VIS_EVERY == 0:
        visualize(model, epoch)

# %% [markdown]
# ## 9. Risultati

# %%
fig, ax = plt.subplots(figsize=(10, 4))
epochs = list(range(1, len(history["train"]) + 1))
ax.plot(epochs, history["train"], label="train", linewidth=2)
ax.plot(epochs, history["val"],   label="val",   linewidth=2, linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss totale")
ax.set_title("Pre-training loss"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Carica miglior checkpoint e mostra risultati finali
ckpt_path = CHECKPOINT_DIR / "pretrain_best.pth"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # aggiungi k_star al checkpoint se manca
    ckpt.setdefault("k_star", k_star)
    torch.save(ckpt, ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nMiglior checkpoint: epoch {ckpt.get('epoch','?')}  val={ckpt.get('metric',0):.4f}")

visualize(model, epoch="best", n_show=min(3, len(val_sub)))

print(f"\n✓  Pre-training completato")
print(f"   Checkpoint → {ckpt_path}")
print(f"   Prossimo step: esegui  01_meta_train.py")
