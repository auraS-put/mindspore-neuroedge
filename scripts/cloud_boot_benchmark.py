"""ModelArts benchmark boot script — trains all 5 models sequentially.

Workflow:
1. Install deps from bundled wheels (fast, ~1-2 min)
2. Extract code tarball
3. Load data ONCE, pre-slice into train/val/test subsets
4. For each model: train N epochs, evaluate val+test after each epoch
5. Upload progress.json to OBS after each epoch (live monitoring)
6. Save checkpoints per model to OBS

Memory management (32 GiB VM):
- Pre-slice data into subsets (~6.9 GB) then free full array (saves ~7 GB)
- python_multiprocessing=False avoids fork (no COW memory growth)
- Explicit del + gc.collect() between dataset phases

Environment variables:
    RUN_MODE    : dry_run | full (default: dry_run)
    EPOCHS      : number of epochs per model (default: 5)
    MODELS      : comma-separated model list (default: all 5)
    SESSION_ID  : unique session identifier (auto-generated if not set)
"""
import subprocess, sys, os, tarfile, json, time, datetime, traceback, gc

print("=== cloud_boot_benchmark.py start ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

ON_CLOUD = os.path.exists("/home/ma-user") or os.path.exists("/opt/ml")
ON_MODELARTS = os.path.exists("/home/ma-user")
ON_SAGEMAKER = os.path.exists("/opt/ml")
SESSION_ID = os.environ.get("SESSION_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

# Model → training config mapping (paper-verified hyperparameters)
MODEL_TRAINING_MAP = {
    "multiscale_cnn": "multiscale_cnn",       # Paper 12: Focal loss, Adam, LR=1e-3, batch=64, cosine
    "cnn_bilstm_attn": "conv_snn",           # Paper 18: AdamW, LR=1e-3, batch=64, OneCycleLR, SSWCE
    "cnn_informer": "cnn_informer",           # Paper 10: Adam, LR=1e-4, batch=32, cosine, Focal
    "pyramidal_cnn_bilstm": "lightweight",   # Paper 04: Adam, LR=2e-5, batch=32, cosine, WCE
}

ALL_MODELS = [
    "multiscale_cnn",
    "cnn_bilstm_attn",
    "pyramidal_cnn_bilstm",
]

# ── Install dependencies ─────────────────────────────────────────────────
if ON_MODELARTS:
    code_dir = "/home/ma-user/modelarts/user-job-dir/code"

    print("[1/3] Installing from bundled wheels...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "mindspore-gpu", "mindspore", "-y"],
                   capture_output=True)

    # Download wheels from OBS
    import moxing as mox
    wheels_obs = "obs://auras-experiments/wheels/wheels.tar.gz"
    wheels_local = "/tmp/wheels.tar.gz"
    wheels_dir = "/tmp/wheels"
    print(f"  Downloading wheels from OBS...", flush=True)
    mox.file.copy(wheels_obs, wheels_local)
    os.makedirs(wheels_dir, exist_ok=True)
    with tarfile.open(wheels_local) as tf:
        tf.extractall(wheels_dir)
    os.remove(wheels_local)  # free 966 MB

    # Install from local wheels (no network needed)
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", wheels_dir,
        "mindspore==2.0.0", "Pillow", "omegaconf", "scikit-learn",
        "protobuf", "psutil", "-q"
    ], check=True)
    print("  Wheels installed.", flush=True)

    # Extract code
    print("[2/3] Extracting code...", flush=True)
    tarball = os.path.join(code_dir, "auras_code.tar.gz")
    extract_to = "/tmp/auras"
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tarball) as tf:
        tf.extractall(extract_to)
    os.chdir(extract_to)
    sys.path.insert(0, extract_to)

elif ON_SAGEMAKER:
    # SageMaker: custom Docker image has MindSpore pre-installed
    # Code is at /opt/ml/input/data/code/ (from code channel)
    code_dir = os.environ.get("SM_CHANNEL_CODE", "/opt/ml/input/data/code")
    extract_to = "/opt/ml/code"

    # Extract code tarball if present
    tarball = os.path.join(code_dir, "auras_code.tar.gz")
    if os.path.exists(tarball):
        print("[1/2] Extracting code tarball...", flush=True)
        os.makedirs(extract_to, exist_ok=True)
        with tarfile.open(tarball) as tf:
            tf.extractall(extract_to)
    os.chdir(extract_to)
    sys.path.insert(0, extract_to)
    print("[2/2] SageMaker environment ready.", flush=True)
else:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    sys.path.insert(0, os.path.join(repo_root, "src"))

# ── Setup MindSpore ──────────────────────────────────────────────────────
import mindspore as ms
import numpy as np
print(f"MindSpore version: {ms.__version__}", flush=True)

if ON_CLOUD:
    # Use PYNATIVE_MODE — GRAPH_MODE in MindSpore causes OOM due to
    # pre-allocation of full computation graph memory (4GB+ for LSTM/attention).
    try:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
        print("Device: GPU (PYNATIVE)", flush=True)
    except RuntimeError:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        print("Device: CPU (GPU not available)", flush=True)
else:
    ms.set_context(mode=ms.PYNATIVE_MODE)

# ── Configuration ────────────────────────────────────────────────────────
run_mode = os.environ.get("RUN_MODE", "dry_run")
n_epochs = int(os.environ.get("EPOCHS", "5"))
models_str = os.environ.get("MODELS", ",".join(ALL_MODELS))
model_names = [m.strip() for m in models_str.split(",") if m.strip()]

if ON_MODELARTS:
    data_dir = os.environ.get("data", "/home/ma-user/modelarts/inputs/data_0")
elif ON_SAGEMAKER:
    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
else:
    data_dir = "data/processed"

if ON_MODELARTS:
    output_base = "/tmp/benchmark_output"
elif ON_SAGEMAKER:
    output_base = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
else:
    output_base = "experiments/benchmark_output"
os.makedirs(output_base, exist_ok=True)

print(f"\n{'='*60}", flush=True)
print(f"BENCHMARK SESSION: {SESSION_ID}", flush=True)
print(f"  Mode:   {run_mode}", flush=True)
print(f"  Epochs: {n_epochs}", flush=True)
print(f"  Models: {model_names}", flush=True)
print(f"  Data:   {data_dir}", flush=True)
print(f"  Output: {output_base}", flush=True)
print(f"{'='*60}\n", flush=True)

# ── Load data ONCE, pre-slice into subsets ───────────────────────────────
import psutil
from pathlib import Path
from omegaconf import OmegaConf

data_cfg = OmegaConf.load("configs/data/siena.yaml")
data_cfg.processed_dir = data_dir
data_cfg.name = "siena_sop_merged"

if run_mode == "full":
    data_cfg.dry_run_max_windows = None
else:
    data_cfg.dry_run_max_windows = int(os.environ.get("MAX_WINDOWS", "500"))

proc = psutil.Process()
print(f"Memory before data load: {proc.memory_info().rss / 1024**2:.0f} MB", flush=True)

npz_path = Path(data_dir) / "siena_sop_merged.npz"
print(f"Loading data from {npz_path}...", flush=True)
t0 = time.time()
preloaded = np.load(str(npz_path))  # full load into RAM (mmap not supported for npz)
print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)
print(f"  Memory after load: {proc.memory_info().rss / 1024**2:.0f} MB", flush=True)

label_key = data_cfg.get("target_key", "y")
y_full = preloaded[label_key]
print(f"  Total windows: {len(y_full):,}", flush=True)
print(f"  Positive: {int(y_full.sum()):,}  Negative: {int(len(y_full) - y_full.sum()):,}", flush=True)

# ── Build train/val/test splits ONCE ────────────────────────────────────
from sklearn.model_selection import GroupShuffleSplit

seed = 42
np.random.seed(seed)

n_full = len(y_full)
max_w = data_cfg.get("dry_run_max_windows", n_full) or n_full
subjects_full = preloaded["subjects"] if "subjects" in preloaded else None
abs_indices = np.arange(n_full)
if len(abs_indices) > max_w:
    rng_cap = np.random.default_rng(42)
    abs_indices = rng_cap.choice(abs_indices, int(max_w), replace=False)
    abs_indices.sort()

y = y_full[abs_indices]
subjects = subjects_full[abs_indices] if subjects_full is not None else None
n = len(y)
pos_indices = np.arange(n)

test_size = data_cfg.split.get("test_size", 0.2)
val_size = data_cfg.split.get("val_size", 0.1)

if subjects is not None and len(np.unique(subjects)) >= 4:
    # Subject-aware split: no patient leakage between train/val/test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_pos, test_pos = next(gss_test.split(pos_indices, y, groups=subjects))

    relative_val = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_pos, val_pos = next(gss_val.split(train_val_pos, y[train_val_pos],
                                             groups=subjects[train_val_pos]))
    pos_train = train_val_pos[train_pos]
    pos_val = train_val_pos[val_pos]
    pos_test = test_pos
    print(f"  Split method: subject-aware (GroupShuffleSplit)", flush=True)
    print(f"  Test subjects: {np.unique(subjects[pos_test])}", flush=True)
else:
    # Fallback for dry_run with very few subjects
    from sklearn.model_selection import train_test_split
    pos_train_val, pos_test = train_test_split(
        pos_indices, test_size=test_size, stratify=y, random_state=seed)
    relative_val = val_size / (1 - test_size)
    pos_train, pos_val = train_test_split(
        pos_train_val, test_size=relative_val, stratify=y[pos_train_val], random_state=seed)
    print(f"  Split method: stratified (fallback — insufficient subjects)", flush=True)

train_idx = abs_indices[pos_train]
val_idx = abs_indices[pos_val]
test_idx = abs_indices[pos_test]

# ── Pre-slice data into subsets to avoid fork OOM ────────────────────────
# Fancy indexing copies data, so after slicing we can free the full array.
# This avoids the 7 GB array being inherited by GeneratorDataset workers via COW.
print("\n  Pre-slicing data into train/val/test subsets...", flush=True)
X_full = preloaded["X"]
train_X = X_full[train_idx].copy()
train_y = y_full[train_idx].copy()
val_X = X_full[val_idx].copy()
val_y = y_full[val_idx].copy()
test_X = X_full[test_idx].copy()
test_y = y_full[test_idx].copy()

# Free the full 7 GB array — we only need the slices from here on
del preloaded, X_full, y_full
gc.collect()
print(f"  Memory after pre-slice + free: {proc.memory_info().rss / 1024**2:.0f} MB", flush=True)
print(f"  Subset sizes: train_X={train_X.nbytes/1024**3:.2f} GB, "
      f"val_X={val_X.nbytes/1024**3:.2f} GB, test_X={test_X.nbytes/1024**3:.2f} GB", flush=True)

meta = {
    "train_samples": len(train_X),
    "val_samples": len(val_X),
    "test_samples": len(test_X),
    "train_positive": int(train_y.sum()),
    "train_negative": int(len(train_y) - train_y.sum()),
}

print(f"\nSplits: {meta['train_samples']} train / {meta['val_samples']} val / {meta['test_samples']} test")
print(f"Class balance (train): {meta['train_positive']} pos / {meta['train_negative']} neg\n", flush=True)


# ── Lightweight dataset (no fork, no preloaded dict) ─────────────────────
import mindspore.dataset as mds


class SlicedDataset:
    """Dataset wrapper over pre-sliced numpy arrays. No file I/O, no indexing."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return self._X[idx].astype(np.float32), np.array(self._y[idx], dtype=np.int32)


def build_sliced_dataset(X, y, batch_size, shuffle=True, num_workers=1):
    """Create a GeneratorDataset from pre-sliced arrays without forking."""
    source = SlicedDataset(X, y)
    dataset = mds.GeneratorDataset(
        source=source,
        column_names=["eeg", "label"],
        shuffle=shuffle,
        num_parallel_workers=num_workers,
        python_multiprocessing=False,  # threading only — avoids fork + COW OOM
    )
    drop = len(source) >= batch_size
    dataset = dataset.batch(batch_size, drop_remainder=drop)
    return dataset


# ── Progress tracking ────────────────────────────────────────────────────
progress = {
    "session_id": SESSION_ID,
    "run_mode": run_mode,
    "n_epochs": n_epochs,
    "models": model_names,
    "splits": meta,
    "start_time": datetime.datetime.now().isoformat(),
    "status": "running",
    "current_model": None,
    "model_results": {},
}


def upload_progress():
    """Upload progress.json to cloud storage for live monitoring."""
    progress_path = os.path.join(output_base, "progress.json")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    if ON_MODELARTS:
        import moxing as mox
        obs_path = f"obs://auras-experiments/output/benchmark_{SESSION_ID}/progress.json"
        mox.file.copy(progress_path, obs_path)
    elif ON_SAGEMAKER:
        # SageMaker auto-syncs SM_MODEL_DIR to S3 on completion;
        # for live monitoring, use SM output channel
        pass


def upload_checkpoint(model_name, filename):
    """Upload a checkpoint file to cloud storage."""
    if ON_MODELARTS:
        import moxing as mox
        local = os.path.join(output_base, model_name, filename)
        obs_path = f"obs://auras-experiments/output/benchmark_{SESSION_ID}/{model_name}/{filename}"
        mox.file.copy(local, obs_path)
    elif ON_SAGEMAKER:
        # Checkpoints are in SM_MODEL_DIR, auto-uploaded to S3 on completion
        pass


# ── Training loop per model ──────────────────────────────────────────────
from auras.models.factory import create_model
from auras.training.evaluator import evaluate_epoch
from auras.training.losses import build_loss
from auras.training.lr_schedulers import build_lr_schedule
from auras.utils.reproducibility import seed_everything

import mindspore.ops as ops

n_channels = len(data_cfg.channels.selected)
window_len_s = float(data_cfg.window.seconds)

upload_progress()

for model_idx, model_name in enumerate(model_names):
    print(f"\n{'='*60}", flush=True)
    print(f"MODEL {model_idx+1}/{len(model_names)}: {model_name}", flush=True)
    print(f"{'='*60}\n", flush=True)

    progress["current_model"] = model_name
    progress["model_results"][model_name] = {
        "status": "training",
        "start_time": datetime.datetime.now().isoformat(),
        "epochs": [],
    }
    upload_progress()

    model_output_dir = Path(output_base) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load per-model training config (paper-verified hyperparameters)
        tcfg_name = MODEL_TRAINING_MAP[model_name]
        training_cfg = OmegaConf.load(f"configs/training/{tcfg_name}.yaml")
        training_cfg.epochs = n_epochs
        if run_mode == "full":
            training_cfg.num_workers = 1

        # Allow batch size override via env var (to better utilize GPU)
        batch_override = int(os.environ.get("BATCH_SIZE", "0"))
        if batch_override > 0:
            orig_batch = training_cfg.batch_size
            training_cfg.batch_size = batch_override
            # Linear LR scaling: scale LR proportionally to batch size increase
            scale = batch_override / orig_batch
            training_cfg.learning_rate = training_cfg.learning_rate * scale
            print(f"  Batch override: {orig_batch} → {batch_override} (LR scaled {scale:.1f}×)", flush=True)

        batch_size = training_cfg.batch_size
        num_workers = max(training_cfg.get("num_workers", 4), 1)

        print(f"  Config: {tcfg_name}.yaml — LR={training_cfg.learning_rate} "
              f"batch={batch_size} opt={training_cfg.get('optimizer','adam')} "
              f"sched={training_cfg.scheduler.name}", flush=True)

        # Build model
        seed_everything(seed)
        model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
        model = create_model(model_cfg, num_channels=n_channels)
        n_params = model.count_params()
        print(f"  Parameters: {n_params:,}", flush=True)

        # Loss, LR, optimizer
        loss_fn = build_loss(training_cfg, meta["train_positive"], meta["train_negative"])
        steps_per_epoch = max(meta["train_samples"] // batch_size, 1)
        lr_schedule = build_lr_schedule(training_cfg, steps_per_epoch)

        opt_name = training_cfg.get("optimizer", "adam")
        wd = training_cfg.weight_decay
        if opt_name == "adamw":
            optimizer = ms.nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_schedule, weight_decay=wd)
        elif opt_name == "adam":
            optimizer = ms.nn.Adam(model.trainable_params(), learning_rate=lr_schedule, weight_decay=wd)
        else:
            optimizer = ms.nn.SGD(model.trainable_params(), learning_rate=lr_schedule, weight_decay=wd, momentum=0.9)

        clip_norm = training_cfg.get("gradient_clip_norm", 1.0)

        def forward_fn(data, label):
            return loss_fn(model(data), label)

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

        # Resume from checkpoint if exists
        best_val = -float("inf")
        no_improve = 0
        best_epoch = 0
        start_epoch = 0

        resume_path = model_output_dir / "train_state.ckpt"
        if resume_path.exists():
            state = ms.load_checkpoint(str(resume_path))
            param_dict = {k: v for k, v in state.items() if not k.startswith("__meta_")}
            ms.load_param_into_net(model, param_dict, strict_load=False)
            start_epoch = int(state.get("__meta_epoch", ms.Tensor(0)).asnumpy())
            best_val = float(state.get("__meta_best_val", ms.Tensor(best_val)).asnumpy())
            best_epoch = int(state.get("__meta_best_epoch", ms.Tensor(0)).asnumpy())
            no_improve = int(state.get("__meta_no_improve", ms.Tensor(0)).asnumpy())
            print(f"  Resumed from epoch {start_epoch} (best_val={best_val:.4f})", flush=True)

        patience = training_cfg.early_stopping.patience
        log_every = max(1, training_cfg.get("log_every_steps", 10))

        # ── Epoch loop ───────────────────────────────────────────────
        for epoch in range(start_epoch, n_epochs):
            epoch_start = time.time()
            model.set_train(True)
            epoch_loss = 0.0
            n_batches = 0

            # Build fresh train dataset each epoch (GeneratorDataset is consumed once)
            train_ds = build_sliced_dataset(train_X, train_y, batch_size,
                                            shuffle=True, num_workers=num_workers)

            for X, label in train_ds.create_tuple_iterator():
                loss, grads = grad_fn(X, label)
                grads = ops.clip_by_global_norm(grads, clip_norm)
                optimizer(grads)
                batch_loss = float(loss.asnumpy())
                epoch_loss += batch_loss
                n_batches += 1

                # Progress bar
                elapsed = time.time() - epoch_start
                eta = (elapsed / n_batches) * (steps_per_epoch - n_batches) if n_batches < steps_per_epoch else 0
                pct = min(100, n_batches * 100 // steps_per_epoch)
                bar_len = 20
                filled = bar_len * n_batches // steps_per_epoch
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:3d}% | batch {n_batches}/{steps_per_epoch} | "
                      f"loss={batch_loss:.4f} avg={epoch_loss/n_batches:.4f} | "
                      f"ETA {eta:.0f}s", end="", flush=True)

            print("", flush=True)  # newline after progress bar

            # Free train dataset before evaluation
            del train_ds
            gc.collect()

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - epoch_start

            # ── Validation evaluation ────────────────────────────────
            model.set_train(False)
            val_ds = build_sliced_dataset(val_X, val_y, batch_size,
                                          shuffle=False, num_workers=num_workers)
            val_result = evaluate_epoch(model, val_ds.create_tuple_iterator(),
                                        window_length_s=window_len_s)
            # Compute val loss
            val_loss_total = 0.0
            val_n = 0
            val_ds2 = build_sliced_dataset(val_X, val_y, batch_size,
                                           shuffle=False, num_workers=num_workers)
            for X, label in val_ds2.create_tuple_iterator():
                val_loss_total += float(loss_fn(model(X), label).asnumpy())
                val_n += 1
            del val_ds, val_ds2
            gc.collect()
            val_loss = val_loss_total / max(val_n, 1)

            val_recall = val_result.segment.recall
            val_f1 = val_result.segment.f1
            improved = val_recall > best_val

            if improved:
                best_val = val_recall
                no_improve = 0
                best_epoch = epoch + 1
                # Save best checkpoint
                ms.save_checkpoint(model, str(model_output_dir / "best.ckpt"))
                upload_checkpoint(model_name, "best.ckpt")
            else:
                no_improve += 1

            # ── Test evaluation ──────────────────────────────────────
            test_ds = build_sliced_dataset(test_X, test_y, batch_size,
                                           shuffle=False, num_workers=num_workers)
            test_result = evaluate_epoch(model, test_ds.create_tuple_iterator(),
                                         window_length_s=window_len_s)
            # Compute test loss
            test_loss_total = 0.0
            test_n = 0
            test_ds2 = build_sliced_dataset(test_X, test_y, batch_size,
                                            shuffle=False, num_workers=num_workers)
            for X, label in test_ds2.create_tuple_iterator():
                test_loss_total += float(loss_fn(model(X), label).asnumpy())
                test_n += 1
            del test_ds, test_ds2
            gc.collect()
            test_loss = test_loss_total / max(test_n, 1)

            # ── Log epoch results ────────────────────────────────────
            marker = " *BEST*" if improved else ""
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n  [{ts}] {model_name} Epoch {epoch+1}/{n_epochs} DONE "
                  f"({epoch_time:.0f}s){marker}", flush=True)
            print(f"  ┌─────────────┬──────────┬──────────┐", flush=True)
            print(f"  │  Metric     │   Val    │   Test   │", flush=True)
            print(f"  ├─────────────┼──────────┼──────────┤", flush=True)
            print(f"  │ Loss        │ {val_loss:8.4f} │ {test_loss:8.4f} │", flush=True)
            print(f"  │ Accuracy    │ {val_result.segment.accuracy:8.4f} │ {test_result.segment.accuracy:8.4f} │", flush=True)
            print(f"  │ Recall      │ {val_result.segment.recall:8.4f} │ {test_result.segment.recall:8.4f} │", flush=True)
            print(f"  │ Precision   │ {val_result.segment.precision:8.4f} │ {test_result.segment.precision:8.4f} │", flush=True)
            print(f"  │ Specificity │ {val_result.segment.specificity:8.4f} │ {test_result.segment.specificity:8.4f} │", flush=True)
            print(f"  │ F1          │ {val_result.segment.f1:8.4f} │ {test_result.segment.f1:8.4f} │", flush=True)
            print(f"  ├─────────────┼──────────┼──────────┤", flush=True)
            print(f"  │ FP/h        │ {val_result.fp_per_hour:8.2f} │ {test_result.fp_per_hour:8.2f} │", flush=True)
            print(f"  │ SDR         │ {val_result.seizure_detection_rate:8.3f} │ {test_result.seizure_detection_rate:8.3f} │", flush=True)
            print(f"  └─────────────┴──────────┴──────────┘", flush=True)
            print(f"    Train loss: {avg_loss:.4f} | No-improve: {no_improve}/{patience}\n", flush=True)

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "epoch_time_s": epoch_time,
                "val": val_result.to_dict(),
                "test": test_result.to_dict(),
                "best_val_recall": best_val,
                "no_improve": no_improve,
            }
            progress["model_results"][model_name]["epochs"].append(epoch_record)

            # ── Save training state for resume ───────────────────────
            params_to_save = [{"name": p.name, "data": p} for p in model.get_parameters()]
            params_to_save.append({"name": "__meta_epoch", "data": ms.Tensor(epoch + 1, ms.int32)})
            params_to_save.append({"name": "__meta_best_val", "data": ms.Tensor(best_val, ms.float32)})
            params_to_save.append({"name": "__meta_best_epoch", "data": ms.Tensor(best_epoch, ms.int32)})
            params_to_save.append({"name": "__meta_no_improve", "data": ms.Tensor(no_improve, ms.int32)})
            ms.save_checkpoint(params_to_save, str(model_output_dir / "train_state.ckpt"))
            upload_checkpoint(model_name, "train_state.ckpt")

            # Upload progress after every epoch
            upload_progress()

            # Early stopping
            if no_improve >= patience:
                print(f"  EARLY STOP at epoch {epoch+1} (best_val_recall={best_val:.4f} @ epoch {best_epoch})", flush=True)
                break

        # ── Model complete ───────────────────────────────────────────
        ms.save_checkpoint(model, str(model_output_dir / "final.ckpt"))
        upload_checkpoint(model_name, "final.ckpt")

        progress["model_results"][model_name]["status"] = "completed"
        progress["model_results"][model_name]["end_time"] = datetime.datetime.now().isoformat()
        progress["model_results"][model_name]["n_params"] = n_params
        progress["model_results"][model_name]["best_epoch"] = best_epoch
        progress["model_results"][model_name]["best_val_recall"] = best_val
        upload_progress()

        print(f"\n  {model_name} COMPLETE — best_val_recall={best_val:.4f} @ epoch {best_epoch}", flush=True)

        # Free model memory
        del model, optimizer, grad_fn, loss_fn, lr_schedule

    except Exception as e:
        print(f"\n  ERROR training {model_name}: {e}", flush=True)
        traceback.print_exc()
        progress["model_results"][model_name]["status"] = "failed"
        progress["model_results"][model_name]["error"] = str(e)
        upload_progress()
        continue

# ── Final summary ────────────────────────────────────────────────────────
progress["status"] = "completed"
progress["end_time"] = datetime.datetime.now().isoformat()
total_time = time.time() - t0
progress["total_time_s"] = total_time

# Save final comprehensive results
upload_progress()

# Upload entire output directory to cloud storage
if ON_MODELARTS:
    import moxing as mox
    obs_output = f"obs://auras-experiments/output/benchmark_{SESSION_ID}/"
    mox.file.copy_parallel(output_base, obs_output)
    print(f"\nAll results uploaded to {obs_output}", flush=True)
elif ON_SAGEMAKER:
    print(f"\nResults at {output_base} — will sync to S3 on job completion.", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"BENCHMARK COMPLETE", flush=True)
print(f"  Session:    {SESSION_ID}", flush=True)
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)", flush=True)
print(f"  Models trained: {sum(1 for m in progress['model_results'].values() if m['status']=='completed')}/{len(model_names)}", flush=True)
print(f"{'='*60}", flush=True)
