"""ModelArts training job entry point.

This script:
1. Installs MindSpore 2.0.0 + deps (on cloud only)
2. Extracts the code tarball
3. Resolves configs from yaml files
4. Runs a 1-epoch dry-run training
"""
import subprocess, sys, os, tarfile

print("=== cloud_boot.py start ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

# ── On cloud: install deps ───────────────────────────────────────────────
ON_CLOUD = os.path.exists("/home/ma-user")

if ON_CLOUD:
    print("[1/4] Upgrading pip...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print("[2/4] Removing pre-installed MindSpore...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "mindspore-gpu", "mindspore", "-y"],
                   capture_output=True)  # ignore errors if not installed
    print("[3/4] Installing MindSpore 2.0.0...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "mindspore==2.0.0", "Pillow<10", "-q"],
                   check=True)
    print("[4/4] Installing deps...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "omegaconf", "scikit-learn",
                    "protobuf>=3.13", "psutil", "-q"], check=True)
    print("Deps installed.", flush=True)

# ── Extract code (on cloud) or use repo root (local) ────────────────────
if ON_CLOUD:
    code_dir = "/home/ma-user/modelarts/user-job-dir/code"
    tarball = os.path.join(code_dir, "auras_code.tar.gz")
    extract_to = "/tmp/auras"
    os.makedirs(extract_to, exist_ok=True)
    print(f"Extracting {tarball}...")
    with tarfile.open(tarball) as tf:
        tf.extractall(extract_to)
    os.chdir(extract_to)
    sys.path.insert(0, extract_to)
else:
    # Local: assume we're run from the repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    sys.path.insert(0, os.path.join(repo_root, "src"))

# ── Setup MindSpore ──────────────────────────────────────────────────────
import mindspore as ms
print(f"MindSpore version: {ms.__version__}")

if ON_CLOUD:
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
else:
    ms.set_context(mode=ms.PYNATIVE_MODE)

# ── Resolve data path ────────────────────────────────────────────────────
if ON_CLOUD:
    data_dir = os.environ.get("data", "/home/ma-user/modelarts/inputs/data_0")
else:
    data_dir = "data/processed"

print(f"Data dir: {data_dir}")
print(f"Data dir contents: {os.listdir(data_dir)}")

# ── Build fully resolved config ──────────────────────────────────────────
from omegaconf import OmegaConf

# Load sub-configs from yaml (same resolution logic as trainer.py)
model_cfg = OmegaConf.load("configs/model/cnn_bilstm_attn.yaml")
training_cfg = OmegaConf.load("configs/training/conv_snn.yaml")
data_cfg = OmegaConf.load("configs/data/siena.yaml")

# Override data paths for cloud
data_cfg.processed_dir = data_dir
data_cfg.name = "siena_sop_merged"

# Mode: DRY_RUN (default) or FULL
run_mode = os.environ.get("RUN_MODE", "dry_run")  # dry_run | full
if run_mode == "full":
    # Full dataset — no window cap, 1 worker to avoid fork OOM
    data_cfg.dry_run_max_windows = None
    training_cfg.num_workers = 1
    training_cfg.epochs = int(os.environ.get("EPOCHS", "1"))
    print(f"  MODE: FULL (epochs={training_cfg.epochs}, num_workers=1)")
else:
    data_cfg.dry_run_max_windows = 500
    training_cfg.epochs = 1
    print(f"  MODE: DRY_RUN (500 windows, 1 epoch)")

cfg = OmegaConf.create({
    "seed": 42,
    "project_name": "auraS",
    "output_dir": "/tmp/auras_output" if ON_CLOUD else "experiments/cloud_test",
    "model": model_cfg,
    "training": training_cfg,
    "data": data_cfg,
})

print(f"Config summary:")
print(f"  Model: {cfg.model.name}")
print(f"  Training epochs: {cfg.training.epochs}")
print(f"  Data: {cfg.data.processed_dir}/{cfg.data.name}.npz")
print(f"  Channels: {cfg.data.channels.selected}")
print(f"  Dry run windows: {cfg.data.dry_run_max_windows}")

# ── Run training ─────────────────────────────────────────────────────────
import psutil, time
proc = psutil.Process()
print(f"  Memory before training: {proc.memory_info().rss / 1024**2:.0f} MB", flush=True)
t0 = time.time()

from auras.training.trainer import train

train(cfg)

elapsed = time.time() - t0
print(f"  Memory after training:  {proc.memory_info().rss / 1024**2:.0f} MB", flush=True)
print(f"  Training wall time:     {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

# ── Upload results to OBS (cloud only) ───────────────────────────────────
if ON_CLOUD:
    print("Uploading results to OBS...", flush=True)
    import moxing as mox
    output_dir = cfg.output_dir
    obs_output = "obs://auras-experiments/output/"
    mox.file.copy_parallel(output_dir, obs_output)
    print(f"  Copied {output_dir} -> {obs_output}")

print("=== cloud_boot.py DONE ===")
