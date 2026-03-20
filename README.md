# auraS — AI-Powered Seizure Early Warning System

> **"A Sixth Sense for Epilepsy Patients"** — Real-time seizure prediction
> from EEG signals using MindSpore, optimized for edge deployment on Huawei devices.

**Huawei ICT Competition 2025–2026 · Innovation Track · Topic 1: MindSpore**
**Team Pierogi** — Poznan University of Technology

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Training](#training)
- [Experiment Framework](#experiment-framework)
- [Hyperparameter Search](#hyperparameter-search)
- [Monitoring](#monitoring)
- [Deployment](#deployment)
- [Execution Plan](#execution-plan)
- [Technology Stack](#technology-stack)
- [Team](#team)

---

## Overview

Epilepsy affects ~50 million people worldwide. The most debilitating aspect is
the **unpredictability** of seizures. auraS bridges the gap between clinical
EEG monitoring and consumer wearables by developing an intelligent
**privacy-first Early Warning System** that:

1. **Predicts** seizures minutes before onset using a lightweight AI model
2. **Alerts** patients with a high-contrast "Sixth Sense" UI
3. **Notifies** caregivers with GPS location via HarmonyOS services
4. **Runs locally** on-device using MindSpore Lite (no cloud dependency)

### Key Differentiators

| Dimension | Approach |
|---|---|
| **Algorithmic** | Comparative ablation across 7 architectures (LSTM → Autoformer) |
| **Medical-grade** | Optimized for **Recall** (sensitivity), not just accuracy |
| **Edge-first** | GhostNet/MobileNetV3 targeting <10ms inference on Kirin NPU |
| **Reproducible** | Hydra configs + LOSO cross-validation + Optuna HP search |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLOUD LAYER (ModelArts)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  Ascend 910  │  │   Re-train   │  │  WandB / Optuna         │ │
│  │  Training    │→ │   Pipeline   │→ │  Experiment Tracking    │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└──────────────────────────┬───────────────────────────────────────┘
                           │ .ms model
┌──────────────────────────▼───────────────────────────────────────┐
│                      EDGE LAYER (HarmonyOS)                      │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────┐ │
│  │ BT/EEG   │→ │  Preprocess  │→ │ MindSpore  │→ │  Alert    │ │
│  │ Wearable  │  │  Pipeline    │  │ Lite Infer │  │  System   │ │
│  └──────────┘  └──────────────┘  └────────────┘  └───────────┘ │
└──────────────────────────────────────────────────────────────────┘
                                                    │
┌───────────────────────────────────────────────────▼──────────────┐
│                     SERVICE LAYER                                │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │ Location Kit │  │   Push Kit    │  │  SMS / Emergency      │ │
│  │ (GPS coords) │  │ (Caregiver)   │  │  Notification         │ │
│  └──────────────┘  └───────────────┘  └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### ML Pipeline

```
Raw EDF ──→ Band-pass ──→ Channel ──→ Sliding ──→ Augment ──→ Train
 files      + Notch       Selection   Window       + Noise     MindSpore
            Filter        (4-6 ch)    (4s × 256Hz)             nn.Cell
                                                                  │
                                          ┌───────────────────────┘
                                          ▼
                                    Evaluate ──→ Export ──→ Quantize ──→ Deploy
                                    (LOSO CV)    MindIR    INT8/.ms     MindSpore
                                    Recall/F1                           Lite
```

---

## Project Structure

```
auraS/
├── README.md                          ← You are here
├── pyproject.toml                     # Dependencies, build config, entry-points
├── Makefile                           # Common commands (make train, make data, ...)
├── .env.example                       # API keys template (Kaggle, WandB, ModelArts)
│
├── configs/                           # Hierarchical YAML configs (Hydra / OmegaConf)
│   ├── config.yaml                    #   Top-level defaults
│   ├── data/                          #   Per-dataset config
│   │   ├── siena.yaml                 #     Siena Scalp EEG (primary)
│   │   ├── chbmit.yaml               #     CHB-MIT Scalp EEG
│   │   └── tuh.yaml                  #     TUH Seizure Corpus
│   ├── model/                         #   Per-architecture config
│   │   ├── lstm.yaml
│   │   ├── bilstm.yaml
│   │   ├── resnet1d.yaml
│   │   ├── mobilenetv3_1d.yaml
│   │   ├── ghostnet1d.yaml            #   ★ Huawei Noah's Ark
│   │   ├── mobilevit_1d.yaml
│   │   └── autoformer.yaml
│   ├── training/                      #   Training hyperparameters
│   │   ├── default.yaml               #     Local GPU/CPU training
│   │   └── modelarts.yaml             #     Ascend 910 cloud training
│   ├── experiment/                    #   Experiment plans
│   │   ├── baseline.yaml              #     All models × all datasets
│   │   └── ablation.yaml              #     Channel/window/loss ablations
│   └── optuna/
│       └── search.yaml                #   Optuna HP search space
│
├── scripts/                           # Standalone CLI scripts
│   ├── download_siena.py              #   Kaggle dataset download
│   ├── prepare_dataset.py             #   Raw EDF → processed .npz
│   └── export_lite.py                 #   Model → MindSpore Lite
│
├── src/auras/                         # Main Python package
│   ├── __init__.py
│   ├── data/                          # ── Data Pipeline ──────────────
│   │   ├── channels.py                #   Wearable channel selection
│   │   ├── loaders.py                 #   EDF file discovery
│   │   ├── preprocess.py              #   Z-score, sliding window
│   │   ├── dataset.py                 #   MindSpore GeneratorDataset
│   │   ├── augmentations.py           #   Noise, shift, scale, dropout
│   │   └── parsers/                   #   Seizure annotation parsers
│   │       ├── siena.py               #     ✅ Implemented
│   │       ├── chbmit.py              #     ⬜ Placeholder
│   │       └── tuh.py                 #     ⬜ Placeholder
│   │
│   ├── models/                        # ── Model Architectures ────────
│   │   ├── base.py                    #   Abstract BaseSeizureModel
│   │   ├── lstm.py                    #   LSTM baseline
│   │   ├── bilstm.py                  #   Bidirectional LSTM
│   │   ├── resnet1d.py                #   1-D ResNet-18
│   │   ├── mobilenetv3_1d.py          #   MobileNetV3-Small 1-D
│   │   ├── ghostnet1d.py              #   ★ GhostNet 1-D (Huawei)
│   │   ├── mobilevit_1d.py            #   MobileViT 1-D hybrid
│   │   ├── autoformer.py              #   Autoformer time-series
│   │   └── factory.py                 #   Model registry & create_model()
│   │
│   ├── training/                      # ── Training Infrastructure ────
│   │   ├── trainer.py                 #   Main training loop
│   │   ├── losses.py                  #   Weighted CE, Focal Loss
│   │   ├── metrics.py                 #   Recall, F1, FPR, AUC-ROC
│   │   ├── callbacks.py               #   Logging, early stop, checkpoints
│   │   ├── sampler.py                 #   Weighted sampling (imbalance)
│   │   └── lr_schedulers.py           #   Cosine annealing, warmup
│   │
│   ├── experiment/                    # ── Experiment Management ──────
│   │   ├── runner.py                  #   Multi-model experiment orchestrator
│   │   ├── optuna_search.py           #   Optuna HP optimization
│   │   └── cross_validation.py        #   LOSO & stratified K-fold
│   │
│   ├── monitoring/                    # ── Monitoring Backends ────────
│   │   ├── base_logger.py             #   Abstract interface + ConsoleLogger
│   │   ├── wandb_logger.py            #   Weights & Biases integration
│   │   └── modelarts_logger.py        #   Huawei ModelArts metrics
│   │
│   ├── deployment/                    # ── Edge Deployment ────────────
│   │   ├── converter.py               #   .ckpt → .mindir export
│   │   ├── quantizer.py               #   INT8 quantization (placeholder)
│   │   └── benchmark.py               #   Latency & throughput profiling
│   │
│   └── utils/                         # ── Shared Utilities ───────────
│       ├── config.py                  #   OmegaConf helpers
│       ├── reproducibility.py         #   Seed everything
│       └── io.py                      #   Checkpoint save/load
│
├── tests/                             # Unit & integration tests
│   ├── test_data/
│   ├── test_models/
│   └── test_training/
│
├── notebooks/                         # Jupyter exploration
├── experiments/                       # Run outputs & Optuna DBs (gitignored)
│   ├── runs/
│   └── optuna/
│
├── data/                              # Dataset storage (gitignored except metadata)
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
└── docs/                              # Extended documentation
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <repo-url> && cd auraS
make setup

# Or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,kaggle]"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your Kaggle and WandB API keys
```

### 3. Download & Process Data

```bash
make download-siena      # Download from Kaggle
make prepare-siena       # Process EDF → .npz windows
```

### 4. Train a Model

```bash
# Train with defaults (GhostNet on Siena)
make train

# Train a specific model
python -m auras.training.trainer model=lstm data=siena

# Run all baseline experiments
make train-baseline
```

### 5. Hyperparameter Search

```bash
make search              # Run Optuna HP optimization
make optuna-dashboard    # Launch web UI at localhost:8080
```

### 6. Export for Edge

```bash
make export-lite ARGS="--checkpoint experiments/runs/ghostnet1d/final.ckpt --model ghostnet1d"
make benchmark ARGS="--checkpoint experiments/runs/ghostnet1d/final.ckpt --model ghostnet1d"
```

---

## Data Pipeline

### Datasets

| Dataset | Source | Subjects | Channels | Sample Rate | Seizures |
|---------|--------|----------|----------|-------------|----------|
| **Siena** (primary) | Kaggle | 14 | 31 → **4** | 512 Hz → 256 Hz | ~47 |
| CHB-MIT | PhysioNet | 23 | 23 → **4** | 256 Hz | ~198 |
| TUH Seizure | IEDSS | 100+ | 19-23 → **4** | 250 Hz | ~3000+ |

### Spatial Subsampling (Channel Reduction)

Clinical EEG uses 19-31 channels. Consumer wearables have 4-6 electrodes.
We select **T7, T8, F7, F8** (temporal + frontal) — the channels most
informative for seizure detection AND most likely present on wearable devices.

### Processing Pipeline

```
EDF file → MNE load → Channel selection → Resample (256 Hz)
         → Band-pass (0.5-45 Hz) → Notch (50/60 Hz)
         → Z-score normalization → Sliding window (4s, 1s stride)
         → Seizure labeling → .npz output
```

---

## Models

Seven architectures spanning the complexity spectrum:

| # | Architecture | Type | Why | Edge Suitability |
|---|---|---|---|---|
| 1 | **LSTM** | RNN | Clinical baseline, strong temporal modeling | ⬛⬛⬜⬜⬜ |
| 2 | **BiLSTM** | RNN | Bidirectional context | ⬛⬛⬜⬜⬜ |
| 3 | **ResNet-1D** | CNN | Proven residual learning adapted to 1-D | ⬛⬛⬛⬜⬜ |
| 4 | **MobileNetV3-1D** | CNN | Efficiency standard (inverted residuals + SE) | ⬛⬛⬛⬛⬜ |
| 5 | **GhostNet-1D** ★ | CNN | Huawei Noah's Ark — ghost features from cheap ops | ⬛⬛⬛⬛⬛ |
| 6 | **MobileViT-1D** | Hybrid | CNN local + Transformer global attention | ⬛⬛⬛⬛⬜ |
| 7 | **Autoformer** | Transformer | Series decomposition + auto-correlation | ⬛⬛⬛⬜⬜ |

All models accept input shape `(B, C, T)` and output logits `(B, 2)`.
Instantiation through the factory: `create_model(cfg, num_channels=4)`.

---

## Training

### Class Imbalance Strategy

Seizures are rare (~0.8% of data). We combat this with:
- **Weighted Cross-Entropy** / **Focal Loss** (penalize false negatives)
- **Weighted Random Sampling** (balanced mini-batches)
- **Recall-optimized early stopping** (save best sensitivity, not accuracy)

### Training Modes

| Mode | Command | Hardware |
|---|---|---|
| Local | `make train` | CPU / NVIDIA GPU |
| ModelArts | `training=modelarts` | Ascend 910 |

---

## Experiment Framework

### Baseline Comparison

Trains all 7 models on each dataset with 3 repetitions for statistical
significance. Uses **Leave-One-Subject-Out (LOSO)** cross-validation —
the gold standard for patient-independent seizure detection.

```bash
python -m auras.experiment.runner --config configs/experiment/baseline.yaml
```

### Ablation Study

Isolates the impact of individual design choices:
- **Channel count**: 2, 4, 6 channels
- **Window size**: 2s, 4s, 8s, 16s
- **Loss function**: Weighted CE vs Focal Loss
- **Preprocessing**: with/without bandpass, with/without z-score

---

## Hyperparameter Search

[Optuna](https://optuna.org/) with TPE sampler and Median pruner:

| Parameter | Search Range |
|---|---|
| Learning rate | 1e-5 — 1e-2 (log) |
| Batch size | {64, 128, 256, 512} |
| Dropout | 0.0 — 0.5 |
| Hidden size | {64, 128, 256} |
| Weight decay | 1e-6 — 1e-2 (log) |
| Loss function | {weighted_ce, focal} |
| Focal gamma | 1.0 — 5.0 |

```bash
make search                # Run 100 trials
make optuna-dashboard      # Monitor at http://localhost:8080
```

---

## Monitoring

Three monitoring backends that all implement the same `BaseLogger` interface:

| Backend | Use Case | Setup |
|---|---|---|
| **WandB** | Primary experiment tracking, charts, tables | Set `WANDB_API_KEY` |
| **ModelArts** | Huawei Cloud training console | Auto on ModelArts jobs |
| **Console** | Local development fallback | Always available |

Training automatically logs: loss, learning rate, recall, F1, FPR, AUC-ROC.

---

## Deployment

### Export Pipeline

```
.ckpt → MindIR (.mindir) → Quantize (INT8) → MindSpore Lite (.ms)
```

### Benchmark Metrics

| Metric | Target | Why |
|---|---|---|
| **Inference latency** | < 10 ms | Real-time 4-second sliding window |
| **Model size** | < 5 MB | Fit in mobile app bundle |
| **Battery** | < 1 mAs/inference | 24/7 monitoring on single charge |

---

## Execution Plan

### Phase 1: Foundation ✅
- [x] Project structure & architecture design
- [x] Config system (Hydra/OmegaConf)
- [x] Data pipeline (download, parse, preprocess)
- [x] All 7 model architectures (MindSpore nn.Cell)
- [x] Training infrastructure (losses, metrics, callbacks)
- [x] Monitoring backends (WandB, ModelArts, Console)
- [x] Experiment framework (runner, Optuna, LOSO CV)
- [x] Deployment pipeline (converter, quantizer, benchmark)

### Phase 2: Validate & Train
- [ ] Validate Siena data pipeline end-to-end
- [ ] Train LSTM baseline on Siena (smoke test)
- [ ] Verify WandB logging works
- [ ] Train all 7 baselines on Siena
- [ ] Compute LOSO cross-validation results
- [ ] Ablation study (channels, windows, losses)

### Phase 3: Optimize
- [ ] Run Optuna HP search on top 3 models
- [ ] Fine-tune GhostNet for NPU efficiency
- [ ] Study latest research papers for architecture improvements
- [ ] Implement any novel techniques from literature

### Phase 4: Deploy
- [ ] Export best model to MindIR
- [ ] INT8 quantization
- [ ] Benchmark on target hardware
- [ ] HarmonyOS app prototype (ArkTS)
- [ ] Integration with Location Kit + Push Kit

### Phase 5: Report & Present
- [ ] Compile results tables & visualizations
- [ ] Write competition technical report
- [ ] Prepare demo video
- [ ] Final presentation

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Framework** | MindSpore 2.3+ | Training & inference |
| **EEG Processing** | MNE-Python, SciPy | EDF loading, filtering |
| **Config** | Hydra, OmegaConf | Composable YAML configs |
| **HP Search** | Optuna + Dashboard | Bayesian optimization with pruning |
| **Tracking** | Weights & Biases | Experiment tracking & visualization |
| **Cloud** | Huawei ModelArts | Ascend 910 training |
| **Edge** | MindSpore Lite | On-device inference |
| **Mobile** | HarmonyOS (ArkTS) | Application UI |
| **Services** | HMS Core | Location Kit, Push Kit, Health Kit |
| **Data** | NumPy, pandas, scikit-learn | Processing, splitting, metrics |
| **Testing** | pytest, ruff | Quality assurance |

### Additional Tools Worth Considering

- **SHAP / Captum** — Model interpretability (which channels/time-points drive predictions)
- **ONNX** — Cross-framework model exchange (backup export path)
- **TensorBoard** — Alternative visualization (MindSpore has native support)
- **DVC** — Data version control for large EDF datasets
- **MLflow** — Alternative experiment tracking if WandB is unavailable
- **Grad-CAM for 1-D** — Visual explanations of seizure predictions for clinicians

---

## Team

| Role | Name |
|---|---|
| Instructor | Piotr Zwierzykowski |
| Team Captain | Alicja Augustyniak |
| Member | Patryk Maciejewski |
| Member | Filip Domański |

**Poznan University of Technology** · Team Pierogi 🥟

---

## License

This project is developed for the Huawei ICT Competition 2025-2026.
