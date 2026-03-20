# ───────────────────────────────────────────────────────────────────
# auraS Makefile — common workflows
# ───────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help
PYTHON := .venv/bin/python
PIP := .venv/bin/pip

# ── Setup ─────────────────────────────────────────────────────────
.PHONY: setup
setup: ## Create venv and install project in editable mode
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,kaggle]"
	$(PYTHON) -m pre_commit install || true
	@echo "✔ Setup complete. Activate with:  source .venv/bin/activate"

# ── Data ──────────────────────────────────────────────────────────
.PHONY: download-siena
download-siena: ## Download Siena scalp-EEG dataset from Kaggle
	$(PYTHON) scripts/download_siena.py

.PHONY: prepare-siena
prepare-siena: ## Process raw Siena EDF files → model-ready .npz
	$(PYTHON) scripts/prepare_dataset.py --config configs/data/siena.yaml

.PHONY: data
data: download-siena prepare-siena ## Full data pipeline (download + preprocess)

# ── Training ──────────────────────────────────────────────────────
.PHONY: train
train: ## Train default model (override with ARGS="...")
	$(PYTHON) -m auras.training.trainer $(ARGS)

.PHONY: train-baseline
train-baseline: ## Train all baseline models sequentially
	$(PYTHON) -m auras.experiment.runner --config configs/experiment/baseline.yaml

# ── Hyperparameter search ─────────────────────────────────────────
.PHONY: search
search: ## Run Optuna hyperparameter search
	$(PYTHON) -m auras.experiment.optuna_search $(ARGS)

.PHONY: optuna-dashboard
optuna-dashboard: ## Launch Optuna dashboard UI
	$(PYTHON) -m optuna_dashboard sqlite:///experiments/optuna/study.db

# ── Evaluation ────────────────────────────────────────────────────
.PHONY: evaluate
evaluate: ## Evaluate a checkpoint (pass CKPT=path/to/model.ckpt)
	$(PYTHON) -m auras.training.trainer --mode eval --checkpoint $(CKPT)

# ── Export / Deployment ───────────────────────────────────────────
.PHONY: export-lite
export-lite: ## Convert best model to MindSpore Lite .ms format
	$(PYTHON) scripts/export_lite.py $(ARGS)

.PHONY: benchmark
benchmark: ## Benchmark inference latency & throughput
	$(PYTHON) -m auras.deployment.benchmark $(ARGS)

# ── Quality ───────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/ scripts/

.PHONY: format
format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format src/ tests/ scripts/

.PHONY: test
test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=auras --cov-report=html

# ── Utilities ─────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove caches and build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .mypy_cache build dist *.egg-info

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
