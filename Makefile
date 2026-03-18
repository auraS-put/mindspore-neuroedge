PYTHON ?= python3

.PHONY: install test lint train eval export

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

lint:
	$(PYTHON) -m py_compile src/auras/__init__.py

train:
	bash scripts/run_experiment.sh configs/model/lstm.yaml configs/train/default.yaml

eval:
	bash scripts/evaluate_model.sh experiments/runs/latest

export:
	bash scripts/export_lite.sh experiments/runs/latest/checkpoints/best.ckpt
