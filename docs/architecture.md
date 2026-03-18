# auraS Technical Architecture

## 1. System Pattern

auraS uses Cloud-Train, Edge-Infer:

- Training and model selection occur in cloud workflows (ModelArts)
- Inference and safety actions execute on-device for low latency and privacy

## 2. Runtime Dataflow

1. Wearable stream arrives to smartphone via Bluetooth.
2. Preprocessing normalizes, filters, and windows the signal.
3. Feature tensor is passed to a quantized MindSpore Lite model.
4. Model score is calibrated into a risk probability.
5. Decision engine applies threshold + cooldown policy.
6. Alert manager triggers local warning and escalation if required.

## 3. Core Components

- `src/auras/data`: dataset adapters, channel mapping, preprocessing
- `src/auras/models`: model families (LSTM/CNN/Transformer candidates)
- `src/auras/training`: training loop, weighted loss, metrics, checkpoints
- `src/auras/inference`: sliding windows, post-processing, thresholding
- `src/auras/deployment`: conversion and quantization helpers

## 4. Non-Functional Priorities

- Reliability: prioritize Recall, robust thresholding, fail-safe alerts
- Efficiency: mobile-oriented architectures, quantization, profiling
- Privacy: local inference by default, minimal sensitive data transfer
- Reproducibility: config-driven experiments and strict run artifact logging

## 5. Experiment Governance

Every run should produce:

- immutable config snapshot
- dataset and split fingerprint
- checkpoint and metric report
- inference latency and memory profile
- threshold calibration details

## 6. Integration with HarmonyOS

- App receives live sensor windows
- Embedded runtime loads `.ms` model
- UI state machine transitions: Monitoring -> Warning -> Escalation
- HMS kits provide push and location services during escalation
