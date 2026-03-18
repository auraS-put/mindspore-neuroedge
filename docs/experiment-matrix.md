# Experiment Matrix

## Objective

Compare architectures under the same preprocessing and evaluation protocol.

## Base Evaluation Protocol

- Patient-aware split where possible
- Rolling validation for temporal robustness
- Class-weighted loss for imbalance
- Early stopping on Recall/F1 composite

## Primary Metrics

- Recall (Sensitivity)
- False Alarm Rate (FAR)
- F1-score
- AUC-ROC
- Inference latency (ms/window)
- Peak memory (MB)
- Energy proxy (relative score)

## Planned Runs

| Run ID | Family | Variant | Seq Len | Channels | Loss Weights | Notes |
|---|---|---|---:|---:|---|---|
| R01 | LSTM | 2-layer | 512 | 4 | 1:8 | baseline |
| R02 | BiLSTM | 2-layer | 512 | 4 | 1:8 | temporal baseline |
| R03 | MobileNetV3-1D | small | 512 | 4 | 1:8 | efficiency candidate |
| R04 | GhostNet-1D | x1.0 | 512 | 4 | 1:8 | Huawei-oriented candidate |
| R05 | MobileViT-1D | tiny | 768 | 6 | 1:10 | long-context candidate |
| R06 | Autoformer | compact | 768 | 6 | 1:10 | long-range dependencies |

## Acceptance Gates

- Gate A (Model quality): Recall must exceed baseline by pre-agreed margin
- Gate B (Usability): FAR must remain below target threshold
- Gate C (Runtime): Latency suitable for near real-time alerting
- Gate D (Deployability): memory footprint acceptable for edge target

## Logging Contract

Each run stores:

- config snapshot
- random seed
- metrics.json
- per-class confusion matrix
- calibration curve and threshold rationale
