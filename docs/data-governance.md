# Data Governance and Privacy Notes

## Principles

- Use open research datasets for training and benchmarking
- Keep personally identifiable information out of model artifacts
- Store only data necessary for reproducibility

## Dataset Intake Checklist

- verify citation and usage terms
- record source version and checksum when possible
- map channels to standardized naming scheme

## Handling Sensitive Signals

- local preprocessing for wearable streams
- avoid uploading raw patient streams by default
- anonymize metadata before cloud-side retraining

## Model Safety Positioning

auraS is a decision support prototype. It is not a standalone clinical diagnosis tool.

## Retention and Audit

- maintain run metadata and model version history
- document threshold changes and rationale
- preserve traceability for competition demo claims
