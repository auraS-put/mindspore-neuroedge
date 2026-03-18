# Data Directory

Do not commit proprietary or sensitive datasets to git.

Expected layout:

- `raw/`: untouched source files
- `interim/`: cleaned intermediate artifacts
- `processed/`: model-ready windows/features
- `external/`: third-party metadata and references

Use dataset-specific ingestion scripts to preserve reproducibility.
