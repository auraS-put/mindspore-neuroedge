# Execution Plan

## Milestone 0 - Foundation (Week 1)

- Finalize repository scaffold and coding conventions
- Prepare dataset access checklist and legal/ethics constraints
- Define baseline success metrics and acceptance thresholds

Deliverables:

- repository skeleton
- experiment matrix draft
- data schema contract

## Milestone 1 - Data Pipeline (Weeks 2-3)

- Build dataset adapters for CHB-MIT, TUH, Siena
- Implement channel reduction and metadata harmonization
- Add denoising + windowing pipeline and sanity checks

Deliverables:

- standardized samples and labels
- reproducible split manifest
- preprocessing smoke tests

## Milestone 2 - Baselines (Weeks 4-5)

- Implement and train LSTM/BiLSTM baseline
- Calibrate thresholds by maximizing Recall under FAR constraints
- Establish benchmark report template

Deliverables:

- baseline checkpoints
- baseline metric table
- failure mode notes

## Milestone 3 - Efficient Models (Weeks 6-7)

- Implement MobileNetV3-1D and GhostNet-1D
- Profile latency and memory against baseline
- Select candidate for mobile deployment

Deliverables:

- model comparison report
- edge feasibility summary

## Milestone 4 - Advanced Models (Weeks 8-9)

- Evaluate MobileViT-1D and Autoformer variants
- Run ablations on sequence length/channel subsets
- Decide final model shortlist

Deliverables:

- ablation table
- finalist model shortlist

## Milestone 5 - Edge Deployment (Weeks 10-11)

- Convert checkpoints to MindSpore Lite format
- Quantize and validate model drift
- Integrate with HarmonyOS inference bridge

Deliverables:

- deployable `.ms` artifact
- edge benchmark report

## Milestone 6 - Safety Loop and Demo (Weeks 12-13)

- Implement warning UX and escalation workflow
- Perform end-to-end reliability tests
- Freeze final narrative and demo script

Deliverables:

- working end-to-end prototype
- competition demo package

## Milestone 7 - Buffer and Submission (Week 14)

- Retest on frozen build
- Polish documentation and final KPI table
- Submit and rehearse presentation
