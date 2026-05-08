# Project Charter — AOI Defect Detection MVP

## Goal
Two-week sprint producing a portfolio-grade industrial AOI defect detection MVP
on the NVIDIA inference stack, targeting NVIDIA Metropolis (Manufacturing) TPM
role applications.

## Scope
- In: PyTorch fine-tuning, ONNX export, TensorRT FP32/FP16/INT8, DeepStream
  multi-stream pipeline, ISP-aware robustness study, documentation.
- Out: production hardening, customer-specific deployment, multi-tenant
  inference orchestration.

## Success Criteria
- mAP@50 ≥ 0.5 on at least one MVTec class.
- TensorRT FP16 ≥ 2× speedup vs FP32.
- INT8 calibration end-to-end functional.
- DeepStream pipeline running TRT engine on mock factory video.
- Repo public-ready with README, ADRs, benchmarks, demo video.

## Stakeholders
- Sprint owner: Yu-Hsuan Tseng (engineering + TPM)
- Audience: NVIDIA Metropolis (Manufacturing) hiring team

## Timeline
| Day | Stage | Status |
|---|---|---|
| D1 | Environment + Repo | Done |
| D2 | Dataset + Training | Done |
| D3 | ONNX Export | Done |
| D4 | TensorRT | Done |
| D5-D7 | DeepStream | In Progress |
| D8-D9 | ISP Robustness | Planned |
| D10-D14 | Docs + Submit | Planned |
