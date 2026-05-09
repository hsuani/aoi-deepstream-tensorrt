# Day 5 — GCP L4 + NGC DeepStream 9.0 Setup

## Done
- GCP $300 trial activated; project for AOI sprint
- L4 Spot VM (g2-standard-8, 1× L4 GPU) on us-central1-a, ~$0.28/hr
- Ubuntu 24.04 + NVIDIA driver 590 + CUDA 12.4 verified
- Docker + nvidia-container-toolkit installed
- NGC API key generated (Personal API Key, 1 yr expiry)
- DS9 container pulled: `nvcr.io/nvidia/deepstream:9.0-samples-multiarch`
- DS9 smoke tests passed:
  - `deepstream-app --version` → DeepStreamSDK 9.0.0
  - `gst-inspect-1.0 nvinfer` → plugin recognized
  - `gst-inspect-1.0 nvstreammux` → plugin recognized
- ONNX (45 MB) + INT8 calibration cache (15 KB) uploaded via `gcloud compute scp`
- Repo cloned to VM via `gh auth login` + `gh repo clone`
- Symlinks set: `~/aoi/models/*.onnx`, `~/aoi/engines/*.cache` → `~/onnx/`
- Mac-side: deepstream-byovm + deepstream-dev skills installed in `~/.claude/skills/`

## Container Tag

Note: actual DS9 tag at NGC differed from my initial assumption.
Used: `nvcr.io/nvidia/deepstream:9.0-samples-multiarch`
(not `9.0-gc-triton-devel` which doesn't exist for DS9 series).

## Standard Container Run Command

```bash
docker run --rm --gpus all -it \
  --network host --privileged \
  -v ~/onnx:/onnx \
  -v ~/aoi:/aoi \
  -w /aoi \
  nvcr.io/nvidia/deepstream:9.0-samples-multiarch \
  bash
