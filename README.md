# aoi-deepstream-tensorrt

> **Status**: 🚧 Active development — 2-week sprint started 2026-05-05. README is updated daily; results land progressively.

End-to-end Automated Optical Inspection (AOI) defect detection MVP on the NVIDIA inference stack: **PyTorch → ONNX → TensorRT → DeepStream**. Built to validate latency, throughput, and image-quality robustness for industrial inspection workloads.

## Project Management

- **[GitHub Project Board](https://github.com/users/hsuani/projects/1)** — Epics, Stories, Sprint tracking
- [Project Charter](docs/project-charter.md) — goal, scope, timeline
- [Risk Register](docs/risk-register.md) — open / mitigated / accepted risks
- [Architecture Decision Records](docs/adr/) — 6 ADRs covering model choice, calibration, precision selection, deployment scope
- [Day-by-day notes](notes/) — daily logs, retros, hyperparameter ablation

## Why this project

Smartphone ISP work taught me one thing the AOI literature undersells: **detection accuracy is upstream-bounded by sensor and pipeline image quality**. A model that hits 99% AUC on clean MVTec data can collapse under realistic factory conditions — exposure drift, sensor noise, alignment jitter — which the standard benchmarks ignore.

This repo benchmarks an AOI pipeline twice: once on the canonical MVTec AD test set, and once under ISP-style perturbations that mimic real production-line variance. The delta is the story.

## Architecture

```
                  +-------------------+        +------------------+
   MVTec AD --->  |  PyTorch training |  --->  |  ONNX (opset 17) |
                  |  (YOLOv8-seg +    |        +------------------+
                  |   EfficientAD)    |                 |
                  +-------------------+                 v
                                              +------------------+
                                              | TensorRT engine  |
                                              | FP32 / FP16 /    |
                                              | INT8 (entropy)   |
                                              +------------------+
                                                       |
                                                       v
            +-------------------------------------------------------------+
            |  DeepStream 9.0 pipeline (Python)                           |
            |  uridecodebin -> nvstreammux -> nvinfer -> nvdsosd -> sink  |
            +-------------------------------------------------------------+
                                                       |
                                                       v
                                          Annotated mock factory stream
```

## Stack

| Layer | Tool |
|---|---|
| Training | PyTorch · Ultralytics YOLOv8 · EfficientAD |
| Export | ONNX (opset 17, dynamic batch) |
| Optimization | TensorRT 10.x (10.14 in DS 9.0 container) · `trtexec` · entropy calibration |
| Streaming | DeepStream 9.0 · `pyservicemaker` · GStreamer · `nvinfer` · `nvdsosd` |
| Dataset | MVTec AD (transistor, cable, metal_nut) |
| Hardware | Mac M5 (MPS) for training · Tesla T4 (Kaggle) for TRT bench · L4 (GCP) for DeepStream pipeline · A10 / L40S as production targets · CUDA 12.x (Kaggle) / 13.1 (DS 9.0 container) |

## Results

### Live Demo

![AOI defect detection demo](docs/deepstream_demo.gif)

> 28-sec clip showing per-frame defect detection on 115 MVTec metal_nut test
> images. Generated locally on Mac (Ultralytics + MPS backend) for portfolio
> embedding; production DeepStream pipeline (D7) produces visually equivalent
> overlays at sub-2 ms latency on NVIDIA L4 GPU (see DeepStream Multi-Stream
> table below).
> [Full HD mp4](docs/deepstream_demo.mp4).

### Latency / Throughput — TensorRT (NVIDIA Tesla T4, batch=1, imgsz=640)

| Precision | Engine size | Build time | Latency mean | Throughput | Speedup |
|-----------|-------------|------------|--------------|------------|---------|
| FP32      | 58.0 MB     | 57 s       | 10.91 ms     | 91.6 qps   | 1.00×   |
| FP16      | 25.3 MB     | 319 s      | 3.49 ms      | 286.5 qps  | **3.13×** |
| INT8      | 15.7 MB     | 480 s      | **2.09 ms**  | **478.1 qps** | **5.22×** |

![Benchmark](docs/trt_benchmark_t4.png)

INT8 calibration with `IInt8EntropyCalibrator2` over 335 MVTec metal_nut images.
Functional output drift: FP16 < 0.001% vs FP32, INT8 8.6%. Per-precision mAP
on val set deferred to production hardware verification.

Full benchmark methodology + reproducibility: [`benchmarks/trt-t4.md`](benchmarks/trt-t4.md).

### DeepStream Multi-Stream — NVIDIA L4 GPU (GCP), DS 9.0, INT8, batch=4

| Run | fps / stream | total img/s | RT @ 30 fps/stream |
|-----|--------------|-------------|---------------------|
| 1   | 116.17       | 464.68      | ✅ 3.8× headroom    |
| 2   | 116.43       | 465.72      | ✅                  |

L4 vs T4 (D4): INT8 BS=1 throughput 478 → 540 qps (~1.13× speedup).
DeepStream pipeline overhead absorbs ~57% of raw `trtexec` throughput (808 → 465
img/s) — within typical DS production envelope (decode + memcpy + nvinfer + osd
+ sink). KITTI dump confirms parser correctness with 9 876 metal_nut defect
detections across the 4-stream run.

Skill-orchestrated via NVIDIA `deepstream-byovm` agentic skill (det-only path
per [ADR-0006](docs/adr/0006-detection-only-deployment.md)).

Full PDF report: [`benchmark_report.pdf`](src/deepstream/models/yolov8s_seg_metal_nut/reports/benchmark_report_yolov8s_seg_metal_nut.pdf).

### ISP-aware Robustness Study (D8)

Three perturbation modules ([`src/isp_aug/`](src/isp_aug/)) calibrated for
production-line spectrum, all linear-domain physics (de-gamma → perturb →
re-gamma):

- **noise**: Poisson shot + Gaussian read + ISO/gain push + Bayer-pattern asymmetry
- **exposure**: linear gain + WB drift (per-100K R/B asymmetry) + gamma offset
- **alignment**: rotation + translation + anisotropic scale (single affine)

13-cell mAP@50 (mask) matrix, baseline = 0.7502:

| Perturbation | s1 | s2 | s3 |
|---|---|---|---|
| noise | 0.034 (-95%) | 0.030 | 0.001 (-99.8%) |
| exposure | 0.741 (-1%) | 0.749 | 0.733 (-2%) |
| alignment | 0.698 (-7%) | 0.309 | 0.126 (-83%) |
| combined | 0.018 | 0.005 | 0.000 |

**Empirical sensitivity ranking**: NOISE >>> ALIGNMENT > EXPOSURE
(predicted in [ADR-0007](docs/adr/0007-isp-aware-perturbation-hypotheses.md) was
NOISE > EXPOSURE > ALIGNMENT — partially wrong, exposure/alignment swapped).

**Top finding**: production-line normal noise (SNR 30-40 dB) collapses mAP
from 0.75 → 0.03 (-95%). Training had zero sensor-noise augmentation →
any noise input is fully OOD. Mitigation path: integrate `noise.apply()`
(same Poisson + Gaussian linear-domain model used for D4 INT8 calibration)
as a training-time augmentation transform — calibration cache and training
noise model share one physics, bridging quantization design with
deployment robustness.

**Visual proof**:
- ![Robustness moderate](docs/robustness_grid_moderate.png)

  5 sample images × 5 cells at moderate severity. Clean detects defects
  cleanly; noise / combined produce "blue-blob" oversized masks; exposure
  preserves detection; alignment partial.

- [Severity progression on `scratch_007`](docs/robustness_grid_severity.png)
  shows per-perturbation degradation across s1/s2/s3.

**Per-defect insights**: see
[`benchmarks/robustness.md`](benchmarks/robustness.md) §6 +
[`benchmarks/robustness_per_defect.csv`](benchmarks/robustness_per_defect.csv).
- `flip` is **alignment-invariant** (mAP 0.995 across all alignment
  severities) — model learned rotation-equivariant features for
  orientation-defect class.
- `color` baseline is the bottleneck (0.396), but **slightly improves**
  under moderate/severe exposure (gain push enhances tonal contrast cue).
- `scratch + noise` is the most fragile pairing (high-frequency cue drowns
  in additive noise).

Full methodology: [`benchmarks/robustness.md`](benchmarks/robustness.md).

## Repo layout

```
.
├── src/
│   ├── data/
│   │   ├── mvtec_to_yolo.py          # MVTec → YOLOv8-seg format converter
│   │   └── visualize_yolo_label.py   # polygon overlay sanity check
│   ├── train.py                      # YOLOv8s-seg fine-tune entry (recipe v1/v2/v3)
│   ├── export_onnx.py                # ONNX export with onnxruntime verification
│   ├── deepstream/
│   │   └── models/yolov8s_seg_metal_nut/   # skill bundle: parser .cpp + Dockerfile + scripts + reports + bench logs
│   └── isp_aug/                      # ISP-aware augmentation (D8-D9, planned)
├── notebooks/
│   └── aoi-tensorrt-benchmark.ipynb  # Kaggle T4 FP32/FP16/INT8 build + benchmark
├── benchmarks/
│   └── trt-t4.md                     # Tesla T4 benchmark methodology + results
├── engines/
│   └── metal_nut_int8.cache          # entropy calibration cache (reproducible re-build)
├── notes/                            # day-by-day logs, retros, hyperparameter ablation
└── docs/
    ├── adr/                          # 6 architecture decision records (incl. ADR-0006 det-only deployment)
    ├── project-charter.md
    ├── risk-register.md
    └── trt_benchmark_t4.png          # latency / throughput / size bar plot
```

## Running

### 1. Train baseline (Mac MPS or NVIDIA GPU)

```bash
git clone git@github.com:hsuani/aoi-deepstream-tensorrt.git
cd aoi-deepstream-tensorrt
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Convert MVTec → YOLO format (download MVTec AD first; see notes/day-02-training_summary.md)
python src/data/mvtec_to_yolo.py --src data/mvtec --dst data/yolo

# Train metal_nut (recipe v1: lr=auto, full augmentation, ~30 min on M5 / T4)
python src/train.py --class metal_nut --epochs 50 --recipe v1
```

### 2. Export ONNX

```bash
python src/export_onnx.py \
  --weights runs/segment/metal_nut_v1/weights/best.pt \
  --imgsz 640 --dynamic --simplify
```

### 3. Build TRT engines + benchmark (Kaggle T4)

Open [`notebooks/aoi-tensorrt-benchmark.ipynb`](notebooks/aoi-tensorrt-benchmark.ipynb)
on a Kaggle notebook with T4 GPU + Internet enabled. Upload the ONNX as a Kaggle
Dataset and the calibration data (MVTec metal_nut subset). Run cells top-to-bottom
(~30 min total: FP32 1m + FP16 5m + INT8 8m + benchmarks).

### 4. DeepStream 9.0 multi-stream pipeline (NVIDIA L4 GPU)

```bash
# scp ONNX + calibration cache + scripts to GPU host (e.g. GCP L4 VM)
gcloud compute scp --recurse --zone us-central1-a \
  src/deepstream/models/yolov8s_seg_metal_nut \
  <vm>:~/ds-metal-nut

# On GPU host:
cd ~/ds-metal-nut
docker build -t aoi-metal-nut:ds9 -f docker/Dockerfile .
docker run --rm --gpus all \
  -v "$PWD/model":/app/model \
  -v "$PWD/benchmarks":/app/benchmarks \
  -v "$PWD/samples":/app/samples \
  -v "$PWD/sources":/app/sources \
  -v "$PWD/reports":/app/reports \
  -v ~/mvtec/metal_nut/test:/data/mvtec/metal_nut/test:ro \
  aoi-metal-nut:ds9 all
```

Mac-side ffmpeg used to pre-generate 4 × 120 s test loop mp4 (DS 9.0 samples-multiarch
container lacks `x264enc` / `openh264enc` / `/dev/v4l2-nvenc`).

Skill-orchestrated via NVIDIA `deepstream-byovm` agentic skill (det-only, ADR-0006).

## Roadmap

- [x] **D1** — Repo skeleton, MVTec → YOLO converter, dev environment
- [x] **D2** — MVTec AD baseline (3 classes; metal_nut mAP@50 0.75; transistor / cable cross-class study)
- [x] **D3** — ONNX export with dynamic batch + onnxruntime verification
- [x] **D4** — TensorRT FP32 / FP16 / INT8 build + benchmark on Tesla T4 (5.22× INT8 speedup)
- [x] **D5** — GCP L4 VM + NGC DeepStream 9.0 environment (driver 590, Ubuntu 24.04, NV stack alive)
- [x] **D6** — `deepstream-byovm` skill autonomous mode generated end-to-end pipeline scaffolding (det-only path per ADR-0006)
- [x] **D7** — DS 9.0 multi-stream pipeline on L4 (4 streams × 116 fps = 465 img/s aggregate, INT8); skill-orchestrated; PDF benchmark report committed
- [x] **D8** — ISP-aware robustness study (3 perturbation modules + combined-apply, 13-cell mAP matrix + per-defect breakdown; ADR-0007 hypothesis verdicts)
- [ ] **D9** — Heatmap viz + per-precision INT8/FP16 cross-check (deferred to GCP L4 v2)
- [ ] **D10-D14** — Documentation polish, blog post, NVIDIA Inception application
- [ ] _Stretch_ — EfficientAD secondary baseline; Jetson Orin Nano deployment

## Limitations & honest caveats

- 2-week scope: 3 MVTec classes, single-model 4-stream DeepStream config. Multi-model cascaded inference (e.g. PGIE + SGIE) is future work.
- INT8 calibration uses 335 representative MVTec metal_nut images (train/good + test/good + test/defect across 4 defect types); production deployment would require calibration on representative line data.
- ISP-aware augmentation is parametric, not derived from real factory captures.

## Author

Yu-Hsuan (Shane) Tseng — ex-Qualcomm ISP/Camera, building toward NVIDIA Metropolis (Manufacturing). Prior experience: smartphone ISP tuning, MediaPipe geometric analysis, serverless ML deployment.

## License

MIT
