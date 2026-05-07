# aoi-deepstream-tensorrt

> **Status**: 🚧 Active development — 2-week sprint started 2026-05-05. README is updated daily; results land progressively.

End-to-end Automated Optical Inspection (AOI) defect detection MVP on the NVIDIA inference stack: **PyTorch → ONNX → TensorRT → DeepStream**. Built to validate latency, throughput, and image-quality robustness for industrial inspection workloads.

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
            |  DeepStream 7.x pipeline (Python)                           |
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
| Optimization | TensorRT 10.x · `trtexec` · entropy calibration |
| Streaming | DeepStream 7.x · GStreamer · `nvinfer` · `nvdsosd` |
| Dataset | MVTec AD (transistor, cable, metal_nut) |
| Hardware | NVIDIA A10 / L40S (cloud) · CUDA 12.x |

## Results

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

### ISP-aware Robustness Study _(planned, week 2)_

Re-benchmark on perturbed test set:
- Sensor noise (Gaussian + Poisson, three intensities)
- Exposure drift (±2 EV)
- Alignment jitter (rotation ±3°, translation ±5 px)

Hypothesis: INT8 quantization amplifies sensitivity to low-SNR inputs. To be measured.

## Repo layout

```
.
├── src/
│   ├── train.py            # PyTorch training entry
│   ├── export_onnx.py      # ONNX export
│   ├── build_engine.py     # trtexec wrapper, calibration
│   ├── deepstream/         # gst pipeline + config
│   └── isp_aug/            # ISP-aware augmentation module
├── benchmarks/             # results, plots
├── configs/                # nvinfer + DeepStream configs
├── notebooks/              # exploration
└── docs/                   # architecture diagrams, design notes
```

## Running

```bash
# Pull NGC DeepStream container
docker pull nvcr.io/nvidia/deepstream:7.0-gc-triton-devel

# Inside container
git clone <this-repo>
cd aoi-deepstream-tensorrt
pip install -r requirements.txt

# 1. Train baseline
python src/train.py --class transistor --epochs 50

# 2. Export + build engine
python src/export_onnx.py --ckpt models/yolov8s_transistor.pt
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# 3. Run DeepStream pipeline
python src/deepstream/run.py --config configs/aoi_pipeline.txt
```

Full setup notes in [`docs/setup.md`](docs/setup.md).

## Roadmap

- [x] Repo skeleton, NGC container verified
- [ ] MVTec AD baseline (3 classes)
- [ ] ONNX export + FP32/FP16 engines
- [ ] INT8 calibration + 3-precision benchmark
- [ ] DeepStream pipeline + 30-sec demo video
- [ ] ISP-aware augmentation + robustness re-benchmark
- [ ] EfficientAD secondary baseline
- [ ] Jetson Orin Nano deployment (stretch)

## Limitations & honest caveats

- 2-week scope: 3 MVTec classes, single-stream DeepStream config. Multi-stream + multi-model orchestration is future work.
- INT8 calibration uses MVTec train split; production deployment would require calibration on representative line data.
- ISP-aware augmentation is parametric, not derived from real factory captures.

## Author

Hsuani — ex-Qualcomm ISP/Camera, building toward NVIDIA Metropolis (Manufacturing). Prior experience: smartphone ISP tuning, MediaPipe geometric analysis, serverless ML deployment.

## License

MIT
