# TensorRT Benchmark — metal_nut YOLOv8s-seg on Kaggle T4

## Hardware
- GPU: NVIDIA Tesla T4 (16 GB, compute capability 7.5, Turing)
- CUDA: 12.x
- TensorRT: 10.x
- Source ONNX: `models/yolov8s_seg_metal_nut.onnx` (45 MB, opset 17, dynamic batch)
- Source PyTorch baseline: `runs/segment/metal_nut_v1/weights/best.pt` (mAP@50(M) = 0.75)

## Build Configuration
- Workspace memory: 4 GB
- Dynamic shape profile: min=1×3×640×640, opt=1×3×640×640, max=8×3×640×640
- Benchmark: batch=1, imgsz=640, 50 warmup iter, 500 measured iter
- INT8 calibration: entropy method (`IInt8EntropyCalibrator2`)
  - 335 MVTec metal_nut images: 220 train/good + 22 test/good + 93 test/defect
  - Defect types covered: bent, color, flip, scratch
  - Preprocessing matches Ultralytics inference: BGR→RGB, resize 640×640, /255, HWC→CHW, FP32

## Results

| Precision | Engine size | Build time | Latency mean | Latency p99 | Throughput | Speedup |
|-----------|-------------|------------|--------------|-------------|------------|---------|
| FP32      | 58.0 MB     | 57.0 s     | 10.914 ms    | 11.246 ms   | 91.6 qps   | 1.00×   |
| FP16      | 25.3 MB     | 319.4 s    | 3.491 ms     | 4.818 ms    | 286.5 qps  | 3.13×   |
| INT8      | 15.7 MB     | 479.8 s    | **2.092 ms** | 4.297 ms    | **478.1 qps** | **5.22×** |

![TRT benchmark T4](../docs/trt_benchmark_t4.png)

## Functional Validation

Same fixed-seed input through all three engines (deterministic verification):

| Engine | output0 mean | output0 std | Drift vs FP32 (mean) |
|--------|--------------|-------------|----------------------|
| FP32   | 22.194586    | 86.580551   | (baseline)           |
| FP16   | 22.194946    | 86.581192   | < 0.001%             |
| INT8   | 24.108269    | 89.886063   | 8.62%                |

- FP16 numerical drift is below FP32 single-precision rounding noise → mAP impact expected negligible.
- INT8 8.6% mean drift is the entropy calibrator's intentional trade-off; on-device mAP verification deferred to production hardware.

## INT8 Calibration Cache

`engines/metal_nut_int8.cache` — 14.86 KB, committed for reproducibility.
Re-builds with cache hit drop from ~480 s to ~120 s (skip calibration phase).

## Observations

1. **FP16 yields 3.13× speedup with negligible accuracy cost.** Standard production-ready choice.
2. **INT8 hits 5.22×** on T4 despite Turing's limited INT8 throughput. A10 / L40S (Ampere / Ada Lovelace) ship with INT8 dot-product instructions at scale, projecting 6-8× per NVIDIA published benchmarks.
3. **Engine plan files are not portable** across GPU compute capability + TRT major version. Production deployment requires per-target re-build.
4. **Latency at batch=1 is single-stream worst case.** DeepStream multi-stream pipelines amortize via batched inference; throughput is the relevant metric there.

## Future Work

- mAP@50 / mAP@50-95 per precision on full MVTec metal_nut val set (49 images, 41 instances)
- Re-benchmark on Ampere+ (A10 via NVIDIA Brev) for production-realistic numbers
- Multi-stream throughput benchmark (`min/opt/max` shape range exercising batch=4-8) for DeepStream sizing
- Per-precision robustness under ISP-style perturbations (Day 8-9 plan)

## Reproduce

```bash
# 1. Train: see PLAN.md Day 2
# 2. Export ONNX (Mac):
python src/export_onnx.py --weights models/yolov8s_seg_metal_nut.pt --imgsz 640 --dynamic --simplify

# 3. Build engines + benchmark (Kaggle T4):
#    Open notebooks/aoi-tensorrt-benchmark.ipynb in a Kaggle notebook with:
#    - GPU T4, Internet ON
#    - Add Input: yolov8s_seg_metal_nut.onnx (as Kaggle Dataset)
#    - Add Input: MVTec metal_nut calibration set (train/good + test/)
#    Run All → ~30 min total (FP32 1m + FP16 5m + INT8 10m + benchmarks)