```markdown
# Day 4 — TensorRT Optimization

## Done
- Built FP32 / FP16 / INT8 TensorRT engines on Kaggle T4
- Implemented `IInt8EntropyCalibrator2` with MVTec metal_nut calibration set (335 images)
- Latency benchmarks: 50 warmup + 500 measured iterations per engine, GPU event timing
- Functional output validation (deterministic forward, drift quantification)
- Calibration cache saved (`engines/metal_nut_int8.cache`, 14.86 KB)
- Bar plot `docs/trt_benchmark_t4.png` for README + portfolio

## Headline Numbers (Kaggle T4, batch=1, imgsz=640)
- FP32: 10.91 ms / 91.6 qps  (baseline)
- FP16: **3.49 ms / 286.5 qps  (3.13× over FP32)**
- INT8: **2.09 ms / 478.1 qps  (5.22× over FP32)**

## Functional Drift (vs FP32 output0 mean)
- FP16: < 0.001%
- INT8: 8.62%

## What I Learned
- TRT layer fusion + tactic auto-tuning materially exceeds onnxruntime-CUDA EP performance.
- INT8 calibration is engineering-heavy: own preprocessing match, calibration set composition,
  `IInt8EntropyCalibrator2` lifecycle (memory pre-alloc, cursor pagination, cache I/O).
- T4 INT8 acceleration capped (Turing has no INT8 DP4A at scale); production target hardware
  is as much a model-design decision as the architecture itself.
- Engine plan files are bound to (GPU compute capability, CUDA version, TRT major version) triple.
- Build heuristics (Ampere+) noticeably absent on T4 — explicit profile shapes matter more.

## Issues Hit + Resolutions
- macOS AppleDouble files (`._*`) in calibration tar inflated image count from 335 → 670 and
  failed `cv2.imread` on shadow files. Fix: glob filter `name.startswith("._")`.
- `set_calibration_profile` / `int8_calibrator` deprecated in TRT 10.1 (warnings). PTQ path
  still functional. Migration to QAT / explicit quantization deferred.
- TRT logger warnings about "differs from one already registered" are global-state noise; ignored.

## Time Spent
- Kaggle setup + ONNX upload: 30 min
- FP32 + FP16 build + benchmark: 30 min
- INT8 calibrator code + debug + build + benchmark: 90 min
- Validation + plot + writeup: 30 min
- Total: ~3 hr

## Next
- Day 5: DeepStream pipeline (NV LaunchPad / Brev with DeepStream 7.x container)
- Day 8: ISP-aware augmentation robustness study per precision
- D5+ blocked on getting LaunchPad lab approved (1-3 days) or burning Brev signup credit ($25)