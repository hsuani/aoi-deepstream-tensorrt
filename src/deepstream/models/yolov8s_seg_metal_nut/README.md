# YOLOv8-seg metal_nut — DeepStream 9.0 (det-only path)

Single-class defect detection on MVTec metal_nut. Skill scope is object detection
only; YOLOv8-seg `output1` (32 prototype masks) is **bound but not consumed**, so
runtime is bbox-only. To recover masks, switch to `network-type=3` plus a custom
postprocess library (out of scope for `deepstream-byovm`).

## Inputs you bring

| File | Where it lives in the container |
|------|---------------------------------|
| `yolov8s_seg_metal_nut.onnx` | `model/` |
| `metal_nut_int8.cache` | `model/` |
| MVTec `metal_nut/test/` | mounted at `/data/mvtec/metal_nut/test` (read-only) |

## Layout (skill-conformant)

```
src/deepstream/models/yolov8s_seg_metal_nut/
  model/            ONNX + INT8 cache (mount at runtime)
  parser/           nvdsinfer_custombboxparser_yolov8s_seg_metal_nut.cpp + Makefile + .so
  config/           nvinfer config + labels.txt
  scripts/          build_engine.sh, prep_test_loops.sh,
                    run_pyservicemaker_mock_factory.py,
                    bench_multistream.sh, generate_report.py
  benchmarks/
    engines/        yolov8s_seg_metal_nut_dynamic_b4.engine, timing.cache
    b1/             trtexec_b1.log
    b4/             trtexec_b4.log
    ds/             ds_app_4stream.txt, config_infer_ds_*.txt,
                    ds_s4_run1.log, ds_s4_run2.log, perf_summary.json
  samples/          metal_nut_mock.{mp4,ogv}, kitti_output/*.txt
  sources/test_images/  test_loop_{0..3}.{mp4,ogv}
  docker/           Dockerfile, entrypoint.sh, .dockerignore
  reports/          benchmark_report.{md,html,pdf}, benchmark_data.json,
                    charts/chart_*.png
```

## Build + run

```bash
cd src/deepstream/models/yolov8s_seg_metal_nut

# 1. Drop ONNX + cache into model/
cp /path/to/yolov8s_seg_metal_nut.onnx model/
cp /path/to/metal_nut_int8.cache       model/

# 2. Build image
docker build -t aoi-metal-nut:ds9 -f docker/Dockerfile .

# 3. Full pipeline (engine -> loops -> bench -> report)
docker run --rm --gpus all \
  -v "$PWD/model":/app/model \
  -v "$PWD/benchmarks":/app/benchmarks \
  -v "$PWD/samples":/app/samples \
  -v "$PWD/sources":/app/sources \
  -v "$PWD/reports":/app/reports \
  -v /path/to/mvtec/metal_nut/test:/data/mvtec/metal_nut/test:ro \
  aoi-metal-nut:ds9 all

# 4. Single-stream pyservicemaker mock factory (after engine exists)
docker run --rm --gpus all \
  -v "$PWD/model":/app/model \
  -v "$PWD/benchmarks":/app/benchmarks \
  -v "$PWD/samples":/app/samples \
  -v /path/to/mvtec/metal_nut/test:/data/mvtec/metal_nut/test:ro \
  aoi-metal-nut:ds9 mock-factory
```

## Critical rules in this build

- Engine name is `yolov8s_seg_metal_nut_dynamic_b4.engine`. Never bare.
- `batch-size == num_streams` for the multi-stream run (4 = 4).
- Log filenames fixed: `trtexec_b{1,4}.log`, `ds_s4_run{1,2}.log`.
- Parser uses `NvDsInferObjectDetectionInfo obj = {};` (zero-init, DS 9.0 OBB safe).
- Encoder fallback: NVENC -> `.mp4`; else `theoraenc + oggmux` -> `.ogv`.

## Single GPU L4 expectations

INT8 YOLOv8s @ 640 typically lands around 600-800 img/s on L4 trtexec
`--noDataTransfers`; DS 4-stream tends to run at 60-80% of that. Numbers in the
report come from the actual `**PERF:` lines, not estimates.
