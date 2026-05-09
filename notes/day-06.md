# Day 6 — Skill-Orchestrated DeepStream Pipeline (det-only path)

## Done
- Ran NVIDIA `deepstream-byovm` skill (autonomous mode) on Mac against
  `yolov8s_seg_metal_nut.onnx` + `metal_nut_int8.cache`.
- Skill chose Path A (detection-only) per ADR-0006: output1 prototype masks
  bound but not consumed; output0 sliced [1, 5, 8400] (4 bbox + 1 cls).
- Generated 17-file artifact bundle under
  `src/deepstream/models/yolov8s_seg_metal_nut/`.

## Skill artifacts

| # | Item | File(s) |
|---|------|---------|
| 1 | pyservicemaker mock factory pipeline | `scripts/run_pyservicemaker_mock_factory.py` |
| 2 | Det-only YOLOv8-seg parser (output0 ch 0..4) | `parser/nvdsinfer_custombboxparser_yolov8s_seg_metal_nut.cpp` + `Makefile` |
| 3 | nvinfer configs (single-stream + ds-bench batch=4), labels | `config/config_infer_primary_*.txt`, `benchmarks/ds/config_infer_ds_*.txt`, `benchmarks/ds/ds_app_4stream.txt`, `config/labels.txt` |
| 3a | Engine build script (ONNX → INT8 inside container, BS=1 + BS=4) | `scripts/build_engine.sh` |
| 4 | Dockerfile (DS 9.0 base, parser built at image build, multi-mode entrypoint) | `docker/Dockerfile` + `docker/entrypoint.sh` |
| 5 | Multi-stream bench (4 streams, 2 runs, **PERF parsing) | `scripts/bench_multistream.sh` + `scripts/prep_test_loops.sh` |
| 6 | PDF report generator (5 charts, 12-section md, html, PDF) | `scripts/generate_report.py` |
| — | Operator README | `src/deepstream/models/yolov8s_seg_metal_nut/README.md` |

## Skill-conformance properties

- Engine name fixed: `yolov8s_seg_metal_nut_dynamic_b4.engine` (no bare suffix).
- `batch-size == num_streams` (= 4) for multi-stream run.
- Log filenames fixed: `trtexec_b{1,4}.log`, `ds_s4_run{1,2}.log`.
- Parser uses `NvDsInferObjectDetectionInfo obj = {};` (DS 9.0 zero-init for OBB safety).
- Encoder fallback: NVENC → `.mp4`; otherwise `theoraenc + oggmux` → `.ogv`.
- Mock factory: `multifilesrc` walks `metal_nut/test/*/*.png` at 10 fps, loop=true.

## Out of scope (explicitly per skill rules)

- **No instance-mask metadata.** Mask recovery requires `network-type=3` plus
  custom postprocess library (sigmoid(proto · coef) → upsample → bbox crop)
  not in skill template registry. Tracked as D7 stretch goal.
- **KITTI accuracy gate skipped.** Skill rule 5 disables gate when path forks
  at "strip seg head". Inserting `ds-kitti-dump.sh` before
  `bench_multistream.sh` would re-enable; deferred.

## Pending

- Skill output is **scaffolding only**. Actual engine build + multi-stream
  bench + PDF report require execution on the GCP L4 VM (Day 7 first task).
- ONNX + INT8 cache must be copied into
  `src/deepstream/models/yolov8s_seg_metal_nut/model/` before scp.

## Time spent

- Skill orchestration (Mac): ~1.5 hr
- Reading skill output + ADR-0006 decision: ~30 min
- Notes + commit: ~20 min
- Total: ~2.5 hr (vs ~6-10 hr estimated for hand-written seg pipeline)

## Commit
`<placeholder — filled after push>`
