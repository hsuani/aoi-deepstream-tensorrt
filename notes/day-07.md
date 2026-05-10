# Day 7 — Pipeline Run + Multi-Stream Benchmark

## Done
- `aoi-metal-nut:ds9` Docker image built on GCP L4 VM (DS 9.0 samples-multiarch, custom YOLOv8-seg det-only parser, CUDA 13.1 dev headers).
- Skill-generated prep + Dockerfile patched to fit DS 9.0 + L4 reality:
  - `cuda-cudart-dev-13-1` + `cuda-cccl-13-1` + `cuda-crt-13-1` apt added (parser .cpp build dep).
  - Makefile include path: `/usr/local/cuda-12.8` → `/usr/local/cuda` symlink.
  - DS9 base image: `9.0-gc-triton-devel` (deprecated) → `9.0-samples-multiarch`.
- INT8 engine rebuilt in container from ONNX + calibration cache (cache hit ~120 s).
- Loop test videos generated Mac-side via ffmpeg + libx264 (120 s @ 15 fps, 1280×720). Container theora encoder produced 3-sec broken outputs; container lacked x264enc / openh264enc / `/dev/v4l2-nvenc`. Mac ffmpeg path was unblock.
- `ds_app_4stream.txt` patched: `.ogv` → `.mp4`; invalid `file-loop=1` removed (DS9 `[source*]` group does not accept).
- 4-stream multi-stream bench captured `**PERF` samples.

## Headline Numbers (L4 INT8, batch=4 = num_streams)

| Metric | Run 1 | Run 2 |
|---|---|---|
| fps / stream | 116.17 | 116.43 |
| total img/s | 464.68 | 465.72 |
| RT @ 30 fps/stream | ✅ 3.8× headroom | ✅ |

| trtexec | throughput | latency mean |
|---|---|---|
| BS=1 | 540 qps | 1.78 ms |
| BS=4 | 808 img/s | 4.80 ms / batch |

DeepStream pipeline efficiency: 808 raw → 465 aggregate ≈ 57% loss to decode + memcpy + nvinfer + osd + sink overhead (typical for production DS pipeline).

## L4 vs T4 (D4) Comparison

| Hardware | INT8 BS=1 latency | INT8 BS=1 qps | Speedup vs T4 |
|---|---|---|---|
| T4 INT8 (Kaggle D4) | 2.09 ms | 478 | 1.00× |
| **L4 INT8 (GCP D7)** | **1.78 ms** | **540** | **1.13×** |

## Out of Scope (per ADR-0006)
- No instance mask metadata in pipeline (det-only path).
- KITTI accuracy gate skipped per skill rule 5.
- DS9 `[source*]` `file-loop=1` not honored — workaround: 120 s source video so first-pass EOS arrives well past PERF capture window.

## Issues + Resolutions

| Issue | Cause | Fix |
|---|---|---|
| `cuda_runtime_api.h` not found | CUDA 12.8 path; DS9 ships CUDA 13.1; runtime image lacks dev headers | apt cuda-cudart-dev-13-1 etc; samples-multiarch base; Makefile `/usr/local/cuda` symlink |
| DS pipeline EOS at 3 sec | theora encoder produced 3-sec broken; `[tests]` `file-loop=1` ignored | Mac ffmpeg + libx264 120 s + ds_app config swap |
| `file-loop=1` warn per `[source*]` | Invalid key for DS9 | Remove key, rely on long source |
| GCP VM Spot reclaim | Spot scheduling | Cold restart; consider STANDARD model |
| `gcloud ssh` from VM 4003 error | VM SA missing compute scope | gcloud from Mac only |

## Cost
~$8 / $300 GCP trial cumulative (L4 Spot × ~25 hr Day 5-7 + storage).

## Files
- `reports/benchmark_data.json`, `benchmark_report.{md,html,pdf}`, `charts/*.png`
- `benchmarks/{b1,b4,ds}/*.log`

## Next (D8-D9)
- ISP-aware aug module (`src/isp_aug/`): signal-dependent noise + ±2 EV exposure + alignment jitter
- Per-precision robustness 9-cell mAP matrix (FP32/FP16/INT8 × 3 perturbations × 3 severities)
- Hypothesis: INT8 sensitivity to low-SNR inputs amplifies under sensor noise

## Time
~2.8 hr (rebuild + dep fixes 1.5 + Mac video gen + scp + DS patch 0.5 + bench + report 0.3 + notes 0.5)
