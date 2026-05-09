# ADR-0006: Detection-only deployment (skill-driven), seg head reserved

**Status**: Accepted
**Date**: 2026-05-09

## Context
deepstream-byovm skill autonomous mode does not yet support YOLOv8-seg
output (output1 prototype masks + 32 mask coefficients). Three paths:
A. Strip seg head, deploy detection subset (4 bbox + 1 cls) via skill autonomous flow.
B. Bypass skill, write custom YOLOv8-seg parser + manual nvinfer config + Dockerfile.
C. Wait for skill seg support (not available, time-bounded).

## Decision
Path A: skill-orchestrated detection-only deployment.

## Rationale
- D6 / D7 time budget tight. Skill autonomy yields PDF report + multi-stream
  benchmark + Docker image in ~2 hr; manual seg pipeline 6-10 hr risks D7
  spill.
- bbox-only is production-realistic for AOI line decisions (PASS / FAIL
  binary). Mask is QA-time luxury, not real-time gate.
- Training metric mAP@50(M) 0.75 preserved as offline reference; pipeline
  metric is bbox mAP at deploy time.
- Skill output is canonical (auto-scaling, KITTI gate, NV-recommended layout)
  vs hand-rolled parser carrying maintenance debt.
- Custom seg parser (path B) deferred as D7 stretch goal once detection
  pipeline benchmarks land.

## Consequences
- README / CV reflect dual-headed model: trained as seg, deployed as det.
- Demo video shows bbox + class label overlay (no mask polygon).
- Mask channel available offline for ground-truth IoU comparison and future
  seg-aware DeepStream parser when skill gains support.
