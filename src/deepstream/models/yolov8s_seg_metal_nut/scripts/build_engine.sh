#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX="$MODEL_DIR/model/yolov8s_seg_metal_nut.onnx"
ENGINE_DIR="$MODEL_DIR/benchmarks/engines"
B1_DIR="$MODEL_DIR/benchmarks/b1"
BMAX_DIR="$MODEL_DIR/benchmarks/b4"
ENGINE="$ENGINE_DIR/yolov8s_seg_metal_nut_dynamic_b4.engine"
CACHE="$ENGINE_DIR/timing.cache"
MAX_BS=4

mkdir -p "$ENGINE_DIR" "$B1_DIR" "$BMAX_DIR"

[ -f "$ONNX" ] || { echo "ERROR: missing ONNX $ONNX"; exit 1; }

echo "=== Building FP16 engine (BS=1..$MAX_BS) on $(date) ==="
echo "Note: INT8 deferred — TRT 10.14 lacks tactic for YOLOv8-seg proto Conv+Sigmoid+Mul fusion."
echo "FP16 baseline serves DeepStream multi-stream demo; D4 Kaggle T4 INT8 numbers in benchmarks/trt-t4.md."

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --timingCacheFile="$CACHE" \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:${MAX_BS}x3x640x640 \
  --maxShapes=images:${MAX_BS}x3x640x640 \
  --memPoolSize=workspace:4096M \
  --noDataTransfers \
  --useCudaGraph \
  2>&1 | tee "$BMAX_DIR/trtexec_b${MAX_BS}.log"

[ -s "$ENGINE" ] || { echo "ERROR: engine empty after build"; exit 1; }

echo
echo "=== Profiling BS=1 perf on built engine ==="

trtexec \
  --loadEngine="$ENGINE" \
  --shapes=images:1x3x640x640 \
  --memPoolSize=workspace:4096M \
  --noDataTransfers \
  --useCudaGraph \
  2>&1 | tee "$B1_DIR/trtexec_b1.log"

QPS=$(grep -oP 'Throughput:\s*\K[0-9.]+' "$BMAX_DIR/trtexec_b${MAX_BS}.log" | tail -1 || echo "n/a")
echo
echo "Engine size: $(du -h "$ENGINE" | cut -f1)"
echo "BS=$MAX_BS QPS: $QPS"
