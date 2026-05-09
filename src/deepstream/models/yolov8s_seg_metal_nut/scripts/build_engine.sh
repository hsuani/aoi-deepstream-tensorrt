#!/usr/bin/env bash
# Build TRT engine from yolov8s_seg_metal_nut.onnx using the supplied INT8 calibration cache.
# Outputs:
#   benchmarks/engines/yolov8s_seg_metal_nut_dynamic_b4.engine
#   benchmarks/engines/timing.cache
#   benchmarks/b1/trtexec_b1.log
#   benchmarks/b4/trtexec_b4.log

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="yolov8s_seg_metal_nut"
MAX_BS="${MAX_BS:-4}"

ONNX="$MODEL_DIR/model/${MODEL_NAME}.onnx"
CALIB="$MODEL_DIR/model/metal_nut_int8.cache"
ENGINE_DIR="$MODEL_DIR/benchmarks/engines"
ENGINE="$ENGINE_DIR/${MODEL_NAME}_dynamic_b${MAX_BS}.engine"
TIMING="$ENGINE_DIR/timing.cache"
B1_DIR="$MODEL_DIR/benchmarks/b1"
BMAX_DIR="$MODEL_DIR/benchmarks/b${MAX_BS}"

mkdir -p "$ENGINE_DIR" "$B1_DIR" "$BMAX_DIR"

[ -f "$ONNX"  ] || { echo "ERROR: missing ONNX $ONNX"; exit 1; }
[ -f "$CALIB" ] || { echo "ERROR: missing INT8 cache $CALIB"; exit 1; }

echo "=== Building INT8 engine (BS=1..$MAX_BS) ==="
trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --timingCacheFile="$TIMING" \
  --int8 \
  --calib="$CALIB" \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:${MAX_BS}x3x640x640 \
  --maxShapes=images:${MAX_BS}x3x640x640 \
  --memPoolSize=workspace:4096M \
  --noDataTransfers \
  --useCudaGraph \
  2>&1 | tee "$ENGINE_DIR/build.log"

[ -f "$ENGINE" ] || { echo "ERROR: engine not produced"; exit 1; }

echo "=== Benchmark BS=1 ==="
trtexec \
  --loadEngine="$ENGINE" \
  --shapes=images:1x3x640x640 \
  --warmUp=2000 --duration=10 --iterations=200 \
  --noDataTransfers --useCudaGraph \
  --avgRuns=100 \
  2>&1 | tee "$B1_DIR/trtexec_b1.log"

echo "=== Benchmark BS=$MAX_BS ==="
trtexec \
  --loadEngine="$ENGINE" \
  --shapes=images:${MAX_BS}x3x640x640 \
  --warmUp=2000 --duration=10 --iterations=200 \
  --noDataTransfers --useCudaGraph \
  --avgRuns=100 \
  2>&1 | tee "$BMAX_DIR/trtexec_b${MAX_BS}.log"

QPS=$(grep -oP 'Throughput:\s*\K[0-9.]+' "$BMAX_DIR/trtexec_b${MAX_BS}.log" | tail -1)
IMGS=$(python3 -c "print(round(${QPS:-0} * $MAX_BS, 2))")
echo "=== Engine ready: $ENGINE"
echo "    BS=$MAX_BS throughput=${QPS} batches/s = ${IMGS} img/s"
