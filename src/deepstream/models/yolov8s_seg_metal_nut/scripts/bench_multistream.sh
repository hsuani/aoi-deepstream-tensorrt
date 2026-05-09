#!/usr/bin/env bash
# Multi-stream DeepStream benchmark at batch=4 (= num_streams).
# Wraps deepstream-app with enable-perf-measurement=1 and parses **PERF: lines.
#
# Outputs:
#   benchmarks/ds/ds_s4_run1.log
#   benchmarks/ds/ds_s4_run2.log
#   benchmarks/ds/perf_summary.json
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS_DIR="$MODEL_DIR/benchmarks/ds"
DS_APP_CFG="$DS_DIR/ds_app_4stream.txt"
NVINFER_CFG="$DS_DIR/config_infer_ds_yolov8s_seg_metal_nut.txt"
N=4

[ -f "$DS_APP_CFG" ]  || { echo "ERROR: missing $DS_APP_CFG"; exit 1; }
[ -f "$NVINFER_CFG" ] || { echo "ERROR: missing $NVINFER_CFG"; exit 1; }

run_once () {
  local LOG="$1"
  echo "=== deepstream-app run -> $LOG ==="
  ( cd "$DS_DIR" && timeout 60s deepstream-app -c "$(basename "$DS_APP_CFG")" ) \
    2>&1 | tee "$LOG" || true
}

LOG1="$DS_DIR/ds_s${N}_run1.log"
LOG2="$DS_DIR/ds_s${N}_run2.log"

run_once "$LOG1"
run_once "$LOG2"

parse_fps () {
  local LOG="$1"
  python3 - "$LOG" << 'PY'
import re, sys
path = sys.argv[1]
vals = []
with open(path, 'r', errors='ignore') as f:
    for line in f:
        # **PERF:  fps_a (avg_a)  fps_b (avg_b) ...
        if '**PERF:' in line:
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*\(', line):
                vals.append(float(m.group(1)))
if not vals:
    print(0.0); sys.exit(0)
# average per-stream FPS across last few measurement windows
tail = vals[-min(40, len(vals)):]
print(round(sum(tail) / len(tail), 2))
PY
}

FPS1=$(parse_fps "$LOG1")
FPS2=$(parse_fps "$LOG2")
TOT1=$(python3 -c "print(round(${FPS1:-0} * $N, 2))")
TOT2=$(python3 -c "print(round(${FPS2:-0} * $N, 2))")
RT=$(python3 -c "print('YES' if ${FPS2:-0} >= 30 else 'NO')")

cat > "$DS_DIR/perf_summary.json" << JSON
{
  "num_streams": $N,
  "run1": {"fps_per_stream": $FPS1, "total_fps": $TOT1, "log": "$(basename "$LOG1")"},
  "run2": {"fps_per_stream": $FPS2, "total_fps": $TOT2, "log": "$(basename "$LOG2")"},
  "real_time_30fps": "$RT"
}
JSON

echo "DS Run 1: $N streams | $FPS1 fps/stream | total=$TOT1 img/s"
echo "DS Run 2: $N streams | $FPS2 fps/stream | total=$TOT2 img/s | RT: $RT"
echo "Summary: $DS_DIR/perf_summary.json"
