#!/usr/bin/env bash
# Pre-bundle metal_nut/test PNGs into 4 loop video files for the 4-stream
# deepstream-app benchmark. Each stream uses an identical loop file (same
# distribution; deepstream-app feeds them into 4 parallel sources).
#
# Encoder: NVENC -> .mp4 if available, else theoraenc -> .ogv.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES_DIR="$MODEL_DIR/sources/test_images"
IMAGES_ROOT="${IMAGES_ROOT:-/data/mvtec/metal_nut/test}"
FPS="${FPS:-15}"
DURATION_SEC="${DURATION_SEC:-30}"
NUM_BUF=$((FPS * DURATION_SEC))

mkdir -p "$SOURCES_DIR"
[ -d "$IMAGES_ROOT" ] || { echo "ERROR: $IMAGES_ROOT not found"; exit 1; }

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

i=0
while IFS= read -r src; do
  ln -s "$src" "$STAGING/img_$(printf '%05d' $i).png"
  i=$((i + 1))
done < <(find "$IMAGES_ROOT" -type f -name '*.png' | sort)
[ "$i" -eq 0 ] && { echo "ERROR: no PNGs under $IMAGES_ROOT"; exit 1; }
echo "Staged $i frames into $STAGING"

if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 && [ -e /dev/v4l2-nvenc ]; then
  EXT=mp4
else
  EXT=ogv
fi
echo "Encoder mode: $EXT"

OUT_MASTER="$SOURCES_DIR/test_loop_master.${EXT}"
if [ "$EXT" = "mp4" ]; then
  gst-launch-1.0 -e \
    multifilesrc location="$STAGING/img_%05d.png" caps="image/png,framerate=${FPS}/1" loop=true num-buffers=$NUM_BUF ! \
    pngdec ! videoconvert ! videoscale ! "video/x-raw,width=1280,height=720,format=I420" ! \
    nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12" ! \
    nvv4l2h264enc bitrate=4000000 ! h264parse ! mp4mux ! \
    filesink location="$OUT_MASTER"
else
  gst-launch-1.0 -e \
    multifilesrc location="$STAGING/img_%05d.png" caps="image/png,framerate=${FPS}/1" loop=true num-buffers=$NUM_BUF ! \
    pngdec ! videoconvert ! videoscale ! "video/x-raw,width=1280,height=720,format=I420" ! \
    theoraenc quality=48 ! oggmux ! \
    filesink location="$OUT_MASTER"
fi

for s in 0 1 2 3; do
  cp -f "$OUT_MASTER" "$SOURCES_DIR/test_loop_${s}.${EXT}"
done
ls -l "$SOURCES_DIR"

# Patch ds_app config if extension differs from default .mp4
DS_CFG="$MODEL_DIR/benchmarks/ds/ds_app_4stream.txt"
if [ "$EXT" = "ogv" ] && [ -f "$DS_CFG" ]; then
  sed -i.bak 's|test_loop_\([0-9]\).mp4|test_loop_\1.ogv|g' "$DS_CFG"
  echo "Patched $DS_CFG to use .ogv sources"
fi

echo "Loop sources ready: $SOURCES_DIR/test_loop_{0,1,2,3}.${EXT}"
