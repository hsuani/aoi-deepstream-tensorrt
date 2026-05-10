#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES_DIR="$MODEL_DIR/sources/test_images"
IMAGES_ROOT="${IMAGES_ROOT:-/data/mvtec/metal_nut/test}"
FPS="${FPS:-15}"
DURATION_SEC="${DURATION_SEC:-120}"
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
echo "Staged $i frames into $STAGING; target NUM_BUF=$NUM_BUF (~${DURATION_SEC}s @ ${FPS}fps)"

EXT=mp4
OUT_MASTER="$SOURCES_DIR/test_loop_master.${EXT}"

if gst-inspect-1.0 x264enc >/dev/null 2>&1; then
  ENC_CHAIN="x264enc bitrate=4000 speed-preset=ultrafast tune=zerolatency key-int-max=30 ! h264parse ! mp4mux"
elif gst-inspect-1.0 openh264enc >/dev/null 2>&1; then
  ENC_CHAIN="openh264enc bitrate=4000000 ! h264parse ! mp4mux"
else
  echo "ERROR: no h264 encoder available"
  exit 1
fi
echo "Encoder chain: $ENC_CHAIN"

gst-launch-1.0 -e \
  multifilesrc location="$STAGING/img_%05d.png" caps="image/png,framerate=${FPS}/1" \
    loop=true num-buffers=$NUM_BUF ! \
  pngdec ! videoconvert ! videoscale ! \
  "video/x-raw,width=1280,height=720,format=I420" ! \
  $ENC_CHAIN ! \
  filesink location="$OUT_MASTER"

ls -lh "$OUT_MASTER"

for s in 0 1 2 3; do
  cp -f "$OUT_MASTER" "$SOURCES_DIR/test_loop_${s}.${EXT}"
done
ls -lh "$SOURCES_DIR"

# Patch ds_app config: .ogv -> .mp4 + drop invalid file-loop key
DS_CFG="$MODEL_DIR/benchmarks/ds/ds_app_4stream.txt"
sed -i.bak 's|test_loop_\([0-9]\)\.ogv|test_loop_\1.mp4|g' "$DS_CFG"
sed -i '/^file-loop=1/d' "$DS_CFG"
echo "Patched $DS_CFG"

echo "DONE: $SOURCES_DIR/test_loop_{0,1,2,3}.${EXT}"
