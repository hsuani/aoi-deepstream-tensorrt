#!/usr/bin/env python3
"""
Mock factory pipeline: read metal_nut/test PNGs at a fixed cadence, run YOLOv8-seg-det
inference, draw OSD, encode to .ogv (theoraenc fallback) or .mp4 (NVENC if present).

Usage:
  python3 run_pyservicemaker_mock_factory.py \
      --images-root /data/mvtec/metal_nut/test \
      --out /output/metal_nut_mock.ogv \
      --fps 10

The script flattens metal_nut/test/{good,scratch,bent,...}/*.png into a single
numbered sequence under /tmp/mock_factory/img_%05d.png, then drives the GStreamer
pipeline via pyservicemaker.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

CONFIG_REL = "../config/config_infer_primary_yolov8s_seg_metal_nut.txt"


def gather_images(images_root: Path, staging: Path) -> int:
    staging.mkdir(parents=True, exist_ok=True)
    pngs = sorted(glob(str(images_root / "*" / "*.png")))
    if not pngs:
        pngs = sorted(glob(str(images_root / "*.png")))
    if not pngs:
        raise RuntimeError(f"No PNG images found under {images_root}")
    for i, src in enumerate(pngs):
        dst = staging / f"img_{i:05d}.png"
        if not dst.exists():
            os.symlink(os.path.abspath(src), dst)
    return len(pngs)


def has_nvenc() -> bool:
    return os.path.exists("/dev/v4l2-nvenc") or shutil.which("gst-inspect-1.0") and \
        subprocess.run(
            ["gst-inspect-1.0", "nvv4l2h264enc"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0


def build_pipeline(staging: Path, fps: int, out_path: Path, nvinfer_cfg: Path):
    from pyservicemaker import Pipeline

    p = Pipeline("metal-nut-mock-factory")

    p.add("multifilesrc", "src", {
        "location": str(staging / "img_%05d.png"),
        "caps": f"image/png,framerate={fps}/1",
        "loop": True,
    })
    p.add("pngdec", "dec")
    p.add("videoconvert", "vc1")
    p.add("videoscale", "vs")
    p.add("capsfilter", "cf1", {"caps": "video/x-raw,width=1280,height=720"})
    p.add("nvvideoconvert", "nvc1")
    p.add("capsfilter", "cf2", {"caps": "video/x-raw(memory:NVMM),format=NV12"})

    p.add("nvstreammux", "mux", {
        "batch-size": 1,
        "width": 1280,
        "height": 720,
        "live-source": 0,
        "batched-push-timeout": 40000,
    })

    p.add("nvinfer", "pgie", {"config-file-path": str(nvinfer_cfg)})

    p.add("nvvideoconvert", "nvc2")
    p.add("nvdsosd", "osd")
    p.add("nvvideoconvert", "nvc3")

    if has_nvenc() and out_path.suffix == ".mp4":
        p.add("capsfilter", "cf_enc", {"caps": "video/x-raw(memory:NVMM),format=NV12"})
        p.add("nvv4l2h264enc", "enc")
        p.add("h264parse", "parse")
        p.add("mp4mux", "mux_out")
    else:
        if out_path.suffix == ".mp4":
            print("WARN: NVENC unavailable; switching output to .ogv (theoraenc fallback)",
                  file=sys.stderr)
            out_path = out_path.with_suffix(".ogv")
        p.add("capsfilter", "cf_enc", {"caps": "video/x-raw,format=I420"})
        p.add("theoraenc", "enc", {"quality": 48})
        p.add("oggmux", "mux_out")

    p.add("filesink", "sink", {"location": str(out_path), "sync": 0})

    p.link(["src", "dec", "vc1", "vs", "cf1", "nvc1", "cf2"])
    p.link("cf2", "mux", "sink_0")
    p.link(["mux", "pgie", "nvc2", "osd", "nvc3", "cf_enc", "enc", "mux_out", "sink"])
    return p, out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out", default=Path("/output/metal_nut_mock.ogv"), type=Path)
    ap.add_argument("--fps", default=10, type=int)
    ap.add_argument(
        "--nvinfer-config",
        default=(Path(__file__).resolve().parent / CONFIG_REL).resolve(),
        type=Path,
    )
    ap.add_argument("--staging", default=Path(tempfile.gettempdir()) / "mock_factory",
                    type=Path)
    args = ap.parse_args()

    if not args.nvinfer_config.exists():
        print(f"ERROR: nvinfer config not found: {args.nvinfer_config}", file=sys.stderr)
        return 1
    if not args.images_root.exists():
        print(f"ERROR: images root not found: {args.images_root}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = gather_images(args.images_root, args.staging)
    print(f"Mock factory: {n} frames staged at {args.staging}, fps={args.fps}, "
          f"nvinfer={args.nvinfer_config}, out={args.out}")

    pipeline, out_path = build_pipeline(args.staging, args.fps, args.out, args.nvinfer_config)
    pipeline.start()
    try:
        pipeline.wait()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
