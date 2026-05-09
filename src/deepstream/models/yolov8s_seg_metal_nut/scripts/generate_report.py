#!/usr/bin/env python3
"""
Generate benchmark_data.json + 5 PNG charts + benchmark_report.{md,html,pdf}.

Inputs (paths relative to model dir):
  benchmarks/b1/trtexec_b1.log
  benchmarks/b4/trtexec_b4.log
  benchmarks/ds/ds_s4_run1.log
  benchmarks/ds/ds_s4_run2.log
  samples/kitti_output/*.txt        (optional)

Outputs:
  reports/benchmark_data.json
  reports/charts/chart_throughput.png
  reports/charts/chart_latency.png
  reports/charts/chart_ds_efficiency.png
  reports/charts/chart_perf_timeline.png
  reports/charts/chart_class_hist.png
  reports/benchmark_report.md
  reports/benchmark_report.html
  reports/benchmark_report_yolov8s_seg_metal_nut.pdf
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "yolov8s_seg_metal_nut"
MODEL_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = MODEL_DIR / "reports"
CHART_DIR = REPORT_DIR / "charts"


def parse_trtexec(log: Path) -> dict[str, float]:
    if not log.exists():
        return {"throughput_qps": 0.0, "latency_mean_ms": 0.0, "latency_median_ms": 0.0}
    text = log.read_text(errors="ignore")
    qps = re.findall(r"Throughput:\s*([0-9.]+)", text)
    mean = re.findall(r"Latency:.*?mean\s*=\s*([0-9.]+)\s*ms", text)
    med = re.findall(r"Latency:.*?median\s*=\s*([0-9.]+)\s*ms", text)
    return {
        "throughput_qps": float(qps[-1]) if qps else 0.0,
        "latency_mean_ms": float(mean[-1]) if mean else 0.0,
        "latency_median_ms": float(med[-1]) if med else 0.0,
    }


def parse_perf_log(log: Path) -> tuple[float, list[float]]:
    """Return (avg fps/stream over last 40 measurement windows, full timeline)."""
    if not log.exists():
        return 0.0, []
    timeline: list[float] = []
    with log.open(errors="ignore") as f:
        for line in f:
            if "**PERF:" not in line:
                continue
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\(", line):
                timeline.append(float(m.group(1)))
    if not timeline:
        return 0.0, []
    tail = timeline[-min(40, len(timeline)):]
    return round(sum(tail) / len(tail), 2), timeline


def parse_kitti(kitti_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not kitti_dir.exists():
        return counts
    for f in kitti_dir.glob("*.txt"):
        for line in f.read_text(errors="ignore").splitlines():
            parts = line.split()
            if parts:
                counts[parts[0]] = counts.get(parts[0], 0) + 1
    return counts


def chart_throughput(data: dict[str, Any], out: Path) -> None:
    bs = ["BS=1", "BS=4"]
    qps = [data["trtexec"]["b1"]["throughput_qps"],
           data["trtexec"]["b4"]["throughput_qps"]]
    img_s = [qps[0], qps[1] * 4]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(bs, img_s, color=["#76b900", "#1f77b4"])
    ax.set_ylabel("Images / second")
    ax.set_title("trtexec Throughput (INT8, --noDataTransfers)")
    for b, v in zip(bars, img_s):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_latency(data: dict[str, Any], out: Path) -> None:
    bs = ["BS=1", "BS=4"]
    mean = [data["trtexec"]["b1"]["latency_mean_ms"],
            data["trtexec"]["b4"]["latency_mean_ms"]]
    med = [data["trtexec"]["b1"]["latency_median_ms"],
           data["trtexec"]["b4"]["latency_median_ms"]]
    x = list(range(len(bs)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - 0.18 for i in x], mean, width=0.36, label="mean", color="#76b900")
    ax.bar([i + 0.18 for i in x], med, width=0.36, label="median", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(bs)
    ax.set_ylabel("ms")
    ax.set_title("trtexec Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_ds_efficiency(data: dict[str, Any], out: Path) -> None:
    trt_imgs = data["trtexec"]["b4"]["throughput_qps"] * 4
    runs = ["DS Run 1", "DS Run 2"]
    totals = [data["ds"]["run1"]["total_fps"], data["ds"]["run2"]["total_fps"]]
    eff = [round((t / trt_imgs * 100) if trt_imgs else 0, 1) for t in totals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(runs, totals, color=["#1f77b4", "#76b900"])
    ax.axhline(trt_imgs, color="red", linestyle="--", label=f"trtexec ceiling ({trt_imgs:.0f})")
    ax.set_ylabel("Aggregate img/s")
    ax.set_title("DeepStream 4-Stream vs trtexec Ceiling")
    for b, v, e in zip(bars, totals, eff):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}\n({e}%)",
                ha="center", va="bottom")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_perf_timeline(data: dict[str, Any], out: Path) -> None:
    t1 = data["ds"]["run1"]["timeline"]
    t2 = data["ds"]["run2"]["timeline"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if t1:
        ax.plot(t1, label="Run 1", color="#1f77b4")
    if t2:
        ax.plot(t2, label="Run 2", color="#76b900")
    ax.axhline(30, color="red", linestyle="--", label="30 fps real-time")
    ax.set_xlabel("PERF measurement index")
    ax.set_ylabel("fps / stream")
    ax.set_title("DeepStream PERF Timeline (per-stream)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_class_hist(data: dict[str, Any], out: Path) -> None:
    counts = data["kitti"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if counts:
        labels = list(counts.keys())
        vals = [counts[l] for l in labels]
        ax.bar(labels, vals, color="#76b900")
        ax.set_ylabel("Detections")
        ax.set_title("KITTI Detection Histogram")
    else:
        ax.text(0.5, 0.5, "No KITTI dump available", ha="center", va="center")
        ax.set_title("KITTI Detection Histogram (empty)")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_markdown(data: dict[str, Any], out: Path) -> None:
    trt_imgs_b4 = data["trtexec"]["b4"]["throughput_qps"] * 4
    eff1 = (data["ds"]["run1"]["total_fps"] / trt_imgs_b4 * 100) if trt_imgs_b4 else 0
    eff2 = (data["ds"]["run2"]["total_fps"] / trt_imgs_b4 * 100) if trt_imgs_b4 else 0
    rt = "YES" if data["ds"]["run2"]["fps_per_stream"] >= 30 else "NO"

    md = f"""# Benchmark Report: {MODEL_NAME}

## 1. Model
- ONNX: `model/{MODEL_NAME}.onnx`
- Architecture: YOLOv8-seg (single class, det-only at runtime)
- Input: `images` [1,3,640,640], BGR->RGB, /255
- Outputs (engine binds both, parser reads only `output0`):
  - `output0` [1,37,8400] = 4 bbox + 1 cls + 32 mask coef
  - `output1` [1,32,160,160] = prototype masks (UNUSED)

## 2. Hardware
- Target: GCP VM `aoi-d5` (us-central1-a), L4 GPU
- DeepStream 9.0 container

## 3. Engine Build
- Precision: INT8 (`metal_nut_int8.cache`)
- Dynamic batch [1..4]
- Engine: `benchmarks/engines/{MODEL_NAME}_dynamic_b4.engine`

## 4. trtexec BS=1
- Throughput: {data['trtexec']['b1']['throughput_qps']:.2f} qps
- Latency mean / median: {data['trtexec']['b1']['latency_mean_ms']:.2f} / {data['trtexec']['b1']['latency_median_ms']:.2f} ms

## 5. trtexec BS=4
- Throughput: {data['trtexec']['b4']['throughput_qps']:.2f} batches/s = {trt_imgs_b4:.2f} img/s
- Latency mean / median: {data['trtexec']['b4']['latency_mean_ms']:.2f} / {data['trtexec']['b4']['latency_median_ms']:.2f} ms

![Throughput](charts/chart_throughput.png)
![Latency](charts/chart_latency.png)

## 6. DeepStream Multi-Stream (batch=4, 4 streams)
| Run | Streams | fps/stream | total img/s | DS efficiency |
|-----|---------|------------|-------------|---------------|
| Run 1 | {data['ds']['run1']['num_streams']} | {data['ds']['run1']['fps_per_stream']:.2f} | {data['ds']['run1']['total_fps']:.2f} | {eff1:.1f}% |
| Run 2 | {data['ds']['run2']['num_streams']} | {data['ds']['run2']['fps_per_stream']:.2f} | {data['ds']['run2']['total_fps']:.2f} | {eff2:.1f}% |

Real-time @ 30 fps/stream: **{rt}**

![DS Efficiency](charts/chart_ds_efficiency.png)
![PERF Timeline](charts/chart_perf_timeline.png)

## 7. Detection Sanity (KITTI dump)
![Class Histogram](charts/chart_class_hist.png)

## 8. Pipeline Stages
1. Engine build (`scripts/build_engine.sh`)
2. Loop sources (`scripts/prep_test_loops.sh`)
3. Single-stream mock factory (`scripts/run_pyservicemaker_mock_factory.py`)
4. Multi-stream bench (`scripts/bench_multistream.sh`)
5. Report (this script)

## 9. Parser Notes
- Det-only: 32 mask coefficients in `output0` channels 5..36 are skipped.
- `output1` (prototype masks) bound but not consumed.
- `cluster-mode=2` (DS NMS), `pre-cluster-threshold=0.25`.

## 10. Known Limitations
- No mask metadata produced; instance segmentation requires `network-type=3` plus a sigmoid+matmul postprocess that this skill does not cover.
- INT8 calibration cache reused as-is; rebuild if class distribution shifts.

## 11. Encoder Mode
- Primary: `nvv4l2h264enc` -> `.mp4`
- Fallback: `theoraenc + oggmux` -> `.ogv`

## 12. Files
- Engine: `benchmarks/engines/{MODEL_NAME}_dynamic_b4.engine`
- Logs: `benchmarks/b1/trtexec_b1.log`, `benchmarks/b4/trtexec_b4.log`,
  `benchmarks/ds/ds_s4_run1.log`, `benchmarks/ds/ds_s4_run2.log`
- Sample video: `samples/metal_nut_mock.{{mp4,ogv}}`
"""
    out.write_text(md)


def md_to_html(md_path: Path, html_path: Path) -> None:
    import base64
    import markdown as md

    body = md.markdown(md_path.read_text(), extensions=["tables", "fenced_code"])

    # inline charts as base64 so the HTML is self-contained
    def inline_img(match: re.Match) -> str:
        rel = match.group(1)
        p = (md_path.parent / rel).resolve()
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f'<img src="data:image/png;base64,{b64}" style="max-width:760px;">'
        return match.group(0)

    body = re.sub(r'<img alt="[^"]*" src="([^"]+)"\s*/?>', inline_img, body)

    css = """
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
     max-width:880px;margin:32px auto;padding:0 20px;color:#222;line-height:1.5;}
h1,h2,h3{color:#76b900}
table{border-collapse:collapse;margin:1em 0}
th,td{border:1px solid #ccc;padding:6px 10px}
th{background:#f3f3f3;text-align:left}
img{display:block;margin:14px 0}
code{background:#f6f6f6;padding:2px 4px;border-radius:3px}
"""
    html = f"<!doctype html><meta charset=utf-8><style>{css}</style>{body}"
    html_path.write_text(html)


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    if not shutil.which("wkhtmltopdf"):
        print("WARN: wkhtmltopdf not found; skipping PDF", file=sys.stderr)
        return
    subprocess.run(
        ["wkhtmltopdf", "--enable-local-file-access",
         "--margin-top", "10mm", "--margin-bottom", "10mm",
         str(html_path), str(pdf_path)],
        check=True,
    )


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fps1, tl1 = parse_perf_log(MODEL_DIR / "benchmarks/ds/ds_s4_run1.log")
    fps2, tl2 = parse_perf_log(MODEL_DIR / "benchmarks/ds/ds_s4_run2.log")

    data: dict[str, Any] = {
        "model": MODEL_NAME,
        "trtexec": {
            "b1": parse_trtexec(MODEL_DIR / "benchmarks/b1/trtexec_b1.log"),
            "b4": parse_trtexec(MODEL_DIR / "benchmarks/b4/trtexec_b4.log"),
        },
        "ds": {
            "run1": {
                "num_streams": 4,
                "fps_per_stream": fps1,
                "total_fps": round(fps1 * 4, 2),
                "timeline": tl1,
                "log": "benchmarks/ds/ds_s4_run1.log",
            },
            "run2": {
                "num_streams": 4,
                "fps_per_stream": fps2,
                "total_fps": round(fps2 * 4, 2),
                "timeline": tl2,
                "log": "benchmarks/ds/ds_s4_run2.log",
            },
        },
        "kitti": parse_kitti(MODEL_DIR / "samples/kitti_output"),
    }

    (REPORT_DIR / "benchmark_data.json").write_text(json.dumps(data, indent=2))

    chart_throughput(data, CHART_DIR / "chart_throughput.png")
    chart_latency(data, CHART_DIR / "chart_latency.png")
    chart_ds_efficiency(data, CHART_DIR / "chart_ds_efficiency.png")
    chart_perf_timeline(data, CHART_DIR / "chart_perf_timeline.png")
    chart_class_hist(data, CHART_DIR / "chart_class_hist.png")

    md_path = REPORT_DIR / "benchmark_report.md"
    html_path = REPORT_DIR / "benchmark_report.html"
    pdf_path = REPORT_DIR / f"benchmark_report_{MODEL_NAME}.pdf"
    write_markdown(data, md_path)
    md_to_html(md_path, html_path)
    html_to_pdf(html_path, pdf_path)

    print(f"Report:  {md_path}")
    print(f"HTML:    {html_path}")
    print(f"PDF:     {pdf_path}")
    print(f"Data:    {REPORT_DIR / 'benchmark_data.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
