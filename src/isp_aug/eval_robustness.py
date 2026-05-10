"""Run model.val() across baseline + 12 perturbed cells; emit a CSV table."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path,
                    default=Path("models/yolov8s_seg_metal_nut.pt"))
    ap.add_argument("--baseline-data", type=Path,
                    default=Path("data/yolo/metal_nut/data.yaml"))
    ap.add_argument("--perturbed-root", type=Path,
                    default=Path("data/yolo/metal_nut_perturbed"))
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-csv", type=Path,
                    default=Path("benchmarks/robustness_matrix.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    rows = []

    cells = [("baseline_s0", args.baseline_data)]
    for perturbation in ("noise", "exposure", "alignment", "combined"):
        for severity in (1, 2, 3):
            cell_name = f"{perturbation}_s{severity}"
            data_yaml = args.perturbed_root / cell_name / "data.yaml"
            if not data_yaml.exists():
                print(f"[skip] {cell_name}: missing {data_yaml}", file=sys.stderr)
                continue
            cells.append((cell_name, data_yaml))

    for cell_name, data_yaml in cells:
        print(f"\n=== {cell_name} ({data_yaml}) ===")
        results = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            device=args.device,
            split="val",
            verbose=False,
            save=False,
            save_json=False,
            project="runs/segment",
            name=f"robustness_{cell_name}",
            exist_ok=True,
        )
        d = results.results_dict
        rows.append({
            "cell": cell_name,
            "box_p":   round(float(d.get("metrics/precision(B)", 0.0)), 4),
            "box_r":   round(float(d.get("metrics/recall(B)", 0.0)), 4),
            "box_map50":   round(float(d.get("metrics/mAP50(B)", 0.0)), 4),
            "box_map5095": round(float(d.get("metrics/mAP50-95(B)", 0.0)), 4),
            "mask_p":   round(float(d.get("metrics/precision(M)", 0.0)), 4),
            "mask_r":   round(float(d.get("metrics/recall(M)", 0.0)), 4),
            "mask_map50":   round(float(d.get("metrics/mAP50(M)", 0.0)), 4),
            "mask_map5095": round(float(d.get("metrics/mAP50-95(M)", 0.0)), 4),
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] wrote {args.out_csv}")

    # Print pretty
    print("\n=== Robustness matrix ===")
    print(f"{'cell':<16} | {'mAP50(M)':>9} | {'mAP50-95(M)':>11} | {'mAP50(B)':>9} | {'recall(M)':>9}")
    print("-" * 70)
    for r in rows:
        print(f"{r['cell']:<16} | {r['mask_map50']:>9.4f} | {r['mask_map5095']:>11.4f} | "
              f"{r['box_map50']:>9.4f} | {r['mask_r']:>9.4f}")


if __name__ == "__main__":
    main()
