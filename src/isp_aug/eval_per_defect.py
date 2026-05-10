"""Per-defect-type breakdown of robustness eval.

The model was trained single_cls=True (all 4 defect types -> one class), so
Ultralytics validator can't natively report per-defect-type mAP. We instead
filter the val set by filename prefix (bent_*, color_*, flip_*, scratch_*),
build per-defect sub-val dirs (mirroring the cell layout), and run model.val()
on each.

Output: benchmarks/robustness_per_defect.csv with rows
  cell x defect_type, columns mAP@50(M), mAP@50-95(M), recall(M).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import yaml


DEFECT_PREFIXES = ("bent", "color", "flip", "scratch")


def build_defect_subvals(
    src_images: Path,
    src_labels: Path,
    out_root: Path,
    cell_name: str,
) -> list[Path]:
    """For one cell's val, split into 4 sub-vals by filename prefix."""
    written = []
    for prefix in DEFECT_PREFIXES:
        sub_dir = out_root / cell_name / prefix
        img_dir = sub_dir / "images" / "val"
        lbl_dir = sub_dir / "labels" / "val"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Symlink matching files
        n = 0
        for src_img in src_images.glob(f"{prefix}_*"):
            (img_dir / src_img.name).unlink(missing_ok=True)
            (img_dir / src_img.name).symlink_to(src_img.resolve())
            label = src_labels / src_img.with_suffix(".txt").name
            if label.exists():
                (lbl_dir / label.name).unlink(missing_ok=True)
                (lbl_dir / label.name).symlink_to(label.resolve())
            n += 1
        if n == 0:
            shutil.rmtree(sub_dir)
            continue

        # Per-defect data.yaml
        data_yaml = {
            "path": str(sub_dir.resolve()),
            "train": "images/val",
            "val": "images/val",
            "nc": 1,
            "names": ["defect"],
        }
        with open(sub_dir / "data.yaml", "w") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        written.append(sub_dir)
    return written


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path,
                    default=Path("models/yolov8s_seg_metal_nut.pt"))
    ap.add_argument("--baseline-images", type=Path,
                    default=Path("data/yolo/metal_nut/images/val"))
    ap.add_argument("--baseline-labels", type=Path,
                    default=Path("data/yolo/metal_nut/labels/val"))
    ap.add_argument("--perturbed-root", type=Path,
                    default=Path("data/yolo/metal_nut_perturbed"))
    ap.add_argument("--workspace-root", type=Path,
                    default=Path("data/yolo/metal_nut_per_defect"),
                    help="Where the filtered sub-val dirs live")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-csv", type=Path,
                    default=Path("benchmarks/robustness_per_defect.csv"))
    args = ap.parse_args()

    # 1. Build per-defect sub-vals for baseline + 12 perturbed cells
    print("=== building per-defect sub-vals ===")
    cells: dict[str, list[Path]] = {}

    cells["baseline"] = build_defect_subvals(
        args.baseline_images,
        args.baseline_labels,
        args.workspace_root,
        "baseline",
    )

    for perturbation in ("noise", "exposure", "alignment", "combined"):
        for severity in (1, 2, 3):
            cell_name = f"{perturbation}_s{severity}"
            cell_imgs = args.perturbed_root / cell_name / "images" / "val"
            if not cell_imgs.is_dir():
                print(f"[skip] missing {cell_imgs}", file=sys.stderr)
                continue
            cells[cell_name] = build_defect_subvals(
                cell_imgs,
                args.baseline_labels,  # labels symlinked from baseline
                args.workspace_root,
                cell_name,
            )

    # 2. Run model.val() per cell per defect type
    print("\n=== running model.val() per cell per defect type ===")
    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    rows = []
    for cell_name, sub_dirs in cells.items():
        for sub_dir in sub_dirs:
            defect_type = sub_dir.name
            data_yaml = sub_dir / "data.yaml"
            print(f"  {cell_name} / {defect_type} ...")
            results = model.val(
                data=str(data_yaml),
                imgsz=args.imgsz,
                device=args.device,
                split="val",
                verbose=False,
                save=False,
                save_json=False,
                project="runs/segment",
                name=f"perdef_{cell_name}_{defect_type}",
                exist_ok=True,
            )
            d = results.results_dict
            rows.append({
                "cell": cell_name,
                "defect": defect_type,
                "mask_map50":   round(float(d.get("metrics/mAP50(M)", 0.0)), 4),
                "mask_map5095": round(float(d.get("metrics/mAP50-95(M)", 0.0)), 4),
                "mask_recall":  round(float(d.get("metrics/recall(M)", 0.0)), 4),
                "box_map50":    round(float(d.get("metrics/mAP50(B)", 0.0)), 4),
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] wrote {args.out_csv}")

    # Pretty print pivot table: cell rows, defect cols (mAP@50 mask)
    print("\n=== Per-defect mAP@50 (M) pivot ===")
    cell_order = ["baseline"] + [
        f"{p}_s{s}"
        for p in ("noise", "exposure", "alignment", "combined")
        for s in (1, 2, 3)
    ]
    cell_order = [c for c in cell_order if any(r["cell"] == c for r in rows)]
    print(f"{'cell':<16} | " + " | ".join(f"{d:>8}" for d in DEFECT_PREFIXES))
    print("-" * (16 + 3 + (10 + 3) * len(DEFECT_PREFIXES)))
    for c in cell_order:
        cell_rows = {r["defect"]: r["mask_map50"] for r in rows if r["cell"] == c}
        cells_str = " | ".join(f"{cell_rows.get(d, 0.0):>8.4f}" for d in DEFECT_PREFIXES)
        print(f"{c:<16} | {cells_str}")


if __name__ == "__main__":
    main()
