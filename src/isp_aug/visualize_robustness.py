"""Visual proof: model inference (bbox + mask overlay) on baseline + 12 perturbed cells.

Produces two grids:

  docs/robustness_grid_moderate.png
    5 sample images (one per defect type) x 5 cells:
      clean | noise_s2 | exposure_s2 | alignment_s2 | combined_s2

  docs/robustness_grid_severity.png
    1 sample (scratch_007) x 4x4:
      Rows: noise / exposure / alignment / combined
      Cols: clean | s1 | s2 | s3

Both pull pre-generated perturbed images from
data/yolo/metal_nut_perturbed/<cell>/images/val/, so visuals match the
eval matrix (benchmarks/robustness_matrix.csv).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


SAMPLES_MULTI = [
    ("good_test_010", "good"),
    ("bent_006", "bent"),
    ("scratch_007", "scratch"),
    ("color_001", "color"),
    ("flip_001", "flip"),
]

SAMPLE_DEEP = "scratch_007"  # rich multi-defect sample for severity progression

CLEAN_DIR = Path("data/yolo/metal_nut/images/val")
PERTURBED_ROOT = Path("data/yolo/metal_nut_perturbed")


def annotate_label(img: np.ndarray, text: str, font_scale: float = 0.6) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        out, text, (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def predict_and_overlay(model, img_path: Path, target_h: int = 320) -> np.ndarray:
    """Run prediction; return annotated image (with bbox + mask) at target_h."""
    results = model.predict(
        source=str(img_path),
        save=False,
        verbose=False,
        conf=0.25,
        imgsz=640,
        device="mps",
    )
    annotated = results[0].plot(line_width=2)  # numpy BGR with bbox + mask drawn
    h, w = annotated.shape[:2]
    scale = target_h / h
    return cv2.resize(annotated, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def make_multi_sample_grid(model, out_path: Path) -> None:
    """5 samples x 5 cells (clean + noise/exposure/alignment/combined at s2)."""
    cell_specs = [
        ("clean", CLEAN_DIR),
        ("noise_s2", PERTURBED_ROOT / "noise_s2" / "images" / "val"),
        ("exposure_s2", PERTURBED_ROOT / "exposure_s2" / "images" / "val"),
        ("alignment_s2", PERTURBED_ROOT / "alignment_s2" / "images" / "val"),
        ("combined_s2", PERTURBED_ROOT / "combined_s2" / "images" / "val"),
    ]
    rows = []
    for stem, label in SAMPLES_MULTI:
        cells = []
        for cell_name, cell_dir in cell_specs:
            img_path = cell_dir / f"{stem}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"missing {img_path}")
            ann = predict_and_overlay(model, img_path)
            ann = annotate_label(ann, f"{label} | {cell_name}")
            cells.append(ann)
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[ok] wrote {out_path} shape={grid.shape}")


def make_severity_grid(model, out_path: Path) -> None:
    """1 sample x 4 perturb rows x 4 severity cols."""
    perturbations = ["noise", "exposure", "alignment", "combined"]
    rows = []
    clean_path = CLEAN_DIR / f"{SAMPLE_DEEP}.png"
    clean_ann = annotate_label(predict_and_overlay(model, clean_path), f"{SAMPLE_DEEP} | clean")
    for perturbation in perturbations:
        cells = [clean_ann.copy()]
        for s in (1, 2, 3):
            img_path = PERTURBED_ROOT / f"{perturbation}_s{s}" / "images" / "val" / f"{SAMPLE_DEEP}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"missing {img_path}")
            ann = predict_and_overlay(model, img_path)
            ann = annotate_label(ann, f"{perturbation} s{s}")
            cells.append(ann)
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[ok] wrote {out_path} shape={grid.shape}")


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from ultralytics import YOLO

    model = YOLO("models/yolov8s_seg_metal_nut.pt")

    make_multi_sample_grid(model, Path("docs/robustness_grid_moderate.png"))
    make_severity_grid(model, Path("docs/robustness_grid_severity.png"))


if __name__ == "__main__":
    main()
