"""CLI: apply ISP perturbation to a YOLO val image dir + emit a data.yaml.

Output structure (per cell):

  <output-root>/<cell-name>/
    images/val/    (perturbed images, same filename as source)
    labels/val/    (symlink to source labels — see caveat for alignment)
    data.yaml      (Ultralytics format, points val to perturbed images)

Caveat: for alignment + combined cells, GT polygon labels are NOT transformed
to follow the affine warp. mAP measurement therefore conflates "model
robustness" with "label-image misalignment under spatial drift". This is
intentional — it matches the deployment scenario where camera fixture drifts
between calibration and inference, AND the inspection app expects fixed
defect-coordinate references. For pure model-robustness measurement, separate
follow-up runs would transform polygons under the same affine matrix.

Perturbations + severities form 13 cells per precision:
  baseline (clean) + {noise, exposure, alignment, combined} x {1, 2, 3}
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import yaml

from src.isp_aug import alignment, apply_combined, exposure, noise


PERTURBATIONS = {
    "noise": noise.apply,
    "exposure": exposure.apply,
    "alignment": alignment.apply,
    "combined": apply_combined,
}


def perturb_dir(
    input_dir: Path,
    output_dir: Path,
    perturbation: str,
    severity: int,
    seed: int,
) -> int:
    """Read every image in input_dir, apply perturbation, write to output_dir.

    Per-image seed is `seed * 10000 + image_index`, so output is reproducible
    AND each image gets a different random draw (avoids identical noise on
    every image which would be unrealistic).
    """
    fn = PERTURBATIONS[perturbation]
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    for idx, src_path in enumerate(images):
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"[skip] {src_path}", file=sys.stderr)
            continue
        img_seed = seed * 10000 + idx
        out = fn(img, severity=severity, seed=img_seed)
        cv2.imwrite(str(output_dir / src_path.name), out)
    return len(images)


def build_cell(
    input_root: Path,
    output_root: Path,
    perturbation: str,
    severity: int,
    seed: int,
) -> Path:
    """Build one perturbed cell with images/val + labels/val + data.yaml.

    Args:
        input_root:  e.g. data/yolo/metal_nut (must have images/val + labels/val)
        output_root: e.g. data/yolo/metal_nut_perturbed
        perturbation: noise | exposure | alignment | combined
        severity:    1 / 2 / 3
        seed:        per-cell seed (per-image is derived inside perturb_dir)

    Returns: cell directory path containing data.yaml.
    """
    cell_name = f"{perturbation}_s{severity}"
    cell_dir = output_root / cell_name

    src_images = input_root / "images" / "val"
    src_labels = input_root / "labels" / "val"
    if not src_images.is_dir():
        raise FileNotFoundError(f"missing source images dir: {src_images}")
    if not src_labels.is_dir():
        raise FileNotFoundError(f"missing source labels dir: {src_labels}")

    dst_images = cell_dir / "images" / "val"
    dst_labels = cell_dir / "labels" / "val"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.parent.mkdir(parents=True, exist_ok=True)

    # Symlink labels (NOT transformed under alignment — see module docstring caveat)
    if dst_labels.exists() or dst_labels.is_symlink():
        dst_labels.unlink()
    dst_labels.symlink_to(src_labels.resolve())

    n = perturb_dir(src_images, dst_images, perturbation, severity, seed)

    # Write data.yaml — train pointed to val (Ultralytics requires train key but we never train)
    data_yaml = {
        "path": str(cell_dir.resolve()),
        "train": "images/val",
        "val": "images/val",
        "nc": 1,
        "names": ["defect"],
    }
    with open(cell_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"[ok] {cell_name}: {n} images perturbed at {dst_images}")
    return cell_dir


def build_all_cells(
    input_root: Path,
    output_root: Path,
    seed: int,
) -> list[Path]:
    """Build all 12 cells: {noise, exposure, alignment, combined} x {1, 2, 3}."""
    cells = []
    for perturbation in ("noise", "exposure", "alignment", "combined"):
        for severity in (1, 2, 3):
            cells.append(
                build_cell(input_root, output_root, perturbation, severity, seed)
            )
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path,
                    default=Path("data/yolo/metal_nut"),
                    help="Source root with images/val + labels/val + data.yaml")
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/yolo/metal_nut_perturbed"),
                    help="Output root; each cell gets a subdir.")
    ap.add_argument("--perturbation", type=str, default="all",
                    choices=("noise", "exposure", "alignment", "combined", "all"))
    ap.add_argument("--severity", type=int, default=0,
                    help="0 = all severities (1, 2, 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.perturbation == "all":
        if args.severity != 0:
            print("--perturbation all forces all severities", file=sys.stderr)
        build_all_cells(args.input_root, args.output_root, args.seed)
    else:
        if args.severity == 0:
            for s in (1, 2, 3):
                build_cell(args.input_root, args.output_root,
                           args.perturbation, s, args.seed)
        else:
            build_cell(args.input_root, args.output_root,
                       args.perturbation, args.severity, args.seed)


if __name__ == "__main__":
    main()
