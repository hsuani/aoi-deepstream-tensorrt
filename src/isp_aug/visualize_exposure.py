"""Visual sanity check for exposure.py — compose 3x4 grid of (sample, severity)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.isp_aug import exposure


SAMPLES = [
    ("good", "data/mvtec/metal_nut/test/good/000.png"),
    ("scratch", "data/mvtec/metal_nut/test/scratch/000.png"),
    ("color", "data/mvtec/metal_nut/test/color/000.png"),
]


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        out, text, (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def make_grid(out_path: Path, seed: int = 42, target_h: int = 320) -> None:
    rows = []
    for label, path in SAMPLES:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"missing {path}")

        h, w = img.shape[:2]
        scale = target_h / h
        new_size = (int(w * scale), target_h)
        img_small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        cells = [annotate(img_small, f"{label}: clean")]
        for s in (1, 2, 3):
            perturbed = exposure.apply(img_small, severity=s, seed=seed)
            cells.append(annotate(perturbed, f"{label}: exposure s{s}"))
        rows.append(np.hstack(cells))

    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[ok] wrote {out_path} shape={grid.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/exposure_examples.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-h", type=int, default=320)
    args = ap.parse_args()
    make_grid(args.out, seed=args.seed, target_h=args.target_h)


if __name__ == "__main__":
    main()
