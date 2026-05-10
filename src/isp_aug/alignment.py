"""ISP-aware alignment perturbation — affine: rotation + translation + anisotropic scale.

Models industrial AOI mechanical pose drift:

  - Rotation: fixture / camera mount drift (around image center)
  - Translation: conveyor / fixture wear (independent dX, dY)
  - Anisotropic scale: lens focus drift / thermal expansion (sx != sy)

Single affine matrix applied via cv2.warpAffine to avoid multi-warp
interpolation accumulation. Black border (BORDER_CONSTANT value=0) matches
metal_nut near-black background.

No perspective transform — top-down macro AOI setup has no perspective
component (camera and workpiece plane are parallel by design).

Severity table (metal_nut-tuned, with explicit training-augmentation
overlap as a sanity floor):

  | Severity     | rotation | translation | scale | Real-world analog                          |
  | -----------  | -------- | ----------- | ----- | ------------------------------------------ |
  | 1 mild       | +/-2     | +/-3 px     | +/-2% | Inside training aug range (sanity floor)   |
  | 2 moderate   | +/-7     | +/-12 px    | +/-7% | Slightly OOD - fixture wear                |
  | 3 severe     | +/-15    | +/-25 px    | +/-15%| Clearly OOD - camera misalignment          |

Training reference: D2 v1 used degrees=10, translate=0.1 (~+/-70 px on
700-px image), scale=0.3. Severity s1 sits well inside; s3 is clearly OOD.
"""
from __future__ import annotations

import cv2
import numpy as np


# Severity -> {deg, tx_px, scale_pct}
# tx_px and scale are independent per-axis (X != Y); see apply() body.
SEVERITY_TABLE: dict[int, dict[str, float]] = {
    1: {"deg": 2.0,  "tx_px": 3.0,  "scale_pct": 2.0},
    2: {"deg": 7.0,  "tx_px": 12.0, "scale_pct": 7.0},
    3: {"deg": 15.0, "tx_px": 25.0, "scale_pct": 15.0},
}


def apply(
    img: np.ndarray,
    severity: int,
    seed: int | None = None,
) -> np.ndarray:
    """Apply affine alignment perturbation to a single uint8 BGR image.

    Args:
        img:       uint8, shape (H, W, 3), BGR (OpenCV convention).
        severity:  1 (mild), 2 (moderate), 3 (severe).
        seed:      Optional RNG seed. Module-local RNG; does not pollute
                   global numpy state.

    Returns:
        uint8, shape (H, W, 3), BGR.

    Raises:
        ValueError: if severity not in {1, 2, 3} or img dtype/shape wrong.
    """
    if severity not in SEVERITY_TABLE:
        raise ValueError(f"severity must be in {{1, 2, 3}}, got {severity}")
    if img.dtype != np.uint8:
        raise ValueError(f"img dtype must be uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img shape must be (H, W, 3), got {img.shape}")

    cfg = SEVERITY_TABLE[severity]
    rng = np.random.default_rng(seed)

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Sample affine parameters with random sign (independent per axis)
    angle = float(rng.uniform(-cfg["deg"], cfg["deg"]))
    tx = float(rng.uniform(-cfg["tx_px"], cfg["tx_px"]))
    ty = float(rng.uniform(-cfg["tx_px"], cfg["tx_px"]))  # same range, independent sample
    sx = 1.0 + float(rng.uniform(-cfg["scale_pct"], cfg["scale_pct"])) / 100.0
    sy = 1.0 + float(rng.uniform(-cfg["scale_pct"], cfg["scale_pct"])) / 100.0

    # Build 2x3 affine matrix: rotation + uniform scale (sx) about center
    M = cv2.getRotationMatrix2D((cx, cy), angle, sx)

    # Decouple sy from sx by scaling row 1 of the matrix (y-output components)
    # by sy/sx. Result: x-axis scale = sx, y-axis scale = sy.
    M[1, :] *= sy / sx

    # Apply translation (post-rotation/scale)
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out


def severity_summary() -> str:
    """Human-readable severity table for CLI / docs."""
    lines = ["severity | rotation | translation | scale | analog"]
    lines.append("---------|----------|-------------|-------|--------")
    analogs = {
        1: "Inside training aug range (sanity floor)",
        2: "Slightly OOD - fixture wear",
        3: "Clearly OOD - camera misalignment",
    }
    for s, cfg in SEVERITY_TABLE.items():
        lines.append(
            f"{s}        | +/-{cfg['deg']:>4.0f}deg | "
            f"+/-{cfg['tx_px']:>4.0f} px   | "
            f"+/-{cfg['scale_pct']:>3.0f}% | {analogs[s]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(severity_summary())
