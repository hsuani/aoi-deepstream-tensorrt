"""ISP-aware exposure drift — linear-domain gain + WB + gamma offset.

Models industrial AOI exposure instability sources:

  - Linear gain (EV): aperture / shutter / ISO drift
  - WB drift (color temperature): LED aging + ambient light mix
  - Gamma offset (display calibration): monitor / ISP calibration drift

Pipeline:
  sRGB uint8 BGR
    -> de-gamma (^2.2) -> linear [0,1]
    -> *2^EV                            # linear gain
    -> *WB_per_channel (R *= 1+f, B *= 1-f, G unchanged)
    -> clip [0,1]
    -> re-gamma (^(1/(2.2 + delta_gamma)))
    -> sRGB uint8

Severity table (metal_nut-tuned, production line spectrum):

  | Severity     | EV range | gamma offset | WB shift (K) | Real-world analog                       |
  | -----------  | -------- | ------------ | ------------ | --------------------------------------- |
  | 1 mild       | +/-0.5   | +/-0.05      | +/-200       | Daily ambient drift / auto-exp adjust   |
  | 2 moderate   | +/-1.5   | +/-0.15      | +/-500       | LED 6 mo aged + ambient sun spillover   |
  | 3 severe     | +/-2.0   | +/-0.30      | +/-1000      | LED 2-yr decay + monitor calib failure  |

WB modeling: per 100K shift = 0.5% R/B channel asymmetry (Bradford-like first
order). Positive K (warmer) -> R rises, B falls. Negative K (cooler) -> opposite.

Gamma offset only applies to re-gamma encoding (display side). De-gamma side
is fixed at 2.2 (sensor JPEG output assumed standard).
"""
from __future__ import annotations

import numpy as np


GAMMA_BASE = 2.2

# Severity -> {ev_range, gamma_offset, wb_temp_shift_K}
SEVERITY_TABLE: dict[int, dict[str, float]] = {
    1: {"ev_range": 0.5, "gamma_offset": 0.05, "wb_temp_shift": 200.0},
    2: {"ev_range": 1.5, "gamma_offset": 0.15, "wb_temp_shift": 500.0},
    3: {"ev_range": 2.0, "gamma_offset": 0.30, "wb_temp_shift": 1000.0},
}

# Per 100K WB shift -> 0.5% R/B asymmetry (Bradford-like first-order approx)
WB_FACTOR_PER_K = 0.00005  # 0.5 / 100 / 100


def apply(
    img: np.ndarray,
    severity: int,
    seed: int | None = None,
    enable_gain: bool = True,
    enable_wb: bool = True,
    enable_gamma: bool = True,
) -> np.ndarray:
    """Apply ISP-aware exposure drift to a single uint8 BGR image.

    Args:
        img:           uint8, shape (H, W, 3), BGR (OpenCV convention).
        severity:      1 (mild), 2 (moderate), 3 (severe).
        seed:          Optional RNG seed. Module-local RNG; does not pollute
                       global numpy state.
        enable_gain:   If False, skip EV gain step (ablation flag).
        enable_wb:     If False, skip WB drift step (ablation flag).
        enable_gamma:  If False, skip gamma offset (use fixed 2.2 on re-gamma).

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

    # Sample drift parameters with random sign (drift can go either direction)
    ev = float(rng.uniform(-cfg["ev_range"], cfg["ev_range"])) if enable_gain else 0.0
    delta_gamma = (
        float(rng.uniform(-cfg["gamma_offset"], cfg["gamma_offset"]))
        if enable_gamma
        else 0.0
    )
    wb_shift_K = (
        float(rng.uniform(-cfg["wb_temp_shift"], cfg["wb_temp_shift"]))
        if enable_wb
        else 0.0
    )

    # 1. cast to float32 + normalize
    img_norm = img.astype(np.float32) / 255.0

    # 2. de-gamma to linear domain (fixed 2.2)
    img_lin = np.power(img_norm, GAMMA_BASE)

    # 3. linear gain (EV)
    if enable_gain:
        gain = 2.0 ** ev
        img_lin = img_lin * gain

    # 4. WB drift — per-channel R/B scaling
    if enable_wb:
        wb_factor = wb_shift_K * WB_FACTOR_PER_K  # ~0.5% per 100K
        # OpenCV BGR: [..., 0]=B, [..., 1]=G, [..., 2]=R
        img_lin[..., 2] = img_lin[..., 2] * (1.0 + wb_factor)  # R
        img_lin[..., 0] = img_lin[..., 0] * (1.0 - wb_factor)  # B
        # G unchanged

    # 5. clip to [0, 1] BEFORE re-gamma (negatives -> NaN under fractional power)
    img_lin = np.clip(img_lin, 0.0, 1.0)

    # 6. re-gamma with shifted gamma exponent
    gamma_out = GAMMA_BASE + delta_gamma  # always > 0 since |delta| <= 0.30
    img_out_norm = np.power(img_lin, 1.0 / gamma_out)

    # 7. de-normalize + cast to uint8
    img_out = (img_out_norm * 255.0).round().clip(0, 255).astype(np.uint8)
    return img_out


def severity_summary() -> str:
    """Human-readable severity table for CLI / docs."""
    lines = ["severity | EV range | gamma offset | WB shift (K) | analog"]
    lines.append("---------|----------|--------------|--------------|--------")
    analogs = {
        1: "Daily ambient drift / auto-exp micro-adjust",
        2: "LED 6 mo aged + ambient sun spillover",
        3: "LED 2-yr decay + monitor calib failure",
    }
    for s, cfg in SEVERITY_TABLE.items():
        lines.append(
            f"{s}        | +/-{cfg['ev_range']:>5.2f} | "
            f"+/-{cfg['gamma_offset']:>10.2f} | "
            f"+/-{cfg['wb_temp_shift']:>10.0f} | {analogs[s]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(severity_summary())
