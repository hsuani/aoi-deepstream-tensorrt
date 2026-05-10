"""ISP-aware sensor noise model — linear-domain Poisson + Gaussian.

Models industrial AOI camera noise sources:

  - Poisson photon shot noise (signal-dependent)
  - Gaussian read noise (signal-independent)
  - ISO/gain push (multiplicative on both)
  - Per-channel sigma asymmetry (Bayer pattern: R/B noisier than G)

Pipeline:
  sRGB uint8 BGR
    -> de-gamma (^2.2) -> linear [0,1]
    -> *gain (push)
    -> +Poisson(scale=shot_scale)   # signal-dependent shot noise
    -> +Gaussian(sigma=read_sigma)  # signal-independent read noise
    -> /gain (reverse — keep "same scene, more noise" framing)
    -> clip [0,1]
    -> re-gamma (^(1/2.2)) -> sRGB uint8

Note: uses simplified gamma = 2.2 (not piecewise sRGB curve). Error <1%
except deep shadows where photon-noise dominates anyway.

Severity table (calibrated for MVTec metal_nut DN range 30-150):

  | Severity     | shot_scale | read_sigma | gain | Real-world analog          |
  | -----------  | ---------- | ---------- | ---- | -------------------------- |
  | 1 mild       | 1000       | 0.005      | 1.0x | Production lab, base ISO   |
  | 2 moderate   | 200        | 0.02       | 2.0x | Aged LED + ISO 200-400     |
  | 3 severe     | 50         | 0.05       | 4.0x | Low-light line, ISO 800+   |

Larger shot_scale = more photons per DN = LESS Poisson noise (counter-intuitive).
"""
from __future__ import annotations

import numpy as np


GAMMA = 2.2  # simplified sRGB gamma


# Severity -> {shot_scale, read_sigma, gain}
SEVERITY_TABLE: dict[int, dict[str, float]] = {
    1: {"shot_scale": 1000.0, "read_sigma": 0.005, "gain": 1.0},
    2: {"shot_scale": 200.0,  "read_sigma": 0.02,  "gain": 2.0},
    3: {"shot_scale": 50.0,   "read_sigma": 0.05,  "gain": 4.0},
}

# Per-channel sigma multiplier (BGR order; G is 50% of Bayer pixels -> less noise)
PER_CHANNEL_SCALE_BGR = np.array([1.3, 0.8, 1.3], dtype=np.float32)


def apply(
    img: np.ndarray,
    severity: int,
    seed: int | None = None,
    per_channel: bool = True,
) -> np.ndarray:
    """Apply ISP-aware sensor noise to a single uint8 BGR image.

    Args:
        img:         uint8, shape (H, W, 3), BGR (OpenCV convention).
        severity:    1 (mild), 2 (moderate), 3 (severe).
        seed:        Optional RNG seed for reproducibility. Module-local RNG
                     used; does not pollute global numpy state.
        per_channel: If True (default), apply per-channel sigma asymmetry
                     (B and R get 1.3x sigma, G gets 0.8x — Bayer pattern).
                     Only takes effect at severity >= 2.

    Returns:
        uint8, shape (H, W, 3), BGR — same shape, perturbed.

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
    shot_scale = cfg["shot_scale"]
    read_sigma = cfg["read_sigma"]
    gain = cfg["gain"]

    rng = np.random.default_rng(seed)

    # 1. cast to float32 + normalize
    img_norm = img.astype(np.float32) / 255.0

    # 2. de-gamma to linear domain
    img_lin = np.power(img_norm, GAMMA)

    # 3. apply gain (ISO push) — operates on real linear signal
    img_lin = img_lin * gain

    # 4. Poisson shot noise: signal-dependent
    #    Number of photons per DN = img_lin * shot_scale.
    #    Sample observed photons -> divide by shot_scale to bring back to
    #    normalized DN range.
    photons = rng.poisson(np.clip(img_lin, 0.0, None) * shot_scale).astype(np.float32)
    img_lin = photons / shot_scale

    # 5. Gaussian read noise: signal-independent
    if per_channel and severity >= 2:
        # Broadcast (3,) over (H, W, 3)
        sigma_per_pixel = read_sigma * PER_CHANNEL_SCALE_BGR
        gauss = rng.normal(0.0, 1.0, img_lin.shape).astype(np.float32) * sigma_per_pixel
    else:
        gauss = rng.normal(0.0, read_sigma, img_lin.shape).astype(np.float32)
    img_lin = img_lin + gauss

    # 6. reverse gain — "same scene, more noise" (keeps DN range comparable
    #    to clean baseline; isolates noise as the only varying factor).
    img_lin = img_lin / gain

    # 7. clip to [0, 1] BEFORE re-gamma — negative values would give NaN under
    #    fractional power.
    img_lin = np.clip(img_lin, 0.0, 1.0)

    # 8. re-gamma back to sRGB perceptual domain
    img_out_norm = np.power(img_lin, 1.0 / GAMMA)

    # 9. de-normalize + cast back to uint8
    img_out = (img_out_norm * 255.0).round().astype(np.uint8)
    return img_out


def severity_summary() -> str:
    """Human-readable severity table for CLI / docs."""
    lines = ["severity | shot_scale | read_sigma | gain | analog"]
    lines.append("---------|------------|------------|------|--------")
    analogs = {
        1: "Lab production, base ISO",
        2: "Aged LED + ISO 200-400",
        3: "Low-light line, ISO 800+",
    }
    for s, cfg in SEVERITY_TABLE.items():
        lines.append(
            f"{s}        | {cfg['shot_scale']:>10.0f} | {cfg['read_sigma']:>10.3f} | "
            f"{cfg['gain']:>4.1f} | {analogs[s]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(severity_summary())
