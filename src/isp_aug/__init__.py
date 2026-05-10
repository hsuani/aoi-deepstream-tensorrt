"""ISP-aware augmentation modules for AOI robustness study (D8-D9).

Three perturbation types calibrated for MVTec metal_nut:
- noise:     linear-domain Poisson + Gaussian (signal-dependent + read noise)
- exposure:  linear-domain gain + WB drift + gamma offset
- alignment: rotation + translation + anisotropic scale (single affine)

Severity levels 1 / 2 / 3 (mild / moderate / severe).

Combined-apply helper: ISP-realistic order alignment -> exposure -> noise
(workpiece pose -> lens/illumination -> sensor sampling).
"""
from __future__ import annotations

import numpy as np

from . import alignment, exposure, noise

__all__ = ["noise", "exposure", "alignment", "apply_combined"]


def apply_combined(
    img: np.ndarray,
    severity: int,
    seed: int | None = None,
) -> np.ndarray:
    """Apply alignment + exposure + noise in ISP-realistic order.

    Pipeline order matches real-world AOI signal chain:
      1. alignment  (workpiece pose drift, before camera)
      2. exposure   (lens / illumination, mid pipeline)
      3. noise      (sensor sampling, final stage)

    Each sub-module gets a distinct derived seed so individual sub-perturbations
    are deterministic but uncorrelated within the same combined call.
    """
    if seed is None:
        sa = sb = sc = None
    else:
        sa, sb, sc = seed, seed + 100003, seed + 200003  # large primes spaced

    img = alignment.apply(img, severity, seed=sa)
    img = exposure.apply(img, severity, seed=sb)
    img = noise.apply(img, severity, seed=sc)
    return img

