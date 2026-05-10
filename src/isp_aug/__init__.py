"""ISP-aware augmentation modules for AOI robustness study (D8-D9).

Three perturbation types calibrated for MVTec metal_nut:
- noise:     linear-domain Poisson + Gaussian (signal-dependent + read noise)
- exposure:  linear-domain gain + WB drift + gamma offset
- alignment: rotation + translation + anisotropic scale (single affine)

Severity levels 1 / 2 / 3 (mild / moderate / severe).
"""
from . import alignment, exposure, noise

__all__ = ["noise", "exposure", "alignment"]
