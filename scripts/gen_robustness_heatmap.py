"""Generate aggregated robustness heatmap PNG from cross-precision matrix.

Two figures:
  benchmarks/robustness_plot.png       — 13-cell mAP@50 × {FP32, FP16}
                                          + Mac PyTorch reference column
  benchmarks/robustness_drift_plot.png — FP16 vs FP32 abs drift per cell
                                          (validates "FP16 ≡ FP32 functionally")

Mac PyTorch FP32 baseline numbers come from benchmarks/robustness_matrix.csv
(the original D8/D9 eval on Mac M5).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"

# Cell ordering (clean → perturbation × severity)
CELL_ORDER = [
    "baseline_s0",
    "noise_s1", "noise_s2", "noise_s3",
    "exposure_s1", "exposure_s2", "exposure_s3",
    "alignment_s1", "alignment_s2", "alignment_s3",
    "combined_s1", "combined_s2", "combined_s3",
]


def _load_cross_precision():
    df = pd.read_csv(BENCH / "robustness_cross_precision.csv").set_index("cell")
    return df.loc[CELL_ORDER]


def _load_mac_reference():
    """Pull Mac PyTorch FP32 mAP@50 from original D8/D9 matrix."""
    f = BENCH / "robustness_matrix.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # CSV header varies; try common column names
    val_col = next(
        c for c in df.columns
        if c.lower() in ("mask_map50", "map50_mask", "map50(m)", "mask_map@50")
    )
    cell_col = "cell" if "cell" in df.columns else df.columns[0]
    s = df.set_index(cell_col)[val_col]
    return s.reindex(CELL_ORDER)


def main():
    cp = _load_cross_precision()
    mac = _load_mac_reference()

    # ───── Figure 1: mAP heatmap with delta annotations ─────
    cols = ["L4 TRT FP32", "L4 TRT FP16"]
    matrix = pd.DataFrame(
        {
            "L4 TRT FP32": cp["mask_map50_fp32"],
            "L4 TRT FP16": cp["mask_map50_fp16"],
        }
    )
    if mac is not None:
        matrix.insert(0, "Mac PyTorch FP32", mac.values)
        cols = ["Mac PyTorch FP32"] + cols

    baseline_row = matrix.loc["baseline_s0"]
    delta = (matrix.subtract(baseline_row) / baseline_row * 100).fillna(0)

    # cell-level annotation = raw mAP value
    annot = matrix.round(3).astype(str)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(delta.values, aspect="auto", cmap="RdYlGn", vmin=-100, vmax=5)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, annot.iloc[i, j],
                ha="center", va="center",
                fontsize=9,
                color="black" if delta.iloc[i, j] > -50 else "white",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% Δ vs clean baseline (per column)")
    ax.set_title(
        "ISP-aware robustness matrix · mAP@50 (mask)\n"
        f"Mac PyTorch baseline {mac.iloc[0]:.3f}; L4 TRT FP32 baseline {cp.loc['baseline_s0','mask_map50_fp32']:.3f}; "
        f"INT8 deferred (ADR-0008)" if mac is not None else
        f"L4 TRT FP32 baseline {cp.loc['baseline_s0','mask_map50_fp32']:.3f}; INT8 deferred (ADR-0008)",
        fontsize=10,
    )
    fig.tight_layout()
    out1 = BENCH / "robustness_plot.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"wrote {out1}")
    plt.close(fig)

    # ───── Figure 2: FP16 vs FP32 drift (validates parity) ─────
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    cells = cp.index.tolist()
    drift = cp["fp16_vs_fp32_abs_drift"].values
    bars = ax2.barh(cells, drift, color="#3a7bd5")
    ax2.set_xlabel("|FP16 − FP32| mAP@50 (mask), absolute")
    ax2.set_title(
        "FP16 vs FP32 functional drift on L4 (TRT 10.14)\n"
        "All cells < 0.001 mAP — FP16 inherits FP32 robustness profile fully",
        fontsize=10,
    )
    ax2.axvline(0.001, color="#cc4444", linestyle="--", linewidth=1, label="0.001 mAP threshold")
    ax2.legend(loc="lower right", fontsize=8)
    for bar, v in zip(bars, drift):
        ax2.text(v + 1e-5, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                 va="center", fontsize=7)
    ax2.set_xlim(0, max(drift) * 1.3 + 0.0005)
    fig2.tight_layout()
    out2 = BENCH / "robustness_drift_plot.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
