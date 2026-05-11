"""Merge per-precision robustness CSVs into single cross-precision matrix.

Inputs:
  /tmp/d9_l4_results/results/robustness_fp32.csv
  /tmp/d9_l4_results/results/robustness_fp16.csv
  (INT8 deferred — see ADR-0008)

Output:
  benchmarks/robustness_cross_precision.csv

Columns per cell:
  - mask_map50_fp32, mask_map50_fp16
  - fp16_vs_fp32_abs_drift  (|fp16 - fp32|)
  - fp16_vs_fp32_pct_drift  (|fp16 - fp32| / fp32 * 100)
  - delta_vs_baseline_fp32_pct, delta_vs_baseline_fp16_pct
"""

import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FP32 = Path("/tmp/d9_l4_results/results/robustness_fp32.csv")
FP16 = Path("/tmp/d9_l4_results/results/robustness_fp16.csv")
OUT = REPO / "benchmarks" / "robustness_cross_precision.csv"


def main():
    fp32 = pd.read_csv(FP32).set_index("cell")[["mask_map50"]].rename(
        columns={"mask_map50": "mask_map50_fp32"}
    )
    fp16 = pd.read_csv(FP16).set_index("cell")[["mask_map50"]].rename(
        columns={"mask_map50": "mask_map50_fp16"}
    )
    df = fp32.join(fp16)

    df["fp16_vs_fp32_abs_drift"] = (df["mask_map50_fp16"] - df["mask_map50_fp32"]).abs()
    df["fp16_vs_fp32_pct_drift"] = (
        df["fp16_vs_fp32_abs_drift"] / df["mask_map50_fp32"].replace(0, float("nan")) * 100
    )

    base_fp32 = df.loc["baseline_s0", "mask_map50_fp32"]
    base_fp16 = df.loc["baseline_s0", "mask_map50_fp16"]
    df["delta_vs_baseline_fp32_pct"] = (df["mask_map50_fp32"] - base_fp32) / base_fp32 * 100
    df["delta_vs_baseline_fp16_pct"] = (df["mask_map50_fp16"] - base_fp16) / base_fp16 * 100

    df = df.round(4)
    df.to_csv(OUT)
    print(f"wrote {OUT}")
    print(df)


if __name__ == "__main__":
    main()
