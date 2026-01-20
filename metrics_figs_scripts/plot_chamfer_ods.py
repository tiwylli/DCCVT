#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_COLUMN = "label"
METRIC_COLUMN = "chamfer_distance_2"
LABEL_SUFFIX = "_b32_ups_DCCVT_10_final_intDCCVT_cvt100_sdfsmooth100"
LABEL_PATTERN = re.compile(
    rf"^(?:\D*?)(?P<mesh_id>\d+){re.escape(LABEL_SUFFIX)}$"
)


def load_ods(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    return pd.read_excel(path, engine="odf", sheet_name=sheet)


def build_series(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if LABEL_COLUMN not in df.columns or METRIC_COLUMN not in df.columns:
        raise ValueError(
            "Expected columns not found. "
            f"Required: {LABEL_COLUMN}, {METRIC_COLUMN}. "
            f"Available: {list(df.columns)}"
        )

    name_series = df[LABEL_COLUMN].astype(str).str.strip()
    mask = name_series.str.fullmatch(LABEL_PATTERN)
    filtered_df = df[mask]

    mesh_id = pd.to_numeric(
        name_series[mask].str.extract(LABEL_PATTERN, expand=False),
        errors="coerce",
    )
    metric = pd.to_numeric(filtered_df[METRIC_COLUMN], errors="coerce")

    cleaned = (
        pd.DataFrame({"mesh_id": mesh_id, label: metric})
        .dropna(subset=["mesh_id", label])
        .groupby("mesh_id", as_index=False)
        .mean(numeric_only=True)
    )
    cleaned["mesh_id"] = cleaned["mesh_id"].astype(int)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Chamferdistance 2 from two ODS files aligned by mesh id."
        )
    )
    parser.add_argument("ods1", type=Path)
    parser.add_argument("ods2", type=Path)
    parser.add_argument("--sheet1", default=0, help="Sheet name or index.")
    parser.add_argument("--sheet2", default=0, help="Sheet name or index.")
    parser.add_argument("--out", default="chamfer_compare.svg")
    args = parser.parse_args()

    df1 = load_ods(args.ods1, sheet=args.sheet1)
    df2 = load_ods(args.ods2, sheet=args.sheet2)

    label1 = Path(args.ods1).stem
    label2 = Path(args.ods2).stem

    series1 = build_series(df1, label1)
    series2 = build_series(df2, label2)

    merged = series1.merge(series2, on="mesh_id", how="inner").sort_values(
        "mesh_id"
    )
    if merged.empty:
        raise SystemExit("No matching mesh_id values between the two files.")

    x_positions = list(range(len(merged)))

    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, merged[label1], marker="o", label="Converged")
    plt.plot(x_positions, merged[label2], marker="o", label="Unconverged")
    plt.xlabel("Mesh ID")
    plt.ylabel("CD")
    plt.title("Ours NU Converged vs Unconverged Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(x_positions, merged["mesh_id"].astype(str), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out, format="svg")


if __name__ == "__main__":
    main()
