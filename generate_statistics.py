"""
Generate statistical results for the Heart Disease Dataset.
Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download the dataset from Kaggle and place the CSV (e.g. heart.csv) in this
directory or in a 'data' subfolder. Then run: python generate_statistics.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Default paths to look for the dataset
DATA_PATHS = [
    Path("heart.csv"),
    Path("data/heart.csv"),
    Path("heart-disease-dataset/heart.csv"),
]


def find_dataset() -> Path:
    """Locate the heart disease CSV file."""
    base = Path(__file__).resolve().parent
    for rel in DATA_PATHS:
        path = base / rel
        if path.exists():
            return path
    return None


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset and normalize column names if needed."""
    df = pd.read_csv(path)
    # Some versions use 'num' as target, others use 'target'
    if "num" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"num": "target"})
    return df


def print_section(title: str, char: str = "=") -> None:
    """Print a section header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def generate_statistics(df: pd.DataFrame) -> None:
    """Compute and print comprehensive statistical results."""
    print_section("HEART DISEASE DATASET - STATISTICAL SUMMARY", "=")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # -------------------------------------------------------------------------
    print_section("MISSING VALUES", "-")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values.")
    else:
        for col in df.columns:
            n = missing[col]
            if n > 0:
                pct = 100 * n / len(df)
                print(f"  {col}: {n} ({pct:.1f}%)")

    # -------------------------------------------------------------------------
    print_section("DATA TYPES", "-")
    print(df.dtypes.to_string())

    # -------------------------------------------------------------------------
    print_section("DESCRIPTIVE STATISTICS (numeric)", "-")
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        print(numeric.describe().round(2).to_string())
    else:
        print("No numeric columns.")

    # -------------------------------------------------------------------------
    if "target" in df.columns:
        print_section("TARGET DISTRIBUTION (heart disease)", "-")
        counts = df["target"].value_counts().sort_index()
        # Binary: 0 = no disease, 1+ = disease (UCI uses 0-4)
        binary = (df["target"] > 0).astype(int)
        print("Original target (num) values:")
        print(counts.to_string())
        print(f"\nBinary (0=no disease, 1=disease):")
        print(binary.value_counts().sort_index().to_string())
        print(f"\nClass balance: {binary.mean() * 100:.1f}% positive (disease)")

    # -------------------------------------------------------------------------
    print_section("CATEGORICAL / DISCRETE VALUE COUNTS", "-")
    categorical_candidates = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_candidates:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts().sort_index().to_string())

    # -------------------------------------------------------------------------
    print_section("CORRELATION MATRIX (numeric)", "-")
    if not numeric.empty:
        corr = numeric.corr()
        print(corr.round(3).to_string())
        if "target" in corr.columns:
            print("\nCorrelation with target (absolute, sorted):")
            target_corr = corr["target"].drop("target", errors="ignore").abs().sort_values(ascending=False)
            print(target_corr.to_string())

    # -------------------------------------------------------------------------
    print_section("ADDITIONAL STATISTICS", "-")
    for col in numeric.columns:
        print(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
              f"min={df[col].min()}, max={df[col].max()}")

    print()


def save_statistics(df: pd.DataFrame, output_path: Path) -> None:
    """Save key statistics to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("HEART DISEASE DATASET - STATISTICAL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        f.write("DESCRIPTIVE STATISTICS:\n")
        f.write(df.describe().round(4).to_string())
        f.write("\n\nTARGET VALUE COUNTS:\n")
        if "target" in df.columns:
            f.write(df["target"].value_counts().sort_index().to_string())
        f.write("\n")
    print(f"Summary saved to: {output_path}")


def main() -> None:
    csv_path = find_dataset()
    if csv_path is None:
        print("Heart disease CSV not found. Please download the dataset from:")
        print("  https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("\nPlace 'heart.csv' in one of these locations:")
        for p in DATA_PATHS:
            print(f"  - {p}")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    generate_statistics(df)

    out_file = Path(__file__).resolve().parent / "statistics_summary.txt"
    save_statistics(df, out_file)


if __name__ == "__main__":
    main()
