from __future__ import annotations

import argparse
from physmoe.data import load_csv_or_folder


def main():
    p = argparse.ArgumentParser(description="Inspect one CSV or a station folder of CSV files.")
    p.add_argument("--csv", required=True)
    p.add_argument("--timestamp_col", default="date")
    args = p.parse_args()
    df = load_csv_or_folder(args.csv, args.timestamp_col)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Head:")
    print(df.head())
    print("Tail:")
    print(df.tail())
    if args.timestamp_col in df.columns:
        print("Timestamp range:", df[args.timestamp_col].min(), "->", df[args.timestamp_col].max())
        print("Duplicate timestamps:", df[args.timestamp_col].duplicated().sum())
    print("Missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()
