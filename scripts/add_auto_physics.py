from __future__ import annotations

import argparse
from pathlib import Path
from physmoe.data import load_csv_or_folder, add_auto_physics_columns


def main():
    p = argparse.ArgumentParser(description="Generate auto_Cmax and auto_night columns and save a prepared CSV.")
    p.add_argument("--csv", required=True, help="Input CSV or station folder")
    p.add_argument("--out", required=True)
    p.add_argument("--target_col", default="OT")
    p.add_argument("--timestamp_col", default="date")
    p.add_argument("--night_start", type=float, default=18.0)
    p.add_argument("--night_end", type=float, default=6.0)
    p.add_argument("--cmax_quantile", type=float, default=0.98)
    p.add_argument("--cmax_margin", type=float, default=1.05)
    p.add_argument("--cmax_smooth_slots", type=int, default=9)
    args = p.parse_args()
    df = load_csv_or_folder(args.csv, args.timestamp_col)
    df, cmax_col, night_col = add_auto_physics_columns(
        df,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        night_start=args.night_start,
        night_end=args.night_end,
        cmax_quantile=args.cmax_quantile,
        cmax_margin=args.cmax_margin,
        cmax_smooth_slots=args.cmax_smooth_slots,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved prepared CSV to {out}")
    print(f"Added columns: {cmax_col}, {night_col}")


if __name__ == "__main__":
    main()
