from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="examples/synthetic_station/synthetic.csv")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.days * 96  # 15-min resolution
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    hour = ts.hour + ts.minute / 60
    daylight = ((hour >= 6) & (hour <= 18)).astype(float)
    solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    daily_cloud = rng.beta(2, 5, size=args.days)
    cloud = np.repeat(daily_cloud, 96) + 0.15 * rng.normal(size=n)
    cloud = np.clip(cloud, 0, 1)
    # abrupt cloud blocks
    for _ in range(args.days * 2):
        center = rng.integers(0, n)
        width = rng.integers(2, 12)
        cloud[max(0, center - width): min(n, center + width)] = np.clip(
            cloud[max(0, center - width): min(n, center + width)] + rng.uniform(0.3, 0.8), 0, 1
        )
    temp = 18 + 12 * solar + 3 * rng.normal(size=n)
    humidity = 60 + 20 * cloud - 10 * solar + 5 * rng.normal(size=n)
    wind = np.clip(2 + rng.normal(size=n), 0, None)
    power = 2000 * solar * (1 - 0.75 * cloud) * daylight
    power += 40 * rng.normal(size=n)
    power = np.clip(power, 0, None)
    df = pd.DataFrame({
        "date": ts,
        "OT": power,
        "Ratio_0": cloud,
        "Ratio_1": solar,
        "Ratio_2": temp,
        "Ratio_3": humidity,
        "Wind": wind,
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved synthetic data to {out}")


if __name__ == "__main__":
    main()
