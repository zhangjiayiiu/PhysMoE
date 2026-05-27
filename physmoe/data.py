from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    path: str
    seq_len: int = 96
    pred_len: int = 32
    target_col: str = "OT"
    timestamp_col: Optional[str] = "date"
    feature_cols: Optional[Sequence[str]] = None
    cmax_col: Optional[str] = None
    night_col: Optional[str] = None
    fill_method: str = "ffill"
    auto_physics: bool = False
    night_start: float = 18.0
    night_end: float = 6.0
    cmax_quantile: float = 0.98
    cmax_margin: float = 1.05
    cmax_smooth_slots: int = 9


def load_csv_or_folder(path: str, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    """Load a single CSV or all CSV files in one station folder.

    Folder mode reads all files matching *.csv, sorted by filename, then sorts by
    timestamp_col if available. This matches datasets stored as one folder per
    PV station and multiple chronological CSV files under that folder.
    """
    p = Path(path)
    if p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
        frames = []
        print(f"Found {len(csv_files)} CSV file(s) under {path}")
        for f in csv_files:
            print(f"Reading: {f}")
            frames.append(pd.read_csv(f))
        df = pd.concat(frames, ignore_index=True)
    elif p.is_file():
        print(f"Reading single CSV: {path}")
        df = pd.read_csv(p)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        df = df.drop_duplicates(subset=[timestamp_col], keep="first").reset_index(drop=True)
    return df


def add_auto_physics_columns(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    night_start: float = 18.0,
    night_end: float = 6.0,
    cmax_quantile: float = 0.98,
    cmax_margin: float = 1.05,
    cmax_smooth_slots: int = 9,
    cmax_col: str = "auto_Cmax",
    night_col: str = "auto_night",
) -> Tuple[pd.DataFrame, str, str]:
    """Add empirical Cmax and timestamp-derived night columns.

    auto_Cmax is an estimated clear-sky upper envelope using a high quantile of
    historical PV power at each intra-day time slot. This is an auxiliary soft
    regularization signal, not a measured clear-sky physical target.
    """
    if timestamp_col not in df.columns:
        raise ValueError("timestamp_col is required for --auto_physics")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")

    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    hour_float = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    if night_start > night_end:
        night = ((hour_float >= night_start) | (hour_float < night_end)).astype(float)
    else:
        night = ((hour_float >= night_start) & (hour_float < night_end)).astype(float)
    out[night_col] = night.to_numpy(dtype=np.float32)

    # Intra-day slot: minute of day. Works for 15-min data and other regular resolutions.
    slot = ts.dt.hour * 60 + ts.dt.minute
    power = pd.to_numeric(out[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp = pd.DataFrame({"slot": slot, "power": power})
    envelope = tmp.groupby("slot")["power"].quantile(cmax_quantile).sort_index()
    if cmax_smooth_slots and len(envelope) > 2:
        envelope = envelope.rolling(int(cmax_smooth_slots), center=True, min_periods=1).mean()
    cmax_map = envelope.to_dict()
    global_q = float(power.quantile(cmax_quantile)) if len(power) else 0.0
    out[cmax_col] = slot.map(cmax_map).fillna(global_q).to_numpy(dtype=np.float32) * float(cmax_margin)
    out[cmax_col] = np.maximum(out[cmax_col].to_numpy(dtype=np.float32), 0.0)
    return out, cmax_col, night_col


def prepare_dataframe(cfg: DataConfig) -> Tuple[pd.DataFrame, DataConfig]:
    df = load_csv_or_folder(cfg.path, cfg.timestamp_col)
    if cfg.auto_physics:
        if cfg.timestamp_col is None:
            raise ValueError("--auto_physics requires --timestamp_col")
        df, cmax_col, night_col = add_auto_physics_columns(
            df,
            target_col=cfg.target_col,
            timestamp_col=cfg.timestamp_col,
            night_start=cfg.night_start,
            night_end=cfg.night_end,
            cmax_quantile=cfg.cmax_quantile,
            cmax_margin=cfg.cmax_margin,
            cmax_smooth_slots=cfg.cmax_smooth_slots,
        )
        cfg = DataConfig(**{**cfg.__dict__, "cmax_col": cmax_col, "night_col": night_col})
    return df, cfg


class PVWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DataConfig, start: int = 0, end: Optional[int] = None):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)

        if cfg.feature_cols is None:
            excluded = {c for c in [cfg.timestamp_col, cfg.cmax_col, cfg.night_col] if c is not None}
            feature_cols = [c for c in self.df.columns if c not in excluded]
        else:
            feature_cols = list(cfg.feature_cols)
        if cfg.target_col not in feature_cols:
            feature_cols = [cfg.target_col] + feature_cols
        # Move target to channel 0; model defaults target_idx=0.
        feature_cols = [cfg.target_col] + [c for c in feature_cols if c != cfg.target_col]
        missing = [c for c in feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}. Available: {list(self.df.columns)}")
        self.feature_cols = feature_cols

        values = self.df[feature_cols].apply(pd.to_numeric, errors="coerce")
        if cfg.fill_method == "ffill":
            values = values.ffill().bfill().fillna(0.0)
        elif cfg.fill_method == "zero":
            values = values.fillna(0.0)
        elif cfg.fill_method == "drop":
            values = values.dropna()
        else:
            raise ValueError("fill_method must be one of: ffill, zero, drop")
        self.x = values.to_numpy(dtype=np.float32)

        self.cmax = None
        if cfg.cmax_col and cfg.cmax_col in self.df.columns:
            self.cmax = (
                pd.to_numeric(self.df[cfg.cmax_col], errors="coerce")
                .ffill()
                .bfill()
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        self.night = None
        if cfg.night_col and cfg.night_col in self.df.columns:
            self.night = pd.to_numeric(self.df[cfg.night_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

        max_start = len(self.x) - cfg.seq_len - cfg.pred_len + 1
        if max_start <= 0:
            raise ValueError(f"Not enough rows ({len(self.x)}) for seq_len + pred_len ({cfg.seq_len + cfg.pred_len}).")
        end = max_start if end is None else min(end, max_start)
        self.indices = np.arange(start, end)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = int(self.indices[i])
        x = self.x[s : s + self.cfg.seq_len]
        y_start = s + self.cfg.seq_len
        y = self.x[y_start : y_start + self.cfg.pred_len, 0]
        out: Dict[str, torch.Tensor] = {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }
        if self.cmax is not None:
            out["cmax"] = torch.tensor(self.cmax[y_start : y_start + self.cfg.pred_len], dtype=torch.float32)
        if self.night is not None:
            out["night"] = torch.tensor(self.night[y_start : y_start + self.cfg.pred_len], dtype=torch.float32)
        return out


def contiguous_splits(n: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)
