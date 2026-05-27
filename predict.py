from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from physmoe import DataConfig, PVWindowDataset, PhysMoE, PhysMoEConfig, prepare_dataframe


def parse_args():
    p = argparse.ArgumentParser(description="Run PhysMoE inference on a CSV/folder and save predictions.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--target_col", default="OT")
    p.add_argument("--timestamp_col", default="date")
    p.add_argument("--feature_cols", nargs="*", default=None)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", default="predictions.csv")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = PhysMoEConfig(**ckpt["model_cfg"])
    data_cfg = DataConfig(
        path=args.csv,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        feature_cols=args.feature_cols,
    )
    df, data_cfg = prepare_dataframe(data_cfg)
    ds = PVWindowDataset(df, data_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysMoE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            x = batch["x"].to(device)
            y_hat = model(x).cpu().numpy()
            y_true = batch["y"].numpy()
            for i in range(y_hat.shape[0]):
                rows.append({"window_index": offset + i, **{f"pred_t+{j+1}": y_hat[i, j] for j in range(y_hat.shape[1])}, **{f"true_t+{j+1}": y_true[i, j] for j in range(y_true.shape[1])}})
            offset += y_hat.shape[0]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
