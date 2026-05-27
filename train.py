from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from physmoe import (
    DataConfig,
    PVWindowDataset,
    PhysMoE,
    PhysMoEConfig,
    PhysMoELoss,
    count_parameters,
    prepare_dataframe,
    regression_metrics,
)
from physmoe.data import contiguous_splits
from physmoe.utils import save_json, set_seed


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(
    model: PhysMoE,
    loader: DataLoader,
    criterion: PhysMoELoss,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mse = 0.0
    total_cap = 0.0
    total_night = 0.0
    n_samples = 0
    preds: List[torch.Tensor] = []
    trues: List[torch.Tensor] = []
    persists: List[torch.Tensor] = []

    for batch in loader:
        batch = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        y_hat = model(batch["x"])
        loss, parts = criterion(y_hat, batch["y"], batch.get("cmax"), batch.get("night"))
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = batch["x"].shape[0]
        total_loss += float(loss.detach()) * bs
        total_mse += parts["mse"] * bs
        total_cap += parts["cap"] * bs
        total_night += parts["night"] * bs
        n_samples += bs

        if not is_train:
            preds.append(y_hat.detach().cpu())
            trues.append(batch["y"].detach().cpu())
            # Persistence baseline: repeat the last observed target value.
            persists.append(batch["x"][:, -1:, 0].repeat(1, batch["y"].shape[1]).detach().cpu())

    out = {
        "loss": total_loss / max(1, n_samples),
        "mse_loss": total_mse / max(1, n_samples),
        "cap_loss": total_cap / max(1, n_samples),
        "night_loss": total_night / max(1, n_samples),
    }
    if not is_train and preds:
        out.update(regression_metrics(torch.cat(preds), torch.cat(trues), torch.cat(persists)))
    return out


def append_history(path: Path, epoch: int, split: str, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, "split": split, **metrics}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a best-effort PhysMoE reproduction.")
    p.add_argument("--csv", required=True, help="Path to a single CSV or a station folder containing multiple CSV files.")
    p.add_argument("--target_col", default="OT", help="PV power target column name.")
    p.add_argument("--timestamp_col", default="date", help="Timestamp column used for sorting and auto physics columns.")
    p.add_argument("--feature_cols", nargs="*", default=None, help="Input columns. Include target_col; target is moved to channel 0.")
    p.add_argument("--cmax_col", default=None, help="Optional clear-sky upper envelope column.")
    p.add_argument("--night_col", default=None, help="Optional night indicator column, where night=1 and daylight=0.")
    p.add_argument("--auto_physics", action="store_true", help="Generate auto_Cmax and auto_night from timestamp/target columns.")
    p.add_argument("--night_start", type=float, default=18.0, help="Hour when night starts for auto_night.")
    p.add_argument("--night_end", type=float, default=6.0, help="Hour when night ends for auto_night.")
    p.add_argument("--cmax_quantile", type=float, default=0.98, help="High quantile for empirical clear-sky envelope.")
    p.add_argument("--cmax_margin", type=float, default=1.05, help="Safety margin multiplied into auto_Cmax.")
    p.add_argument("--cmax_smooth_slots", type=int, default=9, help="Centered smoothing window over intra-day slots for auto_Cmax.")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=32)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--router_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--trend_kernel", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--lambda_cap", type=float, default=0.0, help="Weight for optional capacity regularization.")
    p.add_argument("--lambda_night", type=float, default=0.0, help="Weight for optional night-time regularization.")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--fill_method", choices=["ffill", "zero", "drop"], default="ffill")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="runs/physmoe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        path=args.csv,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        feature_cols=args.feature_cols,
        cmax_col=args.cmax_col,
        night_col=args.night_col,
        fill_method=args.fill_method,
        auto_physics=args.auto_physics,
        night_start=args.night_start,
        night_end=args.night_end,
        cmax_quantile=args.cmax_quantile,
        cmax_margin=args.cmax_margin,
        cmax_smooth_slots=args.cmax_smooth_slots,
    )
    df, data_cfg = prepare_dataframe(data_cfg)
    merged_path = out_dir / "_merged_input.csv"
    df.to_csv(merged_path, index=False)
    print(f"Merged/prepared input saved to: {merged_path}")

    full_ds = PVWindowDataset(df, data_cfg)
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = contiguous_splits(len(full_ds), args.train_ratio, args.val_ratio)
    train_ds = PVWindowDataset(df, data_cfg, tr_s, tr_e)
    val_ds = PVWindowDataset(df, data_cfg, va_s, va_e)
    test_ds = PVWindowDataset(df, data_cfg, te_s, te_e)

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False),
    }

    model_cfg = PhysMoEConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_features=len(full_ds.feature_cols),
        target_idx=0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        patch_len=args.patch_len,
        stride=args.stride,
        router_hidden=args.router_hidden,
        dropout=args.dropout,
        trend_kernel=args.trend_kernel,
        router_tau=args.tau,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysMoE(model_cfg).to(device)
    criterion = PhysMoELoss(lambda_cap=args.lambda_cap, lambda_night=args.lambda_night)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_dump = {
        "args": vars(args),
        "data_cfg": asdict(data_cfg),
        "model_cfg": asdict(model_cfg),
        "feature_cols_used": full_ds.feature_cols,
        "num_parameters": count_parameters(model),
        "device": str(device),
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "note": "auto_Cmax/auto_night are auxiliary soft regularization signals, not measured physical supervision.",
    }
    save_json(config_dump, out_dir / "config.json")
    print(f"Feature columns: {full_ds.feature_cols}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Using device: {device}")

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history_path = out_dir / "history.csv"
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = run_epoch(model, loaders["val"], criterion, None, device)
        append_history(history_path, epoch, "train", train_metrics)
        append_history(history_path, epoch, "val", val_metrics)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_MSE={val_metrics.get('MSE', float('nan')):.6f} | "
            f"val_MAE={val_metrics.get('MAE', float('nan')):.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "model_cfg": asdict(model_cfg), "epoch": epoch}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, loaders["test"], criterion, None, device)
    append_history(history_path, best_epoch, "test", test_metrics)
    save_json(test_metrics, out_dir / "test_metrics.json")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
