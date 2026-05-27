# PhysMoE: Physics-guided Dynamic Mixture-of-Experts for PV Forecasting

This repository provides a **paper-faithful, best-effort PyTorch implementation** of **PhysMoE**, a physics-guided dynamic mixture-of-experts model for photovoltaic (PV) power forecasting under meteorological mutations.

The implementation follows the main modules described in the paper:

- RevIN preprocessing.
- Endogenous/exogenous channel decoupling.
- Meteorological dynamic feature extraction with raw variables, first-order differences, and moving averages.
- Patch encoding for endogenous power and exogenous meteorological streams.
- Physics-guided dynamic MoE router driven by meteorological representations.
- Steady-State Expert (SSE) based on endogenous self-attention.
- Meteorological Volatility Expert (MVE) based on decoupled cross-attention where Query = power and Key/Value = weather.
- Dynamic expert aggregation.
- Optional auxiliary physics-informed regularization using an estimated clear-sky envelope and timestamp-derived night indicator.

> Important: This is not an official implementation unless explicitly stated by the paper authors. Some reproduction-critical details are not fully specified in the paper, so this repository exposes the relevant choices as command-line parameters.

---

## Repository structure

```text
physmoe-pv/
├── physmoe/
│   ├── __init__.py
│   ├── data.py          # CSV/folder loading, sliding windows, auto physics columns
│   ├── metrics.py       # MSE, MAE, sMAPE, nRMSE, R2, FS
│   ├── model.py         # PhysMoE model and optional regularization loss
│   └── utils.py
├── scripts/
│   ├── inspect_csv.py
│   └── add_auto_physics.py
├── examples/
│   └── create_synthetic_data.py
├── configs/
│   └── nantong13.yaml
├── train.py
├── predict.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone <your-repo-url>.git
cd physmoe-pv
pip install -r requirements.txt
```

Editable install is also supported:

```bash
pip install -e .
```

---

## Data format

The code supports two data layouts.

### 1. Single CSV

```text
data/nantong13.csv
```

### 2. One folder per PV station, multiple chronological CSV files

```text
data/
├── Nantong1.3/
│   ├── Nantong1.3__2023-01-02_2023-06-17.csv
│   ├── Nantong1.3__2023-06-18_2023-10-01.csv
│   └── ...
├── Nantong1.6/
│   ├── file_1.csv
│   └── ...
└── Tianchang/
    ├── file_1.csv
    └── ...
```

When `--csv` points to a folder, `train.py` automatically reads all `*.csv` files under that folder, sorts them by filename, concatenates them, sorts by `--timestamp_col`, and saves the prepared data to:

```text
<out>/_merged_input.csv
```

For your Nantong-style files, a typical schema is:

```csv
date,Ratio_0,Ratio_1,Ratio_2,Ratio_3,OT
2023-01-02 00:00:00,0.12,0.34,0.56,0.78,0.0
...
```

where `OT` is the PV power target.

---

## Inspect your data

```bash
python scripts/inspect_csv.py \
  --csv ./data/Nantong1.3 \
  --timestamp_col date
```

---

## Quick smoke test with synthetic data

```bash
python examples/create_synthetic_data.py --out ./examples/synthetic_station/synthetic.csv --days 30

python train.py \
  --csv ./examples/synthetic_station \
  --timestamp_col date \
  --target_col OT \
  --feature_cols OT Ratio_0 Ratio_1 Ratio_2 Ratio_3 Wind \
  --seq_len 96 \
  --pred_len 32 \
  --epochs 3 \
  --batch_size 16 \
  --out ./runs/smoke_test
```

---

## Training on a station folder

Example for a Nantong1.3-style dataset:

```bash
python train.py \
  --csv ./data/Nantong1.3 \
  --timestamp_col date \
  --target_col OT \
  --feature_cols OT Ratio_0 Ratio_1 Ratio_2 Ratio_3 \
  --seq_len 96 \
  --pred_len 32 \
  --d_model 64 \
  --n_heads 4 \
  --patch_len 16 \
  --stride 8 \
  --trend_kernel 3 \
  --tau 0.05 \
  --epochs 40 \
  --patience 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --out ./runs/nantong13_8h
```

For 15-minute data, the prediction horizon mapping is:

| Forecast horizon | `--pred_len` |
|---|---:|
| 15 min | 1 |
| 30 min | 2 |
| 1 h | 4 |
| 2 h | 8 |
| 4 h | 16 |
| 8 h | 32 |

---

## Optional auxiliary physics-informed regularization

Some PV datasets do not provide measured clear-sky limits `Cmax(t)` or night indicators `Inight(t)`. This repository therefore treats the physical terms as **optional auxiliary soft regularization**, not mandatory supervision.

Enable automatic generation of:

- `auto_Cmax`: empirical clear-sky upper envelope estimated by a high quantile of historical PV power at each intra-day time slot.
- `auto_night`: timestamp-derived night-time indicator.

```bash
python train.py \
  --csv ./data/Nantong1.3 \
  --timestamp_col date \
  --target_col OT \
  --feature_cols OT Ratio_0 Ratio_1 Ratio_2 Ratio_3 \
  --auto_physics \
  --cmax_quantile 0.98 \
  --cmax_margin 1.05 \
  --night_start 18 \
  --night_end 6 \
  --lambda_cap 0.1 \
  --lambda_night 0.1 \
  --seq_len 96 \
  --pred_len 32 \
  --epochs 40 \
  --batch_size 32 \
  --lr 1e-4 \
  --out ./runs/nantong13_8h_physreg
```

If you have explicit columns in your data, use them directly:

```bash
python train.py \
  --csv ./data/Nantong1.3 \
  --timestamp_col date \
  --target_col OT \
  --feature_cols OT Ratio_0 Ratio_1 Ratio_2 Ratio_3 \
  --cmax_col Cmax \
  --night_col night \
  --lambda_cap 0.1 \
  --lambda_night 0.1
```

Set the weights to zero to disable the regularization:

```bash
--lambda_cap 0 --lambda_night 0
```

---

## Outputs

A training run writes:

```text
runs/<run_name>/
├── _merged_input.csv    # merged/prepared CSV, including auto_Cmax/auto_night if enabled
├── best.pt              # best checkpoint by validation loss
├── config.json          # full run configuration
├── history.csv          # epoch metrics
└── test_metrics.json    # final test metrics
```

---

## Prediction

```bash
python predict.py \
  --checkpoint ./runs/nantong13_8h/best.pt \
  --csv ./data/Nantong1.3 \
  --timestamp_col date \
  --target_col OT \
  --feature_cols OT Ratio_0 Ratio_1 Ratio_2 Ratio_3 \
  --seq_len 96 \
  --pred_len 32 \
  --out ./runs/nantong13_8h/predictions.csv
```

---

## Reproducibility notes

The paper does not disclose every implementation detail, such as exact feature columns, split boundaries, missing-value handling, hidden dimensions, patch length/stride, random seeds, and clear-sky envelope construction. This implementation therefore makes transparent defaults and saves all run settings in `config.json`.

Recommended reporting practice:

- Use chronological train/validation/test splits.
- Report `MSE`, `MAE`, `sMAPE`, `nRMSE`, `R2`, and `FS`.
- Repeat experiments with multiple random seeds where possible.
- Treat `auto_Cmax` and `auto_night` as auxiliary soft regularization signals, not measured physical labels.

---

## Citation

If you use this implementation for academic work, cite the corresponding PhysMoE paper and state that this is a best-effort reproduction.
