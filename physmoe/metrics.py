from __future__ import annotations

from typing import Dict, Optional
import torch


def regression_metrics(y_hat: torch.Tensor, y_true: torch.Tensor, persistence: Optional[torch.Tensor] = None) -> Dict[str, float]:
    y_hat = y_hat.detach().float().reshape(-1)
    y_true = y_true.detach().float().reshape(-1)
    eps = 1e-8
    mse = torch.mean((y_true - y_hat) ** 2)
    mae = torch.mean(torch.abs(y_true - y_hat))
    smape = torch.mean(torch.abs(y_true - y_hat) / ((torch.abs(y_true) + torch.abs(y_hat)) / 2 + eps)) * 100
    nrmse = torch.sqrt(mse) / (torch.mean(y_true) + eps) * 100
    r2 = 1 - torch.sum((y_true - y_hat) ** 2) / (torch.sum((y_true - torch.mean(y_true)) ** 2) + eps)
    out = {
        "MSE": float(mse),
        "MAE": float(mae),
        "sMAPE": float(smape),
        "nRMSE": float(nrmse),
        "R2": float(r2),
    }
    if persistence is not None:
        persistence = persistence.detach().float().reshape(-1)
        mse_p = torch.mean((y_true - persistence) ** 2)
        out["FS"] = float((1 - mse / (mse_p + eps)) * 100)
    return out
