from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PhysMoEConfig:
    """Configuration for the PhysMoE model.

    The implementation is a paper-faithful, best-effort reproduction. Some
    hyperparameters are exposed because the original paper does not disclose
    all implementation details.
    """

    seq_len: int = 96
    pred_len: int = 32
    num_features: int = 5
    target_idx: int = 0
    d_model: int = 64
    n_heads: int = 4
    patch_len: int = 16
    stride: int = 8
    router_hidden: int = 64
    dropout: float = 0.1
    trend_kernel: int = 3
    router_tau: float = 0.05
    revin_affine: bool = True
    use_roll_difference: bool = False

    @property
    def num_exogenous(self) -> int:
        return max(0, self.num_features - 1)

    @property
    def n_patches(self) -> int:
        if self.seq_len < self.patch_len:
            raise ValueError("seq_len must be >= patch_len")
        return 1 + (self.seq_len - self.patch_len) // self.stride


class RevIN(nn.Module):
    """Reversible Instance Normalization for tensors shaped [B, L, C]."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        self._mean: Optional[torch.Tensor] = None
        self._stdev: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            var = x.var(dim=1, keepdim=True, unbiased=False).detach()
            self._stdev = torch.sqrt(var + self.eps)
            out = (x - self._mean) / self._stdev
            if self.affine:
                out = out * self.gamma + self.beta
            return out
        if mode == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError("RevIN denorm called before norm.")
            out = x
            if self.affine:
                out = (out - self.beta[..., : out.shape[-1]]) / (self.gamma[..., : out.shape[-1]] + self.eps)
            return out * self._stdev[..., : out.shape[-1]] + self._mean[..., : out.shape[-1]]
        raise ValueError("mode must be 'norm' or 'denorm'")

    def denorm_target(self, y: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """Denormalize a target prediction shaped [B, H] or [B, H, 1]."""
        if self._mean is None or self._stdev is None:
            raise RuntimeError("RevIN denorm_target called before norm.")
        squeeze = False
        if y.ndim == 2:
            y = y.unsqueeze(-1)
            squeeze = True
        mean = self._mean[:, :, target_idx : target_idx + 1]
        std = self._stdev[:, :, target_idx : target_idx + 1]
        if self.affine:
            gamma = self.gamma[:, :, target_idx : target_idx + 1]
            beta = self.beta[:, :, target_idx : target_idx + 1]
            y = (y - beta) / (gamma + self.eps)
        out = y * std + mean
        return out.squeeze(-1) if squeeze else out


class MeteorologicalFeatureExtractor(nn.Module):
    """Raw + first-difference + moving-average meteorological feature augmentation."""

    def __init__(self, num_exogenous: int, d_model: int, trend_kernel: int = 3, use_roll_difference: bool = False):
        super().__init__()
        self.num_exogenous = num_exogenous
        self.trend_kernel = trend_kernel
        self.use_roll_difference = use_roll_difference
        self.proj = nn.Linear(num_exogenous * 3, d_model)

    def forward(self, x_ex: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_ex: [B, L, C_ex]
        if self.use_roll_difference:
            diff = x_ex - torch.roll(x_ex, shifts=1, dims=1)
        else:
            prev = torch.cat([x_ex[:, :1], x_ex[:, :-1]], dim=1)
            diff = x_ex - prev

        # Average pooling over time for each variable.
        k = max(1, int(self.trend_kernel))
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        x_t = x_ex.transpose(1, 2)  # [B, C_ex, L]
        trend = F.avg_pool1d(F.pad(x_t, (pad_left, pad_right), mode="replicate"), kernel_size=k, stride=1)
        trend = trend.transpose(1, 2)

        aug = torch.cat([x_ex, diff, trend], dim=-1)
        z_ex = self.proj(aug)
        return z_ex, {"diff": diff, "trend": trend}


class PatchEncoder(nn.Module):
    """Patchify [B, L, D] and map each flattened patch to d_model."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (seq_len - patch_len) // stride
        self.proj = nn.Linear(patch_len * in_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, N, P, D]
        patches = z.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold returns [B, N, D, P], swap to [B, N, P, D]
        patches = patches.transpose(-1, -2).contiguous()
        patches = patches.reshape(z.shape[0], self.n_patches, -1)
        h = self.proj(patches) + self.pos
        return self.dropout(h)


class PhysicsGuidedRouter(nn.Module):
    """Router driven only by exogenous meteorological representation."""

    def __init__(self, d_model: int, router_hidden: int, tau: float = 0.05, dropout: float = 0.1):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(d_model, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 2),
        )

    def forward(self, h_ex: torch.Tensor) -> torch.Tensor:
        # h_ex: [B, N, D]; use weather summary as routing trigger.
        pooled = h_ex.mean(dim=1)
        logits = self.net(pooled)
        return torch.softmax(logits / self.tau, dim=-1)  # [B, 2]


class SteadyStateExpert(nn.Module):
    """Endogenous self-attention expert."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_en: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(h_en, h_en, h_en, need_weights=False)
        h = self.norm(h_en + out)
        return self.norm2(h + self.ffn(h))


class MeteorologicalVolatilityExpert(nn.Module):
    """Causal cross-attention: Query = power, Key/Value = weather."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_en: torch.Tensor, h_ex: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross(query=h_en, key=h_ex, value=h_ex, need_weights=False)
        h = self.norm(h_en + out)
        return self.norm2(h + self.ffn(h))


class PhysMoE(nn.Module):
    """PhysMoE model.

    Forward input: x shaped [B, seq_len, num_features]. The target power channel
    is expected to be at target_idx, and dataset utilities move it to index 0 by
    default.
    """

    def __init__(self, cfg: PhysMoEConfig):
        super().__init__()
        if cfg.num_features < 2:
            raise ValueError("PhysMoE requires at least target + one exogenous feature.")
        self.cfg = cfg
        self.revin = RevIN(cfg.num_features, affine=cfg.revin_affine)

        self.en_proj = nn.Linear(1, cfg.d_model)
        self.ex_extractor = MeteorologicalFeatureExtractor(
            cfg.num_exogenous, cfg.d_model, cfg.trend_kernel, cfg.use_roll_difference
        )
        self.en_patch = PatchEncoder(cfg.seq_len, cfg.patch_len, cfg.stride, cfg.d_model, cfg.d_model, cfg.dropout)
        self.ex_patch = PatchEncoder(cfg.seq_len, cfg.patch_len, cfg.stride, cfg.d_model, cfg.d_model, cfg.dropout)

        self.router = PhysicsGuidedRouter(cfg.d_model, cfg.router_hidden, cfg.router_tau, cfg.dropout)
        self.steady = SteadyStateExpert(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.volatile = MeteorologicalVolatilityExpert(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.out_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cfg.n_patches * cfg.d_model, cfg.pred_len),
        )
        self.linear_shortcut = nn.Linear(cfg.seq_len, cfg.pred_len)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x_norm = self.revin(x, mode="norm")
        x_en = x_norm[..., self.cfg.target_idx : self.cfg.target_idx + 1]
        ex_indices = [i for i in range(self.cfg.num_features) if i != self.cfg.target_idx]
        x_ex = x_norm[..., ex_indices]

        z_en = self.en_proj(x_en)
        z_ex, dyn = self.ex_extractor(x_ex)
        h_en = self.en_patch(z_en)
        h_ex = self.ex_patch(z_ex)

        g = self.router(h_ex)  # [B, 2]
        o_sse = self.steady(h_en)
        o_mve = self.volatile(h_en, h_ex)
        g_s = g[:, 0].view(-1, 1, 1)
        g_v = g[:, 1].view(-1, 1, 1)
        o = g_s * o_sse + g_v * o_mve

        y_norm = self.out_proj(o) + self.linear_shortcut(x_en.squeeze(-1))
        y = self.revin.denorm_target(y_norm, self.cfg.target_idx)

        if return_aux:
            return y, {"router": g, "diff": dyn["diff"], "trend": dyn["trend"], "o_sse": o_sse, "o_mve": o_mve}
        return y


class PhysMoELoss(nn.Module):
    """MSE plus optional auxiliary physics-informed regularization.

    L_cap and L_night are soft regularizers. If cmax/night are unavailable or
    lambda is set to zero, they are skipped.
    """

    def __init__(self, lambda_cap: float = 0.0, lambda_night: float = 0.0):
        super().__init__()
        self.lambda_cap = float(lambda_cap)
        self.lambda_night = float(lambda_night)

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        cmax: Optional[torch.Tensor] = None,
        night: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse = F.mse_loss(y_hat, y_true)
        cap = y_hat.new_tensor(0.0)
        night_loss = y_hat.new_tensor(0.0)
        if self.lambda_cap > 0 and cmax is not None:
            cap = F.relu(y_hat - cmax).mean()
        if self.lambda_night > 0 and night is not None:
            night_loss = ((night * y_hat) ** 2).mean()
        total = mse + self.lambda_cap * cap + self.lambda_night * night_loss
        return total, {"mse": float(mse.detach()), "cap": float(cap.detach()), "night": float(night_loss.detach())}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
