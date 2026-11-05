"""Forecasting baselines — from simple NumPy rules to trainable PyTorch models.

All models share the 96h → 72h main-task contract:
    input  : (batch, 96, 1) OR (batch, 96)
    output : (batch, 72, 1)

Simple persistence/seasonal rules are numpy-only and need no training.
GRUForecaster and TCNForecaster are torch.nn.Module with trainable weights.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

# ==========================================================================
# Rule-based NumPy baselines (no training)
# ==========================================================================

def persistence(history, horizon):
    """Last-value carry-forward baseline."""
    if np.asarray(history).size == 0:
        raise ValueError("history is empty")
    if horizon <= 0:
        return np.array([], dtype=float)
    return np.full(horizon, float(np.asarray(history)[-1]), dtype=float)


def seasonal_baseline(history, horizon, period=24):
    """Repeat the most recent seasonal period.

    Default period=24 = 1 day (solar diurnal), well-matched to 96h history.
    For longer histories or multi-day cycles, pass period=144 (6 days,
    semi-diurnal tide). Falls back to persistence when history < period
    AND a halved period is also too short.
    """
    arr = np.asarray(history)
    if arr.size < period and period > 24:
        # Retry with 24-hour period (always fits in 96h history)
        period = 24
    if arr.size < period:
        return persistence(arr, horizon)
    out = np.empty(horizon, dtype=float)
    base = arr[-period:]
    for k in range(horizon):
        out[k] = base[k % period]
    return out


def train_only_scaler(values: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) computed from train values only.

    Input can be numpy or torch, 1D or multi-D (flattened).
    Handles NaNs via np.nanmean / np.nanstd equivalent.
    """
    if isinstance(values, np.ndarray):
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size < 2:
            raise ValueError("need ≥2 finite train values for scaler")
        mean = float(v.mean())
        std = float(v.std())
    else:
        v = torch.as_tensor(values, dtype=torch.float32).ravel()
        v = v[torch.isfinite(v)]
        if v.numel() < 2:
            raise ValueError("need ≥2 finite train values for scaler")
        mean = float(v.mean())
        std = float(v.std())
    if std == 0 or not math.isfinite(std):
        std = 1.0  # degenerate constant series — avoid division by zero
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def make_windows(series: np.ndarray, lookback: int, horizon: int,
                  stride: int = 1, mask_nan: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding (history, target, mask) windows from a 1D hourly series.

    Args:
        series: 1D numpy array of hourly values (may contain NaN).
        lookback: hours of history per window.
        horizon: hours to forecast.
        stride: skip factor between windows; stride=1 → every window.
        mask_nan: if True, windows with ANY NaN in target are excluded
                  (history NaNs produce a missing_mask but are kept).

    Returns:
        history: (n_valid, lookback) float64
        target:  (n_valid, horizon) float64
        missing: (n_valid, lookback) bool — True where history had NaN (replaced by 0).
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D hourly array")
    total = lookback + horizon
    n_possible = series.size - total + 1
    if n_possible <= 0:
        raise ValueError(f"series too short: need >={total} rows, got {series.size}")
    if stride <= 0:
        raise ValueError("stride must be positive")

    origins = np.arange(0, n_possible, stride)
    if origins.size == 0:
        raise ValueError("no windows generated at requested stride")

    hist = np.stack([series[o:o + lookback] for o in origins])
    targ = np.stack([series[o + lookback:o + total] for o in origins])

    # Missing mask from history
    missing = ~np.isfinite(hist)
    # Replace NaN with 0 in history (missing_mask tells model which positions were bad)
    hist = np.where(missing, 0.0, hist)

    # Exclude windows where ANY target cell is NaN (never forecast NaN)
    if mask_nan:
        valid_mask = np.isfinite(targ).all(axis=1)
        hist = hist[valid_mask]
        targ = targ[valid_mask]
        missing = missing[valid_mask]

    return hist.astype(np.float64), targ.astype(np.float64), missing


# ==========================================================================
# GRU — 96 history → 72 forecast, encoder-decoder style
# ==========================================================================

class GRUForecaster(nn.Module):
    """Encoder-Decoder GRU for multi-step forecasting.

    Encodes 96 hours of history into a hidden state, then autoregressively
    decodes 72 hours by feeding each prediction back as input.

    Params ~ 5–10K depending on hidden_size. Matches DDPM budget (27K params).
    """

    def __init__(self, input_dim: int = 1, hidden_size: int = 48,
                 horizon: int = 72, num_layers: int = 2, dropout: float = 0.1,
                 scaler_mean: float = 0.0, scaler_std: float = 1.0):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_dim + 1, hidden_size)  # +1 for missing mask
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_size, 1)

        # Store scaler as non-persistent buffers (not in state_dict)
        self.register_buffer("scaler_mean", torch.tensor(scaler_mean))
        self.register_buffer("scaler_std", torch.tensor(scaler_std))

    def predict(self, history: torch.Tensor, missing: torch.Tensor | None = None
                ) -> torch.Tensor:
        """history: (B, 96, 1) OR (B, 96). Returns (B, 72, 1).

        Computes gradients (for training). Use predict_eval for inference.
        """
        if history.ndim == 2:
            history = history.unsqueeze(-1)
        B = history.shape[0]
        device = history.device

        z = self._encode(history, missing)

        dec_input = torch.zeros(B, 1, self.hidden_size, device=device)
        outputs = []
        h = z
        for _ in range(self.horizon):
            out, h = self.decoder(dec_input, h)
            pred = self.output_proj(out)
            outputs.append(pred)
            dec_input = out

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def predict_eval(self, history: torch.Tensor, missing: torch.Tensor | None = None
                      ) -> torch.Tensor:
        """Inference-mode prediction (no gradient tracking)."""
        return self.predict(history, missing)

    def forward_loss(self, history: torch.Tensor, target: torch.Tensor,
                     missing: torch.Tensor | None = None,
                     target_mask: torch.Tensor | None = None) -> torch.Tensor:
        """MSE loss on normalized scale, masked for NaN targets."""
        if history.ndim == 2:
            history = history.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        # Normalize both using train-only scaler
        h_norm = (history - self.scaler_mean) / self.scaler_std
        t_norm = (target - self.scaler_mean) / self.scaler_std

        pred = self.predict(h_norm, missing)

        if target_mask is not None:
            mask = target_mask.unsqueeze(-1) if target_mask.ndim == 2 else target_mask
            diff = (pred - t_norm) ** 2
            diff = diff * mask.float()
            return diff.sum() / mask.float().sum().clamp(min=1)
        return torch.mean((pred - t_norm) ** 2)

    def _encode(self, x: torch.Tensor, missing: torch.Tensor | None) -> torch.Tensor:
        """Run encoder GRU, return final hidden state."""
        B, L, _C = x.shape
        # Always concatenate a mask channel (zeros if no missing info)
        if missing is not None and missing.ndim == 2:
            mask = missing.unsqueeze(-1).float()
        else:
            mask = torch.zeros(B, L, 1, device=x.device)
        x = torch.cat([x, mask], dim=-1)

        x = self.input_proj(x.view(-1, x.shape[-1])).view(B, L, -1)
        _, h = self.encoder(x)
        return h


# ==========================================================================
# TCN — dilated causal convolutions, non-autoregressive
# ==========================================================================

class Chomp1d(nn.Module):
    """Chomp padding to enforce causality: right-pad then remove future values."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T). Returns same shape but last chomp_size timesteps cropped."""
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu, self.drop1,
                                  self.conv2, self.chomp2, self.relu, self.drop2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = self.downsample(x) if self.downsample is not None else x
        return self.out_relu(out + res)


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network for multi-step forecasting.

    Stack of dilated causal convolutions. Receptive field grows
    exponentially (dilation 1, 2, 4, 8, ...). Final linear layer
    projects the latest feature to all 72 horizon steps at once
    (non-autoregressive, full-horizon direct prediction).

    Params scale with channels; 32 channels × 4 layers ≈ 15–20K.
    """

    def __init__(self, input_dim: int = 1, horizon: int = 72,
                 channels: list[int] = (32, 32, 32, 32),
                 kernel_size: int = 3, dropout: float = 0.1,
                 scaler_mean: float = 0.0, scaler_std: float = 1.0):
        super().__init__()
        self.horizon = horizon

        layers: list[nn.Module] = []
        in_ch = input_dim + 1  # +1 for missing mask channel
        for out_ch in channels:
            dilation = 2 ** (channels.index(out_ch))
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

        # Final linear: take LAST timestep features → all horizon steps
        self.forecast_head = nn.Linear(in_ch, horizon)

        self.register_buffer("scaler_mean", torch.tensor(scaler_mean))
        self.register_buffer("scaler_std", torch.tensor(scaler_std))

    def predict(self, history: torch.Tensor, missing: torch.Tensor | None = None
                ) -> torch.Tensor:
        """history: (B, 96, 1) OR (B, 96). Returns (B, 72, 1).

        Computes gradients (for training). Use predict_eval for inference.
        """
        if history.ndim == 2:
            history = history.unsqueeze(-1)
        B, L, _C = history.shape
        device = history.device

        h = history.transpose(1, 2)  # (B, 1, 96)
        # Always concatenate a mask channel (zeros if no missing info)
        if missing is not None and missing.ndim == 2:
            m = missing.float().unsqueeze(1)  # (B, 1, 96)
        else:
            m = torch.zeros(B, 1, L, device=device)
        h = torch.cat([h, m], dim=1)  # (B, 2, 96) — matches input_dim+1

        feat = self.network(h)  # (B, channels, 96)
        last_feat = feat[:, :, -1]  # (B, channels)
        out = self.forecast_head(last_feat)  # (B, 72)
        return out.unsqueeze(-1)  # (B, 72, 1)

    @torch.no_grad()
    def predict_eval(self, history: torch.Tensor, missing: torch.Tensor | None = None
                      ) -> torch.Tensor:
        """Inference-mode prediction (no gradient tracking)."""
        return self.predict(history, missing)

    def forward_loss(self, history: torch.Tensor, target: torch.Tensor,
                     missing: torch.Tensor | None = None,
                     target_mask: torch.Tensor | None = None) -> torch.Tensor:
        if history.ndim == 2:
            history = history.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        h_norm = (history - self.scaler_mean) / self.scaler_std
        t_norm = (target - self.scaler_mean) / self.scaler_std

        pred = self.predict(h_norm, missing)

        if target_mask is not None:
            mask = target_mask.unsqueeze(-1) if target_mask.ndim == 2 else target_mask
            diff = (pred - t_norm) ** 2 * mask.float()
            return diff.sum() / mask.float().sum().clamp(min=1)
        return torch.mean((pred - t_norm) ** 2)


# ==========================================================================
# Training utilities
# ==========================================================================

def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                     history: torch.Tensor, target: torch.Tensor,
                     missing: torch.Tensor | None = None,
                     batch_size: int = 64, shuffle: bool = True,
                     missing_target_mask: torch.Tensor | None = None
                     ) -> float:
    """Train for one epoch. Returns mean training loss."""
    # Build default masks if not provided
    if missing is None:
        missing = torch.zeros_like(history, dtype=torch.bool)
    if missing_target_mask is None:
        missing_target_mask = torch.ones_like(target, dtype=torch.bool)

    dataset = torch.utils.data.TensorDataset(history, target, missing, missing_target_mask)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=0,
    )
    model.train()
    total_loss = 0.0
    n_batches = 0
    for h, t, m, tm in loader:
        optimizer.zero_grad()
        loss = model.forward_loss(h, t, missing=m, target_mask=tm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)
