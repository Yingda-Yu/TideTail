"""A0 — Real-only forecasting baselines on prime data (96h → 72h).

Compares persistence, seasonal (24h period), and trainable GRU/TCN.
All use train-only scaler. Saves results to data/results/a0_baselines_96_72.json

Usage:
    python3 scripts/run_a0_baselines.py --epochs 10 --regions ny_bight
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")


def load_region(region: str, context: int = 96, horizon: int = 72):
    base = DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    meta = json.loads((base / "train_only_metadata_v1.json").read_text())
    wi = np.load(base / "window_index_v1.npz")
    merged = target[["timestamp_utc", "water_level_m"]].merge(
        tide[["timestamp_utc", "tide_level_m"]], on="timestamp_utc", how="inner"
    )
    water_level = merged["water_level_m"].to_numpy(dtype=np.float32)

    def _win(starts):
        starts = starts.astype(np.int64)
        N = len(starts)
        h = np.zeros((N, context), dtype=np.float32)
        t = np.zeros((N, horizon), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = water_level[s : s + context]
            t[i] = water_level[s + context : s + context + horizon]
        return h, t

    return {
        "train_h": _win(wi["train_start_indices"])[0],
        "train_t": _win(wi["train_start_indices"])[1],
        "test_h": _win(wi["test_start_indices"])[0],
        "test_t": _win(wi["test_start_indices"])[1],
        "scaler_mean": float(meta["scalers"]["water_level_m"]["mean"]),
        "scaler_std": float(meta["scalers"]["water_level_m"]["std"]),
    }


# --- Rule baselines -----------------------------------------------------
def persistence(history, horizon):
    return np.full(horizon, float(history[-1]))


def seasonal(history, horizon, period=24):
    base = history[-period:] if len(history) >= period else np.tile(history, period // len(history) + 1)[-period:]
    return np.array([base[k % period] for k in range(horizon)])


def batch_predict(histories, fn, horizon):
    out = np.zeros((len(histories), horizon), dtype=np.float32)
    for i, h in enumerate(histories):
        out[i] = fn(h, horizon)
    return out


# --- Trainable baselines ------------------------------------------------
class GRUForecaster(nn.Module):
    def __init__(self, context=96, horizon=72, hidden=64):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


class TCNForecaster(nn.Module):
    def __init__(self, context=96, horizon=72, width=32):
        super().__init__()
        self.input_proj = nn.Conv1d(1, width, 3, padding=1)
        self.blocks = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=8, dilation=8), nn.ReLU(),
        )
        self.fc = nn.Linear(width, horizon)
    def forward(self, x):
        h = self.input_proj(x.unsqueeze(1))
        h = self.blocks(h)
        return self.fc(h[:, :, -1])


def train_torch_baseline(model_cls, data, device, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = data["scaler_mean"], data["scaler_std"]
    X = torch.from_numpy((data["train_h"] - mean) / std).to(device)
    Y = torch.from_numpy((data["train_t"] - mean) / std).to(device)
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        loss_ep = 0.0
        for b in range((n + batch_size - 1) // batch_size):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            loss = F.mse_loss(model(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_ep += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy((data["test_h"] - mean) / std).to(device)
        Yp = model(Xt).cpu().numpy() * std + mean
    return Yp


# --- Metrics ------------------------------------------------------------
def metrics(pred, real):
    err = pred - real
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # q95 tail MAE
    tail_mask = np.abs(real) > np.percentile(np.abs(real), 95)
    tail_mae = float(np.mean(np.abs(err[tail_mask]))) if tail_mask.sum() > 0 else float("nan")
    # Peak error per sample
    peak_pred = pred.max(axis=1)
    peak_real = real.max(axis=1)
    peak_mae = float(np.mean(np.abs(peak_pred - peak_real)))
    # Peak timing
    idx_p = pred.argmax(axis=1)
    idx_r = real.argmax(axis=1)
    timing_mae = float(np.mean(np.abs(idx_p - idx_r)))
    return {"mae": mae, "rmse": rmse, "tail95_mae": tail_mae,
            "peak_mae": peak_mae, "timing_mae": timing_mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--regions", nargs="+",
                    default=["ny_bight", "thames_north_sea", "us_gulf_coast",
                             "south_atlantic_charleston", "southeast_australia"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    results = []

    for region in args.regions:
        print(f"\n=== {region} ===")
        data = load_region(region)
        test_h, test_t = data["test_h"], data["test_t"]
        print(f"  test windows: {len(test_t)}, real std={test_t.std():.3f}")

        # Persistence
        pred = batch_predict(test_h, persistence, horizon=72)
        r = metrics(pred, test_t)
        r.update({"model": "persistence", "region": region})
        results.append(r)
        print(f"  persistence:  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"tail95={r['tail95_mae']:.4f}  peak_err={r['peak_mae']:.4f}")

        # Seasonal (24h)
        pred = batch_predict(test_h, lambda h, ho: seasonal(h, ho, 24), horizon=72)
        r = metrics(pred, test_t)
        r.update({"model": "seasonal_24h", "region": region})
        results.append(r)
        print(f"  seasonal_24h: MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"tail95={r['tail95_mae']:.4f}  peak_err={r['peak_mae']:.4f}")

        # GRU
        t0 = time.time()
        pred = train_torch_baseline(GRUForecaster, data, device, args.epochs, args.batch_size, args.lr, seed=42)
        r = metrics(pred, test_t)
        r.update({"model": "gru", "region": region, "runtime_s": time.time() - t0})
        results.append(r)
        print(f"  GRU:          MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"tail95={r['tail95_mae']:.4f}  peak_err={r['peak_mae']:.4f}  "
              f"({r.get('runtime_s', 0):.1f}s)")

        # TCN
        t0 = time.time()
        pred = train_torch_baseline(TCNForecaster, data, device, args.epochs, args.batch_size, args.lr, seed=42)
        r = metrics(pred, test_t)
        r.update({"model": "tcn", "region": region, "runtime_s": time.time() - t0})
        results.append(r)
        print(f"  TCN:          MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"tail95={r['tail95_mae']:.4f}  peak_err={r['peak_mae']:.4f}  "
              f"({r.get('runtime_s', 0):.1f}s)")

    out = RESULTS_DIR / "a0_baselines_96_72.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # Summary table
    print(f"\n{'Model':<18} {'Region':<25} {'MAE':>8} {'RMSE':>8} {'Tail95':>8} {'PeakErr':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<18} {r['region']:<25} "
              f"{r['mae']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['tail95_mae']:>8.4f} {r['peak_mae']:>9.4f}")


if __name__ == "__main__":
    main()
