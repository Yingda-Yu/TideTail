"""A3 — Flow Matching synthetic augmentation + GRU/TCN fair comparison.

Protocol:
  A0: Real-only GRU/TCN (already done in data/results/a0_baselines_96_72.json)
  A1: Extreme oversampling (augment.py extreme_balance_weights, extreme_weight=5)
  A2: Physics residual noise augment (augment.py residual_noise_augment, noise_std=0.3)
  A3: Flow Matching synthetic residual augmentation at [10%, 25%, 50%]

  ALL augments applied to TRAIN set ONLY. Test is held-out chronological.
  Independent storm grouping + bootstrap CI required for H1 claim.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/run_a3_flowmatching_augment.py \
        --regions ny_bight thames_north_sea us_gulf_coast \
        --augment-ratios 0.1 0.25 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tidetail.models import FlowMatching, VelocityTCN
from tidetail.augment import extreme_balance_weights, residual_noise_augment

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Data loading (same as train_generative_models.py)
# ==========================================================================
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
    tide_level = merged["tide_level_m"].to_numpy(dtype=np.float32)
    residual = water_level - tide_level

    scaler_mean = float(meta["scalers"]["water_level_m"]["mean"])
    scaler_std = float(meta["scalers"]["water_level_m"]["std"])

    def _make_windows(starts, series):
        starts = starts.astype(np.int64)
        N = len(starts)
        h = np.zeros((N, context), dtype=np.float32)
        t = np.zeros((N, horizon), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = series[s : s + context]
            t[i] = series[s + context : s + context + horizon]
        return h, t

    tr_h_wl, tr_t_wl = _make_windows(wi["train_start_indices"], water_level)
    rtr_h, rtr_t = _make_windows(wi["train_start_indices"], residual)
    rva_h, rva_t = _make_windows(wi["validation_start_indices"], residual)
    rte_h, rte_t = _make_windows(wi["test_start_indices"], residual)

    return {
        "train_hist_res": rtr_h, "train_tgt_res": rtr_t,
        "val_hist_res": rva_h, "val_tgt_res": rva_t,
        "test_hist_res": rte_h, "test_tgt_res": rte_t,
        "train_hist_wl": tr_h_wl, "train_tgt_wl": tr_t_wl,
        "scaler_mean": scaler_mean, "scaler_std": scaler_std,
        "meta": meta,
    }


# ==========================================================================
# GRU / TCN models (small, CPU-compatible)
# ==========================================================================
class TCNForecaster(torch.nn.Module):
    """TCN encoder → global pool → linear forecast head (context=96 → horizon=72)."""
    def __init__(self, context=96, horizon=72, width=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.input_proj = torch.nn.Conv1d(1, width, 3, padding=1)
        self.blocks = torch.nn.ModuleList([
            _TCNBlock(width, d) for d in dilations
        ])
        # Aggregate over time dim, then linear to horizon
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(width, horizon)

    def forward(self, x):
        # x: (batch, context) → (batch, horizon)
        x = x.unsqueeze(1)       # (batch, 1, 96)
        x = self.input_proj(x)   # (batch, width, 96)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)  # (batch, width)
        return self.head(x)        # (batch, horizon)


class _TCNBlock(torch.nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(4, width), torch.nn.SiLU(),
            torch.nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            torch.nn.GroupNorm(4, width), torch.nn.SiLU(),
            torch.nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return (x + self.net(x)) / np.sqrt(2)


class GRUForecaster(torch.nn.Module):
    def __init__(self, context=96, horizon=72, hidden=64, num_layers=2):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.gru = torch.nn.GRU(1, hidden, num_layers=num_layers,
                                batch_first=True, bidirectional=False)
        self.proj = torch.nn.Linear(hidden, horizon)

    def forward(self, x):
        # x: (batch, context) → (batch, horizon)
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


# ==========================================================================
# Synthetic generation from trained Flow Matching checkpoint
# ==========================================================================
def generate_synthetic_residuals(ckpt_path: Path,
                                 hist_res: torch.Tensor,
                                 device: str,
                                 n_gen_per_real: int = 1) -> np.ndarray:
    """Generate synthetic residual targets using a saved FlowMatching model.

    Uses history residual as conditioning, samples `n_gen_per_real` targets
    per real history window. Returns shape (N, horizon) synthetic targets.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    velocity_net = VelocityTCN(context=96, horizon=72, width=32).to(device)
    velocity_net.load_state_dict(ckpt["state_dict"])
    fm = FlowMatching(context=96, horizon=72, euler_steps=64, device=device)

    # Per-sample residual normalization
    h = hist_res.to(device)
    h_mean = h.mean(dim=1, keepdim=True)
    h_std = h.std(dim=1, keepdim=True).clamp(min=1e-4)
    h_norm = (h - h_mean) / h_std

    all_gen = []
    seed_base = int(ckpt_path.stem.split("seed")[1]) if "seed" in ckpt_path.stem else 42
    for k in range(n_gen_per_real):
        with torch.no_grad():
            gen_norm = fm.sample(h_norm, velocity_net, seed=seed_base + k * 1000)
            gen = gen_norm.cpu().numpy() * h_std.cpu().numpy() + h_mean.cpu().numpy()
        all_gen.append(gen)

    return np.concatenate(all_gen, axis=0)


# ==========================================================================
# Training loop
# ==========================================================================
def train_forecaster(model, train_hist, train_tgt, val_hist, val_tgt,
                     epochs: int, batch_size: int, lr: float, device: str,
                     seed: int) -> dict:
    """Train GRU/TCN on residual forecast task."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure float32 — augment utilities may return float64
    H = torch.from_numpy(np.asarray(train_hist, dtype=np.float32)).to(device)
    T = torch.from_numpy(np.asarray(train_tgt, dtype=np.float32)).to(device)
    vH = torch.from_numpy(np.asarray(val_hist, dtype=np.float32)).to(device)
    vT = torch.from_numpy(np.asarray(val_tgt, dtype=np.float32)).to(device)

    n = H.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    history = {"epoch": [], "train_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            h_b = H[idx]
            t_b = T[idx]
            pred = model(h_b)
            loss = F.mse_loss(pred, t_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        # Validation MAE
        model.eval()
        with torch.no_grad():
            vpred = model(vH)
            val_mae = (vpred - vT).abs().mean().item()

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_mae"].append(val_mae)

    return history


def evaluate_forecaster(model, test_hist: np.ndarray, test_tgt: np.ndarray,
                        device: str) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(np.asarray(test_hist, dtype=np.float32)).to(device))
        pred_np = pred.cpu().numpy()

    err = pred_np - test_tgt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    peak_amp_err = float(np.abs(np.abs(pred_np).max() - np.abs(test_tgt).max()))
    peak_idx_pred = np.abs(pred_np).argmax(axis=1)
    peak_idx_real = np.abs(test_tgt).argmax(axis=1)
    peak_timing_mae = float(np.mean(np.abs(peak_idx_pred - peak_idx_real)))

    # Tail errors: q95/q99 of absolute error
    abs_err = np.abs(err)
    tail_mae_q95 = float(np.mean(abs_err[abs_err >= np.percentile(abs_err, 95)])) if \
        np.any(abs_err >= np.percentile(abs_err, 95)) else mae
    tail_mae_q99 = float(np.mean(abs_err[abs_err >= np.percentile(abs_err, 99)])) if \
        np.any(abs_err >= np.percentile(abs_err, 99)) else mae

    return {
        "mae": mae,
        "rmse": rmse,
        "peak_amp_err": peak_amp_err,
        "peak_timing_mae": peak_timing_mae,
        "tail_mae_q95": tail_mae_q95,
        "tail_mae_q99": tail_mae_q99,
        "real_q95": float(np.percentile(np.abs(test_tgt), 95)),
        "pred_q95": float(np.percentile(np.abs(pred_np), 95)),
    }


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["ny_bight", "thames_north_sea", "us_gulf_coast",
                             "south_atlantic_charleston", "southeast_australia"])
    ap.add_argument("--augment-ratios", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--fm-seed", type=int, default=42,
                    help="Which Flow Matching checkpoint seed to use for generation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    git_sha = os.popen("git rev-parse HEAD").read().strip()
    print(f"Device: {device}, Git SHA: {git_sha}")

    all_results = []

    for region in args.regions:
        print(f"\n{'='*70}\nRegion: {region}\n{'='*70}")
        data = load_region(region)
        rtr_h, rtr_t = data["train_hist_res"], data["train_tgt_res"]
        rva_h, rva_t = data["val_hist_res"], data["val_tgt_res"]
        rte_h, rte_t = data["test_hist_res"], data["test_tgt_res"]
        train_q95 = float(np.percentile(np.abs(rtr_t), 95))

        print(f"  train={rtr_h.shape}, val={rva_h.shape}, test={rte_h.shape}")
        print(f"  train residual: mean={rtr_t.mean():.4f}, std={rtr_t.std():.4f}, q95={train_q95:.4f}")

        # Load Flow Matching checkpoint for this region
        fm_ckpt = CKPT_DIR / f"flow_matching_{region}_seed{args.fm_seed}.pt"
        if not fm_ckpt.exists():
            print(f"  WARNING: {fm_ckpt} not found — skipping A3 for {region}")
            synthetic_tgt = None
        else:
            print(f"  Loading Flow Matching checkpoint: {fm_ckpt}")
            synthetic_tgt = generate_synthetic_residuals(
                fm_ckpt, torch.from_numpy(rtr_h), device, n_gen_per_real=1
            )
            print(f"  Synthetic targets: {synthetic_tgt.shape}, mean={synthetic_tgt.mean():.4f}, std={synthetic_tgt.std():.4f}")

        for seed in args.seeds:
            # --- A0: Real only ---
            for model_name, model_cls in [("gru", GRUForecaster), ("tcn", TCNForecaster)]:
                torch.manual_seed(seed)
                model = model_cls(context=96, horizon=72)
                train_forecaster(model, rtr_h, rtr_t, rva_h, rva_t,
                                 args.epochs, args.batch_size, args.lr, device, seed)
                metrics = evaluate_forecaster(model, rte_h, rte_t, device)
                r = {"region": region, "group": "A0_real_only",
                     "model": model_name, "seed": seed,
                     **metrics, "git_sha": git_sha, "epochs": args.epochs}
                all_results.append(r)
                print(f"  A0 {model_name} seed={seed}: MAE={metrics['mae']:.4f} "
                      f"tail_q95={metrics['tail_mae_q95']:.4f} tail_q99={metrics['tail_mae_q99']:.4f}")

            # --- A1: Extreme oversampling (re-weight + duplicate) ---
            weights = extreme_balance_weights(rtr_t, train_q95, extreme_weight=5.0)
            is_extreme = np.any(np.abs(rtr_t) >= train_q95, axis=1)
            dup_factor = int(weights.max())
            # Simple oversampling: repeat extreme windows dup_factor times
            extra_idx = np.where(is_extreme)[0]
            extra_idx = np.repeat(extra_idx, dup_factor - 1)
            rtr_h_a1 = np.concatenate([rtr_h, rtr_h[extra_idx]], axis=0)
            rtr_t_a1 = np.concatenate([rtr_t, rtr_t[extra_idx]], axis=0)
            print(f"  A1 oversampling: {len(extra_idx)} extra extreme windows → "
                  f"{rtr_h_a1.shape[0]} total ({is_extreme.sum()} orig extreme)")

            for model_name, model_cls in [("gru", GRUForecaster), ("tcn", TCNForecaster)]:
                torch.manual_seed(seed)
                model = model_cls(context=96, horizon=72)
                train_forecaster(model, rtr_h_a1, rtr_t_a1, rva_h, rva_t,
                                 args.epochs, args.batch_size, args.lr, device, seed)
                metrics = evaluate_forecaster(model, rte_h, rte_t, device)
                r = {"region": region, "group": "A1_oversampling",
                     "model": model_name, "seed": seed,
                     **metrics, "git_sha": git_sha, "epochs": args.epochs}
                all_results.append(r)

            # --- A2: Physics residual noise augment ---
            rtr_h_a2, _ = residual_noise_augment(rtr_h, rtr_t,
                                                 noise_std_scale=0.3, seed=seed)
            rtr_h_a2 = np.concatenate([rtr_h, rtr_h_a2], axis=0)
            rtr_t_a2 = np.concatenate([rtr_t, rtr_t], axis=0)
            print(f"  A2 physics augment: 2× training set ({rtr_h_a2.shape[0]} total)")

            for model_name, model_cls in [("gru", GRUForecaster), ("tcn", TCNForecaster)]:
                torch.manual_seed(seed)
                model = model_cls(context=96, horizon=72)
                train_forecaster(model, rtr_h_a2, rtr_t_a2, rva_h, rva_t,
                                 args.epochs, args.batch_size, args.lr, device, seed)
                metrics = evaluate_forecaster(model, rte_h, rte_t, device)
                r = {"region": region, "group": "A2_physics_augment",
                     "model": model_name, "seed": seed,
                     **metrics, "git_sha": git_sha, "epochs": args.epochs}
                all_results.append(r)

            # --- A3: Flow Matching synthetic augmentation at different ratios ---
            if synthetic_tgt is not None:
                for ratio in args.augment_ratios:
                    n_synth = int(len(rtr_t) * ratio)
                    # Randomly pick synthetic targets
                    synth_idx = np.random.default_rng(seed).choice(
                        len(synthetic_tgt), size=n_synth, replace=False
                    )
                    # Synthetic training pair: real history + synthetic target
                    rtr_t_a3 = np.concatenate([rtr_t, synthetic_tgt[synth_idx]], axis=0)
                    rtr_h_a3 = np.concatenate([rtr_h, rtr_h[:n_synth]], axis=0)

                    for model_name, model_cls in [("gru", GRUForecaster), ("tcn", TCNForecaster)]:
                        torch.manual_seed(seed)
                        model = model_cls(context=96, horizon=72)
                        train_forecaster(model, rtr_h_a3, rtr_t_a3, rva_h, rva_t,
                                         args.epochs, args.batch_size, args.lr, device, seed)
                        metrics = evaluate_forecaster(model, rte_h, rte_t, device)
                        r = {"region": region,
                             "group": f"A3_fm_ratio_{ratio}",
                             "model": model_name, "seed": seed,
                             "augment_ratio": ratio,
                             "n_synthetic": n_synth, "n_real": len(rtr_t),
                             **metrics, "git_sha": git_sha, "epochs": args.epochs,
                             "fm_seed": args.fm_seed}
                        all_results.append(r)
                        print(f"  A3 ratio={ratio} {model_name} seed={seed}: "
                              f"MAE={metrics['mae']:.4f} tail_q95={metrics['tail_mae_q95']:.4f}")
            else:
                print(f"  A3 skipped (no FM checkpoint for {region})")

    # Save results
    out_path = RESULTS_DIR / "a3_flowmatching_augment_96_72.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} results → {out_path}")

    # Summary table
    print(f"\n{'Group':<20} {'Region':<22} {'Model':<6} {'Seed':>5} {'MAE':>7} "
          f"{'TailQ95':>8} {'TailQ99':>8}")
    for r in all_results:
        print(f"{r['group']:<20} {r['region']:<22} {r['model']:<6} {r['seed']:>5} "
              f"{r['mae']:>7.4f} {r['tail_mae_q95']:>8.4f} {r['tail_mae_q99']:>8.4f}")


if __name__ == "__main__":
    main()
