"""A3 — Flow Matching synthetic augmentation + GRU/TCN fair comparison.

W1-P1 FIXED VERSION (addresses 6 bugs from 2026-11-01 audit):
  W1-BUG-01: Synthetic conditioning mismatch — FIXED (pair rtr_h[synth_idx] not rtr_h[:n_synth])
  W1-BUG-02: Tail-MAE — FIXED (compute on observed extreme targets, not error percentiles)
  W1-BUG-03: Peak metric — FIXED (per-window, not global max)
  W1-BUG-04: Training-budget fairness — IMPLEMENTED Protocol F1 (equal optimizer steps)
  W1-BUG-05: "Physics" label → renamed to "history noise" augmentation
  W1-BUG-06: Extreme criterion → positive-only + absolute variants

Protocol (explicit):
  F1 — Equal optimizer updates: same total optimizer steps across all groups.
       Matched by varying epochs per group to hit target_steps.
       Real-data exposure and total optimization cost reported explicitly.

Groups:
  A0_real_only          — real training, no augmentation
  A1_oversampling       — extreme-window duplicate oversampling (q95 positive residual criterion)
  A2_history_noise      — Gaussian noise on history input (NOT physics simulation)
  A3_fm_ratio_{ratio}  — Flow Matching synthetic residual targets at given ratio

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/run_a3_flowmatching_augment.py \
        --regions ny_bight \
        --augment-ratios 0.1 \
        --seeds 42 \
        --protocol F1 \
        --target-steps 8000
"""
from __future__ import annotations

import argparse
import json
import math
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

from tidetail.augment import residual_noise_augment
from tidetail.models import FlowMatching, VelocityTCN

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_region(region: str, context: int = 96, horizon: int = 72):
    """Load region data — returns residual-only windows + train q95 thresholds."""
    base = DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    json.loads((base / "train_only_metadata_v1.json").read_text())
    wi = np.load(base / "window_index_v1.npz")

    merged = target[["timestamp_utc", "water_level_m"]].merge(
        tide[["timestamp_utc", "tide_level_m"]], on="timestamp_utc", how="inner"
    )
    water_level = merged["water_level_m"].to_numpy(dtype=np.float32)
    tide_level = merged["tide_level_m"].to_numpy(dtype=np.float32)
    residual = water_level - tide_level

    def _make_windows(starts, series):
        starts = starts.astype(np.int64)
        N = len(starts)
        h = np.zeros((N, context), dtype=np.float32)
        t = np.zeros((N, horizon), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = series[s : s + context]
            t[i] = series[s + context : s + context + horizon]
        return h, t

    rtr_h, rtr_t = _make_windows(wi["train_start_indices"], residual)
    rva_h, rva_t = _make_windows(wi["validation_start_indices"], residual)
    rte_h, rte_t = _make_windows(wi["test_start_indices"], residual)

    # Train-only thresholds
    abs_residual = np.abs(rtr_t)
    train_q95 = float(np.percentile(abs_residual, 95))
    train_q99 = float(np.percentile(abs_residual, 99))
    pos_residual = rtr_t[rtr_t > 0]
    train_q95_pos = float(np.percentile(pos_residual, 95)) if len(pos_residual) > 0 else train_q95
    train_q99_pos = float(np.percentile(pos_residual, 99)) if len(pos_residual) > 0 else train_q99

    return {
        "train_hist_res": rtr_h, "train_tgt_res": rtr_t,
        "val_hist_res": rva_h, "val_tgt_res": rva_t,
        "test_hist_res": rte_h, "test_tgt_res": rte_t,
        "train_q95": train_q95, "train_q99": train_q99,
        "train_q95_pos": train_q95_pos, "train_q99_pos": train_q99_pos,
    }


def generate_synthetic_residuals(ckpt_path: Path, hist_res: torch.Tensor,
                                  device: str, n_gen_per_real: int = 1) -> np.ndarray:
    """Generate synthetic residual targets — row i of output conditioned on row i of input history."""
    ckpt = torch.load(ckpt_path, map_location=device)
    velocity_net = VelocityTCN(context=96, horizon=72, width=32).to(device)
    velocity_net.load_state_dict(ckpt["state_dict"])
    fm = FlowMatching(context=96, horizon=72, euler_steps=64, device=device)

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
# Models (GRU/TCN forecasters)
# ==========================================================================

class TCNForecaster(torch.nn.Module):
    def __init__(self, context=96, horizon=72, width=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.input_proj = torch.nn.Conv1d(1, width, 3, padding=1)
        self.blocks = torch.nn.ModuleList([_TCNBlock(width, d) for d in dilations])
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(width, horizon)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


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
        return (x + self.net(x)) / math.sqrt(2)


class GRUForecaster(torch.nn.Module):
    def __init__(self, context=96, horizon=72, hidden=64, num_layers=2):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.gru = torch.nn.GRU(1, hidden, num_layers=num_layers,
                                batch_first=True, bidirectional=False)
        self.proj = torch.nn.Linear(hidden, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


# ==========================================================================
# Training (returns optimizer_steps for fairness tracking)
# ==========================================================================

def train_forecaster(model, train_hist, train_tgt, val_hist, val_tgt,
                     target_steps: int, batch_size: int, lr: float,
                     device: str, seed: int) -> dict:
    """Train forecaster with EXACTLY target_steps optimizer updates (Protocol F1).

    Returns dict including actual optimizer_steps taken.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    H = torch.from_numpy(np.asarray(train_hist, dtype=np.float32)).to(device)
    T = torch.from_numpy(np.asarray(train_tgt, dtype=np.float32)).to(device)
    vH = torch.from_numpy(np.asarray(val_hist, dtype=np.float32)).to(device)
    vT = torch.from_numpy(np.asarray(val_tgt, dtype=np.float32)).to(device)

    n = H.shape[0]
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    steps_per_epoch = n_batches
    total_epochs = math.ceil(target_steps / steps_per_epoch)
    history = {"epoch": [], "train_loss": [], "val_mae": [], "optimizer_steps": 0}
    actual_steps = 0

    for epoch in range(total_epochs):
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
            actual_steps += 1
            if actual_steps >= target_steps:
                break
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            vpred = model(vH)
            val_mae = (vpred - vT).abs().mean().item()

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_mae"].append(val_mae)
        history["optimizer_steps"] = actual_steps

        if actual_steps >= target_steps:
            break

    return history


def evaluate_forecaster(model, test_hist: np.ndarray, test_tgt: np.ndarray,
                         train_q95: float, train_q99: float,
                         device: str) -> dict:
    """CORRECTED evaluation with target-based Tail-MAE and per-window peak.

    Args:
        train_q95: q95 threshold of TRAIN-ONLY absolute residual targets.
        train_q99: q99 threshold of TRAIN-ONLY absolute residual targets.
    """
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(np.asarray(test_hist, dtype=np.float32)).to(device))
        pred_np = pred.cpu().numpy()

    err = pred_np - test_tgt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # W1-BUG-03 FIX: Per-window peak, not global max
    peak_pred_per_window = np.abs(pred_np).max(axis=1)
    peak_tgt_per_window = np.abs(test_tgt).max(axis=1)
    peak_amp_err = float(np.mean(np.abs(peak_pred_per_window - peak_tgt_per_window)))
    peak_idx_pred = np.abs(pred_np).argmax(axis=1)
    peak_idx_real = np.abs(test_tgt).argmax(axis=1)
    peak_timing_mae = float(np.mean(np.abs(peak_idx_pred - peak_idx_real)))

    # W1-BUG-02 FIX: Tail-MAE at extreme OBSERVED targets, not worst errors
    abs_err = np.abs(err)
    abs_tgt = np.abs(test_tgt)
    mask_q95 = abs_tgt >= train_q95
    mask_q99 = abs_tgt >= train_q99
    tail_mae_q95 = float(abs_err[mask_q95].mean()) if mask_q95.any() else mae
    tail_mae_q99 = float(abs_err[mask_q99].mean()) if mask_q99.any() else mae

    return {
        "mae": mae,
        "rmse": rmse,
        "peak_amp_err": peak_amp_err,
        "peak_timing_mae": peak_timing_mae,
        "tail_mae_q95": tail_mae_q95,
        "tail_mae_q99": tail_mae_q99,
        "tail_mae_q95_n": int(mask_q95.sum()),
        "tail_mae_q99_n": int(mask_q99.sum()),
        "train_q95_threshold": train_q95,
        "train_q99_threshold": train_q99,
    }


def compute_epochs_for_steps(train_size: int, target_steps: int,
                              batch_size: int) -> int:
    """How many epochs to get exactly (or slightly exceed) target_steps."""
    steps_per_epoch = max(1, (train_size + batch_size - 1) // batch_size)
    return math.ceil(target_steps / steps_per_epoch)


# ==========================================================================
# Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["ny_bight", "thames_north_sea", "us_gulf_coast",
                             "south_atlantic_charleston", "southeast_australia"])
    ap.add_argument("--augment-ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--protocol", choices=["F1"], default="F1",
                    help="Comparison protocol. F1 = equal optimizer steps.")
    ap.add_argument("--target-steps", type=int, default=8000,
                    help="Target optimizer steps per group (Protocol F1).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--fm-seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    git_sha = os.popen("git rev-parse HEAD").read().strip()
    print(f"Device: {device}, Git SHA: {git_sha}")
    print(f"Protocol: {args.protocol}, Target steps: {args.target_steps}")

    all_results = []
    total_t0 = time.time()

    for region in args.regions:
        print(f"\n{'='*70}\nRegion: {region}\n{'='*70}")
        data = load_region(region)
        rtr_h, rtr_t = data["train_hist_res"], data["train_tgt_res"]
        rva_h, rva_t = data["val_hist_res"], data["val_tgt_res"]
        rte_h, rte_t = data["test_hist_res"], data["test_tgt_res"]
        train_q95 = data["train_q95"]
        train_q99 = data["train_q99"]
        train_q95_pos = data["train_q95_pos"]

        print(f"  train={rtr_h.shape}, val={rva_h.shape}, test={rte_h.shape}")
        print(f"  train q95(abs)={train_q95:.4f}, q95(pos)={train_q95_pos:.4f}")

        # Load FM checkpoint
        fm_ckpt = CKPT_DIR / f"flow_matching_{region}_seed{args.fm_seed}.pt"
        if not fm_ckpt.exists():
            print(f"  WARNING: {fm_ckpt} not found — skipping A3 for {region}")
            synthetic_tgt = None
        else:
            print(f"  Loading Flow Matching checkpoint: {fm_ckpt}")
            synthetic_tgt = generate_synthetic_residuals(
                fm_ckpt, torch.from_numpy(rtr_h), device, n_gen_per_real=1
            )

        groups_config = []

        # --- A0: Real only ---
        groups_config.append({
            "group_name": "A0_real_only",
            "train_hist": rtr_h.copy(),
            "train_tgt": rtr_t.copy(),
            "train_size": len(rtr_h),
            "real_unique": len(rtr_h),
            "synthetic_count": 0,
            "dup_real_count": 0,
        })

        # --- A1: Extreme oversampling (positive-only criterion) ---
        is_extreme_pos = np.any(rtr_t >= train_q95_pos, axis=1)
        extreme_idx_pos = np.where(is_extreme_pos)[0]
        dup_factor = 5  # extreme_weight = 5
        extra_idx = np.repeat(extreme_idx_pos, dup_factor - 1)
        a1_h = np.concatenate([rtr_h, rtr_h[extra_idx]], axis=0)
        a1_t = np.concatenate([rtr_t, rtr_t[extra_idx]], axis=0)
        groups_config.append({
            "group_name": "A1_oversampling",
            "train_hist": a1_h,
            "train_tgt": a1_t,
            "train_size": len(a1_h),
            "real_unique": len(rtr_h),
            "synthetic_count": 0,
            "dup_real_count": len(extra_idx),
            "extreme_criterion": "positive_residual >= train_q95_pos",
        })

        # --- A2: History noise augmentation (NOT physics simulation) ---
        rtr_h_noisy, _ = residual_noise_augment(rtr_h, rtr_t,
                                                 noise_std_scale=0.3, seed=0)
        a2_h = np.concatenate([rtr_h, rtr_h_noisy], axis=0)
        a2_t = np.concatenate([rtr_t, rtr_t], axis=0)
        groups_config.append({
            "group_name": "A2_history_noise",
            "train_hist": a2_h,
            "train_tgt": a2_t,
            "train_size": len(a2_h),
            "real_unique": len(rtr_h),
            "synthetic_count": 0,
            "dup_real_count": len(rtr_h),
            "augment_type": "gaussian_history_noise (noise_std_scale=0.3)",
            "disclaimer": "NOT physics simulation; models instrument uncertainty",
        })

        # --- A3: Flow Matching synthetic (with W1-BUG-01 fix) ---
        if synthetic_tgt is not None:
            for ratio in args.augment_ratios:
                n_synth = int(len(rtr_t) * ratio)
                rng = np.random.default_rng(100 + int(ratio * 100))
                synth_idx = rng.choice(len(synthetic_tgt), size=n_synth, replace=False)

                # W1-BUG-01 FIX: pair synthetic_tgt[synth_idx] with rtr_h[synth_idx],
                # NOT rtr_h[:n_synth]. This preserves exact conditioning alignment.
                a3_h = np.concatenate([rtr_h, rtr_h[synth_idx]], axis=0)
                a3_t = np.concatenate([rtr_t, synthetic_tgt[synth_idx]], axis=0)

                groups_config.append({
                    "group_name": f"A3_fm_ratio_{ratio}",
                    "train_hist": a3_h,
                    "train_tgt": a3_t,
                    "train_size": len(a3_h),
                    "real_unique": len(rtr_h),
                    "synthetic_count": n_synth,
                    "dup_real_count": 0,
                    "augment_ratio": ratio,
                    "fm_checkpoint": str(fm_ckpt),
                    "fm_seed": args.fm_seed,
                    "conditioning_aligned": True,  # W1-BUG-01 FIX marker
                })

        # --- Train all groups under Protocol F1 (equal optimizer steps) ---
        for group_cfg in groups_config:
            gname = group_cfg["group_name"]
            g_h = group_cfg["train_hist"]
            g_t = group_cfg["train_tgt"]
            g_train_size = group_cfg["train_size"]
            steps_per_epoch = max(1, (g_train_size + args.batch_size - 1) // args.batch_size)
            math.ceil(args.target_steps / steps_per_epoch)

            for model_name, model_cls in [("gru", GRUForecaster), ("tcn", TCNForecaster)]:
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    model = model_cls(context=96, horizon=72)
                    history = train_forecaster(
                        model, g_h, g_t, rva_h, rva_t,
                        target_steps=args.target_steps,
                        batch_size=args.batch_size, lr=args.lr,
                        device=device, seed=seed,
                    )
                    steps_actual = history["optimizer_steps"]
                    metrics = evaluate_forecaster(
                        model, rte_h, rte_t, train_q95, train_q99, device,
                    )
                    (g_train_size + args.batch_size - 1) // args.batch_size
                    r = {
                        "region": region,
                        "group": gname,
                        "model": model_name,
                        "seed": seed,
                        # Fairness tracking (Protocol F1 metadata)
                        "protocol": args.protocol,
                        "target_steps": args.target_steps,
                        "actual_steps": steps_actual,
                        "steps_per_epoch": steps_per_epoch,
                        "epochs_actual": len(history["epoch"]),
                        "train_size": g_train_size,
                        "real_unique": group_cfg["real_unique"],
                        "synthetic_count": group_cfg["synthetic_count"],
                        "dup_real_count": group_cfg["dup_real_count"],
                        # Eval metrics
                        **metrics,
                        "git_sha": git_sha,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "runtime_s": time.time() - total_t0,
                        "device": device,
                    }
                    all_results.append(r)
                    print(f"  {gname:<22} {model_name:<5} seed={seed} "
                          f"steps={steps_actual}/{args.target_steps} "
                          f"MAE={metrics['mae']:.4f} "
                          f"tail95={metrics['tail_mae_q95']:.4f} "
                          f"tail95_n={metrics['tail_mae_q95_n']}")

    # Save results
    out_path = RESULTS_DIR / "a3_flowmatching_augment_96_72.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} results → {out_path}")
    print(f"Total runtime: {time.time() - total_t0:.1f}s")

    # Summary
    print(f"\n{'Group':<22} {'Region':<22} {'Model':<6} {'Steps':>7} {'MAE':>7} "
          f"{'Tail95':>8} {'Tail95n':>8} {'PeakErr':>8}")
    for r in all_results:
        print(f"{r['group']:<22} {r['region']:<22} {r['model']:<6} {r['actual_steps']:>7} "
              f"{r['mae']:>7.4f} {r['tail_mae_q95']:>8.4f} {r['tail_mae_q95_n']:>8} "
              f"{r['peak_amp_err']:>8.4f}")


if __name__ == "__main__":
    main()
