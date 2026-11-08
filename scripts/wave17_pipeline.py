"""Wave 1.7 — Unified 96h→72h forecasting pipeline with prediction save + total-water eval.

All models use identical splits, features, forecast horizon, and evaluation.
Saves per-window predictions via total_water_eval.save_window_predictions()
for every model×region×seed combination.

Protocol F1: equal optimizer steps across all groups (8000 default).

Groups:
  persistence     — last history value carried forward
  seasonal_24h    — 24h period repetition
  gru_standalone  — GRU trained on real residuals only (seed 42)
  tcn_standalone  — TCN trained on real residuals only (seed 42)
  A0_real_only    — identical to gru_standalone, named for comparison
  A1_oversampling — extreme-window duplicate oversampling (q95 positive)
  A2_history_noise — Gaussian history noise augmentation
  A3_fm_ratio_{r} — Flow Matching synthetic augmentation at ratio r

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/wave17_pipeline.py \
        --regions ny_bight thames_north_sea \
        --seeds 42 99 \
        --augment-ratios 0.1 0.25 0.5 \
        --skip-fm-check \
        --git-sha c6a703a
"""
from __future__ import annotations

import argparse
import json
import math
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
from tidetail.total_water_eval import (
    DATUM,
    evaluate_total_water,
    save_window_predictions,
    train_only_water_thresholds,
)

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR = ROOT / "data" / "results"
PRED_DIR = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = [
    "ny_bight", "thames_north_sea", "us_gulf_coast",
    "south_atlantic_charleston", "southeast_australia",
]


# ---------------------------------------------------------------------------
# Load region with residuals + FES2022 tide (for total-water reconstruction)
# ---------------------------------------------------------------------------
def load_region(region: str, context: int = 96, horizon: int = 72) -> dict:
    """Load region — VERIFIED by Wave 1.6 P0 contract (MAE = 0 residual+ = total)."""
    base = DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide_df = pd.read_csv(base / "tide_timeline_v1.csv")
    meta = json.loads((base / "train_only_metadata_v1.json").read_text())
    wi = np.load(base / "window_index_v1.npz")

    merged = target[["timestamp_utc", "water_level_m", "observed_total_water_level_m"]].merge(
        tide_df[["timestamp_utc", "tide_level_m"]], on="timestamp_utc", how="inner"
    )
    residual = merged["water_level_m"].to_numpy(dtype=np.float32)      # already de-tided
    tide_fes = merged["tide_level_m"].to_numpy(dtype=np.float32)        # FES2022
    total_water = merged["observed_total_water_level_m"].to_numpy(dtype=np.float32)

    def _win(starts, series):
        starts = starts.astype(np.int64)
        N = len(starts)
        h = np.zeros((N, context), dtype=np.float32)
        t = np.zeros((N, horizon), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = series[s : s + context]
            t[i] = series[s + context : s + context + horizon]
        return h, t

    rtr_h, rtr_t = _win(wi["train_start_indices"], residual)
    rva_h, rva_t = _win(wi["validation_start_indices"], residual)
    rte_h, rte_t = _win(wi["test_start_indices"], residual)
    rte_tide = _win(wi["test_start_indices"], tide_fes)[1]        # future FES2022 tide
    rte_total = _win(wi["test_start_indices"], total_water)[1]    # future observed total water
    rtr_total = _win(wi["train_start_indices"], total_water)[1]   # train total water for thresholds

    # Train-only thresholds
    abs_res_tgt = np.abs(rtr_t)
    train_q95_res = float(np.percentile(abs_res_tgt, 95))
    train_q99_res = float(np.percentile(abs_res_tgt, 99))
    pos_res = rtr_t[rtr_t > 0]
    train_q95_res_pos = float(np.percentile(pos_res, 95)) if len(pos_res) > 0 else train_q95_res
    train_q99_res_pos = float(np.percentile(pos_res, 99)) if len(pos_res) > 0 else train_q99_res

    # Train-only TOTAL-WATER thresholds (for total_water_eval tail masking)
    tw_thresh = train_only_water_thresholds(rtr_total)

    return {
        "region": region,
        # Residual windows (surge only)
        "train_h": rtr_h, "train_t": rtr_t,
        "val_h": rva_h, "val_t": rva_t,
        "test_h": rte_h, "test_t": rte_t,
        # Tide + total water (for evaluation)
        "test_future_tide": rte_tide,
        "test_obs_total_water": rte_total,
        "train_total_water": rtr_total,
        # Thresholds
        "train_q95_res": train_q95_res,
        "train_q99_res": train_q99_res,
        "train_q95_res_pos": train_q95_res_pos,
        "train_q99_res_pos": train_q99_res_pos,
        "tw_thresholds": tw_thresh,
        # Scaler (train-only)
        "scaler_mean": float(meta["scalers"]["water_level_m"]["mean"]),
        "scaler_std": float(meta["scalers"]["water_level_m"]["std"]),
    }


# ---------------------------------------------------------------------------
# Simple rule baselines
# ---------------------------------------------------------------------------
def _persist(h, ho):
    return np.full(ho, float(h[-1]))

def _seasonal(h, ho, period=24):
    base = h[-period:] if len(h) >= period else np.tile(h, period // len(h) + 1)[-period:]
    return np.array([base[k % period] for k in range(ho)])

def batch_predict(histories, fn, horizon):
    out = np.zeros((len(histories), horizon), dtype=np.float32)
    for i, h in enumerate(histories):
        out[i] = fn(h, horizon)
    return out


# ---------------------------------------------------------------------------
# Models — GRU + TCN (same as run_a3_flowmatching_augment.py)
# ---------------------------------------------------------------------------
class GRUForecaster(torch.nn.Module):
    def __init__(self, context=96, horizon=72, hidden=64, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(1, hidden, num_layers=num_layers,
                                batch_first=True, bidirectional=False)
        self.proj = torch.nn.Linear(hidden, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


class TCNForecaster(torch.nn.Module):
    def __init__(self, context=96, horizon=72, width=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
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


# ---------------------------------------------------------------------------
# Training — Protocol F1 (equal optimizer steps)
# ---------------------------------------------------------------------------
def train_forecaster(model, train_h, train_t, val_h, val_t,
                     target_steps, batch_size, lr, device, seed) -> dict:
    """Train forecaster with EXACTLY target_steps optimizer updates (Protocol F1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    H = torch.from_numpy(np.asarray(train_h, dtype=np.float32)).to(device)
    T = torch.from_numpy(np.asarray(train_t, dtype=np.float32)).to(device)

    n = H.shape[0]
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    steps_per_epoch = n_batches
    total_epochs = math.ceil(target_steps / steps_per_epoch)
    actual_steps = 0

    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            pred = model(H[idx])
            loss = F.mse_loss(pred, T[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            actual_steps += 1
            if actual_steps >= target_steps:
                break
        if actual_steps >= target_steps:
            break

    model.eval()
    with torch.no_grad():
        test_hist = torch.from_numpy(np.asarray(val_h, dtype=np.float32)).to(device)
        val_mae = (model(test_hist) - torch.from_numpy(np.asarray(val_t, dtype=np.float32)).to(device)).abs().mean().item()

    return {"optimizer_steps": actual_steps, "val_mae": val_mae}


def predict_on_test(model, test_h, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(np.asarray(test_h, dtype=np.float32)).to(device)
        return model(X).cpu().numpy()


# ---------------------------------------------------------------------------
# FM synthetic target generation
# ---------------------------------------------------------------------------
def generate_synthetic_residuals(ckpt_path: Path, hist_res: np.ndarray,
                                  device: str, n_gen_per_real: int = 1) -> np.ndarray | None:
    """Generate synthetic residual targets — row i conditioned on row i of history."""
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    velocity_net = VelocityTCN(context=96, horizon=72, width=32).to(device)
    velocity_net.load_state_dict(ckpt["state_dict"])
    fm = FlowMatching(context=96, horizon=72, euler_steps=64, device=device)

    h = torch.from_numpy(np.asarray(hist_res, dtype=np.float32)).to(device)
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


# ---------------------------------------------------------------------------
# Single model training + prediction save + total-water eval
# ---------------------------------------------------------------------------
def run_single_experiment(
    region_data: dict,
    group_name: str,
    model_name: str,          # "gru" or "tcn"
    model_cls,
    train_h: np.ndarray,
    train_t: np.ndarray,
    val_h: np.ndarray,
    val_t: np.ndarray,
    test_h: np.ndarray,
    test_t: np.ndarray,
    future_tide: np.ndarray,
    tw_thresholds: dict,
    region: str,
    seed: int,
    device: str,
    target_steps: int = 8000,
    batch_size: int = 256,
    lr: float = 1e-3,
    git_sha: str = "",
    pred_out_dir: Path = PRED_DIR,
) -> dict:
    """Train ONE model+group, save predictions, run total_water_eval. Return metrics dict."""
    t0 = time.time()
    model = model_cls(context=96, horizon=72)
    history = train_forecaster(model, train_h, train_t, val_h, val_t,
                               target_steps, batch_size, lr, device, seed)
    steps_actual = history["optimizer_steps"]

    # Predict on test set
    pred_residual = predict_on_test(model, test_h, device)

    # Save per-window predictions
    pred_dir = pred_out_dir / "v2"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{region}_{group_name}_{model_name}_seed{seed}.npz"
    meta = {
        "region": region,
        "group": group_name,
        "model": model_name,
        "seed": seed,
        "git_sha": git_sha,
        "target_steps": target_steps,
        "actual_steps": steps_actual,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer_steps": steps_actual,
        "device": device,
        "prediction_run_sha": git_sha,
        "data_pipeline_version": "wave17_v2",
        "total_water_datum": DATUM,
    }
    save_window_predictions(
        pred_path,
        pred_residual=pred_residual,
        obs_residual=test_t,
        future_tide=future_tide,
        meta=meta,
    )

    # Total-water evaluation
    tw_metrics = evaluate_total_water(
        pred_residual=pred_residual,
        obs_residual=test_t,
        future_tide=future_tide,
        train_water_q95_pos=tw_thresholds.get("water_q95_pos"),
        train_water_q99_pos=tw_thresholds.get("water_q99_pos"),
        train_water_q95_abs=tw_thresholds.get("water_q95_abs"),
        train_water_q99_abs=tw_thresholds.get("water_q99_abs"),
    )

    runtime = time.time() - t0

    result = {
        "region": region,
        "group": group_name,
        "model": model_name,
        "seed": seed,
        "prediction_path": str(pred_path),
        "prediction_checksum": "",  # filled later
        # Training metadata
        "target_steps": target_steps,
        "actual_steps": steps_actual,
        "runtime_s": runtime,
        "git_sha": git_sha,
        # Residual-scale metrics
        "res_mae": float(np.mean(np.abs(pred_residual - test_t))),
        "res_rmse": float(np.sqrt(np.mean((pred_residual - test_t) ** 2))),
        # Total-water metrics (primary)
        "tw_mae": tw_metrics["tw_mae"],
        "tw_rmse": tw_metrics["tw_rmse"],
        "tw_bias": tw_metrics["tw_bias"],
        "tw_corr": tw_metrics["tw_corr"],
        # Tail metrics
        **_extract_tail_metrics(tw_metrics),
        # Peak metrics
        **_extract_peak_metrics(tw_metrics),
        "n_windows": tw_metrics["n_windows"],
        "horizon_steps": tw_metrics["horizon_steps"],
        "datum": DATUM,
    }

    # Compute checkpointed prediction file
    import hashlib
    h = hashlib.sha256()
    with open(pred_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    result["prediction_checksum"] = h.hexdigest()

    return result


def _extract_tail_metrics(tw_metrics: dict) -> dict:
    """Flatten nested tail metrics dict into top-level keys."""
    out = {}
    tail = tw_metrics.get("total_water_tail", {})
    for key, val in tail.items():
        if isinstance(val, dict):
            for sub_key in ["tail_mae", "tail_rmse", "n", "defined", "threshold"]:
                if sub_key in val:
                    out[f"{key}_{sub_key}"] = val[sub_key]
    return out


def _extract_peak_metrics(tw_metrics: dict) -> dict:
    peaks = tw_metrics.get("total_water_peaks", {})
    return {f"peak_{k}": v for k, v in peaks.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Wave 1.7 unified pipeline")
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--augment-ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--fm-seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--target-steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--skip-fm", action="store_true", help="Skip A3 groups entirely")
    ap.add_argument("--skip-persistence", action="store_true", help="Skip persistence/seasonal rule baselines")
    args = ap.parse_args()

    git_sha = args.git_sha or __import__("os").popen("git rev-parse HEAD").read().strip()
    print(f"Device: {args.device}, Git SHA: {git_sha}")
    print(f"Regions: {args.regions}, Seeds: {args.seeds}")
    print(f"Protocol F1: target_steps={args.target_steps}")

    device = args.device
    all_results = []
    total_t0 = time.time()

    # Iterate over regions
    for region in args.regions:
        print(f"\n{'='*80}\nREGION: {region}\n{'='*80}")
        rd = load_region(region)
        tw_t = rd["tw_thresholds"]
        print(f"  train={rd['train_h'].shape}, test={rd['test_h'].shape}")
        print(f"  test_future_tide shape: {rd['test_future_tide'].shape}")
        print(f"  total-water thresholds: q95_pos={tw_t.get('water_q95_pos', float('nan')):.3f}, "
              f"q99_pos={tw_t.get('water_q99_pos', float('nan')):.3f}")

        # ---- Persistence + Seasonal (rule baselines, CPU-instant) ----
        if not args.skip_persistence:
            for rule_name, rule_fn in [
                ("persistence", _persist),
                ("seasonal_24h", _seasonal),
            ]:
                print(f"  Running {rule_name}...")
                pred_res = batch_predict(rd["test_h"], rule_fn, horizon=72)
                pred_path = PRED_DIR / "v2" / f"{region}_{rule_name}.npz"
                save_window_predictions(
                    pred_path,
                    pred_residual=pred_res,
                    obs_residual=rd["test_t"],
                    future_tide=rd["test_future_tide"],
                    meta={
                        "region": region, "group": rule_name, "model": rule_name,
                        "git_sha": git_sha, "datum": DATUM,
                        "prediction_run_sha": git_sha,
                    },
                )
                tw_m = evaluate_total_water(
                    pred_residual=pred_res,
                    obs_residual=rd["test_t"],
                    future_tide=rd["test_future_tide"],
                    train_water_q95_pos=tw_t.get("water_q95_pos"),
                    train_water_q99_pos=tw_t.get("water_q99_pos"),
                    train_water_q95_abs=tw_t.get("water_q95_abs"),
                    train_water_q99_abs=tw_t.get("water_q99_abs"),
                )
                import hashlib
                h = hashlib.sha256()
                with open(pred_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)

                r = {
                    "region": region, "group": rule_name, "model": rule_name,
                    "seed": 0, "prediction_path": str(pred_path),
                    "prediction_checksum": h.hexdigest(),
                    "runtime_s": 0.01, "git_sha": git_sha,
                    "res_mae": float(np.mean(np.abs(pred_res - rd["test_t"]))),
                    "res_rmse": float(np.sqrt(np.mean((pred_res - rd["test_t"]) ** 2))),
                    "tw_mae": tw_m["tw_mae"], "tw_rmse": tw_m["tw_rmse"],
                    "tw_bias": tw_m["tw_bias"], "tw_corr": tw_m["tw_corr"],
                    "n_windows": tw_m["n_windows"], "horizon_steps": tw_m["horizon_steps"],
                    "datum": DATUM,
                }
                r.update(_extract_tail_metrics(tw_m))
                r.update(_extract_peak_metrics(tw_m))
                all_results.append(r)
                print(f"    {rule_name}: tw_mae={tw_m['tw_mae']:.4f} tail95={_safe_get(r, 'tw_tail_mae_q95_pos_tail_mae'):.4f}")

        # ---- Prepare training groups ----
        train_h_real = rd["train_h"]
        train_t_real = rd["train_t"]

        groups_config = [
            # A0: real only
            {"group_name": "A0_real_only",
             "train_h": train_h_real.copy(),
             "train_t": train_t_real.copy()},
            # A1: extreme oversampling (positive residual >= train_q95_res_pos)
        ]
        is_extreme = np.any(train_t_real >= rd["train_q95_res_pos"], axis=1)
        extreme_idx = np.where(is_extreme)[0]
        dup_factor = 5
        extra_idx = np.repeat(extreme_idx, dup_factor - 1)
        groups_config.append({
            "group_name": "A1_oversampling",
            "train_h": np.concatenate([train_h_real, train_h_real[extra_idx]], axis=0),
            "train_t": np.concatenate([train_t_real, train_t_real[extra_idx]], axis=0),
        })
        # A2: history noise
        rtr_h_noisy, _ = residual_noise_augment(train_h_real, train_t_real,
                                                 noise_std_scale=0.3, seed=0)
        groups_config.append({
            "group_name": "A2_history_noise",
            "train_h": np.concatenate([train_h_real, rtr_h_noisy], axis=0),
            "train_t": np.concatenate([train_t_real, train_t_real], axis=0),
        })

        # A3: Flow Matching synthetic (only if FM checkpoints exist)
        if not args.skip_fm:
            for fm_seed in args.fm_seeds:
                fm_ckpt = CKPT_DIR / f"flow_matching_{region}_seed{fm_seed}.pt"
                if not fm_ckpt.exists():
                    print(f"  FM checkpoint missing: {fm_ckpt} — A3 will be skipped")
                    continue
                print(f"  Generating synthetic residuals from {fm_ckpt}...")
                synth_tgt = generate_synthetic_residuals(fm_ckpt, train_h_real, device)
                if synth_tgt is None:
                    print(f"  WARNING: synthetic generation failed for {region} seed{fm_seed}")
                    continue
                for ratio in args.augment_ratios:
                    n_synth = int(len(train_t_real) * ratio)
                    rng = np.random.default_rng(100 + fm_seed * 10 + int(ratio * 100))
                    synth_idx = rng.choice(len(synth_tgt), size=n_synth, replace=False)
                    groups_config.append({
                        "group_name": f"A3_fm_s{fm_seed}_ratio_{ratio}",
                        "train_h": np.concatenate([train_h_real, train_h_real[synth_idx]], axis=0),
                        "train_t": np.concatenate([train_t_real, synth_tgt[synth_idx]], axis=0),
                        "fm_seed": fm_seed,
                        "augment_ratio": ratio,
                    })

        # ---- Train all groups × GRU/TCN × seeds ----
        for gcfg in groups_config:
            gname = gcfg["group_name"]
            print(f"\n  Group: {gname}")
            for model_name, model_cls in [("gru", GRUForecaster), ("tcn", TCNForecaster)]:
                for seed in args.seeds:
                    try:
                        r = run_single_experiment(
                            region_data=rd,
                            group_name=gname,
                            model_name=model_name,
                            model_cls=model_cls,
                            train_h=gcfg["train_h"],
                            train_t=gcfg["train_t"],
                            val_h=rd["val_h"],
                            val_t=rd["val_t"],
                            test_h=rd["test_h"],
                            test_t=rd["test_t"],
                            future_tide=rd["test_future_tide"],
                            tw_thresholds=tw_t,
                            region=region,
                            seed=seed,
                            device=device,
                            target_steps=args.target_steps,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            git_sha=git_sha,
                        )
                        all_results.append(r)
                        tw95 = r.get("tw_tail_mae_q95_pos_tail_mae", float("nan"))
                        print(f"    {model_name:<4} seed={seed}: tw_mae={r['tw_mae']:.4f} "
                              f"tw_tail95={tw95:.4f} steps={r['actual_steps']}")
                    except RuntimeError as exc:
                        print(f"    FAILED {gname} {model_name} seed={seed}: {exc}")

        # Also run standalone GRU/TCN on seed=42 (for direct baseline comparison)
        # These are A0_real_only already, so skip duplicating here

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = RESULTS_DIR / "wave17_baselines_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*80}")
    print(f"SAVED {len(all_results)} results → {out_path}")
    print(f"Total runtime: {time.time() - total_t0:.1f}s")

    # Print summary table
    print(f"\n{'Region':<22} {'Group':<22} {'Model':<5} {'Seed':>5} {'tw_mae':>8} {'tw_tail95':>10} {'tw_peak_mag':>12} {'peak_timing':>12}")
    print("-" * 105)
    for r in all_results:
        tw95 = r.get("tw_tail_mae_q95_pos_tail_mae", float("nan"))
        pkm = r.get("peak_tw_peak_mag_mae", float("nan"))
        pkt = r.get("peak_tw_peak_timing_mae_hours", float("nan"))
        print(f"{r['region']:<22} {r['group']:<22} {r['model']:<5} {r['seed']:>5} "
              f"{r['tw_mae']:>8.4f} {tw95:>10.4f} {pkm:>12.4f} {pkt:>12.1f}")


def _safe_get(d, key, default=float("nan")):
    """Get from dict, return default if missing or nan."""
    v = d.get(key, default)
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    return v


if __name__ == "__main__":
    main()
