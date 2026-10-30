"""GPU training script for TideTail generative models (M2 Diffusion-TS, M4 Flow Matching).

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/train_generative_models.py \
        --model flow_matching --regions ny_bight thames_north_sea --epochs 10

Contracts:
    - 96h history → 72h horizon
    - Train-only scaler (water_level_m)
    - Per-sample residual z-score normalization
    - Results saved to data/results/{model}_{region}_{ctx}_{horizon}.json
    - Checkpoints saved to data/checkpoints/{model}_{region}_{seed}.pt
    - Never commits raw data or checkpoints
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

# Ensure src is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tidetail.models import (
    DiffusionTS,
    FlowMatching,
    ResidualDenoiser,
    VelocityTCN,
)
from tidetail.quality import evaluate

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
RESULTS_DIR = ROOT / "data" / "results"
CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_region(region: str, context: int = 96, horizon: int = 72):
    """Load prime data for a region, return train/val/test windows + scaler.

    Returns BOTH water-level and residual windows so different model types
    can pick the right input (DiffusionTS decomposes full water level;
    FlowMatching operates on non-tidal residual directly).
    """
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

    # --- Full water-level windows (for Diffusion-TS which decomposes internally) ---
    tr_h, tr_t = _make_windows(wi["train_start_indices"], water_level)
    va_h, va_t = _make_windows(wi["validation_start_indices"], water_level)
    te_h, te_t = _make_windows(wi["test_start_indices"], water_level)

    # --- Residual windows (for Flow Matching which models residual directly) ---
    rtr_h, rtr_t = _make_windows(wi["train_start_indices"], residual)
    rva_h, rva_t = _make_windows(wi["validation_start_indices"], residual)
    rte_h, rte_t = _make_windows(wi["test_start_indices"], residual)

    return {
        # Full water level (Diffusion-TS)
        "train_hist_wl": tr_h, "train_tgt_wl": tr_t,
        "val_hist_wl": va_h,   "val_tgt_wl": va_t,
        "test_hist_wl": te_h,  "test_tgt_wl": te_t,
        # Non-tidal residual (Flow Matching)
        "train_hist_res": rtr_h, "train_tgt_res": rtr_t,
        "val_hist_res": rva_h,   "val_tgt_res": rva_t,
        "test_hist_res": rte_h,  "test_tgt_res": rte_t,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "meta": meta,
        "total_points": len(residual),
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_residual_stats(x: torch.Tensor):
    """Per-sample mean/std for residual normalization.

    x: (batch, T) → mean (batch, 1), std (batch, 1)
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-4)
    return mean, std


def train_flow_matching(data, region: str, epochs: int, batch_size: int,
                        lr: float, seed: int, device: str):
    """Train M4 Flow Matching on a region's residual data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    context, horizon = 96, 72
    model = FlowMatching(context=context, horizon=horizon, euler_steps=64,
                         device=device)
    velocity_net = VelocityTCN(context=context, horizon=horizon, width=32).to(device)
    optimizer = torch.optim.Adam(velocity_net.parameters(), lr=lr)

    X_hist = torch.from_numpy(data["train_hist_res"]).to(device)
    X_tgt = torch.from_numpy(data["train_tgt_res"]).to(device)

    # Per-sample residual normalization using history stats
    hist_mean, hist_std = compute_residual_stats(X_hist)
    hist_norm = (X_hist - hist_mean) / hist_std
    tgt_norm = (X_tgt - hist_mean) / hist_std

    print(f"  FlowMatching training on {region}: {X_hist.shape[0]} windows, "
          f"{epochs} epochs, batch={batch_size}, params={sum(p.numel() for p in velocity_net.parameters()):,}")
    print(f"  history residual: mean={hist_norm.mean().item():.3f}, std={hist_norm.std().item():.3f}")
    print(f"  target residual:  mean={tgt_norm.mean().item():.3f}, std={tgt_norm.std().item():.3f}")

    n_batches = (X_hist.shape[0] + batch_size - 1) // batch_size
    history = {"epoch": [], "train_loss": [], "time_s": []}
    t0 = time.time()

    velocity_net.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_hist.shape[0], device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            h_b = hist_norm[idx]
            t_b = tgt_norm[idx]
            loss = model.training_loss(h_b, t_b, velocity_net)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= X_hist.shape[0]
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        elapsed = time.time() - t0
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}/{epochs}: loss={epoch_loss:.6f}  "
                  f"elapsed={elapsed:.1f}s  ({elapsed / (epoch + 1):.1f}s/epoch)")

    return velocity_net, history, model


def train_diffusion_ts(data, region: str, epochs: int, batch_size: int,
                       lr: float, seed: int, device: str):
    """Train M2 Diffusion-TS (DDPM residual + trend/seasonal decomposition)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    context, horizon = 96, 72
    model = DiffusionTS(context=context, horizon=horizon, steps=64,
                        schedule="cosine1000_subsample", device=device)
    denoiser = ResidualDenoiser(context=context, horizon=horizon, width=32,
                                use_full_context=True).to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)

    X_hist = torch.from_numpy(data["train_hist_wl"]).to(device)
    X_tgt = torch.from_numpy(data["train_tgt_wl"]).to(device)

    print(f"  DiffusionTS training on {region}: {X_hist.shape[0]} windows, "
          f"{epochs} epochs, batch={batch_size}, params={sum(p.numel() for p in denoiser.parameters()):,}")

    n_batches = (X_hist.shape[0] + batch_size - 1) // batch_size
    history = {"epoch": [], "train_loss": [], "time_s": []}
    t0 = time.time()

    denoiser.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_hist.shape[0], device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            h_b = X_hist[idx]
            t_b = X_tgt[idx]
            loss = model.training_loss(h_b, t_b, denoiser, fourier_weight=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= X_hist.shape[0]
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        elapsed = time.time() - t0
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}/{epochs}: loss={epoch_loss:.6f}  "
                  f"elapsed={elapsed:.1f}s  ({elapsed / (epoch + 1):.1f}s/epoch)")

    return denoiser, history, model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_generation(model_name: str, model_obj, net, data, region: str,
                         seed: int, device: str, n_samples: int = 5000):
    """Generate forecasts on test set and run quality.py evaluator.

    Uses residual test set for FlowMatching (models residual directly)
    and water-level test set for DiffusionTS (models full water level).
    """
    if model_name == "flow_matching":
        test_hist = torch.from_numpy(data["test_hist_res"]).to(device)
        test_tgt = data["test_tgt_res"]
    elif model_name == "diffusion_ts":
        test_hist = torch.from_numpy(data["test_hist_wl"]).to(device)
        test_tgt = data["test_tgt_wl"]
    else:
        raise ValueError(f"unknown model {model_name}")

    # Subsample for evaluation (use all if small)
    n = min(n_samples, test_hist.shape[0])
    idx = torch.arange(n, device=device)
    hist_sub = test_hist[idx]
    real_tgt = test_tgt[:n]

    torch.manual_seed(seed)
    with torch.no_grad():
        if model_name == "flow_matching":
            hist_mean, hist_std = compute_residual_stats(hist_sub)
            hist_norm = (hist_sub - hist_mean) / hist_std
            gen_norm = model_obj.sample(hist_norm, net, seed=seed)
            # Denormalize back to residual scale
            gen = gen_norm.cpu().numpy() * hist_std.cpu().numpy() + hist_mean.cpu().numpy()
            real = real_tgt.cpu().numpy() if isinstance(real_tgt, torch.Tensor) else real_tgt
        elif model_name == "diffusion_ts":
            gen = model_obj.forecast(hist_sub, net, seed=seed).cpu().numpy()
            real = real_tgt.cpu().numpy() if isinstance(real_tgt, torch.Tensor) else real_tgt
        else:
            raise ValueError(f"unknown model {model_name}")

    # Run quality evaluator
    real_flat = real.reshape(-1, real.shape[-1])
    gen_flat = gen.reshape(-1, gen.shape[-1])
    report = evaluate(gen_flat, real_flat)

    # Scale ratio (critical for TideTail)
    real_std = real.std()
    gen_std = gen.std()
    scale_ratio = gen_std / max(real_std, 1e-8)
    peak_ratio = np.abs(gen).max() / max(np.abs(real).max(), 1e-8)

    # Per-step MAE/RMSE
    mae = float(np.mean(np.abs(gen - real)))
    rmse = float(np.sqrt(np.mean((gen - real) ** 2)))

    result = {
        "model": model_name,
        "region": region,
        "seed": seed,
        "n_generated": gen.shape[0],
        "forecast_std": float(gen_std),
        "real_std": float(real_std),
        "scale_ratio": float(scale_ratio),
        "peak_ratio": float(peak_ratio),
        "mae": mae,
        "rmse": rmse,
        "real_q95": float(np.percentile(real, 95)),
        "gen_q95": float(np.percentile(gen, 95)),
        "real_q99": float(np.percentile(real, 99)),
        "gen_q99": float(np.percentile(gen, 99)),
        "gate": report.status,
        "gate_reasons": report.warnings + report.failures,
    }
    # Add quality metrics
    d = report.to_dict()
    for k, v in d.items():
        if k not in ("status", "warnings", "failures", "notes"):
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["flow_matching", "diffusion_ts"],
                    required=True)
    ap.add_argument("--regions", nargs="+",
                    default=["ny_bight", "thames_north_sea", "us_gulf_coast",
                             "south_atlantic_charleston", "southeast_australia"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--git-sha", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: CUDA not available, running on CPU")

    git_sha = args.git_sha or os.popen("git rev-parse HEAD").read().strip()
    print(f"Git SHA: {git_sha}")

    all_results = []
    for region in args.regions:
        print(f"\n{'='*60}")
        print(f"Region: {region}")
        print(f"{'='*60}")

        # Load data once
        print("  Loading data...")
        data = load_region(region)
        print(f"  train_res={data['train_hist_res'].shape}, "
              f"test_res={data['test_hist_res'].shape}")

        for seed in args.seeds:
            print(f"\n  --- Seed {seed} ---")
            t_start = time.time()

            if args.model == "flow_matching":
                net, hist, model_obj = train_flow_matching(
                    data, region, args.epochs, args.batch_size, args.lr, seed, device
                )
            elif args.model == "diffusion_ts":
                net, hist, model_obj = train_diffusion_ts(
                    data, region, args.epochs, args.batch_size, args.lr, seed, device
                )

            # Save checkpoint
            ckpt_path = CKPT_DIR / f"{args.model}_{region}_seed{seed}.pt"
            torch.save({"state_dict": net.state_dict(),
                        "history": hist,
                        "model_config": {"context": 96, "horizon": 72}},
                       ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

            # Evaluate
            print("  Evaluating...")
            result = evaluate_generation(args.model, model_obj, net, data,
                                          region, seed, device,
                                          n_samples=args.max_samples)
            result["git_sha"] = git_sha
            result["epochs"] = args.epochs
            result["batch_size"] = args.batch_size
            result["lr"] = args.lr
            result["runtime_s"] = time.time() - t_start
            result["params"] = sum(p.numel() for p in net.parameters())
            result["gpu_name"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
            result["peak_mem_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                                      if device == "cuda" else 0)

            print(f"  Scale ratio: {result['scale_ratio']:.3f}")
            print(f"  Peak ratio:  {result['peak_ratio']:.3f}")
            print(f"  MAE={result['mae']:.4f} RMSE={result['rmse']:.4f}")
            print(f"  q95  real={result['real_q95']:.3f} gen={result['gen_q95']:.3f}")
            print(f"  q99  real={result['real_q99']:.3f} gen={result['gen_q99']:.3f}")
            print(f"  Gate: {result['gate']}  ({'; '.join(result['gate_reasons']) if result['gate_reasons'] else 'clean'})")
            print(f"  Runtime: {result['runtime_s']:.1f}s")

            all_results.append(result)

            # Reset CUDA memory tracking
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

    # Save aggregate results
    out_path = RESULTS_DIR / f"{args.model}_96_72.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} results → {out_path}")

    # Summary table
    print(f"\n{'Model':<14} {'Region':<22} {'Seed':>5} {'Scale':>7} {'Peak':>7} "
          f"{'q95R':>7} {'q95G':>7} {'q99R':>7} {'q99G':>7} {'Gate':>11} {'MAE':>7}")
    for r in all_results:
        print(f"{r['model']:<14} {r['region']:<22} {r['seed']:>5} "
              f"{r['scale_ratio']:>7.3f} {r['peak_ratio']:>7.3f} "
              f"{r['real_q95']:>7.3f} {r['gen_q95']:>7.3f} "
              f"{r['real_q99']:>7.3f} {r['gen_q99']:>7.3f} "
              f"{r['gate']:>11} {r['mae']:>7.4f}")


if __name__ == "__main__":
    main()
