"""Tail-aware Flow Matching training for TideTail.

Trains standard FM with per-sample tail weights that upweight extreme surge
events during training. q95/q99 thresholds fit on TRAINING data only.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/train_tail_aware_fm.py --region ny_bight --epochs 15 --seed 42
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tidetail.models.flow_matching import FlowMatching, VelocityTCN
from tidetail.physics import tail_weight

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
CKPT_DIR = ROOT / "data" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def load_region(region, context=96, horizon=72):
    base = DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    wi = np.load(base / "window_index_v1.npz")
    residual = target["water_level_m"].to_numpy(dtype=np.float32)
    scaler = json.loads((base / "train_only_metadata_v1.json").read_text())

    def make_windows(starts, series):
        starts = starts.astype(np.int64)
        N = len(starts)
        h = np.zeros((N, context), dtype=np.float32)
        t = np.zeros((N, horizon), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = series[s:s+context]
            t[i] = series[s+context:s+context+horizon]
        return h, t

    tr_h, tr_t = make_windows(wi["train_start_indices"], residual)
    va_h, va_t = make_windows(wi["validation_start_indices"], residual)
    te_h, te_t = make_windows(wi["test_start_indices"], residual)
    return {"train_h": tr_h, "train_t": tr_t, "val_h": va_h, "val_t": va_t, "test_h": te_h, "test_t": te_t,
            "scaler_mean": scaler["scalers"]["water_level_m"]["mean"],
            "scaler_std": scaler["scalers"]["water_level_m"]["std"]}

def train_tail_aware_fm(region, epochs=15, batch_size=256, lr=1e-3, seed=42, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_region(region)
    context, horizon = 96, 72
    model = FlowMatching(context=context, horizon=horizon, euler_steps=64, device=device)
    net = VelocityTCN(context=context, horizon=horizon, width=32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    Xh = torch.from_numpy(data["train_h"]).to(device)
    Xt = torch.from_numpy(data["train_t"]).to(device)

    hist_mean = Xh.mean(dim=1, keepdim=True)
    hist_std = Xh.std(dim=1, keepdim=True).clamp(min=1e-4)
    h_norm = (Xh - hist_mean) / hist_std
    t_norm = (Xt - hist_mean) / hist_std

    # Tail weights on TRAINING residuals (absolute values, extreme positive OR negative)
    train_res_abs = np.abs(data["train_t"].flatten())
    q95 = float(np.percentile(train_res_abs, 95))
    q99 = float(np.percentile(train_res_abs, 99))
    # Per-sample weight: mean of tail_weight over horizon steps
    sample_weights = tail_weight(train_res_abs.reshape(-1, horizon), q95, q99).mean(axis=1)
    w_t = torch.from_numpy(sample_weights.astype(np.float32)).to(device)
    print(f"  Tail thresholds (train-only): q95={q95:.3f}m, q99={q99:.3f}m")
    print(f"  Weight stats: mean={w_t.mean().item():.3f}, >1 frac={((w_t>1).float().mean().item()*100):.1f}%")

    n = Xh.shape[0]
    nb = (n + batch_size - 1) // batch_size
    t0 = time.time()
    net.train()

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for b in range(nb):
            idx = perm[b*batch_size:(b+1)*batch_size]
            hb, tb, wb = h_norm[idx], t_norm[idx], w_t[idx]
            # Per-sample tail-weighted FM loss (manual, no reduce=False support)
            B = hb.shape[0]
            tt = torch.rand(B, 1, 1, device=device)
            x0 = torch.randn(B, 1, horizon, device=device)
            x1 = tb.unsqueeze(1)
            xt = (1 - tt) * x0 + tt * x1
            tgt_v = x1 - x0
            vpred = net(xt, tt.squeeze(-1).squeeze(-1), hb)
            per_sample_mse = ((vpred - tgt_v) ** 2).mean(dim=(1, 2))  # (B,)
            loss = (per_sample_mse * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n
        elapsed = time.time() - t0
        if ep % max(1, epochs//5) == 0 or ep == epochs-1:
            print(f"  [{region}] epoch {ep+1}/{epochs}: loss={ep_loss:.6f} ({elapsed:.1f}s)")

    # Save
    ckpt_path = CKPT_DIR / f"flow_matching_tailaware_{region}_seed{seed}.pt"
    torch.save({"model_state": net.state_dict(), "region": region, "seed": seed,
                "epochs": epochs, "q95": q95, "q99": q99, "tail_weight_formula": "piecewise linear q95→1, q99→2"}, ckpt_path)
    print(f"  Saved → {ckpt_path}")
    return ckpt_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print(f"=== Tail-aware FM: {args.region}, seed={args.seed}, device={args.device} ===")
    train_tail_aware_fm(args.region, args.epochs, args.batch_size, args.lr, args.seed, args.device)
