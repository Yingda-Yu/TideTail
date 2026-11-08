"""Wave 2.7 physics-guided Flow Matching pilot (CPU).

Five variants on identical splits and downstream forecaster:

  standard         history only
  tail_aware       history only + tail-weighted FM loss
  pressure_prior   history + causal ERA5 MSL summaries
  wind_prior       history + causal ERA5 wind summaries
  combined         history + MSL + wind summaries

CAUSAL CONTRACT
---------------
Physics covariates use only the 96 h before the forecast start. Future ERA5
reanalysis is never fed to the model. This is therefore an operational-style
causal experiment, not a retrospective future-forcing experiment.

CAVEATS
-------
Single-region (or two-region) CPU pilot, small epoch budget. It diagnoses
whether a physical prior helps under controlled conditions; it is not a
publication claim about all regions.

Usage:
    python3 scripts/wave27_physics_fm.py --regions south_atlantic_charleston ny_bight

Output:
    data/results/wave27_physics_fm.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tidetail.models.flow_matching import FlowMatching, VelocityTCN
from tidetail.physics import tail_weight
from tidetail.quality import evaluate

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
ERA5_DIR = Path("/data/yingda/tidetail-data/real-sources-v1/era5")
PREDS = ROOT / "data" / "predictions" / "v2"
RESULTS = ROOT / "data" / "results"

ANCHORS = {
    "ny_bight": (40.71, -74.01),
    "us_gulf_coast": (29.33, -94.77),
    "south_atlantic_charleston": (32.78, -79.87),
    "southeast_australia": (-34.50, 150.87),
    "thames_north_sea": (51.45, 0.80),
}
FEATURE_NAMES = {
    "pressure": ["msl_last", "msl_mean24", "msl_tend24", "msl_mean72"],
    "wind": ["u_last", "v_last", "speed_last", "speed_mean24", "speed_mean72"],
}
CONTEXT, HORIZON = 96, 72


# ---------------------------------------------------------------------------
# Data + causal physics features
# ---------------------------------------------------------------------------

def load_region(region: str) -> dict:
    base = DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    merged = target[["timestamp_utc", "water_level_m"]].merge(
        tide[["timestamp_utc", "tide_level_m"]], on="timestamp_utc", how="inner"
    )
    residual = merged["water_level_m"].to_numpy(dtype=np.float32)
    ts = pd.to_datetime(merged["timestamp_utc"], utc=True)
    wi = np.load(base / "window_index_v1.npz")

    def windows(starts, series):
        starts = np.asarray(starts, dtype=np.int64)
        h = np.empty((len(starts), CONTEXT), dtype=np.float32)
        t = np.empty((len(starts), HORIZON), dtype=np.float32)
        for i, s in enumerate(starts):
            h[i] = series[s:s + CONTEXT]
            t[i] = series[s + CONTEXT:s + CONTEXT + HORIZON]
        return h, t

    out = {}
    for split, key in (("train", "train_start_indices"), ("val", "validation_start_indices"),
                       ("test", "test_start_indices")):
        h, t = windows(wi[key], residual)
        out[f"{split}_h"], out[f"{split}_t"] = h, t
        out[f"{split}_starts"] = wi[key].astype(np.int64)
    out["ts"] = ts
    out["series"] = residual
    return out


def load_era5_anchor(region: str) -> pd.DataFrame:
    lat0, lon0 = ANCHORS[region]
    rows = []
    for path in sorted(glob.glob(str(ERA5_DIR / f"era5_{region}_*_u10v10mslp.nc"))):
        with xr.open_dataset(path) as ds:
            lat = ds["latitude"].values
            lon = ds["longitude"].values
            ilat = int(np.argmin(np.abs(lat - lat0)))
            ilon = int(np.argmin(np.abs(lon - lon0)))
            frame = pd.DataFrame({
                "time": pd.to_datetime(ds["valid_time"].values, utc=True),
                "u10": ds["u10"].values[:, ilat, ilon].astype(np.float32),
                "v10": ds["v10"].values[:, ilat, ilon].astype(np.float32),
                "msl": ds["msl"].values[:, ilat, ilon].astype(np.float32),
            })
            rows.append(frame)
    df = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    df["speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    return df


def align_physics(region: str, timestamps: pd.Series) -> pd.DataFrame:
    era = load_era5_anchor(region)
    merged = pd.DataFrame({"time": timestamps}).merge(era, on="time", how="left",
                                                      validate="one_to_one")
    missing = int(merged[["u10", "v10", "msl"]].isna().any(axis=1).sum())
    if missing:
        for col in ("u10", "v10", "msl", "speed"):
            merged[col] = merged[col].interpolate(limit_direction="both")
    if len(merged) != len(timestamps):
        raise RuntimeError(f"ERA5 alignment changed row count for {region}")
    merged.attrs["era5_missing_after_join"] = missing
    return merged


def build_features(starts: np.ndarray, physics: pd.DataFrame, mode: str,
                   train_stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Causal summaries over the 96 h history window [s, s+96) only.

    ``s`` is the start index of the model's conditioning history, so the
    last observation available at forecast origin is ``s + 95``. No value
    from ``>= s + 96`` (the target horizon) enters these features.
    """
    cols = []
    if mode in ("pressure", "combined"):
        cols += FEATURE_NAMES["pressure"]
    if mode in ("wind", "combined"):
        cols += FEATURE_NAMES["wind"]
    if mode == "standard" or mode == "tail_aware":
        return np.zeros((len(starts), 0), dtype=np.float32), train_stats or {}

    raw = np.empty((len(starts), len(cols)), dtype=np.float32)
    msl = physics["msl"].to_numpy(dtype=np.float32)
    u = physics["u10"].to_numpy(dtype=np.float32)
    v = physics["v10"].to_numpy(dtype=np.float32)
    sp = physics["speed"].to_numpy(dtype=np.float32)
    n_series = len(msl)
    for i, s in enumerate(starts):
        s = int(s)
        lo, hi = s, s + CONTEXT          # history window, hi exclusive
        if s < 0 or hi > n_series:
            raise IndexError(f"window [{s},{hi}) outside ERA5 series length {n_series}")
        last = hi - 1                     # last known observation
        vals = {
            "msl_last": msl[last],
            "msl_mean24": msl[hi - 24:hi].mean(),
            "msl_tend24": msl[last] - msl[hi - 25],
            "msl_mean72": msl[hi - 72:hi].mean(),
            "u_last": u[last],
            "v_last": v[last],
            "speed_last": sp[last],
            "speed_mean24": sp[hi - 24:hi].mean(),
            "speed_mean72": sp[hi - 72:hi].mean(),
        }
        raw[i] = [vals[c] for c in cols]

    stats = train_stats or {"mean": raw.mean(axis=0), "std": raw.std(axis=0).clip(min=1e-4)}
    norm = (raw - stats["mean"]) / stats["std"]
    return norm.astype(np.float32), stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_variant(data: dict, feats: dict, mode: str, epochs: int,
                  batch_size: int, device: str) -> dict:
    torch.manual_seed(42)
    physics_dim = feats["train"].shape[1]
    fm = FlowMatching(context=CONTEXT, horizon=HORIZON, euler_steps=64, device=device)
    net = VelocityTCN(context=CONTEXT, horizon=HORIZON, width=32,
                      physics_dim=physics_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    Xh = torch.from_numpy(data["train_h"]).to(device)
    Xt = torch.from_numpy(data["train_t"])
    mean = Xh.mean(dim=1, keepdim=True)
    std = Xh.std(dim=1, keepdim=True).clamp(min=1e-4)
    hn = (Xh - mean) / std
    tn = ((Xt - mean.cpu()) / std.cpu()).to(device)
    ph = torch.from_numpy(feats["train"]).to(device)

    tail_w = None
    if mode == "tail_aware":
        flat = np.abs(data["train_t"].reshape(-1))
        q95, q99 = np.percentile(flat, 95), np.percentile(flat, 99)
        tail_w = torch.from_numpy(
            tail_weight(np.abs(data["train_t"]), q95, q99).mean(axis=1).astype(np.float32)
        ).to(device)

    n = len(hn)
    hist = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        net.train()
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            hb, tb = hn[idx], tn[idx]
            pb = ph[idx] if physics_dim else None
            if tail_w is None:
                loss = fm.training_loss(hb, tb, net, physics=pb)
            else:
                wb = tail_w[idx]
                B = hb.shape[0]
                tt = torch.rand(B, 1, 1, device=device)
                x0 = torch.randn(B, 1, HORIZON, device=device)
                x1 = tb.unsqueeze(1)
                xt = (1 - tt) * x0 + tt * x1
                vpred = net(xt, tt.squeeze(-1).squeeze(-1), hb, pb)
                per = ((vpred - (x1 - x0)) ** 2).mean(dim=(1, 2))
                loss = (per * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / n)

    return {"net": net, "fm": fm, "hist": hist, "physics_dim": physics_dim,
            "params": int(sum(p.numel() for p in net.parameters()))}


@torch.no_grad()
def sample_windows(model: dict, hist: np.ndarray, phys: np.ndarray,
                   idx: np.ndarray, draws: int) -> np.ndarray:
    """Generate `draws` samples for history windows `hist[idx]`, in metres.

    The generator is fitted on per-sample-normalised history and targets
    (``train_variant``). Inference must therefore apply the same per-sample
    statistics to the history and invert them on the output. Handing raw
    history to a normalised model mixes metre-scale inputs with dimensionless
    ones and inflates the generated dispersion; Wave 2.7 audited this path and
    found a 4.95 scale ratio from it against 1.17 once normalised.

    Physics features must be the causal features of the SAME windows; they are
    already standardised with train-only statistics by ``build_features``.
    """
    net, fm = model["net"], model["fm"]
    h = torch.from_numpy(np.ascontiguousarray(hist[idx]))
    mean = h.mean(dim=1, keepdim=True)
    std = h.std(dim=1, keepdim=True).clamp(min=1e-4)
    hn = (h - mean) / std
    p = None
    if model["physics_dim"]:
        p = torch.from_numpy(np.ascontiguousarray(phys[idx]))
    out = []
    for k in range(draws):
        g = fm.sample(hn, net, seed=42 + k, physics=p)
        out.append((g * std + mean).numpy())
    return np.stack(out, axis=0)  # (draws, len(idx), horizon)


# ---------------------------------------------------------------------------
# Downstream GRU proxy
# ---------------------------------------------------------------------------

class GRUForecaster(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.head(out[:, -1])


def train_downstream(data: dict, aug: dict | None, seeds=(42,),
                     steps: int = 120, batch_size: int = 256) -> np.ndarray:
    """Train a GRU forecaster with optional aligned augmentation.

    ``aug`` carries ``hist`` and ``target`` arrays of identical length so that
    each synthetic target is conditioned on the history window it was actually
    generated from (no history/target re-pairing).
    """
    torch.manual_seed(42)
    h_real = data["train_h"]
    t_real = data["train_t"]
    if aug is not None:
        h_aug = np.concatenate([h_real, aug["hist"]], axis=0)
        t_aug = np.concatenate([t_real, aug["target"]], axis=0)
    else:
        h_aug, t_aug = h_real, t_real

    # Per-sample normalization (independent of any other split)
    mean = h_aug.mean(axis=1, keepdims=True)
    std = h_aug.std(axis=1, keepdims=True).clip(min=1e-4)
    hn = torch.from_numpy(((h_aug - mean) / std).astype(np.float32))
    tn = torch.from_numpy(((t_aug - mean) / std).astype(np.float32))

    # Test histories get their own per-sample statistics (never train stats).
    te_mean = data["test_h"].mean(axis=1, keepdims=True)
    te_std = data["test_h"].std(axis=1, keepdims=True).clip(min=1e-4)
    th = torch.from_numpy(((data["test_h"] - te_mean) / te_std).astype(np.float32))

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = GRUForecaster()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n = len(hn)
        for step in range(steps):
            idx = torch.randint(0, n, (batch_size,))
            loss = F.mse_loss(net(hn[idx]), tn[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = net(th).numpy() * te_std + te_mean
        preds.append(out)
    return np.mean(preds, axis=0)


def event_level_mae(pred: np.ndarray, truth: np.ndarray, event_ids: np.ndarray) -> dict:
    err = np.abs(pred - truth).mean(axis=1)
    units = np.unique(event_ids)
    extreme = units[units != -999]
    per_unit = np.array([err[event_ids == u].mean() for u in extreme])
    return {
        "n_extreme_units": int(len(extreme)),
        "extreme_unit_mae": float(per_unit.mean()),
        "all_window_mae": float(err.mean()),
        "per_unit_mae": per_unit.tolist(),
    }


def load_event_ids(region: str, n_test: int) -> np.ndarray | None:
    files = sorted(PREDS.glob(f"{region}_A0_real_only_gru_seed*.npz"))
    if not files:
        return None
    with np.load(files[0], allow_pickle=True) as d:
        ids = d["event_ids"].astype(np.int64)
    return ids if len(ids) == n_test else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["south_atlantic_charleston", "ny_bight"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=2000,
                    help="number of synthetic training windows (CPU budget)")
    ap.add_argument("--draws-test", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=0,
                    help="optional cap on windows used to FIT the generative model "
                         "(0 = all; full data needed to pass fidelity gates)")
    ap.add_argument("--downstream-n-train", type=int, default=8000,
                    help="windows used by the downstream GRU, fixed across arms")
    ap.add_argument("--downstream-steps", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    payload = {"audit_version": "wave27_v1", "causal_contract":
               "ERA5 covariates from 96h history only; no future reanalysis",
               "regions": {}}
    t0 = time.time()

    for region in args.regions:
        print(f"\n{'='*70}\nREGION {region}")
        data = load_region(region)
        # Deterministic train subsample so the pilot is CPU-feasible. Test and
        # validation splits are never subsampled.
        if args.n_train and len(data["train_h"]) > args.n_train:
            ridx = np.sort(np.random.default_rng(7).choice(
                len(data["train_h"]), size=args.n_train, replace=False))
            data["train_h"] = data["train_h"][ridx]
            data["train_t"] = data["train_t"][ridx]
            data["train_starts"] = data["train_starts"][ridx]
        print(f"  train windows used to fit FM: {len(data['train_h'])} / "
              f"test: {len(data['test_h'])}")

        # Fixed downstream budget, identical for A0 and every augmented arm.
        n_down = min(args.downstream_n_train, len(data["train_h"]))
        down_idx = np.sort(np.random.default_rng(17).choice(
            len(data["train_h"]), size=n_down, replace=False))
        down_data = {
            "train_h": data["train_h"][down_idx],
            "train_t": data["train_t"][down_idx],
            "test_h": data["test_h"],
            "test_t": data["test_t"],
        }
        physics = align_physics(region, data["ts"])
        print(f"  ERA5 join missing (before interpolation): "
              f"{physics.attrs.get('era5_missing_after_join')}")

        # Train-only feature statistics; val/test reuse the same transform.
        wi_path = DATA_ROOT / region / "window_index_v1.npz"
        feats = {}
        for mode in ("standard", "tail_aware", "pressure", "wind", "combined"):
            tr, stats = build_features(data["train_starts"], physics, mode)
            va, _ = build_features(data["val_starts"], physics, mode, stats)
            te, _ = build_features(data["test_starts"], physics, mode, stats)
            feats[mode] = {"train": tr, "val": va, "test": te, "stats": stats}

        n_test = min(args.n_test, len(data["test_h"]))
        variants = {}
        for mode in ("standard", "tail_aware", "pressure", "wind", "combined"):
            print(f"  training {mode} (physics_dim={feats[mode]['train'].shape[1]}) ...")
            model = train_variant(data, feats[mode], mode, args.epochs, args.batch_size, args.device)

            gen = sample_windows(model, data["test_h"], feats[mode]["test"],
                                 np.arange(n_test), draws=args.draws_test)
            real = data["test_t"][:n_test]
            g_flat = gen.reshape(-1, HORIZON)
            report = evaluate(real, g_flat, train_ref=data["train_t"])
            variants[mode] = {
                "params": model["params"],
                "physics_dim": model["physics_dim"],
                "final_train_loss": model["hist"]["loss"][-1],
                "n_eval_windows": int(n_test),
                "fidelity_status": report.status,
                "scale_ratio": float(report.scale_ratio),
                "wasserstein1": float(report.wasserstein1),
                "autocorr_rmse": float(report.autocorr_rmse),
                "q95_exceed_ratio": float(report.q95_exceedance_ratio),
                "gen_std_ratio": float(np.std(g_flat) / np.std(real)),
                "warnings": list(report.warnings)[:3],
            }

            # Augmentation set: synthetic targets generated from their OWN
            # history windows. ``n_aug`` is an absolute CPU budget; the
            # augmentation ratio is n_aug / n_train (reported below).
            n_aug = min(args.n_aug, len(down_data["train_h"]))
            gidx = np.random.default_rng(7).choice(len(down_data["train_h"]),
                                                   size=n_aug, replace=False)
            # Condition synthetic targets on the SAME (downstream) history
            # windows and their causal features; no re-pairing.
            gen_train = sample_windows(model, down_data["train_h"],
                                       feats[mode]["train"][down_idx],
                                       gidx, draws=1)[0]
            variants[mode]["_aug"] = {"hist": down_data["train_h"][gidx],
                                      "target": gen_train}
            variants[mode]["augmentation_ratio"] = float(
                n_aug / len(down_data["train_h"]))

        # Downstream proxy on the identical test set
        truth = down_data["test_t"]
        ids = load_event_ids(region, len(truth))
        base_pred = train_downstream(down_data, None, steps=args.downstream_steps,
                                     batch_size=args.batch_size)
        base = event_level_mae(base_pred, truth, ids) if ids is not None else {
            "all_window_mae": float(np.abs(base_pred - truth).mean())}
        print(f"  downstream A0 real-only MAE={base.get('extreme_unit_mae', base.get('all_window_mae')):.4f}")
        for mode in ("standard", "tail_aware", "pressure", "wind", "combined"):
            pred = train_downstream(down_data, variants[mode].pop("_aug"),
                                    steps=args.downstream_steps, batch_size=args.batch_size)
            ev = event_level_mae(pred, truth, ids) if ids is not None else {
                "all_window_mae": float(np.abs(pred - truth).mean())}
            variants[mode]["downstream"] = ev
            base_mae = base.get("extreme_unit_mae", base.get("all_window_mae"))
            d = (ev.get("extreme_unit_mae", ev.get("all_window_mae")) - base_mae) / base_mae * 100
            variants[mode]["downstream_delta_pct_vs_A0"] = float(d)
            print(f"    {mode:12s} downstream delta = {d:+.2f}%")

        payload["regions"][region] = {
            "n_train_fm": int(len(data["train_h"])),
            "n_train_downstream": int(len(down_data["train_h"])),
            "n_test": int(len(data["test_h"])),
            "era5_join_missing_before_interp": int(physics.attrs.get("era5_missing_after_join", 0)),
            "baseline_A0": base,
            "variants": variants,
        }

    payload["elapsed_s"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "wave27_physics_fm.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
