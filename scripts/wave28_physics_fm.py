"""Wave 2.8 P1 — genuine physics-guided Flow Matching pilot (CPU).

Six arms on identical training data, conditioning windows, architecture,
seeds and computational budgets:

  standard        history only, no ERA5, no penalty
  era5_cond       history + 9 causal ERA5 features, NO penalty
                  (conditioning-only control — mandatory)
  pressure_prior  era5_cond + inverse-barometer physical penalty
  wind_prior      era5_cond + coast-normal wind-stress physical penalty
  combined_prior  era5_cond + both physical penalties
  combined_tail   combined penalties + tail-aware FM loss

DESIGN PRINCIPLES (Wave 2.8 P1 spec)
------------------------------------
* Feature conditioning, physically motivated regularization and
  PDE-constrained modeling are SEPARATE. Arms 2-6 all condition on the
  same 9 ERA5 features; arms 3-6 additionally add a physical penalty on
  the generator's implied clean estimate; no PDE solver is used.
* The penalty regularizes the generator toward a train-only linear
  physical prediction of the residual level change across the forecast
  origin:

      Delta := mean(target[0:24h]) - mean(history[72:96h])

  predicted from CAUSAL history-only forcing:

      inverse barometer: ib_tend_m = -0.01 m/hPa * (msl_last - msl_mean72)
      wind setup:        tau_n = rho_a * Cd * |W| * W_onshore
                         (averaged over last 24 h / 72 h of history)

  Coefficients are fitted by OLS on TRAIN windows only. The physically
  justified lagged response is the persistence/decay of wind setup into
  the first 24 h of the forecast horizon (setup decay timescale
  12-48 h) and the instantaneous inverse-barometer level.

* Penalty mechanics: at training time the flow-matching velocity
  prediction implies a clean target estimate x1_hat = x_t + (1-t)*v.
  The penalty is MSE( Delta(x1_hat), Delta_phys_pred ) computed in
  per-sample-normalised space (metre-space predictions divided by the
  sample's history std). The penalty weight is auto-calibrated on the
  FIRST batch as lambda = 0.1 * L_FM / R (clipped to [0.01, 1.0]) and
  then FROZEN — no test-set tuning, no per-arm hand-picking.

* Coast-normal orientation: approximate onshore unit vectors
  (east, north) digitised from regional coastal geography; documented
  per region below. They are approximations, flagged as such.

* Bathymetric dependence: NOT explicit. The per-region fitted wind
  coefficient implicitly absorbs regional fetch/bathymetry; GEBCO
  depth conditioning is deferred (documented limitation, not a claim).

CAUSAL CONTRACT
---------------
ERA5 covariates use only the 96 h before the forecast start. Future
reanalysis never enters conditioning or penalties.

Evaluation per arm: sample fidelity (full quality battery incl. Tail95/
Tail99, peak amplitude KS, peak timing), physical-consistency
diagnostics, downstream GRU AND TCN proxies with seed variability, and
event-level paired sign-flip / bootstrap inference vs the A0 real-only
baseline.

Usage:
    python3 scripts/wave28_physics_fm.py --regions \
        south_atlantic_charleston ny_bight

Output:
    data/results/wave28_physics_fm.json   (incremental, per arm)
    data/results/checkpoints/wave28_physics/<region>_<arm>_seed<k>.pt
    data/results/wave28_physics_samples_<region>.npz
"""
from __future__ import annotations

import argparse
import glob
import itertools
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
CKPT_DIR = RESULTS / "checkpoints" / "wave28_physics"

ANCHORS = {
    "ny_bight": (40.71, -74.01),
    "us_gulf_coast": (29.33, -94.77),
    "south_atlantic_charleston": (32.78, -79.87),
    "southeast_australia": (-34.50, 150.87),
    "thames_north_sea": (51.45, 0.80),
}
# Approximate ONSHORE unit vectors (east, north): the direction toward
# which onshore wind blows. Digitised from coastal geography; flagged as
# approximations in the experiment report. (Normalised to unit length.)
COAST_NORMAL_ONSHORE = {
    "ny_bight": (-0.70710678, 0.70710678),        # Atlantic S/SE of Battery
    "us_gulf_coast": (0.0, 1.0),                  # Gulf south of coast
    "south_atlantic_charleston": (-0.5, 0.8660254),  # Atlantic SE of Chas
    "southeast_australia": (-1.0, 0.0),           # Tasman Sea east of coast
    "thames_north_sea": (-1.0, 0.0),              # North Sea east of Thames
}
RHO_AIR, CD = 1.225, 1.5e-3      # wind-stress bulk formula constants
IB_COEF = -0.01                   # m of residual per hPa of pressure anomaly
CONTEXT, HORIZON = 96, 72

ERA5_FEATURES = ["msl_last", "msl_mean24", "msl_tend24", "msl_mean72",
                 "u_last", "v_last", "speed_last", "speed_mean24",
                 "speed_mean72"]
ARMS = ["standard", "era5_cond", "pressure_prior", "wind_prior",
        "combined_prior", "combined_tail"]
PENALTY_PREDICTORS = {
    "standard": [],
    "era5_cond": [],
    "pressure_prior": ["ib_tend_m"],
    "wind_prior": ["tau_n24", "tau_n72"],
    "combined_prior": ["ib_tend_m", "tau_n24", "tau_n72"],
    "combined_tail": ["ib_tend_m", "tau_n24", "tau_n72"],
}


# ---------------------------------------------------------------------------
# Data + causal physics quantities
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
    for split, key in (("train", "train_start_indices"),
                       ("val", "validation_start_indices"),
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
            rows.append(pd.DataFrame({
                "time": pd.to_datetime(ds["valid_time"].values, utc=True),
                "u10": ds["u10"].values[:, ilat, ilon].astype(np.float32),
                "v10": ds["v10"].values[:, ilat, ilon].astype(np.float32),
                "msl": ds["msl"].values[:, ilat, ilon].astype(np.float32),
            }))
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


def build_cond_features(starts: np.ndarray, physics: pd.DataFrame,
                        use_era5: bool,
                        train_stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Standardised 9-dim causal ERA5 conditioning features (or 0-dim)."""
    if not use_era5:
        return np.zeros((len(starts), 0), dtype=np.float32), train_stats or {}
    msl = physics["msl"].to_numpy(dtype=np.float32)
    u = physics["u10"].to_numpy(dtype=np.float32)
    v = physics["v10"].to_numpy(dtype=np.float32)
    sp = physics["speed"].to_numpy(dtype=np.float32)
    n_series = len(msl)
    raw = np.empty((len(starts), len(ERA5_FEATURES)), dtype=np.float32)
    for i, s in enumerate(starts):
        s = int(s)
        lo, hi = s, s + CONTEXT
        if s < 0 or hi > n_series:
            raise IndexError(f"window [{s},{hi}) outside ERA5 series {n_series}")
        last = hi - 1
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
        raw[i] = [vals[c] for c in ERA5_FEATURES]
    stats = train_stats or {"mean": raw.mean(axis=0),
                            "std": raw.std(axis=0).clip(min=1e-4)}
    return ((raw - stats["mean"]) / stats["std"]).astype(np.float32), stats


def build_physics_raw(starts: np.ndarray, physics: pd.DataFrame,
                      region: str) -> dict:
    """Causal, unit-carrying physical forcing per window.

    ib_tend_m : inverse-barometer residual anomaly implied by the
                pressure anomaly at forecast origin vs the previous 72 h
                mean (metres; negative pressure anomaly -> high water).
    tau_n24/72: coast-normal wind stress averaged over the last 24/72 h
                of history (Pa; positive = onshore setup forcing).
    wn24      : mean onshore wind component (m/s), for diagnostics.
    """
    ne, nn = COAST_NORMAL_ONSHORE[region]
    msl = physics["msl"].to_numpy(dtype=np.float64)
    u = physics["u10"].to_numpy(dtype=np.float64)
    v = physics["v10"].to_numpy(dtype=np.float64)
    n_series = len(msl)
    m = len(starts)
    ib = np.empty(m, dtype=np.float64)
    t24 = np.empty(m, dtype=np.float64)
    t72 = np.empty(m, dtype=np.float64)
    wn24 = np.empty(m, dtype=np.float64)
    for i, s in enumerate(starts):
        s = int(s)
        hi = s + CONTEXT
        if s < 0 or hi > n_series:
            raise IndexError(f"window [{s},{hi}) outside ERA5 series {n_series}")
        last = hi - 1
        # pressure anomaly vs trailing 72 h mean, hPa -> IB metres
        ib[i] = IB_COEF * (msl[last] - msl[hi - 72:hi].mean()) / 100.0
        un, vn = u[hi - 24:hi] * ne, v[hi - 24:hi] * nn
        wn = un + vn
        spd = np.sqrt(u[hi - 24:hi] ** 2 + v[hi - 24:hi] ** 2)
        t24[i] = RHO_AIR * CD * float(np.mean(spd * wn))
        un7, vn7 = u[hi - 72:hi] * ne, v[hi - 72:hi] * nn
        wn7 = un7 + vn7
        spd7 = np.sqrt(u[hi - 72:hi] ** 2 + v[hi - 72:hi] ** 2)
        t72[i] = RHO_AIR * CD * float(np.mean(spd7 * wn7))
        wn24[i] = float(np.mean(wn))
    return {"ib_tend_m": ib, "tau_n24": t24, "tau_n72": t72, "wn24": wn24}


def delta_real(hist: np.ndarray, targ: np.ndarray) -> np.ndarray:
    """Observed residual level change across the forecast origin (m)."""
    return targ[:, :24].mean(axis=1) - hist[:, 72:96].mean(axis=1)


def fit_physics_regression(pred_names: list[str], raw: dict,
                           y: np.ndarray) -> dict:
    """OLS on train windows only; returns predictor names, coefficients,
    R^2 (train) and a closure-free spec for reuse."""
    if not pred_names:
        return {"predictors": [], "coef": [], "r2_train": float("nan")}
    X = np.column_stack([np.ones(len(y))] + [raw[p] for p in pred_names])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"predictors": pred_names,
            "coef": beta.tolist(),          # [intercept, ...]
            "r2_train": 1.0 - ss_res / max(ss_tot, 1e-12)}


def physics_predict(reg: dict, raw: dict) -> np.ndarray:
    if not reg["predictors"]:
        return np.zeros(0)
    X = np.column_stack([np.ones(len(raw["ib_tend_m"]))]
                        + [raw[p] for p in reg["predictors"]])
    return X @ np.asarray(reg["coef"])


# ---------------------------------------------------------------------------
# Generative training with optional physical penalty / tail weighting
# ---------------------------------------------------------------------------

def train_variant(data: dict, feats: dict, raw_train: dict, reg: dict,
                  mode: str, seed: int, epochs: int, batch_size: int,
                  device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    physics_dim = feats["train"].shape[1]
    penalty = mode in ("pressure_prior", "wind_prior", "combined_prior",
                       "combined_tail")
    fm = FlowMatching(context=CONTEXT, horizon=HORIZON, euler_steps=64,
                      device=device)
    net = VelocityTCN(context=CONTEXT, horizon=HORIZON, width=32,
                      physics_dim=physics_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    Xh = torch.from_numpy(data["train_h"]).to(device)
    Xt = torch.from_numpy(data["train_t"])
    mean = Xh.mean(dim=1, keepdim=True)
    std = Xh.std(dim=1, keepdim=True).clamp(min=1e-4)
    hn = (Xh - mean) / std
    tn = ((Xt - mean.cpu()) / std.cpu()).to(device)
    std_dev = std.squeeze(1).to(device)          # (N,)
    hist_last_mean = hn[:, 72:96].mean(dim=1)    # (N,) normalised space
    ph = torch.from_numpy(feats["train"]).to(device)
    if penalty:
        dpred_m = torch.from_numpy(
            physics_predict(reg, raw_train).astype(np.float32)).to(device)
        dpred_n = dpred_m / std_dev              # normalised-space target

    tail_w = None
    if mode == "combined_tail":
        flat = np.abs(data["train_t"].reshape(-1))
        q95, q99 = np.percentile(flat, 95), np.percentile(flat, 99)
        tail_w = torch.from_numpy(
            tail_weight(np.abs(data["train_t"]), q95, q99).mean(axis=1)
            .astype(np.float32)).to(device)

    n = len(hn)
    lam = None
    hist = {"epoch": [], "loss": [], "fm": [], "penalty": []}
    t_start = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot = tot_fm = tot_pen = 0.0
        net.train()
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            hb, tb = hn[idx], tn[idx]
            pb = ph[idx] if physics_dim else None
            B = hb.shape[0]
            tt = torch.rand(B, 1, 1, device=device)
            x0 = torch.randn(B, 1, HORIZON, device=device)
            x1 = tb.unsqueeze(1)
            xt = (1 - tt) * x0 + tt * x1
            vpred = net(xt, tt.squeeze(-1).squeeze(-1), hb, pb)
            if tail_w is None:
                fm_loss = F.mse_loss(vpred, x1 - x0)
            else:
                per = ((vpred - (x1 - x0)) ** 2).mean(dim=(1, 2))
                fm_loss = (per * tail_w[idx]).mean()
            pen = torch.zeros((), device=device)
            if penalty:
                # implied clean estimate: x1_hat = x_t + (1-t) * v
                x1_hat = xt + (1 - tt) * vpred
                d_hat = x1_hat.squeeze(1)[:, :24].mean(dim=1) \
                    - hist_last_mean[idx]
                pen = F.mse_loss(d_hat, dpred_n[idx])
                if lam is None:
                    # first-batch auto-calibration, frozen afterwards
                    lam = float(0.1 * fm_loss.detach() / pen.detach()
                                .clamp(min=1e-6))
                    lam = float(np.clip(lam, 0.01, 1.0))
                    print(f"      lambda_phys auto-calibrated -> {lam:.4f}")
                loss = fm_loss + lam * pen
            else:
                loss = fm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            tot_fm += float(fm_loss.detach()) * len(idx)
            tot_pen += float(pen.detach()) * len(idx)
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / n)
        hist["fm"].append(tot_fm / n)
        hist["penalty"].append(tot_pen / n)

    return {"net": net, "fm": fm, "hist": hist, "physics_dim": physics_dim,
            "params": int(sum(p.numel() for p in net.parameters())),
            "lambda_phys": lam, "train_s": time.time() - t_start}


@torch.no_grad()
def sample_windows(model: dict, hist: np.ndarray, phys: np.ndarray,
                   idx: np.ndarray, draws: int, seed_base: int = 42
                   ) -> np.ndarray:
    """Per-sample-normalised sampling path (Wave 2.7 E26 fix)."""
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
        g = fm.sample(hn, net, seed=seed_base + k, physics=p)
        out.append((g * std + mean).numpy())
    return np.stack(out, axis=0)  # (draws, len(idx), horizon)


# ---------------------------------------------------------------------------
# Downstream proxies (GRU + TCN), matched budgets
# ---------------------------------------------------------------------------

class GRUProxy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.head(out[:, -1])


class TCNProxy(nn.Module):
    """Dilated causal-conv direct multi-horizon predictor (matched budget)."""

    def __init__(self, width: int = 32):
        super().__init__()
        layers, in_ch = [], 1
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(in_ch, width, 3, dilation=d,
                                 padding=0), nn.ReLU()]
            in_ch = width
        self.conv = nn.Sequential(*layers)   # receptive field 31 h
        self.head = nn.Linear(width, HORIZON)

    def forward(self, x):                    # x: (B, 96)
        c = self.conv(x.unsqueeze(1))
        return self.head(c[:, :, -1])


def train_downstream(kind: str, data: dict, aug: dict | None,
                     seeds: tuple, steps: int, batch_size: int) -> np.ndarray:
    torch.manual_seed(42)
    h_real, t_real = data["train_h"], data["train_t"]
    if aug is not None:
        h_aug = np.concatenate([h_real, aug["hist"]], axis=0)
        t_aug = np.concatenate([t_real, aug["target"]], axis=0)
    else:
        h_aug, t_aug = h_real, t_real
    mean = h_aug.mean(axis=1, keepdims=True)
    std = h_aug.std(axis=1, keepdims=True).clip(min=1e-4)
    hn = torch.from_numpy(((h_aug - mean) / std).astype(np.float32))
    tn = torch.from_numpy(((t_aug - mean) / std).astype(np.float32))
    te_mean = data["test_h"].mean(axis=1, keepdims=True)
    te_std = data["test_h"].std(axis=1, keepdims=True).clip(min=1e-4)
    th = torch.from_numpy(((data["test_h"] - te_mean) / te_std).astype(np.float32))

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = GRUProxy() if kind == "gru" else TCNProxy()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n = len(hn)
        for _ in range(steps):
            idx = torch.randint(0, n, (batch_size,))
            loss = F.mse_loss(net(hn[idx]), tn[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(th).numpy() * te_std + te_mean)
    return np.stack(preds, axis=0)   # (n_seeds, n_test, horizon)


def event_level_mae(pred: np.ndarray, truth: np.ndarray,
                    event_ids: np.ndarray) -> dict:
    """pred: (n_seeds, N, H); returns per-unit MAE averaged over seeds."""
    err = np.abs(pred - truth[None]).mean(axis=(0, 2))    # (N,)
    units = np.unique(event_ids)
    extreme = units[units != -999]
    per_unit = np.array([err[event_ids == u].mean() for u in extreme])
    return {"n_extreme_units": int(len(extreme)),
            "extreme_unit_mae": float(per_unit.mean()),
            "all_window_mae": float(err.mean()),
            "per_unit_mae": per_unit.tolist()}


def paired_event_stats(delta: np.ndarray, n_boot: int = 10000,
                       max_exact: int = 16, seed: int = 0) -> dict:
    """Paired per-event differences (variant - A0).

    Null-based two-sided p-value via sign-flip randomisation (exact
    enumeration when n <= max_exact, else Monte Carlo), plus a paired
    bootstrap CI for the mean effect. A bootstrap distribution is NOT
    used as the null — the sign-flip null is.
    """
    d = np.asarray(delta, dtype=np.float64)
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    if n <= max_exact:
        flips = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        stat = np.abs(flips @ d) / n
        p = float(np.mean(stat >= abs(obs) - 1e-12))
    else:
        s = rng.choice([-1.0, 1.0], size=(n_boot, n))
        stat = np.abs(s @ d) / n
        p = float(np.mean(stat >= abs(obs) - 1e-12))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_delta": obs,
            "ci95": [float(lo), float(hi)],
            "signflip_p": p,
            "n_pairs": int(n)}


def load_event_ids(region: str, n_test: int) -> np.ndarray | None:
    files = sorted(PREDS.glob(f"{region}_A0_real_only_gru_seed*.npz"))
    if not files:
        return None
    with np.load(files[0], allow_pickle=True) as d:
        ids = d["event_ids"].astype(np.int64)
    return ids if len(ids) == n_test else None


def physics_consistency(d_real: np.ndarray, d_gen: np.ndarray,
                        dpred: np.ndarray) -> dict:
    r_real = float(np.corrcoef(d_real, dpred)[0, 1])
    r_gen = float(np.corrcoef(d_gen, dpred)[0, 1])
    return {"corr_delta_real_vs_phys": r_real,
            "corr_delta_gen_vs_phys": r_gen,
            "corr_deficit": r_gen - r_real}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["south_atlantic_charleston", "ny_bight"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=2000)
    ap.add_argument("--draws-test", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-train-fm", type=int, default=0,
                    help="cap on windows used to FIT the generator "
                         "(0 = all; full data needed for fidelity gates)")
    ap.add_argument("--downstream-n-train", type=int, default=8000)
    ap.add_argument("--downstream-steps", type=int, default=120)
    ap.add_argument("--fm-seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--downstream-seeds", type=int, nargs="+",
                    default=[42, 99, 7])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "wave28_physics_fm.json"
    payload = {
        "audit_version": "wave28_physics_v1",
        "causal_contract": "ERA5 covariates from 96h history only; "
                           "no future reanalysis in conditioning or penalty",
        "design": {
            "arms": ARMS,
            "conditioning": "arms era5_cond..combined_tail share the same 9 "
                            "causal ERA5 features; only the penalty differs",
            "penalty": "MSE on implied clean estimate x1_hat = x_t + (1-t)v "
                       "between Delta(x1_hat) and a train-only OLS physical "
                       "prediction of the residual level change across the "
                       "forecast origin",
            "lambda_rule": "auto-calibrated on first batch as "
                           "0.1*L_FM/R clipped to [0.01, 1.0], then frozen",
            "coast_normal": "approximate onshore unit vectors, see script",
            "bathymetry": "not explicit; per-region wind coefficient absorbs "
                          "regional fetch effects; GEBCO deferred",
        },
        "regions": {},
    }

    for region in args.regions:
        print(f"\n{'='*70}\nREGION {region}")
        data = load_region(region)
        if args.n_train_fm and len(data["train_h"]) > args.n_train_fm:
            ridx = np.sort(np.random.default_rng(7).choice(
                len(data["train_h"]), size=args.n_train_fm, replace=False))
            for k in ("train_h", "train_t", "train_starts"):
                data[k] = data[k][ridx]
        print(f"  train windows: {len(data['train_h'])}  "
              f"test: {len(data['test_h'])}")

        n_down = min(args.downstream_n_train, len(data["train_h"]))
        down_idx = np.sort(np.random.default_rng(17).choice(
            len(data["train_h"]), size=n_down, replace=False))
        down_data = {"train_h": data["train_h"][down_idx],
                     "train_t": data["train_t"][down_idx],
                     "test_h": data["test_h"],
                     "test_t": data["test_t"]}

        physics = align_physics(region, data["ts"])
        print(f"  ERA5 join missing (before interp): "
              f"{physics.attrs.get('era5_missing_after_join')}")

        # conditioning features (identical for all ERA5 arms)
        tr_feats, stats = build_cond_features(data["train_starts"], physics,
                                              use_era5=True)
        feats = {
            "train": tr_feats,
            "val": build_cond_features(data["val_starts"], physics, True,
                                       stats)[0],
            "test": build_cond_features(data["test_starts"], physics, True,
                                        stats)[0],
        }
        feats_none = {k: np.zeros((len(v), 0), dtype=np.float32)
                      for k, v in feats.items()}

        raw = {s: build_physics_raw(data[f"{s}_starts"], physics, region)
               for s in ("train", "val", "test")}
        d_real = {s: delta_real(data[f"{s}_h"], data[f"{s}_t"])
                  for s in ("train", "val", "test")}

        # A0 baselines (real-only, both forecasters, multi-seed)
        truth = down_data["test_t"]
        ids = load_event_ids(region, len(truth))
        reg_by_arm = {}
        for arm in ARMS:
            names = PENALTY_PREDICTORS[arm]
            reg_by_arm[arm] = fit_physics_regression(
                names, raw["train"], d_real["train"])
        # diagnostics: how much do the physics predictors explain?
        full_pred_test = physics_predict(reg_by_arm["combined_prior"],
                                         raw["test"])
        phys_diag = {}
        if len(full_pred_test):
            phys_diag = physics_consistency(d_real["test"],
                                            d_real["test"],  # placeholder
                                            full_pred_test)
            phys_diag.pop("corr_delta_gen_vs_phys", None)
            phys_diag.pop("corr_deficit", None)

        base_pred = {k: train_downstream(k, down_data, None,
                                         tuple(args.downstream_seeds),
                                         args.downstream_steps,
                                         args.batch_size)
                     for k in ("gru", "tcn")}
        base = {k: (event_level_mae(base_pred[k], truth, ids)
                    if ids is not None else
                    {"all_window_mae": float(np.abs(base_pred[k].mean(0)
                                                    - truth).mean())})
                for k in ("gru", "tcn")}
        for k in ("gru", "tcn"):
            m = base[k].get("extreme_unit_mae", base[k]["all_window_mae"])
            print(f"  A0 {k.upper()} MAE={m:.4f}")

        n_test = min(args.n_test, len(data["test_h"]))
        real = data["test_t"][:n_test]
        arms_out = {}
        samples_store = {}

        for arm in ARMS:
            print(f"\n  ARM {arm}")
            arm_feats = feats_none if arm == "standard" else feats
            per_seed = {}
            aug_for_downstream = None
            fid_keys = ["scale_ratio", "wasserstein1", "q95_exceedance_ratio",
                        "q99_exceedance_ratio", "peak_ks_pvalue",
                        "peak_timing_mae"]
            for seed in args.fm_seeds:
                print(f"    fm seed {seed}")
                model = train_variant(data, arm_feats, raw["train"],
                                      reg_by_arm[arm], arm, seed,
                                      args.epochs, args.batch_size,
                                      args.device)
                torch.save({"net": model["net"].state_dict(),
                            "lambda_phys": model["lambda_phys"],
                            "arm": arm, "seed": seed},
                           CKPT_DIR / f"{region}_{arm}_seed{seed}.pt")

                t0 = time.time()
                gen = sample_windows(model, data["test_h"],
                                     arm_feats["test"], np.arange(n_test),
                                     draws=args.draws_test,
                                     seed_base=1000 * seed)
                infer_s = time.time() - t0
                g_flat = gen.reshape(-1, HORIZON)
                report = evaluate(real, g_flat, train_ref=data["train_t"])
                # physics consistency on generated samples
                d_gen = gen.mean(axis=0)[:, :24].mean(axis=1) \
                    - data["test_h"][:n_test][:, 72:96].mean(axis=1)
                pc = None
                if len(full_pred_test):
                    pc = physics_consistency(d_real["test"][:n_test], d_gen,
                                             full_pred_test[:n_test])
                per_seed[str(seed)] = {
                    "params": model["params"],
                    "physics_dim": model["physics_dim"],
                    "lambda_phys": model["lambda_phys"],
                    "final_train_loss": model["hist"]["loss"][-1],
                    "fidelity_status": report.status,
                    "scale_ratio": float(report.scale_ratio),
                    "wasserstein1": float(report.wasserstein1),
                    "q95_exceedance_ratio": float(report.q95_exceedance_ratio),
                    "q99_exceedance_ratio": float(report.q99_exceedance_ratio),
                    "peak_ks_pvalue": float(report.peak_ks_pvalue),
                    "peak_timing_mae": float(report.peak_timing_mae),
                    "gen_std_ratio": float(np.std(g_flat) / np.std(real)),
                    "physics_consistency": pc,
                    "train_s": round(model["train_s"], 1),
                    "inference_s": round(infer_s, 1),
                    "warnings": list(report.warnings)[:3],
                }
                print(f"      fidelity={report.status} "
                      f"scale={report.scale_ratio:.3f} "
                      f"W1={report.wasserstein1:.4f}")
                samples_store[f"{arm}_seed{seed}"] = gen[0].astype(np.float32)

                # augmentation set (draw 1, own history windows, causal feats)
                n_aug = min(args.n_aug, len(down_data["train_h"]))
                gidx = np.random.default_rng(7).choice(
                    len(down_data["train_h"]), size=n_aug, replace=False)
                tr_phys = arm_feats["train"][down_idx]
                gen_train = sample_windows(model, down_data["train_h"],
                                           tr_phys, gidx, draws=1,
                                           seed_base=2000 * seed)[0]
                # accumulate: average augmentation across fm seeds would
                # double the downstream budget; instead each seed is scored
                # separately and aggregated at the event level below.
                pred = {k: train_downstream(
                    k, down_data,
                    {"hist": down_data["train_h"][gidx], "target": gen_train},
                    tuple(args.downstream_seeds), args.downstream_steps,
                    args.batch_size) for k in ("gru", "tcn")}
                ev = {k: (event_level_mae(pred[k], truth, ids)
                          if ids is not None else
                          {"all_window_mae": float(np.abs(pred[k].mean(0)
                                                          - truth).mean())})
                      for k in ("gru", "tcn")}
                per_seed[str(seed)]["downstream"] = ev
                for k in ("gru", "tcn"):
                    bm = base[k].get("extreme_unit_mae",
                                     base[k]["all_window_mae"])
                    vm = ev[k].get("extreme_unit_mae",
                                   ev[k]["all_window_mae"])
                    print(f"      {k} delta = {(vm - bm) / bm * 100:+.2f}%")

            # aggregate across fm seeds: event-level paired stats vs A0
            agg = {}
            for k in ("gru", "tcn"):
                per_unit_var = np.mean(
                    [np.array(per_seed[s]["downstream"][k]["per_unit_mae"])
                     for s in per_seed], axis=0)
                per_unit_a0 = np.array(base[k]["per_unit_mae"]) \
                    if "per_unit_mae" in base[k] else None
                if per_unit_a0 is not None and len(per_unit_var):
                    stats_ev = paired_event_stats(per_unit_var - per_unit_a0)
                else:
                    stats_ev = None
                deltas_pct = []
                for s in per_seed:
                    bm = base[k].get("extreme_unit_mae",
                                     base[k]["all_window_mae"])
                    vm = per_seed[s]["downstream"][k].get(
                        "extreme_unit_mae",
                        per_seed[s]["downstream"][k]["all_window_mae"])
                    deltas_pct.append((vm - bm) / bm * 100)
                agg[k] = {"mean_delta_pct": float(np.mean(deltas_pct)),
                          "per_seed_delta_pct": deltas_pct,
                          "seed_std_delta_pct": float(np.std(deltas_pct,
                                                             ddof=1))
                          if len(deltas_pct) > 1 else 0.0,
                          "paired_event_stats": stats_ev}
            arms_out[arm] = {"per_fm_seed": per_seed,
                             "augmentation_ratio": float(
                                 min(args.n_aug, n_down) / n_down),
                             "physics_regression": reg_by_arm[arm],
                             "downstream_agg": agg}
            payload["regions"][region] = {
                "n_train_fm": int(len(data["train_h"])),
                "n_train_downstream": int(n_down),
                "n_test": int(len(data["test_h"])),
                "n_eval_windows": int(n_test),
                "era5_join_missing_before_interp": int(
                    physics.attrs.get("era5_missing_after_join", 0)),
                "coast_normal_onshore": list(COAST_NORMAL_ONSHORE[region]),
                "physics_regression_combined": reg_by_arm["combined_prior"],
                "physics_diag_test": phys_diag,
                "fm_seeds": args.fm_seeds,
                "downstream_seeds": args.downstream_seeds,
                "baseline_A0": base,
                "arms": arms_out,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"  [incremental save] {out_path}")

        np.savez_compressed(
            RESULTS / f"wave28_physics_samples_{region}.npz",
            real_test=real.astype(np.float32), **samples_store)

    payload["elapsed_s"] = None
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
