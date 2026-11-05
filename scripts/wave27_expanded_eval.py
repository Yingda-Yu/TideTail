"""Wave 2.7 — expanded three-year evaluation pilot.

Protocol
--------
    2015-2020 train | 2021 validation | 2022-2024 test

* Windows are built AFTER the event/time split. A window is admitted only if
  the whole 96 h history AND the whole 72 h target lie inside one split, with
  no overlap across the boundary.
* Preprocessing uses per-sample statistics only; feature/threshold fitting is
  train-only.
* Model selection uses the validation split only (best validation FM loss).
* The test split is touched once, at the end.

Pilot scope: Standard Flow Matching vs A0 real-only, charleston + ny_bight.

Event units: POT q95 clusters (168 h declustered) from
`data/manifests/multiyear_declustering.json`, relabelled per year because
`cluster_id` restarts each year. Canonical-TC overlay for 2022-2023 is NOT
available in the existing prediction archives; units here are POT-only and this
limitation is carried into the report.

Usage:
    python3 scripts/wave27_expanded_eval.py --regions south_atlantic_charleston ny_bight
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tidetail.models.flow_matching import FlowMatching, VelocityTCN  # noqa: E402
import wave27_physics_fm as base  # noqa: E402
from paired_significance_test import paired_bootstrap, paired_randomization  # noqa: E402

RESULTS = ROOT / "data" / "results"
DECLUSTERING = ROOT / "data" / "manifests" / "multiyear_declustering.json"
CONTEXT, HORIZON = 96, 72
BOUNDS = ("2015-01-01", "2021-01-01", "2022-01-01", "2025-01-01")


def build_expanded_splits(ts: pd.Series, series: np.ndarray) -> dict:
    """Strict-containment window starts for 2015-2020 / 2021 / 2022-2024."""
    marks = [int(ts.searchsorted(pd.Timestamp(b, tz="UTC"))) for b in BOUNDS]
    n = len(series)
    span = CONTEXT + HORIZON

    def starts(lo: int, hi: int) -> np.ndarray:
        last = hi - span
        return np.arange(lo, last + 1, dtype=np.int64) if last >= lo else np.empty(0, np.int64)

    def windows(s: np.ndarray):
        h = np.stack([series[i:i + CONTEXT] for i in s]) if len(s) else np.empty((0, CONTEXT), np.float32)
        t = np.stack([series[i + CONTEXT:i + span] for i in s]) if len(s) else np.empty((0, HORIZON), np.float32)
        return h.astype(np.float32), t.astype(np.float32)

    out = {}
    for name, (lo, hi) in {
        "train": (marks[0], marks[1]),
        "val": (marks[1], marks[2]),
        "test": (marks[2], min(marks[3], n)),
    }.items():
        s = starts(lo, hi)
        h, t = windows(s)
        out[f"{name}_h"], out[f"{name}_t"], out[f"{name}_starts"] = h, t, s
    # Explicit boundary hygiene report
    out["boundary"] = {
        "train_last_target_end": int(out["train_starts"].max()) + span - 1,
        "val_first_history_start": int(out["val_starts"].min()),
        "val_last_target_end": int(out["val_starts"].max()) + span - 1,
        "test_first_history_start": int(out["test_starts"].min()),
        "gap_required_hours": span,
        "train_val_gap_ok": bool(out["train_starts"].max() + span - 1 < out["val_starts"].min()),
        "val_test_gap_ok": bool(out["val_starts"].max() + span - 1 < out["test_starts"].min()),
    }
    return out


def pot_units(region: str, origins: np.ndarray) -> np.ndarray:
    """POT-cluster independent units with year-unique labels."""
    decl = json.loads(DECLUSTERING.read_text())[region]["pot_clusters"]
    years = pd.DatetimeIndex(origins).year
    units = np.full(len(origins), -999, dtype=np.int64)
    origin_idx = pd.DatetimeIndex(origins)
    if origin_idx.tz is not None:
        origin_idx = origin_idx.tz_convert("UTC").tz_localize(None)
    origins_ns = origin_idx.asi8
    for year in np.unique(years):
        clusters = decl.get(str(int(year)), [])
        for c in clusters:
            start = int(pd.Timestamp(c["start"]).value)
            end = int(pd.Timestamp(c["end"]).value)
            mask = (years == int(year)) & (origins_ns >= start) & (origins_ns <= end)
            units[mask] = -(int(year) * 1000 + int(c["cluster_id"]) + 1)
    return units


def train_fm_with_selection(data: dict, epochs: int, batch_size: int, lr: float,
                            seed: int, device: str) -> dict:
    """Train Standard FM; select epoch by VALIDATION loss only."""
    torch.manual_seed(seed)
    fm = FlowMatching(context=CONTEXT, horizon=HORIZON, euler_steps=64, device=device)
    net = VelocityTCN(context=CONTEXT, horizon=HORIZON, width=32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def norm(h, t):
        m = h.mean(axis=1, keepdims=True)
        s = h.std(axis=1, keepdims=True).clip(min=1e-4)
        return (h - m) / s, (t - m) / s

    tr_h, tr_t = norm(data["train_h"], data["train_t"])
    va_h, va_t = norm(data["val_h"], data["val_t"])
    Xh = torch.from_numpy(tr_h).to(device)
    Xt = torch.from_numpy(tr_t).to(device)
    Vh = torch.from_numpy(va_h).to(device)
    Vt = torch.from_numpy(va_t).to(device)

    best, best_state, hist = float("inf"), None, []
    n = len(Xh)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            loss = fm.training_loss(Xh[idx], Xt[idx], net)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            vl = float(fm.training_loss(Vh, Vt, net))
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_loss": vl})
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"    epoch {ep+1}/{epochs} train={tot/n:.4f} val={vl:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    return {"net": net, "fm": fm, "physics_dim": 0, "best_val_loss": best,
            "history": hist,
            "params": int(sum(p.numel() for p in net.parameters())),
            "selected_epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["south_atlantic_charleston", "ny_bight"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--downstream-n-train", type=int, default=8000)
    ap.add_argument("--downstream-steps", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    payload = {
        "audit_version": "wave27_expanded_v1",
        "protocol": "2015-2020 train / 2021 validation / 2022-2024 test",
        "selection": "validation-only (best FM validation loss)",
        "event_units": "POT q95 clusters, year-unique labels; no TC overlay",
        "regions": {},
    }

    for region in args.regions:
        print(f"\n{'='*70}\nREGION {region}")
        raw = base.load_region(region)
        ts, series = raw["ts"], raw["series"]
        spl = build_expanded_splits(ts, series)
        print("  window counts:", {k: len(spl[f'{k}_h']) for k in ("train", "val", "test")})
        print("  boundary:", spl["boundary"])
        if not (spl["boundary"]["train_val_gap_ok"] and spl["boundary"]["val_test_gap_ok"]):
            raise RuntimeError("split boundary overlap")

        model = train_fm_with_selection(spl, args.epochs, args.batch_size, args.lr,
                                        args.seed, args.device)

        # ---- test-set fidelity (touch once) ----
        n_test = min(args.n_test, len(spl["test_h"]))
        phys = np.zeros((n_test, 0), dtype=np.float32)
        gen = base.sample_windows(model, spl["test_h"], phys, np.arange(n_test), draws=3)
        real = spl["test_t"][:n_test]
        rep = base.evaluate(real, gen.reshape(-1, HORIZON), train_ref=spl["train_t"])
        fid = {"status": rep.status, "scale_ratio": float(rep.scale_ratio),
               "wasserstein1": float(rep.wasserstein1),
               "autocorr_rmse": float(rep.autocorr_rmse),
               "q95_exceed_ratio": float(rep.q95_exceedance_ratio)}

        # ---- downstream, fixed budget across arms ----
        n_down = min(args.downstream_n_train, len(spl["train_h"]))
        down_idx = np.sort(np.random.default_rng(17).choice(len(spl["train_h"]),
                                                            size=n_down, replace=False))
        down = {"train_h": spl["train_h"][down_idx], "train_t": spl["train_t"][down_idx],
                "test_h": spl["test_h"], "test_t": spl["test_t"]}
        a0 = base.train_downstream(down, None, steps=args.downstream_steps,
                                   batch_size=args.batch_size)

        n_aug = min(args.n_aug, n_down)
        gidx = np.random.default_rng(7).choice(n_down, size=n_aug, replace=False)
        gen_train = base.sample_windows(model, down["train_h"], phys, gidx, draws=1)[0]
        augmented = base.train_downstream(
            down, {"hist": down["train_h"][gidx], "target": gen_train},
            steps=args.downstream_steps, batch_size=args.batch_size)

        origins = ts.to_numpy()[spl["test_starts"] + CONTEXT]
        units = pot_units(region, origins)
        ev_a0 = base.event_level_mae(a0, spl["test_t"], units)
        ev_fm = base.event_level_mae(augmented, spl["test_t"], units)
        per_unit = np.array(ev_fm["per_unit_mae"]) - np.array(ev_a0["per_unit_mae"])
        bs = paired_bootstrap(per_unit, args.n_boot, 42)
        pm = paired_randomization(per_unit, args.n_boot, 43)

        payload["regions"][region] = {
            "window_counts": {k: int(len(spl[f"{k}_h"])) for k in ("train", "val", "test")},
            "split_boundary": spl["boundary"],
            "fm_selected_epoch": model["selected_epoch"],
            "fm_best_val_loss": float(model["best_val_loss"]),
            "fidelity": fid,
            "augmentation_ratio": float(n_aug / n_down),
            "A0_real_only": ev_a0,
            "FM_augmented": ev_fm,
            "delta_extreme_mae_pct": float(
                (ev_fm["extreme_unit_mae"] - ev_a0["extreme_unit_mae"]) / ev_a0["extreme_unit_mae"] * 100),
            "paired_ci95": [bs["ci95_low"], bs["ci95_high"]],
            "paired_p_boot": bs["p_value"],
            "paired_p_perm": pm["p_value"],
            "n_independent_units": int(ev_a0["n_extreme_units"]),
            "multiplicity_note": (
                "Two regions x one candidate = 2 tests. Bonferroni alpha = 0.025 "
                "is reported alongside raw p; this pilot is not powered for "
                "confirmatory claims at 2022-2024 event counts."),
        }
        print(f"  A0 extreme MAE={ev_a0['extreme_unit_mae']:.4f} "
              f"FM={ev_fm['extreme_unit_mae']:.4f} "
              f"delta={payload['regions'][region]['delta_extreme_mae_pct']:+.2f}% "
              f"p_boot={bs['p_value']:.4f} p_perm={pm['p_value']:.4f}")

    payload["elapsed_s"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "wave27_expanded_evaluation.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
