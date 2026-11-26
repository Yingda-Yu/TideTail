"""Wave 2.8 P2 — expanded-year 6-cell comparison (2015-2020/2021/2022-2024).

Cells per region (identical windows, budgets, frozen augmentation ratio):

    A0 real-only        x {GRU, TCN}
    Standard FM aug     x {GRU, TCN}
    Tail-Aware FM aug   x {GRU, TCN}

Generators are retrained from scratch on 2015-2020, selected by
VALIDATION (2021) loss only; the 2022-2024 test split is touched once at
the end for fidelity + downstream scoring.

Multi-seed: FM seeds {42, 99}; downstream seeds {42, 99, 7}. Event-level
paired differences vs A0 with sign-flip randomisation p-values and paired
bootstrap CIs (null-based test per Wave 2.8 P4; the bootstrap is used
only for the CI, never as the null).

Retest disclosure: the 2022-2024 period partially informed earlier
exploratory waves, so this evaluation is reported as retrospective
validation, not untouched confirmatory inference.

Usage:
    python3 scripts/wave28_expanded_eval.py \
        --regions south_atlantic_charleston ny_bight

Output:
    data/results/wave28_expanded_eval.json (incremental, per cell)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("wave27_physics_fm", ROOT / "scripts" / "wave27_physics_fm.py")
w27e = _load("wave27_expanded_eval", ROOT / "scripts" / "wave27_expanded_eval.py")
w28 = _load("wave28_physics_fm", ROOT / "scripts" / "wave28_physics_fm.py")

from tidetail.models.flow_matching import FlowMatching, VelocityTCN  # noqa: E402
from tidetail.physics import tail_weight  # noqa: E402
from tidetail.quality import evaluate  # noqa: E402

RESULTS = ROOT / "data" / "results"
CONTEXT, HORIZON = 96, 72
VARIANTS = ("standard", "tail_aware")


def train_fm_val_selected(data: dict, tail: bool, epochs: int,
                          batch_size: int, lr: float, seed: int,
                          device: str) -> dict:
    """Standard or tail-aware FM; epoch selected by UNWEIGHTED validation
    loss on 2021 only (identical selection rule across variants)."""
    torch.manual_seed(seed)
    fm = FlowMatching(context=CONTEXT, horizon=HORIZON, euler_steps=64,
                      device=device)
    net = VelocityTCN(context=CONTEXT, horizon=HORIZON, width=32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def norm(h, t):
        m = h.mean(axis=1, keepdims=True)
        s = h.std(axis=1, keepdims=True).clip(min=1e-4)
        return (h - m) / s, (t - m) / s

    tr_h, tr_t = norm(data["train_h"], data["train_t"])
    va_h, va_t = norm(data["val_h"], data["val_t"])
    Xh, Xt = torch.from_numpy(tr_h).to(device), torch.from_numpy(tr_t).to(device)
    Vh, Vt = torch.from_numpy(va_h).to(device), torch.from_numpy(va_t).to(device)
    tail_w = None
    if tail:
        flat = np.abs(data["train_t"].reshape(-1))
        q95, q99 = np.percentile(flat, 95), np.percentile(flat, 99)
        tail_w = torch.from_numpy(
            tail_weight(np.abs(data["train_t"]), q95, q99).mean(axis=1)
            .astype(np.float32)).to(device)

    best, best_state, hist = float("inf"), None, []
    n = len(Xh)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            hb, tb = Xh[idx], Xt[idx]
            B = hb.shape[0]
            tt = torch.rand(B, 1, 1, device=device)
            x0 = torch.randn(B, 1, HORIZON, device=device)
            x1 = tb.unsqueeze(1)
            xt = (1 - tt) * x0 + tt * x1
            v = net(xt, tt.squeeze(-1).squeeze(-1), hb)
            if tail_w is None:
                loss = torch.nn.functional.mse_loss(v, x1 - x0)
            else:
                per = ((v - (x1 - x0)) ** 2).mean(dim=(1, 2))
                loss = (per * tail_w[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            vl = float(fm.training_loss(Vh, Vt, net))   # unweighted
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_loss": vl})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone()
                                    for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return {"net": net, "fm": fm, "physics_dim": 0,
            "best_val_loss": best, "history": hist,
            "selected_epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1)}


def _atomic_write(path: Path, obj) -> None:
    """Atomic JSON replace: never leave a truncated results file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _cell_complete(region_data: dict, variant: str, fm_seeds) -> bool:
    """A variant cell is complete when every requested fm_seed has both
    gru and tcn downstream results (Wave 2.9 P0-05 resume contract)."""
    var = region_data.get("variants", {}).get(variant, {})
    per_seed = var.get("per_fm_seed", {})
    for s in fm_seeds:
        sd = per_seed.get(str(s))
        if not sd:
            return False
        cells = sd.get("downstream", {})
        for k in ("gru", "tcn"):
            c = cells.get(k)
            if not c or c.get("extreme_unit_mae") is None:
                return False
            if not np.all(np.isfinite(
                    np.asarray(c.get("per_unit_mae", [0.0]), dtype=float))):
                return False
    return True


def _a0_protocol_matches(region_data: dict, args) -> bool:
    """Reuse A0 only when protocol fields are identical."""
    rd = region_data
    return (rd.get("fm_seeds") == list(args.fm_seeds)
            and rd.get("downstream_seeds") == list(args.downstream_seeds)
            and bool(rd.get("A0", {}).get("gru", {}).get("extreme_unit_mae"))
            and bool(rd.get("A0", {}).get("tcn", {}).get("extreme_unit_mae")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["south_atlantic_charleston", "ny_bight"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fm-seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--downstream-seeds", type=int, nargs="+",
                    default=[42, 99, 7])
    ap.add_argument("--downstream-n-train", type=int, default=8000)
    ap.add_argument("--downstream-steps", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true",
                    help="load existing wave28_expanded_eval.json and skip "
                         "already-complete region/variant cells")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute the requested regions from scratch "
                         "(deterministic seeds reproduce identical results "
                         "while persisting per-unit arrays); takes "
                         "precedence over --resume for these regions")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    out_path = RESULTS / "wave28_expanded_eval.json"
    if (args.resume or args.recompute) and out_path.exists():
        payload = json.loads(out_path.read_text())
        print(f"[RESUME] Loaded {out_path}")
    else:
        payload = {
        "audit_version": "wave28_expanded_v1",
        "protocol": "2015-2020 train / 2021 validation / 2022-2024 test",
        "selection": "validation-only (unweighted 2021 FM loss, identical "
                     "across variants)",
        "retest_disclosure": "2022-2024 partially informed earlier "
                             "exploratory waves; reported as retrospective "
                             "validation, not untouched confirmatory",
        "cells": "A0 x {GRU,TCN}; standard FM x {GRU,TCN}; tail-aware FM "
                 "x {GRU,TCN} per region",
        "regions": {},
    }

    for region in args.regions:
        existing_rd = payload["regions"].get(region, {}) \
            if args.resume and not args.recompute else {}
        if args.resume and existing_rd:
            variants_todo = [v for v in VARIANTS
                             if not _cell_complete(existing_rd, v,
                                                   args.fm_seeds)]
            reuse_a0 = _a0_protocol_matches(existing_rd, args)
            if not variants_todo and reuse_a0:
                print(f"\n[RESUME] {region}: all cells complete — skipping")
                continue
            print(f"\n[RESUME] {region}: pending variants {variants_todo}, "
                  f"A0 reuse={reuse_a0}")
        else:
            variants_todo = list(VARIANTS)
            reuse_a0 = False
        print(f"\n{'='*70}\nREGION {region}")
        raw = base.load_region(region)
        ts, series = raw["ts"], raw["series"]
        spl = w27e.build_expanded_splits(ts, series)
        print("  windows:", {k: len(spl[f"{k}_h"]) for k in ("train", "val", "test")})
        print("  nan_windows_excluded:",
              {k: spl[f"{k}_nan_windows_excluded"] for k in ("train", "val", "test")})
        for k in ("train", "val", "test"):
            if not (np.isfinite(spl[f"{k}_h"]).all()
                    and np.isfinite(spl[f"{k}_t"]).all()):
                raise RuntimeError(f"{k} split contains non-finite samples")
        if not (spl["boundary"]["train_val_gap_ok"]
                and spl["boundary"]["val_test_gap_ok"]):
            raise RuntimeError("split boundary overlap")

        n_down = min(args.downstream_n_train, len(spl["train_h"]))
        down_idx = np.sort(np.random.default_rng(17).choice(
            len(spl["train_h"]), size=n_down, replace=False))
        down = {"train_h": spl["train_h"][down_idx],
                "train_t": spl["train_t"][down_idx],
                "test_h": spl["test_h"], "test_t": spl["test_t"]}
        truth = down["test_t"]
        origins = ts.to_numpy()[spl["test_starts"] + CONTEXT]
        units = w27e.pot_units(region, origins)

        # ---- A0 cells (both forecasters, multi-seed) ----
        if reuse_a0:
            a0 = existing_rd["A0"]
            print("  [RESUME] A0 reused (protocol fields match)")
        else:
            a0 = {}
            for k in ("gru", "tcn"):
                pred = w28.train_downstream(k, down, None,
                                            tuple(args.downstream_seeds),
                                            args.downstream_steps,
                                            args.batch_size)
                a0[k] = w28.event_level_mae(pred, truth, units)
                print(f"  A0 {k.upper()} extreme MAE="
                      f"{a0[k]['extreme_unit_mae']:.4f} "
                      f"({a0[k]['n_extreme_units']} units)")

        # ---- generator cells ----
        n_test = min(args.n_test, len(spl["test_h"]))
        real = spl["test_t"][:n_test]
        n_aug = min(args.n_aug, n_down)
        gidx = np.random.default_rng(7).choice(n_down, size=n_aug,
                                               replace=False)
        variants_out = {}
        for variant in VARIANTS:
            if variant not in variants_todo and args.resume and existing_rd:
                variants_out[variant] = existing_rd["variants"][variant]
                continue
            per_seed = {}
            for seed in args.fm_seeds:
                print(f"  {variant} fm seed {seed}")
                model = train_fm_val_selected(
                    spl, tail=(variant == "tail_aware"), epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, seed=seed,
                    device=args.device)
                gen = base.sample_windows(model, spl["test_h"],
                                          np.zeros((n_test, 0), np.float32),
                                          np.arange(n_test), draws=3)
                tr_ref = spl["train_t"][:2000]
                rep = evaluate(real, gen.reshape(-1, HORIZON),
                               train_ref=tr_ref)
                gen_train = base.sample_windows(
                    model, down["train_h"],
                    np.zeros((n_down, 0), np.float32), gidx, draws=1)[0]
                aug = {"hist": down["train_h"][gidx], "target": gen_train}
                cells = {}
                for k in ("gru", "tcn"):
                    pred = w28.train_downstream(
                        k, down, aug, tuple(args.downstream_seeds),
                        args.downstream_steps, args.batch_size)
                    ev = w28.event_level_mae(pred, truth, units)
                    delta_pct = (ev["extreme_unit_mae"]
                                 - a0[k]["extreme_unit_mae"]) \
                        / a0[k]["extreme_unit_mae"] * 100
                    stats_ev = w28.paired_event_stats(
                        np.array(ev["per_unit_mae"])
                        - np.array(a0[k]["per_unit_mae"]))
                    cells[k] = {"extreme_unit_mae": ev["extreme_unit_mae"],
                                "delta_pct_vs_A0": float(delta_pct),
                                "paired_event_stats": stats_ev,
                                # persist per-unit arrays for auditability
                                "per_unit_mae": ev["per_unit_mae"],
                                "event_ids": sorted(
                                    set(units[units != -999].tolist()))}
                    p_txt = (f"{stats_ev['signflip_p']:.4f}"
                             if stats_ev.get("valid") else "INVALID")
                    print(f"    {k} delta {delta_pct:+.2f}% "
                          f"p_signflip={p_txt} "
                          f"fid={rep.status} scale={rep.scale_ratio:.3f}")
                per_seed[str(seed)] = {
                    "selected_epoch": model["selected_epoch"],
                    "best_val_loss": model["best_val_loss"],
                    "fidelity": {"status": rep.status,
                                 "scale_ratio": float(rep.scale_ratio),
                                 "wasserstein1": float(rep.wasserstein1),
                                 "q95_exceed_ratio":
                                     float(rep.q95_exceedance_ratio),
                                 "q99_exceed_ratio":
                                     float(rep.q99_exceedance_ratio)},
                    "downstream": cells,
                }
            agg = {}
            for k in ("gru", "tcn"):
                deltas = [per_seed[s]["downstream"][k]["delta_pct_vs_A0"]
                          for s in per_seed]
                agg[k] = {"mean_delta_pct": float(np.mean(deltas)),
                          "per_seed_delta_pct": deltas,
                          "seed_std": float(np.std(deltas, ddof=1))
                          if len(deltas) > 1 else 0.0}
            variants_out[variant] = {"per_fm_seed": per_seed,
                                     "augmentation_ratio": n_aug / n_down,
                                     "downstream_agg": agg}
            payload["regions"][region] = {
                "window_counts": {k: int(len(spl[f"{k}_h"]))
                                  for k in ("train", "val", "test")},
                "nan_windows_excluded":
                    {k: int(spl[f"{k}_nan_windows_excluded"])
                     for k in ("train", "val", "test")},
                "split_boundary": spl["boundary"],
                "fm_seeds": args.fm_seeds,
                "downstream_seeds": args.downstream_seeds,
                "A0": a0,
                "variants": variants_out,
                "multiplicity_note":
                    "2 regions x 2 variants x 2 forecasters = 8 paired "
                    "comparisons; report Bonferroni alpha=0.00625 alongside "
                    "raw p-values. All comparisons are exploratory/retro-"
                    "spective (see retest_disclosure).",
            }
            _atomic_write(out_path, payload)
            print(f"  [incremental save] {out_path}")

    payload["elapsed_s"] = round(time.time() - t0, 1)
    _atomic_write(out_path, payload)
    print(f"\nSaved -> {out_path} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
