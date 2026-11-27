#!/usr/bin/env python3
"""Wave 3.2 P3 — bounded tail-calibration study (exploratory extension).

Mandate (owner, Wave 3.2 PART 4):
  * no new model families; at most six arms:
      1 standard FM              (existing generator)
      2 tail-aware FM            (existing generator)
      3 one quantile-calibration candidate   (post-hoc ECDF mapping, NEW)
      4 one fidelity-aware sample-selection candidate (NEW)
      5 real-event oversampling              (no generator)
      6 simple history-noise control         (no generator)
  * thresholds/hyperparameters chosen on train/validation ONLY; the method is
    frozen (stage A result written to disk) BEFORE any test evaluation.
  * contrasting regions only: southeast_australia (augmentation helps) and
    thames_north_sea (augmentation harms).
  * downstream: GRU only (bounded, declared a priori); frozen five-region
    calendar protocol (2015-2020/2021/2022-2024); event-level paired stats.
  * exploratory extension, not confirmatory: 2022-2024 is retrospective.

Output: data/results/wave32_tail_calibration.json (atomic, incremental).
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
we = _load("wave28_expanded_eval", ROOT / "scripts" / "wave28_expanded_eval.py")

from tidetail.quality import evaluate  # noqa: E402

RESULTS = ROOT / "data" / "results"
CONTEXT, HORIZON = 96, 72
REGIONS = ["southeast_australia", "thames_north_sea"]
ARMS = ["standard_fm", "tail_aware_fm", "quantile_calibrated_fm",
        "fidelity_selected_fm", "oversampling", "history_noise"]


# ---------------------------------------------------------------- candidates
def quantile_map(gen: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-horizon-position ECDF mapping of generated samples onto the train
    target distribution. Fitted on train targets only (no test data)."""
    out = np.empty_like(gen)
    probs_grid = np.linspace(0, 1, 201)
    for j in range(gen.shape[1]):
        g = gen[:, j]
        ref_q = np.quantile(ref[:, j], probs_grid)
        ranks = (np.argsort(np.argsort(g)) + 0.5) / len(g)
        out[:, j] = np.interp(ranks, probs_grid, ref_q)
    return out


def fidelity_select(gen_cands: np.ndarray, ref: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Fidelity-aware sample selection: generate k candidates per window and
    keep the first draw whose window statistics fall inside a train-derived
    plausibility band (window std within [q05, q95] of train window stds;
    window mean within train mean +/- 3 MAD). Falls back to the least
    violating candidate. Band fitted on train targets only."""
    stds = ref.std(axis=1)
    means = ref.mean(axis=1)
    mad = np.median(np.abs(means - np.median(means))) * 1.4826
    lo_s, hi_s = np.quantile(stds, 0.05), np.quantile(stds, 0.95)
    mu = np.median(means)
    k, n, _ = gen_cands.shape
    out = np.empty((n, HORIZON), dtype=np.float32)
    n_pass = 0
    for i in range(n):
        cands = gen_cands[:, i, :]
        ok = [(lo_s <= c.std() <= hi_s) and (abs(c.mean() - mu) <= 3 * mad)
              for c in cands]
        take = next((j for j, o in enumerate(ok) if o), None)
        if take is None:
            viol = [max(abs(c.std() - np.clip(c.std(), lo_s, hi_s)),
                        abs(abs(c.mean() - mu) - 3 * mad)) for c in cands]
            take = int(np.argmin(viol))
        else:
            n_pass += 1
        out[i] = cands[take]
    return out, {"selection_pass_rate": n_pass / n, "n_candidates": k}


def oversample_aug(down: dict, n_aug: int, rng) -> dict:
    """Real-event oversampling: duplicate train windows whose target peak
    exceeds the train q95 (targets are REAL values, no generator)."""
    peaks = np.abs(down["train_t"]).max(axis=1)
    q95 = np.quantile(peaks, 0.95)
    idx = np.where(peaks >= q95)[0]
    if len(idx) == 0:
        idx = np.argsort(peaks)[-n_aug:]
    sel = rng.choice(idx, size=n_aug, replace=True)
    return {"hist": down["train_h"][sel], "target": down["train_t"][sel],
            "source": "real_train_q95_windows", "n_unique_source": len(idx)}


def history_noise_aug(down: dict, n_aug: int, rng) -> dict:
    """Simple history-noise control: real targets, history perturbed with
    N(0, 0.1 * per-window std) (sigma ratio pre-registered, not tuned)."""
    idx = rng.choice(len(down["train_h"]), size=n_aug, replace=False)
    h = down["train_h"][idx].astype(np.float64)
    sigma = 0.1 * h.std(axis=1, keepdims=True)
    h = h + rng.normal(0.0, 1.0, size=h.shape) * sigma
    return {"hist": h.astype(np.float32), "target": down["train_t"][idx],
            "source": "real_train_windows", "sigma_ratio": 0.1}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fm-seeds", type=int, nargs="+", default=[42, 99])
    ap.add_argument("--downstream-seeds", type=int, nargs="+",
                    default=[42, 99, 7])
    ap.add_argument("--downstream-n-train", type=int, default=8000)
    ap.add_argument("--downstream-steps", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-cand", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    out_path = RESULTS / "wave32_tail_calibration.json"
    payload = {
        "audit_version": "wave32_tail_calibration_v1",
        "classification": "exploratory extension (retrospective test years; "
                          "NOT confirmatory)",
        "protocol": "2015-2020 train / 2021 validation / 2022-2024 test "
                    "(frozen five-region calendar protocol)",
        "freezing_rule": "stage A: candidates evaluated on validation "
                         "fidelity only; adopted iff composite "
                         "|log(scale)|+|log(q95r)| improves >=10% over raw "
                         "FM; frozen decision written before test stage",
        "multiplicity": "2 regions x 6 arms = 12 GRU paired comparisons; "
                        "exploratory; Bonferroni alpha=0.05/12=0.0042 "
                        "reported alongside raw p",
        "regions": {},
    }
    if out_path.exists():
        payload.update(json.loads(out_path.read_text()))
        print(f"[RESUME] Loaded {out_path}")

    for region in REGIONS:
        rd = payload["regions"].get(region, {})
        if rd.get("complete"):
            print(f"\n[RESUME] {region} complete — skipping")
            continue
        print(f"\n{'='*70}\nREGION {region}")
        raw = base.load_region(region)
        ts, series = raw["ts"], raw["series"]
        spl = w27e.build_expanded_splits(ts, series)
        for k in ("train", "val", "test"):
            assert np.isfinite(spl[f"{k}_h"]).all(), f"{k} non-finite"
        assert spl["boundary"]["train_val_gap_ok"] \
            and spl["boundary"]["val_test_gap_ok"]

        n_down = min(args.downstream_n_train, len(spl["train_h"]))
        down_idx = np.sort(np.random.default_rng(17).choice(
            len(spl["train_h"]), size=n_down, replace=False))
        down = {"train_h": spl["train_h"][down_idx],
                "train_t": spl["train_t"][down_idx],
                "test_h": spl["test_h"], "test_t": spl["test_t"]}
        truth = down["test_t"]
        origins = ts.to_numpy()[spl["test_starts"] + CONTEXT]
        units = w27e.pot_units(region, origins)
        tr_ref = spl["train_t"][:2000]

        # ---------- stage A: validation-only candidate selection ----------
        n_val = min(args.n_val, len(spl["val_h"]))
        real_val = spl["val_t"][:n_val]
        stage_a = {}
        per_seed_gen = {}
        for seed in args.fm_seeds:
            print(f"  [stage A] standard FM seed {seed}")
            model = we.train_fm_val_selected(
                spl, tail=False, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, seed=seed,
                device=args.device)
            per_seed_gen[seed] = model
            gval = base.sample_windows(
                model, spl["val_h"], np.zeros((n_val, 0), np.float32),
                np.arange(n_val), draws=3)
            cands = {}
            cands["standard_fm"] = gval.reshape(-1, HORIZON)
            cands["quantile_calibrated_fm"] = np.concatenate([
                quantile_map(gval[d], spl["train_t"]) for d in range(3)])
            gs6 = base.sample_windows(
                model, spl["val_h"], np.zeros((n_val, 0), np.float32),
                np.arange(n_val), draws=args.n_cand)
            sel, sel_info = fidelity_select(gs6, spl["train_t"],
                                            np.random.default_rng(seed))
            cands["fidelity_selected_fm"] = sel
            rep = {}
            for name, gmat in cands.items():
                r = evaluate(real_val, gmat, train_ref=tr_ref)
                rep[name] = {"status": r.status,
                             "scale_ratio": float(r.scale_ratio),
                             "wasserstein1": float(r.wasserstein1),
                             "q95_exceed_ratio":
                                 float(r.q95_exceedance_ratio),
                             "q99_exceed_ratio":
                                 float(r.q99_exceedance_ratio)}
                print(f"    val {name:24s} scale={r.scale_ratio:.3f} "
                      f"W1={r.wasserstein1:.3f} q95r={r.q95_exceedance_ratio:.3f}")
            # frozen decision rule
            def composite(x):
                return abs(np.log(max(x["scale_ratio"], 1e-6))) + \
                    abs(np.log(max(x["q95_exceed_ratio"], 1e-6)))
            raw_c = composite(rep["standard_fm"])
            best, best_c, adopted = None, None, {}
            for c in ("quantile_calibrated_fm", "fidelity_selected_fm"):
                cc = composite(rep[c])
                adopted[c] = bool(cc <= 0.9 * raw_c)
                if adopted[c] and (best is None or cc < best_c):
                    best, best_c = c, cc
            decision = {"adopted": adopted, "selected_candidate": best,
                        "raw_composite": raw_c,
                        "sel_pass_rate_val": sel_info["selection_pass_rate"]}
            stage_a[str(seed)] = {"val_fidelity": rep, "decision": decision}
            print(f"    [FROZEN] seed {seed}: adopted={adopted} "
                  f"selected={best}")
        rd["stage_a"] = stage_a
        payload["regions"][region] = rd
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out_path)   # freeze BEFORE test evaluation
        print("  [FROZEN] stage A written; test stage begins")

        # ---------- stage B: frozen test evaluation (GRU only) ----------
        model_cache = {("standard", s): per_seed_gen[s] for s in per_seed_gen}
        n_test = len(spl["test_h"])
        real_test = spl["test_t"][:n_test]
        n_aug = min(args.n_aug, n_down)
        gidx = np.random.default_rng(7).choice(n_down, size=n_aug,
                                               replace=False)
        pred_a0 = w28.train_downstream("gru", down, None,
                                       tuple(args.downstream_seeds),
                                       args.downstream_steps, args.batch_size)
        ev_a0 = w28.event_level_mae(pred_a0, truth, units)
        print(f"  A0 GRU extreme MAE={ev_a0['extreme_unit_mae']:.4f} "
              f"({ev_a0['n_extreme_units']} units)")

        arms_out = {}
        for arm in ARMS:
            per_seed = {}
            for seed in args.fm_seeds:
                if arm in ("oversampling", "history_noise"):
                    rng = np.random.default_rng(1000 + seed)
                    aug = (oversample_aug(down, n_aug, rng) if arm ==
                           "oversampling" else
                           history_noise_aug(down, n_aug, rng))
                    gen_test = None
                else:
                    fam = "tail" if arm == "tail_aware_fm" else "standard"
                    model = model_cache.get((fam, seed))
                    if model is None:
                        model = we.train_fm_val_selected(
                            spl, tail=(arm == "tail_aware_fm"),
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, seed=seed, device=args.device)
                        model_cache[(fam, seed)] = model
                    gidx_arm = gidx
                    if arm == "quantile_calibrated_fm":
                        gtr = base.sample_windows(
                            model, down["train_h"],
                            np.zeros((n_down, 0), np.float32), gidx_arm,
                            draws=1)[0]
                        tgt = quantile_map(gtr, spl["train_t"])
                        gen_test = None
                    elif arm == "fidelity_selected_fm":
                        gtr6 = base.sample_windows(
                            model, down["train_h"],
                            np.zeros((n_down, 0), np.float32), gidx_arm,
                            draws=args.n_cand)
                        tgt, sel_info_tr = fidelity_select(
                            gtr6, spl["train_t"],
                            np.random.default_rng(seed))
                        gen_test = None
                    else:
                        gtr = base.sample_windows(
                            model, down["train_h"],
                            np.zeros((n_down, 0), np.float32), gidx_arm,
                            draws=1)[0]
                        tgt = gtr
                    aug = {"hist": down["train_h"][gidx], "target": tgt}
                pred = w28.train_downstream(
                    "gru", down, aug, tuple(args.downstream_seeds),
                    args.downstream_steps, args.batch_size)
                ev = w28.event_level_mae(pred, truth, units)
                delta_pct = (ev["extreme_unit_mae"]
                             - ev_a0["extreme_unit_mae"]) \
                    / ev_a0["extreme_unit_mae"] * 100
                stats = w28.paired_event_stats(
                    np.array(ev["per_unit_mae"])
                    - np.array(ev_a0["per_unit_mae"]))
                per_seed[str(seed)] = {
                    "extreme_unit_mae": ev["extreme_unit_mae"],
                    "delta_pct_vs_A0": float(delta_pct),
                    "paired_event_stats": stats,
                    "per_unit_mae": ev["per_unit_mae"],
                    "aug_source": aug.get("source", "generator")}
                p_txt = (f"{stats['signflip_p']:.4f}"
                         if stats.get("valid") else "INVALID")
                print(f"    {arm:24s} seed {seed} delta {delta_pct:+.2f}% "
                      f"p={p_txt}")
            deltas = [per_seed[s]["delta_pct_vs_A0"] for s in per_seed]
            arms_out[arm] = {
                "per_seed": per_seed,
                "mean_delta_pct": float(np.mean(deltas)),
                "seed_std": float(np.std(deltas, ddof=1))
                if len(deltas) > 1 else 0.0}
        rd["stage_b"] = {"arms": arms_out,
                         "A0_gru": {"extreme_unit_mae":
                                    ev_a0["extreme_unit_mae"],
                                    "per_unit_mae": ev_a0["per_unit_mae"],
                                    "n_extreme_units":
                                        ev_a0["n_extreme_units"]},
                         "test_fidelity_note":
                             "test-window fidelity of the selected candidate "
                             "is reported in stage_a val_fidelity only for "
                             "validation; retrospective test fidelity is not "
                             "used for any selection"}
        rd["complete"] = True
        payload["regions"][region] = rd
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out_path)
        print(f"  [incremental save] {out_path}")

    payload["elapsed_s"] = round(time.time() - t0, 1)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    print(f"\nSaved -> {out_path} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
