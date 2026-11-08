"""Wave 2.8 P3 — REVISED fidelity evaluation for the seven-family pilot.

Corrects the pilot's fidelity-sample bias: the pilot evaluated the FIRST 200
consecutive test windows (a single early-winter slice). This script evaluates
season-stratified windows instead:

  * four seasons (DJF/MAM/JJA/SON) of the test split;
  * independent extreme windows (target q95 peak, >=168 h separation);
  * background windows (below-median target peak, >=72 h separation);
  * Tail95 / Tail99 exceedance ratios per stratum;
  * TWO generator seeds (42, 99) with per-seed and pooled statistics.

Fairness contract is UNCHANGED from the pilot: fixed defaults, 10 epochs,
batch 256, zero hyperparameter search, NO model selection of any kind (so no
test-set tuning). The pilot artifact is never modified; this writes a
separately-labelled artifact.

Usage:
    python3 scripts/wave28_gen_bench_revised_eval.py \
        --regions south_atlantic_charleston ny_bight \
        [--epochs 10] [--families flow_matching tail_aware_fm wgan_gp ...]

Output:
    data/results/wave28_gen_benchmark_revised.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"

_spec = importlib.util.spec_from_file_location(
    "wave28_gen_benchmark", ROOT / "scripts" / "wave28_gen_benchmark.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["wave28_gen_benchmark"] = bench
_spec.loader.exec_module(bench)

SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8],
           "SON": [9, 10, 11]}
EXTREME_SEP_H, BACKGROUND_SEP_H = 168, 72
N_PER_SEASON = 40          # background windows per season
MAX_EXTREME_PER_SEASON = 30


def load_test_timestamps(region: str, test_starts: np.ndarray) -> pd.DatetimeIndex:
    base = bench.DATA_ROOT / region
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    merged = target[["timestamp_utc", "water_level_m"]].merge(
        tide[["timestamp_utc", "tide_level_m"]], on="timestamp_utc", how="inner")
    ts = pd.to_datetime(merged["timestamp_utc"])
    return pd.DatetimeIndex(ts[test_starts])


def _enforce_separation(order: np.ndarray, starts_h: np.ndarray,
                        sep_h: int) -> list[int]:
    """Greedy by descending peak; two-sided >=sep_h separation from ALL
    chosen windows (directional separation would silently discard events
    earlier in the year)."""
    chosen: list[int] = []
    chosen_h: list[float] = []
    for i in order:
        ok = True
        for ch in chosen_h:
            if abs(starts_h[i] - ch) < sep_h:
                ok = False
                break
        if ok:
            chosen.append(int(i))
            chosen_h.append(float(starts_h[i]))
    return chosen


def build_strata(region: str, test_starts: np.ndarray,
                 test_t: np.ndarray) -> dict:
    """Return {stratum_name: [window positions within test arrays]}."""
    ts = load_test_timestamps(region, test_starts)
    months = np.asarray(ts.month)
    start_h = np.asarray((ts - ts[0]).total_seconds() / 3600.0)
    peaks = test_t.max(axis=1)
    q95 = float(np.percentile(peaks, 95))
    med = float(np.percentile(peaks, 50))

    strata: dict[str, list[int]] = {}
    for sname, months_ in SEASONS.items():
        in_s = np.where(np.isin(months, months_))[0]
        if len(in_s) == 0:
            continue
        # background: below-median peak, spread across season
        bg_pool = in_s[peaks[in_s] < med]
        if len(bg_pool) > N_PER_SEASON:
            sel = np.linspace(0, len(bg_pool) - 1, N_PER_SEASON).astype(int)
            bg_pool = bg_pool[sel]
        strata[f"{sname}_background"] = [int(i) for i in bg_pool]
        # extreme: q95 peak, >=168h independent (POT declustering scale)
        ex_pool = in_s[peaks[in_s] >= q95]
        order = ex_pool[np.argsort(-peaks[ex_pool])]
        chosen = _enforce_separation(order, start_h, EXTREME_SEP_H)
        strata[f"{sname}_extreme"] = chosen[:MAX_EXTREME_PER_SEASON]
    # pooled extreme + pooled background for aggregate gates
    strata["all_extreme"] = sorted(
        sum((strata.get(f"{s}_extreme", []) for s in SEASONS), []))
    strata["all_background"] = sorted(
        sum((strata.get(f"{s}_background", []) for s in SEASONS), []))
    # moderate tail: q90 peak, >=120h separation — statistical power
    # supplement for regions where independent q95 events are very few
    q90 = float(np.percentile(peaks, 90))
    ex_pool = np.where(peaks >= q90)[0]
    order = ex_pool[np.argsort(-peaks[ex_pool])]
    strata["all_moderate_tail"] = _enforce_separation(
        order, start_h, 120)[:40]
    return strata


def eval_stratum(report, real: np.ndarray, gen: np.ndarray,
                 idx: list[int]) -> dict:
    """quality.evaluate on a window subset; gen is (draws, N, H)."""
    if len(idx) < 2:
        # wasserstein needs >=2 samples; record honest empty/thin stratum
        return {"status": "NOT_EVALUATED", "n_windows": len(idx),
                "reason": "fewer than 2 independent windows"}
    sub_real = real[idx]
    sub_gen = gen[:, idx, :].reshape(-1, real.shape[1])
    rep = bench.evaluate(sub_real, sub_gen,
                         train_ref=report["train_res_t"])
    return {"status": rep.status,
            "scale_ratio": float(rep.scale_ratio),
            "wasserstein1": float(rep.wasserstein1),
            "q95_exceedance_ratio": float(rep.q95_exceedance_ratio),
            "q99_exceedance_ratio": float(rep.q99_exceedance_ratio),
            "n_windows": len(idx)}


def train_family(fam: str, data: dict, epochs: int, batch_size: int,
                 seed: int) -> dict:
    """Mirror of the pilot's family dispatch (fixed defaults, no search)."""
    if fam in ("flow_matching", "tail_aware_fm"):
        return bench.train_fm(data["train_res_h"], data["train_res_t"],
                              tail=(fam == "tail_aware_fm"), epochs=epochs,
                              batch_size=batch_size, seed=seed)
    if fam == "cvae":
        return bench.train_cvae(data["train_res_h"], data["train_res_t"],
                                epochs, batch_size, seed)
    if fam == "timegrad":
        return bench.train_timegrad(data["train_res_h"], data["train_res_t"],
                                    epochs, batch_size, seed)
    if fam == "ddpm_cond":
        return bench.train_ddpm(data["train_res_h"], data["train_res_t"],
                                epochs, batch_size, seed)
    if fam == "wgan_gp":
        return bench.train_wgan(data["train_res_h"], data["train_res_t"],
                               epochs, batch_size, seed)
    if fam == "diffusion_ts":
        return bench.train_diffusion_ts(data["train_tot_h"],
                                        data["train_tot_t"], epochs,
                                        batch_size, seed)
    raise ValueError(fam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["south_atlantic_charleston", "ny_bight"])
    ap.add_argument("--families", nargs="+", default=bench.FAMILIES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gen-seeds", nargs="+", type=int, default=[42, 99])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default=str(
        RESULTS / "wave28_gen_benchmark_revised.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation": "REVISED season-stratified fidelity (pilot sample-bias "
                      "correction; pilot artifact preserved unchanged)",
        "differs_from_pilot": "pilot scored test_res_t[:200] (first 200 "
                              "consecutive windows, single early-winter "
                              "season); this scores season-stratified "
                              "windows with independent extremes",
        "fairness_contract": "fixed defaults, 10 epochs, batch 256, ZERO "
                             "hyperparameter search, NO model selection "
                             "(validation or otherwise)",
        "epochs": args.epochs, "batch_size": 256, "draws": args.draws,
        "gen_seeds": args.gen_seeds,
        "extreme_separation_h": EXTREME_SEP_H,
        "background_separation_h": BACKGROUND_SEP_H,
        "strata_definition": {
            "seasonal": "DJF/MAM/JJA/SON; background = below-median "
                        "target peak (40/season, spread); extreme = q95 "
                        "target peak with >=168h declustering (POT scale)",
            "all_moderate_tail": "q90 target peak with >=120h "
                                 "declustering (power supplement; "
                                 "independent-q95 events are few per year)",
        },
        "regions": {},
    }
    _atomic_write(out_path, payload)

    for region in args.regions:
        t_reg = time.time()
        data = bench.load_region(region)
        n_test = len(data["test_res_t"])
        strata = build_strata(region, np.arange(n_test), data["test_res_t"])
        # generate ONLY the stratified windows (~200), not all 8,601
        sel = np.array(sorted({i for idx in strata.values() for i in idx}),
                       dtype=np.int64)
        pos = {int(w): int(p) for p, w in enumerate(sel)}
        strata_pos = {name: [pos[i] for i in idx]
                      for name, idx in strata.items()}
        real_sel = data["test_res_t"][sel]
        print(f"== {region}: {n_test} test windows, {len(sel)} stratified "
              f"selected; strata sizes:",
              {k: len(v) for k, v in strata.items()})
        reg_out: dict = {"strata_sizes": {k: len(v) for k, v in strata.items()},
                         "families": {}}
        payload["regions"][region] = reg_out
        _atomic_write(out_path, payload)

        for fam in args.families:
            per_seed = {}
            for seed in args.gen_seeds:
                t0 = time.time()
                model = train_family(fam, data, args.epochs, 256, seed)
                if fam == "diffusion_ts":
                    gen = model["gen"](data["test_tot_h"], sel,
                                       args.draws, data["test_tide_t"])
                else:
                    gen = model["gen"](data["test_res_h"], sel,
                                       args.draws)
                per_seed[str(seed)] = {
                    "train_s": round(model["train_s"], 1),
                    "strata": {name: eval_stratum(data, real_sel,
                                                  gen, idx)
                               for name, idx in strata_pos.items()}}
                ext = per_seed[str(seed)]["strata"]["all_extreme"]
                print(f"  {fam} seed {seed}: "
                      f"{time.time() - t0:.0f}s "
                      f"pooled_extreme_scale="
                      f"{ext.get('scale_ratio', float('nan')):.3f}")
                _atomic_write(out_path, payload)
            reg_out["families"][fam] = {
                "per_gen_seed": per_seed,
                "pooled_extreme_scale_mean": float(np.mean(
                    [per_seed[s]["strata"]["all_extreme"].get(
                        "scale_ratio", np.nan) for s in per_seed])),
                "pooled_background_scale_mean": float(np.mean(
                    [per_seed[s]["strata"]["all_background"].get(
                        "scale_ratio", np.nan) for s in per_seed])),
                "pooled_moderate_tail_scale_mean": float(np.mean(
                    [per_seed[s]["strata"]["all_moderate_tail"].get(
                        "scale_ratio", np.nan) for s in per_seed])),
            }
            _atomic_write(out_path, payload)
        print(f"== {region} done in {time.time() - t_reg:.0f}s")

    print("WROTE", out_path)


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


if __name__ == "__main__":
    main()
