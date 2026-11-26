#!/usr/bin/env python3
"""Wave 3.0 P4-light mechanism analyses on EXISTING five-region per-unit data.

No new training, no heavy compute (runs alongside the P2 fidelity job).
Three pre-specified descriptive/mechanistic analyses:

1. Regional heterogeneity  — per augmentation arm, are per-unit augmentation
   effects different across regions? Cochran-style Q with cluster bootstrap
   SEs + all pairwise region contrasts (Holm-corrected within arm).
2. Event-severity dependence — per region x arm, Spearman rank correlation
   between augmentation effect and POT-cluster peak residual (excess ratio
   as sensitivity), permutation p (Holm-corrected within analysis family).
3. Seasonal dependence — per region x arm, mean effect per local
   meteorological season (southern-hemisphere labels for SE Australia),
   cluster-bootstrap CIs + across-season permutation p (Holm-corrected).

Inputs
  data/results/wave30_expanded_event_metrics.csv  (1,440 rows, joined to
      event ids below)
  data/manifests/multiyear_declustering.json      (severity + start dates)

Event id rule (scripts/wave27_expanded_eval.py::pot_units):
  unit = -(year*1000 + cluster_id + 1)

All analyses are DESCRIPTIVE mechanism probes on the retrospective test
set; they inform interpretation and are never used for model selection.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data/results/wave30_expanded_event_metrics.csv"
DECL = ROOT / "data/manifests/multiyear_declustering.json"
OUT_JSON = ROOT / "data/results/wave30_mechanism_light.json"

ARMS = ["standard_gru", "standard_tcn", "tail_aware_gru", "tail_aware_tcn"]
SOUTHERN = {"southeast_australia"}
RNG_SEED = 20261126
B_BOOT = 10_000
B_PERM = 5_000


def season_of(month: int, region: str) -> str:
    name = ["DJF", "MAM", "JJA", "SON"][(month % 12) // 3]
    if region in SOUTHERN:  # austral seasons: shift by 6 months
        name = {"DJF": "JJA", "MAM": "SON", "JJA": "DJF", "SON": "MAM"}[name]
    return name


def load_events() -> dict:
    """region -> event_id -> {peak_m, duration_h, excess_ratio, season}."""
    dec = json.loads(DECL.read_text())
    events: dict[str, dict[int, dict]] = defaultdict(dict)
    for region, by_year in dec.items():
        for year, clusters in by_year["pot_clusters"].items():
            for c in clusters:
                eid = -(int(year) * 1000 + int(c["cluster_id"]) + 1)
                month = int(c["start"][5:7])
                events[region][eid] = {
                    "peak_m": float(c["peak_residual_m"]),
                    "duration_h": float(c["duration_hours"]),
                    "excess_ratio": (float(c["peak_residual_m"])
                                     / float(c["threshold_q95"])),
                    "season": season_of(month, region),
                }
    return events


def load_rows(events) -> dict:
    """arm -> region -> event_id -> list of (delta_pct, mae_variant, mae_a0)."""
    out: dict[str, dict[str, dict[int, list]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with CSV_IN.open() as fh:
        for r in csv.DictReader(fh):
            arm = r["cell"]
            if arm not in ARMS:
                continue
            region, eid = r["region"], int(r["event_id"])
            if eid not in events[region]:
                raise KeyError(f"unjoined event {region} {eid}")
            mv, a0 = float(r["mae_variant"]), float(r["mae_a0"])
            dp = (mv - a0) / a0 * 100.0
            out[arm][region][eid].append((dp, mv, a0))
    return out


def region_delta_pct(units: list[tuple[float, float, float]]) -> float:
    """Headline convention: ratio of sums, in percent."""
    sv = sum(v for _, v, _ in units)
    sa = sum(a for _, _, a in units)
    return (sv - sa) / sa * 100.0


def ev_means(deltas: dict[int, list]) -> np.ndarray:
    """Per-event mean delta_pct (averaged across fm seeds)."""
    return np.array([np.mean([t[0] for t in deltas[e]])
                     for e in sorted(deltas)])


def cluster_boot_mean(deltas: dict[int, list], b: int, rng) -> np.ndarray:
    """Bootstrap over events (both fm-seed rows travel together)."""
    per_event = ev_means(deltas)
    n = len(per_event)
    stats = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        stats[i] = per_event[idx].mean()
    return stats


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    events = load_events()
    data = load_rows(events)

    result = {
        "audit_version": "wave30-mechanism-light-1.0",
        "rng_seed": RNG_SEED,
        "n_boot": B_BOOT,
        "n_perm": B_PERM,
        "disclosure": ("Descriptive mechanism probes on retrospective test "
                       "events; not used for model selection. Event severity "
                       "and seasons derived from POT-cluster manifest "
                       "(train-reference thresholds); southern-hemisphere "
                       "season labels for southeast_australia."),
        "arms": {},
    }

    # ---- 1. regional heterogeneity -------------------------------------
    hetero: dict[str, dict] = {}
    for arm in ARMS:
        regs = sorted(data[arm])
        region_rows = {}
        for r in regs:
            deltas = data[arm][r]
            boot = cluster_boot_mean(deltas, B_BOOT, rng)
            point = float(ev_means(deltas).mean())
            all_units = [t for v in deltas.values() for t in v]
            region_rows[r] = {
                "mean_delta_pct_point": point,
                "mean_delta_pct_ratio_of_sums": region_delta_pct(all_units),
                "ci95": [float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))],
                "boot_se": float(boot.std(ddof=1)),
                "n_events": len(deltas),
            }
        # Cochran-style Q with bootstrap SEs (df = n_regions - 1)
        pts = np.array([region_rows[r]["mean_delta_pct_point"] for r in regs])
        ses = np.array([region_rows[r]["boot_se"] for r in regs])
        w = 1.0 / np.maximum(ses, 1e-12) ** 2
        dbar = (w * pts).sum() / w.sum()
        q = float((w * (pts - dbar) ** 2).sum())
        # bootstrap p: Q under region-resampled bootstrap variance is
        # approximated by chi2; additionally report max-pairwise spread CI
        from math import erfc, sqrt
        df = len(regs) - 1
        p_chi2 = float(erfc(sqrt(q / df))) if df == 1 else None
        if df > 1:  # regularized upper tail via survival (sf)
            from scipy.stats import chi2 as _chi2
            p_chi2 = float(_chi2.sf(q, df))
        # pairwise contrasts
        pairs = {}
        for a, b in combinations(regs, 2):
            pa = region_rows[a]
            pb = region_rows[b]
            d = pa["mean_delta_pct_point"] - pb["mean_delta_pct_point"]
            se = float(np.hypot(pa["boot_se"], pb["boot_se"]))
            z = d / max(se, 1e-12)
            p = float(erfc(abs(z) / sqrt(2)))  # two-sided normal
            pairs[f"{a}__vs__{b}"] = {
                "contrast_pct": float(d), "se": se, "z": float(z), "p": p}
        holm_p = holm([v["p"] for v in pairs.values()])
        for k, p in zip(pairs, holm_p):
            pairs[k]["p_holm"] = p
        hetero[arm] = {
            "regions": region_rows,
            "q_stat": q, "df": df, "p_chi2": p_chi2,
            "pairwise": pairs,
            "n_pairwise_survive_holm": int(
                sum(1 for v in pairs.values() if v["p_holm"] < 0.05)),
        }
    result["regional_heterogeneity"] = hetero

    # ---- 2. severity dependence -----------------------------------------
    severity: dict[str, dict] = {}
    all_sp = []
    for arm in ARMS:
        for r in sorted(data[arm]):
            deltas = data[arm][r]
            eids = sorted(deltas)
            ev_mean = ev_means(deltas)
            peak = np.array([events[r][e]["peak_m"] for e in eids])
            exc = np.array([events[r][e]["excess_ratio"] for e in eids])
            from scipy.stats import spearmanr
            rho_p, p_p = spearmanr(ev_mean, peak)
            rho_e, p_e = spearmanr(ev_mean, exc)
            all_sp.append((arm, r, rho_p, p_p, rho_e, p_e,
                           len(eids)))
    # permutation p (shuffle peaks among events within region/arm)
    perm_p = []
    for arm, r, rho_p, _, rho_e, _, n in all_sp:
        deltas = data[arm][r]
        ev_mean = ev_means(deltas)
        peak = np.array([events[r][e]["peak_m"] for e in sorted(deltas)])
        obs = abs(spearmanr(ev_mean, peak).statistic)
        cnt = 0
        for _ in range(B_PERM):
            s = spearmanr(ev_mean, rng.permutation(peak)).statistic
            if abs(s) >= obs - 1e-12:
                cnt += 1
        perm_p.append((cnt + 1) / (B_PERM + 1))
    holm_p = holm(perm_p)
    for (arm, r, rho_p, p_p, rho_e, p_e, n), pp, hp in zip(
            all_sp, perm_p, holm_p):
        severity.setdefault(arm, {})[r] = {
            "spearman_delta_vs_peak": float(rho_p),
            "spearman_p_peak": float(p_p),
            "spearman_delta_vs_excess_ratio": float(rho_e),
            "spearman_p_excess": float(p_e),
            "perm_p_peak": float(pp), "p_holm": float(hp),
            "n_events": n,
        }
    result["severity_dependence"] = severity

    # ---- 3. seasonal dependence -----------------------------------------
    seasonal: dict[str, dict] = {}
    for arm in ARMS:
        for r in sorted(data[arm]):
            deltas = data[arm][r]
            by_season: dict[str, list[float]] = defaultdict(list)
            for e, vals in deltas.items():
                by_season[events[r][e]["season"]].append(
                    float(np.mean([t[0] for t in vals])))
            eids = sorted(deltas)
            seas_arr = np.array([events[r][e]["season"] for e in eids])
            ev_mean = ev_means(deltas)
            uniq = sorted(by_season)
            entry = {"per_season": {}, "n_per_season": {
                s: len(by_season[s]) for s in uniq}}
            for s in uniq:
                boot = cluster_boot_mean(
                    {e: deltas[e] for e in eids
                     if events[r][e]["season"] == s}, B_BOOT, rng)
                entry["per_season"][s] = {
                    "mean_delta_pct": float(np.mean(by_season[s])),
                    "ci95": [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))],
                }
            # across-season permutation F-style statistic
            grand = ev_mean.mean()
            ss_b = sum(len(ev_mean[seas_arr == s])
                       * (ev_mean[seas_arr == s].mean() - grand) ** 2
                       for s in uniq)
            obs_f = ss_b / max(np.var(ev_mean, ddof=1), 1e-12)
            cnt = 0
            for _ in range(B_PERM):
                sp = rng.permutation(ev_mean)
                g = sp.mean()
                sb = sum(len(sp[seas_arr == s])
                         * (sp[seas_arr == s].mean() - g) ** 2
                         for s in uniq)
                if sb / max(np.var(sp, ddof=1), 1e-12) >= obs_f - 1e-12:
                    cnt += 1
            entry["perm_p_across_seasons"] = (cnt + 1) / (B_PERM + 1)
            seasonal.setdefault(arm, {})[r] = entry
    # Holm within arm across regions (5 each)
    for arm in seasonal:
        ps = [seasonal[arm][r]["perm_p_across_seasons"] for r in seasonal[arm]]
        hp = holm(ps)
        for r, h in zip(seasonal[arm], hp):
            seasonal[arm][r]["p_holm"] = h
    result["seasonal_dependence"] = seasonal

    OUT_JSON.write_text(json.dumps(result, indent=1))
    print(f"written {OUT_JSON}")

    # console digest
    print("\n== regional heterogeneity (per arm) ==")
    for arm, h in hetero.items():
        print(f" {arm}: Q={h['q_stat']:.1f} df={h['df']} "
              f"p={h['p_chi2']:.4g} "
              f"surviving pairs={h['n_pairwise_survive_holm']}/10")
    print("\n== severity Spearman (delta vs peak) ==")
    for arm, rs in severity.items():
        for r, v in rs.items():
            flag = "*" if v["p_holm"] < 0.05 else ""
            print(f" {arm:16s} {r:26s} rho={v['spearman_delta_vs_peak']:+.3f} "
                  f"p_holm={v['p_holm']:.4g}{flag} n={v['n_events']}")
    print("\n== season perm p ==")
    for arm, rs in seasonal.items():
        for r, v in rs.items():
            flag = "*" if v["p_holm"] < 0.05 else ""
            print(f" {arm:16s} {r:26s} p={v['perm_p_across_seasons']:.4g}"
                  f"{flag} n={v['n_per_season']}")


if __name__ == "__main__":
    main()
