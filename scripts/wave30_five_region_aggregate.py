"""Wave 3.0 P1-02 — aggregate the five-region expanded eval.

Reads data/results/wave28_expanded_eval.json (five regions, six cells
each) and writes:

  * data/results/wave30_expanded_five_region.json
        per-region protocol metadata, A0 MAE, per-variant deltas,
        seed variability, 95% CI, raw + Bonferroni-adjusted p-values,
        fidelity status;
  * data/results/wave30_expanded_event_metrics.csv
        one row per (region, event unit, cell, fm_seed) with per-unit MAE
        where per-unit arrays were persisted (Wave 3.0 runs);
  * docs/experiments/WAVE30_FIVE_REGION_EVAL.md

Primary comparison family (pre-specified in WAVE30_TASK_QUEUE.json):
5 regions x 2 variants x 2 forecasters = 20 comparisons, each pairing the
seed-averaged per-unit deltas of one augmented variant against the same
region/architecture A0. Bonferroni alpha = 0.05/20 = 0.0025.

All comparisons are retrospective (2022-2024 partially informed earlier
exploratory waves) — see retest_disclosure.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "wave28_physics_fm", ROOT / "scripts" / "wave28_physics_fm.py")
w28 = importlib.util.module_from_spec(_spec)
sys.modules["wave28_physics_fm"] = w28
_spec.loader.exec_module(w28)

RESULTS = ROOT / "data" / "results"
SRC = RESULTS / "wave28_expanded_eval.json"
OUT_JSON = RESULTS / "wave30_expanded_five_region.json"
OUT_CSV = RESULTS / "wave30_expanded_event_metrics.csv"
OUT_MD = ROOT / "docs" / "experiments" / "WAVE30_FIVE_REGION_EVAL.md"

REGIONS = ["south_atlantic_charleston", "ny_bight", "us_gulf_coast",
           "southeast_australia", "thames_north_sea"]
VARIANTS = ["standard", "tail_aware"]
ARCHS = ["gru", "tcn"]
N_PRIMARY = len(REGIONS) * len(VARIANTS) * len(ARCHS)   # 20
BONF_ALPHA = 0.05 / N_PRIMARY


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def per_unit(arch_cell: dict) -> np.ndarray | None:
    pu = arch_cell.get("per_unit_mae")
    if pu is None:
        return None
    arr = np.asarray(pu, dtype=float)
    return arr if np.all(np.isfinite(arr)) else None


def paired_delta(var_cell: dict, a0_cell: dict):
    """Seed-averaged paired per-unit delta (variant - A0) when per-unit
    arrays exist on BOTH sides; otherwise None."""
    va, aa = per_unit(var_cell), per_unit(a0_cell)
    if va is None or aa is None or len(va) != len(aa):
        return None
    return va - aa


def main() -> None:
    src = json.loads(SRC.read_text())
    out = {"audit_version": "wave30_five_region_v1",
           "protocol": src.get("protocol"),
           "selection": src.get("selection"),
           "retest_disclosure": src.get("retest_disclosure"),
           "primary_family": {"n_comparisons": N_PRIMARY,
                              "bonferroni_alpha": BONF_ALPHA,
                              "estimand": "seed-averaged per-event-unit "
                                          "paired MAE delta (variant-A0)"},
           "regions": {}, "primary_comparisons": []}

    csv_rows = ["region,event_id,cell,fm_seed,per_unit_index,mae_variant,"
                "mae_a0,delta"]

    for region in REGIONS:
        rd = src.get("regions", {}).get(region)
        if rd is None:
            out["regions"][region] = {"status": "MISSING"}
            continue
        a0 = rd["A0"]
        reg_out = {
            "window_counts": rd.get("window_counts"),
            "nan_windows_excluded": rd.get("nan_windows_excluded"),
            "fm_seeds": rd.get("fm_seeds"),
            "downstream_seeds": rd.get("downstream_seeds"),
            "A0": {k: {"extreme_unit_mae": a0[k]["extreme_unit_mae"],
                       "n_extreme_units": a0[k]["n_extreme_units"]}
                   for k in ARCHS},
            "variants": {}, "cells_status": {},
        }
        for variant in VARIANTS:
            vd = rd.get("variants", {}).get(variant)
            if vd is None:
                reg_out["cells_status"][variant] = "MISSING"
                continue
            v_out = {"per_fm_seed": {}, "downstream_agg": vd.get("downstream_agg"),
                     "augmentation_ratio": vd.get("augmentation_ratio")}
            seeds = sorted(vd.get("per_fm_seed", {}))
            for s in seeds:
                v_out["per_fm_seed"][s] = {
                    "fidelity": vd["per_fm_seed"][s].get("fidelity"),
                    "selected_epoch": vd["per_fm_seed"][s].get("selected_epoch")}
            for arch in ARCHS:
                # fidelity + downstream per seed
                fid, cell_summary = {}, {}
                for s in seeds:
                    sd = vd["per_fm_seed"][s]
                    fid[s] = sd.get("fidelity")
                    cell = sd.get("downstream", {}).get(arch)
                    if cell is None:
                        continue
                    cell_summary[s] = {
                        "delta_pct_vs_A0": cell.get("delta_pct_vs_A0"),
                        "extreme_unit_mae": cell.get("extreme_unit_mae"),
                        "paired_event_stats": cell.get("paired_event_stats")}
                    # event-level CSV where arrays exist
                    pu_v = per_unit(cell)
                    a0_cell = a0[arch]
                    pu_a = per_unit(a0_cell)
                    if pu_v is not None and pu_a is not None \
                            and len(pu_v) == len(pu_a):
                        eids = cell.get("event_ids")
                        for i, (mv, ma) in enumerate(zip(pu_v, pu_a)):
                            eid = eids[i] if eids and i < len(eids) else f"u{i}"
                            csv_rows.append(
                                f"{region},{eid},{variant}_{arch},{s},{i},"
                                f"{mv:.6f},{ma:.6f},{mv - ma:.6f}")
                # seed-averaged paired delta for primary family
                diffs = [paired_delta(vd["per_fm_seed"][s]["downstream"][arch],
                                      a0[arch]) for s in seeds
                         if s in vd["per_fm_seed"]
                         and arch in vd["per_fm_seed"][s].get("downstream", {})]
                diffs = [d for d in diffs if d is not None]
                prim = None
                if diffs:
                    avg = np.mean(diffs, axis=0)
                    st = w28.paired_event_stats(avg)
                    deltas_pct = [cell_summary[s]["delta_pct_vs_A0"]
                                  for s in cell_summary]
                    prim = {
                        "paired_event_stats": st,
                        "mean_delta_pct": float(np.mean(deltas_pct)) if deltas_pct else None,
                        "per_seed_delta_pct": deltas_pct,
                        "seed_std_pct": (float(np.std(deltas_pct, ddof=1))
                                         if len(deltas_pct) > 1 else 0.0),
                        "n_units": int(len(avg)),
                        "fidelity_status": (fid[seeds[0]] or {}).get("status")
                                           if seeds else None,
                    }
                    out["primary_comparisons"].append({
                        "region": region, "variant": variant, "architecture": arch,
                        **{k: prim[k] for k in
                           ("mean_delta_pct", "seed_std_pct", "n_units",
                            "fidelity_status")},
                        "signflip_p": prim["paired_event_stats"]["signflip_p"],
                        "p_method": prim["paired_event_stats"]["p_method"],
                        "ci95": prim["paired_event_stats"]["ci95"],
                    })
                v_out[f"{arch}_primary"] = prim
                v_out[f"{arch}_per_seed_cells"] = cell_summary
            reg_out["variants"][variant] = v_out
            reg_out["cells_status"][variant] = "COMPLETE" if \
                all(v_out.get(f"{a}_primary") for a in ARCHS) else "PARTIAL"
        out["regions"][region] = reg_out

    # Bonferroni over valid primary comparisons
    valid = [c for c in out["primary_comparisons"]
             if c["signflip_p"] is not None]
    for c in valid:
        c["bonferroni_p"] = min(1.0, c["signflip_p"] * N_PRIMARY)
        c["survives_bonferroni"] = bool(c["signflip_p"] < BONF_ALPHA)
    surv = [c for c in valid if c["survives_bonferroni"]]
    out["summary"] = {
        "n_regions_complete": sum(
            1 for r in out["regions"].values()
            if r.get("cells_status", {}).get("standard") == "COMPLETE"
            and r.get("cells_status", {}).get("tail_aware") == "COMPLETE"),
        "n_primary_valid": len(valid),
        "n_primary_expected": N_PRIMARY,
        "n_survive_bonferroni": len(surv),
        "survivors": [f"{c['region']}/{c['variant']}/{c['architecture']}"
                      for c in surv],
        "interpretation": "survivors are NOT automatically beneficial — "
                          "check sign of mean_delta_pct (>0 means harm)",
    }
    atomic_write(OUT_JSON, json.dumps(out, indent=2))

    # CSV: only rows with persisted per-unit arrays (Wave 3.0 runs)
    atomic_write(OUT_CSV, "\n".join(csv_rows) + "\n")

    # Markdown report
    lines = ["# Wave 3.0 Five-Region Expanded Evaluation",
             "",
             "Protocol: 2015-2020 train / 2021 validation / 2022-2024 test; "
             "identical windows, budgets, seeds and augmentation ratio across "
             "regions. Retrospective evaluation (test years partially informed "
             "earlier exploratory waves).",
             "",
             f"Primary family: {N_PRIMARY} comparisons "
             f"(5 regions x standard/tail-aware x GRU/TCN), Bonferroni "
             f"alpha={BONF_ALPHA:.4f}. Sign-flip p on seed-averaged per-unit "
             "paired deltas.",
             "",
             "| Region | Units GRU/TCN | A0 GRU | A0 TCN | std GRU Δ% | std TCN Δ% | tail GRU Δ% | tail TCN Δ% | fidelity |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for region in REGIONS:
        ro = out["regions"].get(region, {})
        if ro.get("status") == "MISSING" or not ro:
            lines.append(f"| {region} | MISSING | | | | | | | |")
            continue
        a0 = ro.get("A0", {})
        var = ro.get("variants", {})

        def fmt(v, a):
            c = (var.get(v) or {}).get(f"{a}_primary")
            if not c or c.get("mean_delta_pct") is None:
                return "—"
            st = c.get("paired_event_stats") or {}
            p = st.get("signflip_p")
            ps = f"{p:.4f}" if p is not None else "NA"
            return f"{c['mean_delta_pct']:+.2f} (p={ps})"
        fid = None
        if var.get("standard") and var["standard"].get("per_fm_seed"):
            f0 = next(iter(var["standard"]["per_fm_seed"].values()), {})
            fid = (f0.get("fidelity") or {}).get("status")
        units = (a0.get("gru") or {}).get("n_extreme_units")
        lines.append(
            f"| {region} | {units} | "
            f"{(a0.get('gru') or {}).get('extreme_unit_mae', float('nan')):.4f} | "
            f"{(a0.get('tcn') or {}).get('extreme_unit_mae', float('nan')):.4f} | "
            f"{fmt('standard','gru')} | {fmt('standard','tcn')} | "
            f"{fmt('tail_aware','gru')} | {fmt('tail_aware','tcn')} | {fid} |")
    lines += ["", f"Bonferroni survivors: {len(surv)} of {len(valid)} valid.",
              "", "Generated by scripts/wave30_five_region_aggregate.py; "
                  "numbers traceable to data/results/wave28_expanded_eval.json."]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(OUT_MD, "\n".join(lines) + "\n")

    print(json.dumps(out["summary"], indent=2))
    print(f"saved: {OUT_JSON}\n        {OUT_CSV}\n        {OUT_MD}")


if __name__ == "__main__":
    main()
