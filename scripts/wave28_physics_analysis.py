"""Wave 2.8 P4 — pre-specified contrast analysis for the 6-arm physics study.

Comparison family is PRE-SPECIFIED (mechanism isolation), not exploratory:

  PRIMARY (mechanism contrasts, 5 x 2 regions x 2 architectures = 20):
    C1  era5_cond     vs standard       (meteorological conditioning effect)
    C2  pressure_prior vs era5_cond     (pressure regularization effect)
    C3  wind_prior     vs era5_cond     (wind regularization effect)
    C4  combined_prior vs era5_cond     (combined regularization effect)
    C5  combined_tail  vs combined_prior (tail-aware loss on top of combined)
  Bonferroni alpha = 0.05 / 20 = 0.0025

  SECONDARY (vs A0 real-only, 6 x 2 x 2 = 24, exploratory):
    every arm's downstream extreme-event MAE vs the A0 baseline.

Arms are physics-inspired statistical regularizers or meteorological
conditioning (retrospective reanalysis) — NOT PDE constraints, NOT bathymetry.

Per contrast: event-level paired per-unit MAE difference (mean across the 2
FM seeds per arm), sign-flip p (exact/Monte-Carlo as in the parent script),
bootstrap 95% CI, per-seed direction consistency (seed robustness).
Negative results are preserved verbatim.

Usage:
    python3 scripts/wave28_physics_analysis.py [--alpha 0.05]

Output:
    data/results/wave28_physics_analysis.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "wave28_physics_fm", ROOT / "scripts" / "wave28_physics_fm.py")
w28 = importlib.util.module_from_spec(_spec)
sys.modules["wave28_physics_fm"] = w28
_spec.loader.exec_module(w28)

CONTRASTS = [
    ("C1_conditioning", "era5_cond", "standard"),
    ("C2_pressure_reg", "pressure_prior", "era5_cond"),
    ("C3_wind_reg", "wind_prior", "era5_cond"),
    ("C4_combined_reg", "combined_prior", "era5_cond"),
    ("C5_tail_on_combined", "combined_tail", "combined_prior"),
]
ARMS = ["standard", "era5_cond", "pressure_prior", "wind_prior",
        "combined_prior", "combined_tail"]
ARCHS = ["gru", "tcn"]


def per_unit(arm: dict, arch: str) -> dict:
    """{fm_seed: per_unit_mae array} for one arm."""
    return {s: np.asarray(v["downstream"][arch]["per_unit_mae"], float)
            for s, v in arm["per_fm_seed"].items()
            if "downstream" in v and arch in v.get("downstream", {})
            and "per_unit_mae" in v["downstream"][arch]}


def paired_diff(arm_a: dict, arm_b: dict, arch: str):
    """Mean-across-seeds paired per-unit difference a - b plus per-seed."""
    ua, ub = per_unit(arm_a, arch), per_unit(arm_b, arch)
    seeds = sorted(set(ua) & set(ub))
    if not seeds:
        return None
    a_mean = np.mean([ua[s] for s in seeds], axis=0)
    b_mean = np.mean([ub[s] for s in seeds], axis=0)
    diff = a_mean - b_mean
    per_seed = {s: float(np.mean(ua[s] - ub[s])) for s in seeds}
    return {"diff": diff, "per_seed_mean": per_seed, "n_pairs": int(len(diff)),
            "n_fm_seeds": len(seeds)}


def contrast_stats(pd_):
    st = w28.paired_event_stats(pd_["diff"])
    directions = [np.sign(v) for v in pd_["per_seed_mean"].values()]
    return {
        "mean_delta": st["mean_delta"],
        "ci95": st["ci95"],
        "signflip_p": st["signflip_p"],
        "n_pairs": pd_["n_pairs"],
        "n_fm_seeds": pd_["n_fm_seeds"],
        "per_seed_mean_delta": pd_["per_seed_mean"],
        "seed_direction_consistent": bool(
            len(set(directions)) == 1 and directions[0] != 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(
        ROOT / "data" / "results" / "wave28_physics_fm.json"))
    ap.add_argument("--out", default=str(
        ROOT / "data" / "results" / "wave28_physics_analysis.json"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    d = json.loads(Path(args.artifact).read_text())
    if not d.get("regions"):
        sys.exit("physics artifact incomplete — no regions key")

    n_primary = len(CONTRASTS) * 2 * len(ARCHS)
    n_secondary = len(ARMS) * 2 * len(ARCHS)
    out = {
        "analysis": "P4 pre-specified mechanism contrasts (Wave 2.8 6-arm)",
        "primary_family_size": n_primary,
        "primary_bonferroni_alpha": args.alpha / n_primary,
        "secondary_family_size": n_secondary,
        "secondary_bonferroni_alpha": args.alpha / n_secondary,
        "interpretation": "era5_cond = meteorological conditioning "
                          "(retrospective reanalysis); prior arms = "
                          "physics-inspired statistical regularization; "
                          "NOT PDE constraints; NOT bathymetry",
        "regions": {},
    }
    for region, rd in d["regions"].items():
        arms = rd.get("arms", {})
        missing = [a for a in ARMS if a not in arms]
        if missing:
            sys.exit(f"{region}: missing arms {missing} — artifact "
                     f"incomplete; refusing partial analysis")
        reg_out = {"primary": {}, "secondary_vs_A0": {}}

        for cname, a, b in CONTRASTS:
            for arch in ARCHS:
                pd_ = paired_diff(arms[a], arms[b], arch)
                if pd_ is None:
                    reg_out["primary"][f"{cname}_{arch}"] = {
                        "status": "NOT_EVALUATED",
                        "reason": "no overlapping fm seeds with per-unit data"}
                    continue
                st = contrast_stats(pd_)
                st.update({"contrast": f"{a} - {b}", "arch": arch})
                reg_out["primary"][f"{cname}_{arch}"] = st

        for arm_name in ARMS:
            for arch in ARCHS:
                ua = per_unit(arms[arm_name], arch)
                base = rd.get("baseline_A0", {})
                basek = base.get(arch, {}) if isinstance(base, dict) else {}
                ub = {}
                if "per_unit_mae" in basek:
                    for s in ua:            # A0 is generator-independent
                        ub[s] = np.asarray(basek["per_unit_mae"], float)
                if ua and ub:
                    pd_ = paired_diff({"per_fm_seed": {
                        s: {"downstream": {arch: {"per_unit_mae": ua[s]}}}
                        for s in ua}},
                        {"per_fm_seed": {
                            s: {"downstream": {arch: {"per_unit_mae": ub[s]}}}
                            for s in ub}}, arch)
                    st = contrast_stats(pd_)
                    st.update({"contrast": f"{arm_name} - A0",
                               "arch": arch})
                    reg_out["secondary_vs_A0"][f"{arm_name}_{arch}"] = st
                else:
                    reg_out["secondary_vs_A0"][f"{arm_name}_{arch}"] = {
                        "status": "NOT_EVALUATED",
                        "reason": "A0 per-unit data unavailable"}

        out["regions"][region] = reg_out

    # family-level verdicts
    surv_primary, surv_secondary = [], []
    a_p, a_s = out["primary_bonferroni_alpha"], out["secondary_bonferroni_alpha"]
    for region, reg_out in out["regions"].items():
        for k, st in reg_out["primary"].items():
            if st.get("signflip_p") is not None and st["signflip_p"] < a_p:
                surv_primary.append((region, k, st))
        for k, st in reg_out["secondary_vs_A0"].items():
            if st.get("signflip_p") is not None and st["signflip_p"] < a_s:
                surv_secondary.append((region, k, st))
    out["verdict"] = {
        "primary_survivors": [
            {"region": r, "contrast": k,
             "mean_delta": st["mean_delta"], "p": st["signflip_p"],
             "seed_robust": st["seed_direction_consistent"]}
            for r, k, st in surv_primary],
        "secondary_survivors": [
            {"region": r, "contrast": k,
             "mean_delta": st["mean_delta"], "p": st["signflip_p"],
             "seed_robust": st["seed_direction_consistent"]}
            for r, k, st in surv_secondary],
        "note": "survivors additionally require FM-seed direction "
                "consistency for any robustness claim; single-generator "
                "draws per seed are a documented limitation",
    }

    tmp = Path(args.out).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(args.out)
    print("WROTE", args.out)
    print("primary survivors:",
          [(r, k) for r, k, _ in surv_primary] or "NONE")
    print("secondary survivors:",
          [(r, k) for r, k, _ in surv_secondary] or "NONE")


if __name__ == "__main__":
    main()
