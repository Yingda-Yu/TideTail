"""Wave 3.0 P4 — paired GRU/TCN architecture-interaction analysis.

Question: does the augmentation effect (variant - A0, per event unit)
differ between TCN and GRU on the SAME event units, beyond what region
level deltas suggest?

For each region x variant:
    d_T[i] = per-unit MAE delta (TCN, variant seed-avg - A0 TCN)
    d_G[i] = per-unit MAE delta (GRU, variant seed-avg - A0 GRU)
    interaction delta_i = d_T[i] - d_G[i]      (paired on event unit)
Effect = mean(delta). Sign-flip randomisation p (exact/MC floor), paired
bootstrap CI. Multiplicity: 5 regions x 2 variants = 10 interaction tests,
Bonferroni alpha = 0.005.

Units are matched by event ID across architectures (identical test windows
and POT units; arrays are aligned on the sorted event-unit list, verified
by event_ids equality).

Output: data/results/wave30_architecture_interaction.json
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

SRC = ROOT / "data" / "results" / "wave28_expanded_eval.json"
OUT = ROOT / "data" / "results" / "wave30_architecture_interaction.json"

REGIONS = ["south_atlantic_charleston", "ny_bight", "us_gulf_coast",
           "southeast_australia", "thames_north_sea"]
VARIANTS = ["standard", "tail_aware"]
N_TESTS = 10
BONF_ALPHA = 0.05 / N_TESTS
N_BOOT = 10000


def seed_avg_unit_delta(region_data: dict, variant: str, arch: str,
                        base_arch: dict):
    """Seed-averaged per-unit delta (variant - A0) aligned on event IDs."""
    vd = region_data["variants"][variant]["per_fm_seed"]
    diffs, eids = [], None
    for s in sorted(vd):
        cell = vd[s]["downstream"][arch]
        pu_v = np.asarray(cell["per_unit_mae"], dtype=float)
        pu_a = np.asarray(base_arch[arch]["per_unit_mae"], dtype=float)
        if len(pu_v) != len(pu_a):
            raise AssertionError(
                f"{variant} {arch} seed {s}: unit count mismatch "
                f"({len(pu_v)} vs {len(pu_a)})")
        # A0 cells share the same POT units/window order as variant cells
        # (identical units array); event IDs are stored on variant cells
        eids = cell.get("event_ids", eids)
        diffs.append(pu_v - pu_a)
    assert eids is not None, "no event_ids found on variant cells"
    return np.mean(diffs, axis=0), eids


def paired_stats(delta: np.ndarray, seed: int = 0) -> dict:
    st = w28.paired_event_stats(delta, n_boot=N_BOOT, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(N_BOOT, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    st["ci95"] = [float(lo), float(hi)]
    st["bonferroni_p"] = (min(1.0, st["signflip_p"] * N_TESTS)
                          if st["signflip_p"] is not None else None)
    st["survives_bonferroni"] = bool(
        st["bonferroni_p"] is not None and st["bonferroni_p"] < BONF_ALPHA)
    return st


def main() -> None:
    src = json.loads(SRC.read_text())
    out = {"audit_version": "wave30_arch_interaction_v2",
           "correction_note": "v1 compared raw signflip_p against "
                              "BONF_ALPHA (bug); v2 compares the "
                              "Bonferroni-adjusted p. Corrected count "
                              "3/10 survivors (SE-Aus tail_aware "
                              "p_bonf=0.008 does not pass).",
           "estimand": "TCN augmentation effect minus GRU augmentation "
                       "effect, per event unit (seed-averaged), paired on "
                       "event ID; positive = TCN more positive (more harm/"
                       "less benefit) than GRU",
           "n_tests": N_TESTS, "bonferroni_alpha": BONF_ALPHA,
           "n_boot": N_BOOT, "comparisons": [], "summary": {}}

    n_done = 0
    for region in REGIONS:
        rd = src["regions"][region]
        a0 = rd["A0"]
        for variant in VARIANTS:
            d_g, eids = seed_avg_unit_delta(rd, variant, "gru", a0)
            d_t, eids_t = seed_avg_unit_delta(rd, variant, "tcn", a0)
            assert eids == eids_t
            inter = d_t - d_g
            st = paired_stats(inter)
            out["comparisons"].append({
                "region": region, "variant": variant,
                "n_units": int(len(inter)),
                "mean_interaction": float(inter.mean()),
                "grueffect_mean": float(d_g.mean()),
                "tcn_effect_mean": float(d_t.mean()),
                "tcn_minus_gru": float(inter.mean()),
                **st,
            })
            n_done += 1

    valid = [c for c in out["comparisons"]
             if c["signflip_p"] is not None]
    surv = [c for c in valid if c["survives_bonferroni"]]
    out["summary"] = {
        "n_comparisons": n_done, "n_valid": len(valid),
        "n_survive_bonferroni": len(surv),
        "survivors": [f"{c['region']}/{c['variant']}" for c in surv],
        "sign_consistency": {
            f"{c['region']}/{c['variant']}":
                ("tcn_more_positive" if c["mean_interaction"] > 0
                 else "tcn_more_negative")
            for c in valid},
        "note": "A non-significant interaction means the 4-of-5 regional "
                "'TCN responds more' pattern is DESCRIPTIVE, not a proven "
                "universal architecture effect.",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    for c in out["comparisons"]:
        print(f"{c['region']:28} {c['variant']:10} "
              f"inter={c['mean_interaction']*1000:+.2f}mm "
              f"p={c['signflip_p']:.4f} bonf={c['bonferroni_p']:.3f} "
              f"ci=[{c['ci95'][0]*1000:+.2f},{c['ci95'][1]*1000:+.2f}]mm")


if __name__ == "__main__":
    main()
