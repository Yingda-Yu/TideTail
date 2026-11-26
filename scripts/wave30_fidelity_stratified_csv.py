"""Wave 3.0 P2 — flatten the stratified fidelity JSON to a CSV.

Reads data/results/wave30_generator_fidelity.json (written incrementally
by wave28_gen_bench_revised_eval.py) and emits one row per
(region, family, gen_seed, stratum) with every persisted fidelity metric.

Output: data/results/wave30_generator_fidelity_stratified.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "results" / "wave30_generator_fidelity.json"
OUT = ROOT / "data" / "results" / "wave30_generator_fidelity_stratified.csv"

ID_COLS = ["region", "family", "gen_seed", "stratum", "status", "n_windows",
           "train_s", "sample_s", "param_count", "optimizer_steps", "epochs",
           "device", "cpu_threads", "draws", "checkpoint_sha256"]
METRIC_COLS = ["scale_ratio", "range_ratio", "wasserstein1", "mean_bias",
               "q95_exceedance_ratio", "q99_exceedance_ratio",
               "autocorr_rmse", "psd_deviation", "peak_ks_stat",
               "peak_ks_pvalue", "peak_timing_mae_h", "peak_duration_ratio",
               "peak_duration_gen_h", "peak_duration_real_h",
               "sample_diversity_1mnn", "exact_dup_ratio",
               "near_dup_ratio_0995", "nn_distance_to_train"]


def main() -> None:
    src = json.loads(SRC.read_text())
    rows = []
    for region, rd in src.get("regions", {}).items():
        meta_seed = {}
        for fam, fd in rd.get("families", {}).items():
            for seed, sd in fd.get("per_gen_seed", {}).items():
                common = {k: sd.get(k) for k in
                          ("train_s", "sample_s", "param_count",
                           "optimizer_steps", "epochs", "device",
                           "cpu_threads", "draws", "checkpoint_sha256")}
                for stratum, st in sd.get("strata", {}).items():
                    row = {"region": region, "family": fam,
                           "gen_seed": seed, "stratum": stratum,
                           "status": st.get("status"),
                           "n_windows": st.get("n_windows"), **common}
                    row.update({k: st.get(k) for k in METRIC_COLS})
                    rows.append(row)

    cols = ID_COLS + METRIC_COLS
    OUT.write_text("")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
