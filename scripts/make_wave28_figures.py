"""Wave 2.8 figures — generated ONLY from result artifacts.

Figures:
  W28_physics_ablation  6-arm physics factorial: fidelity + downstream
  W28_benchmark         7-family generative benchmark
  W28_expanded_cells    expanded-year 6-cell comparison

Every plotted value is read from data/results/*.json produced by the
Wave 2.8 scripts; nothing is hand-typed. Run after the corresponding
experiments complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
FIGS = ROOT / "paper" / "figures"

ARM_LABELS = {
    "standard": "Standard\n(no ERA5)",
    "era5_cond": "ERA5 cond.\nonly (control)",
    "pressure_prior": "ERA5 +\npressure prior",
    "wind_prior": "ERA5 +\nwind prior",
    "combined_prior": "ERA5 +\ncombined prior",
    "combined_tail": "Combined +\ntail-aware",
}


def _load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_physics_ablation() -> bool:
    d = _load("wave28_physics_fm.json")
    if not d or not d.get("regions"):
        return False
    regions = list(d["regions"])
    fig, axes = plt.subplots(len(regions), 2,
                             figsize=(11, 3.2 * len(regions)),
                             squeeze=False)
    for r, region in enumerate(regions):
        v = d["regions"][region]
        arms = [a for a in ARM_LABELS if a in v.get("arms", {})]
        x = np.arange(len(arms))
        # left: fidelity (scale ratio, log-ish band)
        ax = axes[r][0]
        scales = [np.mean([s["scale_ratio"]
                           for s in v["arms"][a]["per_fm_seed"].values()])
                  for a in arms]
        w1s = [np.mean([s["wasserstein1"]
                        for s in v["arms"][a]["per_fm_seed"].values()])
               for a in arms]
        ax.bar(x, scales, color="#4477aa", alpha=0.85)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.axhspan(0.9, 1.1, color="green", alpha=0.08)
        for xi, w1 in zip(x, w1s):
            ax.annotate(f"W1={w1:.3f}", (xi, scales[xi]), ha="center",
                        va="bottom", fontsize=7)
        ax.set_xticks(x, [ARM_LABELS[a] for a in arms], fontsize=7)
        ax.set_ylabel("Scale ratio (gen/real)")
        ax.set_title(f"{region} — generator fidelity", fontsize=10)
        # right: downstream deltas with sign-flip p
        ax = axes[r][1]
        w = 0.38
        for j, fore in enumerate(("gru", "tcn")):
            agg = [v["arms"][a]["downstream_agg"][fore]["mean_delta_pct"]
                   for a in arms]
            ax.bar(x + (j - 0.5) * w, agg, width=w,
                   label=fore.upper(),
                   color="#4477aa" if fore == "gru" else "#ee6677",
                   alpha=0.85)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x, [ARM_LABELS[a] for a in arms], fontsize=7)
        ax.set_ylabel("Δ extreme-event MAE vs A0 (%)")
        ax.set_title(f"{region} — downstream (negative = better)", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Wave 2.8 physics-guided Flow Matching — 6-arm ablation "
                 "(2 FM seeds, GRU+TCN downstream)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "W28_physics_ablation.pdf")
    plt.close(fig)
    return True


def fig_benchmark() -> bool:
    d = _load("wave28_gen_benchmark.json")
    if not d or not d.get("regions"):
        return False
    regions = list(d["regions"])
    fig, axes = plt.subplots(1, len(regions),
                             figsize=(6.2 * len(regions), 4.6), squeeze=False)
    for r, region in enumerate(regions):
        v = d["regions"][region]
        fams = [f for f in v.get("families", {})
                if v["families"][f].get("status") == "OK"]
        x = np.arange(len(fams))
        ax = axes[0][r]
        scale = [v["families"][f]["scale_ratio"] for f in fams]
        w1 = [v["families"][f]["wasserstein1"] for f in fams]
        colors = ["#228833" if 0.9 <= s <= 1.1 else
                  "#ccbb44" if 0.5 <= s <= 1.5 else "#cc3311"
                  for s in scale]
        ax.bar(x, scale, color=colors, alpha=0.9)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        for xi, (s, w) in enumerate(zip(scale, w1)):
            ax.annotate(f"W1={w:.3f}", (xi, s), ha="center", va="bottom",
                        fontsize=7)
        ax.set_xticks(x, [f.replace("_", "\n") for f in fams], fontsize=7)
        ax.set_ylabel("Scale ratio (gen/real)")
        ax.set_title(f"{region} — 7-family fidelity\n"
                     "(green ≤±10%, amber ≤±50%, red beyond)", fontsize=10)
    fig.suptitle("Wave 2.8 fair generative benchmark (residual space; "
                 "Diffusion-TS reconstructed via tide)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "W28_benchmark.pdf")
    plt.close(fig)
    return True


def fig_expanded_cells() -> bool:
    d = _load("wave28_expanded_eval.json")
    if not d or not d.get("regions"):
        return False
    regions = list(d["regions"])
    fig, axes = plt.subplots(1, len(regions),
                             figsize=(6.2 * len(regions), 4.6), squeeze=False)
    variants = ("standard", "tail_aware")
    labels = ["A0 (real only)", "Standard FM", "Tail-aware FM"]
    colors = ["#999999", "#4477aa", "#ee6677"]
    for r, region in enumerate(regions):
        v = d["regions"][region]
        ax = axes[0][r]
        width = 0.38
        for j, fore in enumerate(("gru", "tcn")):
            vals = [v["A0"][fore]["extreme_unit_mae"]]
            for var in variants:
                per = v["variants"][var]["per_fm_seed"]
                vals.append(np.mean([per[s]["downstream"][fore]
                                     ["extreme_unit_mae"] for s in per]))
            ax.bar(np.arange(3) + (j - 0.5) * width, vals, width=width,
                   label=fore.upper(), color=colors, alpha=0.9 if j == 0
                   else 0.6, edgecolor="k", linewidth=0.4)
        ax.set_xticks(np.arange(3), labels, fontsize=8)
        ax.set_ylabel("Extreme-event unit MAE (m)")
        ax.set_title(f"{region} — expanded-year 6 cells\n"
                     f"(2015–2020/2021/2022–2024, 2 FM seeds)", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Wave 2.8 expanded-year comparison: does augmentation help?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "W28_expanded_cells.pdf")
    plt.close(fig)
    return True


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = {
        "W28_physics_ablation": fig_physics_ablation(),
        "W28_benchmark": fig_benchmark(),
        "W28_expanded_cells": fig_expanded_cells(),
    }
    for name, ok in made.items():
        print(f"{'MADE' if ok else 'SKIPPED (artifact absent)'}: "
              f"{FIGS / (name + '.pdf')}")


if __name__ == "__main__":
    main()
