"""Wave 1.9 corrected figure generation — scientific integrity fixes.

F3 now shows independent-unit counts (168h-embargo observ segments + canonical TCs),
NOT calendar-day pseudo-events.

F6 properly labeled as forecast distribution diagnostic (not "synthetic").
Actual generator fidelity figure will be regenerated once FM.sample() outputs exist.

All figures preserve Wave 1.8 numerical accuracy but fix conceptual framing.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = ROOT / "paper" / "figures"
FIG_OUT.mkdir(exist_ok=True)
PREDS_DIR = ROOT / "data" / "predictions" / "v2"
BOOT_JSON = ROOT / "data" / "results" / "wave19_storm_bootstrap.json"

REGIONS_SHORT = {
    "ny_bight": "NY Bight",
    "thames_north_sea": "Thames\nNorth Sea",
    "us_gulf_coast": "US Gulf",
    "south_atlantic_charleston": "Charleston\nSC",
    "southeast_australia": "SE Australia",
}
REGIONS_ORDER = ["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
                 "southeast_australia", "thames_north_sea"]

def fig_f3_independent_units():
    """Correct F3: independent-unit counts from 168h-embargo analysis."""
    results = json.loads(open(BOOT_JSON).read())
    regions = [REGIONS_SHORT[r["region"]] for r in results]
    total = [r["independent_units"] for r in results]
    tc = [r["canonical_tcs"] for r in results]
    seg = [r["observ_segments"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("F3 — Independent Units Per Region (Wave 1.9, CORRECTED)",
                 fontsize=13, fontweight="bold", y=1.02)
    x = np.arange(len(regions))
    w = 0.6
    ax.bar(x, seg, w, color="#4C78A8", label="Observ segments (168h embargo)")
    ax.bar(x, tc, w, bottom=seg, color="#F58518", label="Canonical TCs (IBTrACS 2024)")
    for i, (s, t) in enumerate(zip(seg, tc)):
        ax.text(i, s+t+0.05, str(s+t), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=9)
    ax.set_ylabel("Independent units (Coles 2001 declustering)", fontsize=10)
    ax.set_title("2024 test split only — NO calendar-day mislabeling", fontsize=9, style="italic")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, max(total) * 1.2)

    ax.text(0.02, 0.92,
            "CRITICAL: 1 observ segment = 1 continuous period\n"
            "of 168h-contiguous forecast windows.\n"
            "Most regions have 2024-only test split → 1 segment.",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.9})

    out = FIG_OUT / "F3_independent_units_corrected.pdf"
    fig.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  F3 corrected → {out}")

def fig_f8_with_significance_caveats():
    """F8 augmentation comparison with explicit significance caveats."""
    with open(BOOT_JSON) as f:
        results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("F8 — Augmentation Effect on Total-Water MAE (Wave 1.9)",
                 fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(REGIONS_ORDER))
    w = 0.25

    # A0 baseline MAE + A3 best per region (numerical values, NOT statistical claims)
    a0_maes = []
    a3_best_maes = []
    a0_cis = []
    for r_name in REGIONS_ORDER:
        r = next(x for x in results if x["region"] == r_name)
        a0 = r["bootstrap_results"]["A0_real_only gru seed42"]
        a0_maes.append(a0["point_mae"])
        a0_cis.append((a0["ci_low"], a0["ci_high"]))
        # Best A3 = lowest point_mae across all A3 configs
        a3_vals = [(k, v["point_mae"], v["ci_low"], v["ci_high"])
                   for k, v in r["bootstrap_results"].items() if k.startswith("A3") and "gru" in k]
        a3_vals.sort(key=lambda t: t[1])
        best = a3_vals[0]
        a3_best_maes.append((best[1], best[2], best[3], best[0]))

    ax1.bar(x - w/2, a0_maes, w, label="A0 baseline", color="#4C78A8", alpha=0.85, edgecolor="black")
    a3_heights = [a[0] for a in a3_best_maes]
    ax1.bar(x + w/2, a3_heights, w, label="A3 best config", color="#F58518", alpha=0.85, edgecolor="black")

    for i, (a3v, a3l, a3h, cfg) in enumerate(a3_best_maes):
        ci_low, ci_high = a0_cis[i]
        a3_low, a3_high = a3l, a3h
        overlap = not (a3_high < ci_low or a3_low > ci_high)
        marker = "↔ overlap" if overlap else "✅" if a3_high < ci_low else "❌"
        ax1.annotate(marker, (x[i] + w/2, a3v), xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=8, fontweight="bold",
                     color="#E74C3C" if overlap else "#2ECC71")

    ax1.set_xticks(x)
    ax1.set_xticklabels([REGIONS_SHORT[r] for r in REGIONS_ORDER], fontsize=9)
    ax1.set_ylabel("Independent-unit point MAE (m)", fontsize=10)
    ax1.set_title("(a) Absolute MAE with CI overlap markers", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Δ% numerical (NOT statistical)
    deltas = [(a3v - a0m) / a0m * 100 for a0m, (a3v, *_ ) in zip(a0_maes, a3_best_maes)]
    bar_colors = ["#2ECC71" if d < 0 else "#E74C3C" for d in deltas]
    bars = ax2.bar(x, deltas, color=bar_colors, alpha=0.85, edgecolor="black")
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if d > 0 else -1),
                 f"{d:+.1f}%", ha="center", va="bottom" if d > 0 else "top", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([REGIONS_SHORT[r] for r in REGIONS_ORDER], fontsize=9)
    ax2.set_ylabel("Numerical Δ% vs A0", fontsize=10)
    ax2.set_title("(b) Numerical improvement (NO statistical claim)", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.text(0.5, -0.02,
             "⚠️ All A3 improvements are NUMERICAL, not statistically confirmed.\n"
             "Wave 1.9 independent-unit bootstrap CI overlaps prevent definitive claims.",
             ha="center", fontsize=9, style="italic", color="#888")

    out = FIG_OUT / "F8_augmentation_corrected.pdf"
    fig.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  F8 corrected → {out}")

if __name__ == "__main__":
    print("Wave 1.9 corrected figures...")
    fig_f3_independent_units()
    fig_f8_with_significance_caveats()
    print("Done.")
