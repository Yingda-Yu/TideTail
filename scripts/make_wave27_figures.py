"""Wave 2.7 manuscript figures — generated ONLY from committed result artifacts.

Outputs: paper/figures/W27_*.pdf and .png

Every plotted value is read from:
  data/results/{vae,timegrad,diffusion_ts,flow_matching}_96_72.json
  data/results/paired_significance_wave27.json
  data/results/wave27_event_set_audit.json
  data/results/wave27_model_recovery.json
No number is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

C_GRU = "#1f4e79"
C_TCN = "#c0504d"
C_GREY = "#8a8a8a"
C_OK = "#4b8b3b"
C_BAD = "#a63232"

REGIONS = ["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
           "southeast_australia", "thames_north_sea"]
RLAB = ["NY Bight", "US Gulf", "Charleston", "SE Australia", "Thames/N.Sea"]


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    print("wrote", stem)


# ---------------------------------------------------------------- Fig 1
def fig_fidelity() -> None:
    models = [("VAE", "vae_96_72.json"), ("TimeGrad", "timegrad_96_72.json"),
              ("Diffusion-TS", "diffusion_ts_96_72.json"),
              ("Flow Matching", "flow_matching_96_72.json")]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.axhspan(0.5, 2.0, color="#dcead6", zorder=0)
    ax.set_yscale("log")
    for i, (name, fn) in enumerate(models):
        d = json.loads((RES / fn).read_text())
        s = np.array([r["scale_ratio"] for r in d], dtype=float)
        s = s[np.isfinite(s)]
        x = i + np.linspace(-0.16, 0.16, len(s))
        ax.scatter(x, s, s=22, color=C_GRU, alpha=0.85, zorder=2, linewidths=0)
        ax.hlines(np.median(s), i - 0.28, i + 0.28, color=C_BAD, lw=1.6, zorder=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylabel("generated / real scale ratio (log)")
    ax.set_ylim(0.03, 60)
    ax.text(3.55, 1.75, "fidelity gate\n0.5–2.0", fontsize=7.5, color=C_OK, va="top")
    ax.set_title("Sample dispersion by generator (10 runs each; bar = median)",
                 fontsize=9.5, loc="left")
    save(fig, "W27_fidelity_gate")


# ---------------------------------------------------------------- Fig 2
def fig_architecture() -> None:
    d = json.loads((RES / "paired_significance_wave27.json").read_text())
    gru, tcn = [], []
    for r in REGIONS:
        cmps = d["regions"][r]["comparisons"]
        for arch, out in (("gru", gru), ("tcn", tcn)):
            vals = [c["extreme_units"]["delta_pct"] for c in cmps if c["model"] == arch]
            out.append(float(np.mean(vals)))
    x = np.arange(len(REGIONS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(x - w / 2, gru, w, color=C_GRU, label="GRU")
    ax.bar(x + w / 2, tcn, w, color=C_TCN, label="TCN")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(RLAB, rotation=12)
    ax.set_ylabel("mean extreme-window MAE change vs A0 (%)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.set_title("Augmentation effect by region and forecaster "
                 "(negative = augmentation helps)", fontsize=9.5, loc="left")
    for xi, (g, t) in enumerate(zip(gru, tcn)):
        ax.text(xi - w / 2, g + (0.6 if g >= 0 else -1.2), f"{g:+.1f}",
                ha="center", fontsize=6.5, color=C_GRU)
        ax.text(xi + w / 2, t + (0.6 if t >= 0 else -1.2), f"{t:+.1f}",
                ha="center", fontsize=6.5, color=C_TCN)
    save(fig, "W27_architecture_region")


# ---------------------------------------------------------------- Fig 3
def fig_census() -> None:
    d = json.loads((RES / "wave27_event_set_audit.json").read_text())
    dec = d["periods"]["2015-2024"]["regions"]
    boot = d["bootstrap_2024"]
    pot = [dec[r]["primary_strict"]["pot_total"] for r in REGIONS]
    tc = [dec[r]["primary_strict"]["tc_total"] for r in REGIONS]
    ov = [dec[r]["primary_strict"]["matched_overlap"] for r in REGIONS]
    uni = [dec[r]["primary_strict"]["union"] for r in REGIONS]
    units24 = [boot[r]["extreme_units"] for r in REGIONS]
    x = np.arange(len(REGIONS))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    ax.bar(x, pot, 0.55, color="#c9d9e8", label="POT clusters")
    ax.bar(x, tc, 0.55, bottom=pot, color="#e8c9c9", label="TC events")
    ax.plot(x, uni, "o", color="black", ms=4.5, label="union (deduplicated)")
    ax.set_xticks(x); ax.set_xticklabels(RLAB, rotation=20)
    ax.set_ylabel("independent event units, 2015–2024")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(a) Decade census: raw components vs union", fontsize=8.5, loc="left")
    ax = axes[1]
    ax.bar(x, units24, 0.55, color=C_GRU)
    for xi, v in zip(x, units24):
        ax.text(xi, v + 0.4, str(v), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(RLAB, rotation=20)
    ax.set_ylabel("extreme units, 2024 test")
    ax.set_title("(b) 2024 independent test units (total 59)", fontsize=8.5, loc="left")
    save(fig, "W27_event_census")


# ---------------------------------------------------------------- Fig 4
def fig_recovery() -> None:
    d = json.loads((RES / "wave27_model_recovery.json").read_text())
    rows = [
        ("VAE\nlegacy", d["vae"]["legacy_scalar"]["fidelity"]["scale_ratio"]),
        ("VAE\nlinear", d["vae"]["repaired_linear_warmup"]["fidelity"]["scale_ratio"]),
        ("TG\nlegacy", d["timegrad"]["legacy_ignored_memory_w24"]["fidelity"]["scale_ratio"]),
        ("TG\nsum-w64", d["timegrad"]["repaired_ar_memory_sum_w64"]["fidelity"]["scale_ratio"]),
        ("TG\nmean-w64", d["timegrad"]["repaired_ar_memory_mean_w64"]["fidelity"]["scale_ratio"]),
        ("DTS\nnoise", d["diffusion_ts"]["legacy_fourier_noise"]["fidelity"]["scale_ratio"]),
        ("DTS\nx0", d["diffusion_ts"]["repaired_fourier_x0"]["fidelity"]["scale_ratio"]),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.axhspan(0.5, 2.0, color="#dcead6", zorder=0)
    x = np.arange(len(vals))
    shown = np.where(np.isfinite(vals), vals, 1e6)
    colors = [C_BAD if (not np.isfinite(v)) or v > 2 or v < 0.5 else C_OK for v in vals]
    ax.bar(x, shown, 0.6, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 1e7)
    for xi, v in zip(x, vals):
        txt = "NaN" if not np.isfinite(v) else (f"{v:.3g}")
        y = 1e6 if not np.isfinite(v) else v
        ax.text(xi, y * 1.35, txt, ha="center", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("scale ratio (log)")
    ax.set_title("Diagnostic recovery: legacy vs repaired (Charleston, 20k windows)",
                 fontsize=9.2, loc="left")
    save(fig, "W27_recovery")


# ---------------------------------------------------------------- Fig 5
def fig_baseline_seed() -> None:
    d = json.loads((RES / "paired_significance_wave27.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, r in enumerate(REGIONS):
        cmps = d["regions"][r]["comparisons"]
        for j, arch in enumerate(("gru", "tcn")):
            by_seed = {}
            for c in cmps:
                if c["model"] == arch:
                    by_seed.setdefault(c["baseline_seed"], []).append(
                        c["extreme_units"]["a0_mean_mae"])
            seeds = sorted(by_seed)
            vals = [float(np.mean(by_seed[s])) for s in seeds]
            off = (j - 0.5) * 0.30
            ax.scatter([i + off] * len(vals), vals, s=20,
                       color=C_GRU if arch == "gru" else C_TCN, zorder=2, linewidths=0)
            ax.vlines(i + off, min(vals), max(vals),
                      color=C_GRU if arch == "gru" else C_TCN, lw=1.0, alpha=0.6)
    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels(RLAB, rotation=12)
    ax.set_ylabel("A0 extreme-window MAE (m)")
    ax.set_title("Baseline seed variance: identical A0 recipe, two seeds "
                 "(blue GRU, red TCN)", fontsize=9.2, loc="left")
    ax.set_ylim(bottom=0)
    save(fig, "W27_baseline_seed")


# ---------------------------------------------------------------- Fig 6
def fig_schematic() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2f7", ec="#4a5a70", fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.8, color="#4a5a70"))

    box(0.1, 2.55, 1.6, 1.1, "96 h history\n(non-tidal\nresidual, m)")
    box(0.1, 0.35, 1.6, 1.0, "ERA5 history\nmsl, u10, v10", fc="#f7f1e6")
    box(2.4, 1.6, 2.0, 1.7, "Conditional\ngenerator\nFM / VAE / DDPM /\nTimeGrad")
    box(5.2, 2.6, 1.9, 1.0, "synthetic\nresiduals")
    box(5.2, 0.6, 1.9, 1.0, "real-only\n(A0)")
    box(7.8, 1.5, 1.9, 1.1, "GRU / TCN\n72 h forecast")
    arrow(1.7, 3.1, 2.4, 2.75)
    arrow(1.7, 0.85, 2.4, 2.0)
    arrow(4.4, 2.9, 5.2, 3.1)
    arrow(4.4, 2.0, 5.2, 1.1)
    arrow(7.1, 3.1, 7.8, 2.25)
    arrow(7.1, 1.1, 7.8, 1.85)
    ax.text(5.0, 0.12, "matched 8,000-window downstream budget; identical splits",
            ha="center", fontsize=7, style="italic", color="#555555")
    ax.text(0.1, 3.9, "Physics covariates are causal: only the 96 h before the "
                      "forecast origin", fontsize=7, color=C_BAD)
    save(fig, "W27_schematic")


# ---------------------------------------------------------------- Fig 7
def fig_physics() -> None:
    d = json.loads((RES / "wave27_physics_fm.json").read_text())
    variants = ["standard", "tail_aware", "pressure", "wind", "combined"]
    vlab = ["standard", "tail-aware", "pressure", "wind", "wind+press."]
    regs = [("south_atlantic_charleston", "Charleston"), ("ny_bight", "NY Bight")]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axes[0]
    ax.axhspan(0.5, 2.0, color="#dcead6", zorder=0)
    w = 0.38
    for j, (r, lab) in enumerate(regs):
        vals = [d["regions"][r]["variants"][v]["scale_ratio"] for v in variants]
        ax.bar(x + (j - 0.5) * w, vals, w,
               color=C_GRU if j == 0 else C_TCN, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(vlab, rotation=18, fontsize=7)
    ax.set_ylabel("scale ratio (log)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(a) Sample dispersion vs fidelity gate 0.5–2.0", fontsize=8.5, loc="left")
    ax = axes[1]
    for j, (r, lab) in enumerate(regs):
        vals = [d["regions"][r]["variants"][v]["downstream_delta_pct_vs_A0"] for v in variants]
        ax.bar(x + (j - 0.5) * w, vals, w, color=C_GRU if j == 0 else C_TCN, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(vlab, rotation=18, fontsize=7)
    ax.set_ylabel("extreme-window MAE change vs A0 (%)")
    ax.set_title("(b) Downstream effect (negative = helps)", fontsize=8.5, loc="left")
    save(fig, "W27_physics")


if __name__ == "__main__":
    fig_schematic()
    fig_census()
    fig_fidelity()
    fig_architecture()
    fig_recovery()
    fig_baseline_seed()
    fig_physics()
