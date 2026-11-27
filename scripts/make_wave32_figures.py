#!/usr/bin/env python3
"""Wave 3.2 final manuscript figures (F7-F12).

Every figure is generated from an immutable experiment artifact; no
hand-edited values. See paper/FIGURE_REGISTRY.md for the source mapping.

Usage:
    python3 scripts/make_wave32_figures.py            # build all available
    python3 scripts/make_wave32_figures.py F7 F9      # build a subset
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "paper", "figures")

REGION_LABELS = {
    "ny_bight": "New York Bight",
    "us_gulf_coast": "US Gulf Coast",
    "south_atlantic_charleston": "Charleston",
    "southeast_australia": "Southeast Australia",
    "thames_north_sea": "Thames / North Sea",
}
REGION_ORDER = [
    "southeast_australia",
    "thames_north_sea",
    "us_gulf_coast",
    "south_atlantic_charleston",
    "ny_bight",
]
CELL_LABELS = ["Std-GRU", "Std-TCN", "Tail-GRU", "Tail-TCN"]

BENEFIT = "#1b6ca8"   # negative delta (error decrease)
HARM = "#c0392b"      # positive delta (error increase)
NS_GREY = "#9aa0a6"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, name)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {name}")


def _load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


# ---------------------------------------------------------------- F7
def fig_F7_five_region_effects():
    d = _load("data/results/wave30_expanded_five_region.json")
    comps = {(c["region"], c["variant"], c["architecture"]): c
             for c in d["primary_comparisons"]}

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = 0
    xticks, xlabels, xdiv = [], [], []
    for region in REGION_ORDER:
        xdiv.append(x - 0.6)
        for vi, variant in enumerate(["standard", "tail_aware"]):
            for ai, arch in enumerate(["gru", "tcn"]):
                c = comps[(region, variant, arch)]
                val = c["mean_delta_pct"]
                err = c.get("seed_std_pct", 0.0)
                surv = c["survives_bonferroni"]
                color = BENEFIT if val < 0 else HARM
                ax.errorbar(x, val, yerr=err, fmt="o", color=color,
                            markerfacecolor=color if surv else "white",
                            markeredgewidth=1.4, markeredgecolor=color,
                            markersize=7, capsize=3, elinewidth=1.2,
                            zorder=3 if surv else 2)
                xticks.append(x)
                xlabels.append(CELL_LABELS[vi * 2 + ai])
                x += 1
        x += 1.4  # gap between regions
    for xd in xdiv[1:]:
        ax.axvline(xd - 0.4, color="0.85", lw=0.8, zorder=1)
    ax.axhline(0, color="k", lw=1.0, zorder=1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlim(-0.8, x - 0.6)
    ax.set_ylabel("Forecast error change vs. no augmentation (%)\nbelow 0 improves, above 0 harms")
    # bold region names centred beneath each group's cell labels
    region_short = {
        "southeast_australia": "SE Australia",
        "thames_north_sea": "Thames / N. Sea",
        "us_gulf_coast": "US Gulf Coast",
        "south_atlantic_charleston": "Charleston",
        "ny_bight": "NY Bight",
    }
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, region in enumerate(REGION_ORDER):
        centre = i * 5.4 + 1.5
        ax.text(centre, -0.34, region_short[region], transform=trans,
                ha="center", va="top", fontsize=8, fontweight="bold",
                clip_on=False)
    fig.subplots_adjust(bottom=0.30)
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="k",
                   markeredgecolor="k", label="Bonferroni survivor (filled)"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="0.4", label="not significant (open)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    _save(fig, "W32_five_region_effects.pdf")


# ---------------------------------------------------------------- F8
def fig_F8_arch_interaction():
    d = _load("data/results/wave30_architecture_interaction.json")
    comps = {(c["region"], c["variant"]): c for c in d["comparisons"]}

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.36
    for ri, region in enumerate(REGION_ORDER):
        for vi, variant in enumerate(["standard", "tail_aware"]):
            c = comps[(region, variant)]
            val = c["tcn_minus_gru"] * 1000.0  # m -> mm
            surv = c["survives_bonferroni"]
            x = ri + (vi - 0.5) * width
            ax.bar(x, val, width=width * 0.92,
                   color=("#2c7fb8" if variant == "standard" else "#a6bddb"),
                   edgecolor="k" if surv else "none", linewidth=1.2,
                   zorder=3)
            if surv:
                ax.text(x, val * 0.55, f"p$_B$={c['bonferroni_p']:.3f}",
                        ha="center", va="center", rotation=90, fontsize=7,
                        color="white" if abs(val) > 0.8 else "k", zorder=4)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(REGION_ORDER)))
    ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER],
                       rotation=20, ha="right")
    ax.set_ylabel("TCN minus GRU augmentation effect (mm)")
    ymin, ymax = ax.get_ylim()
    ax.text(0.02, 0.97,
            f"{d['summary']['n_survive_bonferroni']}/10 survive "
            f"Bonferroni $\\alpha$={d['bonferroni_alpha']}",
            transform=ax.transAxes, fontsize=8.5, va="top")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2c7fb8", label="standard FM"),
        plt.Rectangle((0, 0), 1, 1, color="#a6bddb", label="tail-aware FM"),
        plt.Line2D([], [], marker="s", ls="", markerfacecolor="white",
                   markeredgecolor="k", label="black edge = Bonferroni survivor"),
    ]
    ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=7.5,
              ncol=3, bbox_to_anchor=(0.5, 1.14))
    _save(fig, "W32_arch_interaction.pdf")


# ---------------------------------------------------------------- F9
def fig_F9_heterogeneity_severity():
    d = _load("data/results/wave30_mechanism_light.json")
    arms = ["standard_gru", "standard_tcn", "tail_aware_gru", "tail_aware_tcn"]
    arm_labels = ["Std-GRU", "Std-TCN", "Tail-GRU", "Tail-TCN"]
    M = np.full((len(arms), len(REGION_ORDER)), np.nan)
    P = np.full_like(M, np.nan)
    for ai, arm in enumerate(arms):
        for ri, region in enumerate(REGION_ORDER):
            cell = d["severity_dependence"][arm][region]
            M[ai, ri] = cell["spearman_delta_vs_peak"]
            P[ai, ri] = cell.get("p_holm", cell.get("spearman_p_peak", 1.0))

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for ai in range(len(arms)):
        for ri in range(len(REGION_ORDER)):
            sig = P[ai, ri] < 0.05
            ax.text(ri, ai, f"{M[ai, ri]:+.2f}",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if sig else "normal",
                    color=("k" if sig else "0.45"),
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="none" if sig else "none",
                              ec="k" if sig else "none", lw=1.0))
    ax.set_xticks(range(len(REGION_ORDER)))
    ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER],
                       rotation=20, ha="right")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arm_labels)
    ax.set_title("Spearman $\\rho$: augmentation effect vs. event severity (peak residual)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("$\\rho$  (+ = harm grows with severity)")
    fig.subplots_adjust(bottom=0.34)
    fig.text(0.02, 0.015,
             "Boxed values: Holm-adjusted $p<0.05$. Negative $\\rho$: harm concentrated in "
             "weak events (Thames, Charleston); positive: harm concentrated in the most "
             "severe events (Southeast Australia).",
             fontsize=7.5)
    _save(fig, "W32_heterogeneity_severity.pdf")


# ---------------------------------------------------------------- F10
def fig_F10_representative_extremes():
    """Historical candidate-zoo forecasts (retrospective test), SE-Aus vs Thames."""
    cases = [
        ("southeast_australia", "A0_real_only", "A3_fm_s42_ratio_0.1"),
        ("thames_north_sea", "A0_real_only", "A3_fm_s42_ratio_0.1"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=False)

    for ax, (region, a0, a3) in zip(axes, cases):
        f0 = os.path.join(ROOT, f"data/predictions/v2/{region}_{a0}_gru_seed42.npz")
        f3 = os.path.join(ROOT, f"data/predictions/v2/{region}_{a3}_gru_seed42.npz")
        z0, z3 = np.load(f0), np.load(f3)
        obs0 = z0["obs_residual"]
        eids = z0["event_ids"]
        # representative window: highest observed residual peak among genuine
        # independent event units (background windows carry the sentinel -999)
        evmask = eids != -999
        peaks = obs0.max(axis=1)
        vmask = np.where(evmask)[0]
        k = int(vmask[np.argmax(peaks[vmask])])
        eid = int(eids[k])
        assert int(z3["event_ids"][k]) == eid, "window alignment mismatch"
        h = np.arange(72)
        ax.plot(h, obs0[k], color="k", lw=2.0, label="observed residual")
        ax.plot(h, z0["pred_residual"][k], color="#c0392b", lw=1.4,
                label="A0 real-only GRU")
        ax.plot(h, z3["pred_residual"][k], color="#1b6ca8", lw=1.4,
                label="A3 FM-augmented GRU")
        r0 = float(np.sqrt(np.mean((z0["pred_residual"][k] - obs0[k]) ** 2)))
        r3 = float(np.sqrt(np.mean((z3["pred_residual"][k] - obs0[k]) ** 2)))
        ax.set_title(f"{REGION_LABELS[region]} — event {eid} "
                     f"(peak {peaks[k]:.2f} m; RMSE A0 {r0:.3f} vs A3 {r3:.3f} m)")
        ax.set_ylabel("residual (m)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("forecast horizon (h)")
    fig.text(0.01, -0.02,
             "Historical candidate-zoo archives (retrospective 2022-2024 test "
             "window; highest-peak independent event unit per region, illustrative "
             "only). The Thames and Southeast Australia residual series carry "
             "regional datum offsets inherited from the source archives.",
             fontsize=7.5)
    _save(fig, "W32_representative_extremes.pdf")


# ---------------------------------------------------------------- F11 (P2)
def fig_F11_fidelity_stratified():
    src = os.path.join(ROOT, "data/results/wave30_generator_fidelity.json")
    if not os.path.exists(src):
        print("[SKIP] F11: wave30_generator_fidelity.json not final yet (P2 running)")
        return None
    d = _load("data/results/wave30_generator_fidelity.json")
    # Filled in after P2 acceptance; structure mirrors W27_fidelity_gate.
    raise NotImplementedError("F11 implementation lands at P2 acceptance")


# ---------------------------------------------------------------- F12 (P3)
def fig_F12_tail_calibration():
    src = os.path.join(ROOT, "data/results/wave32_tail_calibration.json")
    if not os.path.exists(src):
        print("[SKIP] F12: wave32_tail_calibration.json not created yet (P3)")
        return None
    raise NotImplementedError("F12 implementation lands at P3 completion")


FIGS = {"F7": fig_F7_five_region_effects,
        "F8": fig_F8_arch_interaction,
        "F9": fig_F9_heterogeneity_severity,
        "F10": fig_F10_representative_extremes,
        "F11": fig_F11_fidelity_stratified,
        "F12": fig_F12_tail_calibration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figures", nargs="*", default=None)
    args = ap.parse_args()
    names = args.figures or list(FIGS)
    for name in names:
        FIGS[name]()


if __name__ == "__main__":
    main()
