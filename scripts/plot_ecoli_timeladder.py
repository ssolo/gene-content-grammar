#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_ecoli_timeladder.py -- E. coli recovery vs evolutionary divergence time.

Each fine-tune of the marginal-HQ denoiser holds out a nested clade around the
E. coli genome (phylum > class > order > ... > the genome itself), so the
NEAREST relative left in training sits at a known divergence time.  This plots
ground-truth recovery (MCC / recall / precision) against that time, in the Fig. 1
style of scripts/plot_results.py.

The RUNGS table below is the authoritative ladder.  Its deep rungs are dated on
the Davin et al. (2025) time-calibrated tree, in which E. coli K-12 is a leaf;
the shallow rungs are literature estimates, because E. coli is the sole
Enterobacteriaceae tip in that backbone tree and so cannot date them.  Their
published ranges are drawn as horizontal error bars.

Reads recover_ecoli_ecolizoom_<rung>_ho3_fn468.tsv (columns
COG_ID/fn/truth/input_present/denoised_prob); present = denoised_prob > 0.5.
Where the noise-instance replicates reps_ecoli_ecolizoom_<rung>_ho3_fn468.tsv
exist (per-(fn, rep) TP/FP/FN/TN from scripts/score_ecoli_replicates.py) the
marker is their MEDIAN and a nested 95%/68% highest-density-interval band is
shaded behind the curve, so marker and band come from one consistent eval and
the marker sits at the centre of its band by construction.  Without replicates
the marker falls back to the single draw in recover_*.tsv, which can land
off-centre.  Rungs absent from --indir are skipped.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9',
         'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
         'verm': '#D55E00', 'purple': '#CC79A7'}

# Divergence time is 2x the MRCA node age: E. coli and the relative each evolve
# for T since their split, so 2T of evolution separates them.
DIV_FACTOR = 2.0
# (rung key, clade label, MRCA node age T [Ma], published (lo, hi) range in Ma,
# dated on the Davin tree).  Plotted x = DIV_FACTOR * T; ordered by age so the
# staggered top labels do not collide.
RUNGS = [
    # Escherichia crown, E. coli vs the cryptic/sister Escherichia (Walk 2009).
    ("species",      "Escherichia",                  25, (19, 31),     False),
    # Enterobacteriaceae stem, the enterobacterial radiation (Ochman & Wilson 1999).
    ("family",       "Enterobacteriaceae",          300, (250, 500),   False),
    ("order",        "Enterobacterales",            929, (929, 929),   True),
    ("intermediate", "Gammaproteobacteria\nsub-clade", 1563, (1436, 1765), True),
    ("class",        "Gammaproteobacteria",        2526, (2526, 2526), True),
    ("phylum",       "Proteobacteria",             2749, (2749, 2749), True),
]
# (fn, colour, marker) -- one curve per false-negative level
FN_STYLE = [(0.4, OKABE['sky'], 'o'), (0.6, OKABE['green'], 's'),
            (0.8, OKABE['verm'], 'D')]
PANELS = [("MCC", "MCC  (higher = better)"),
          ("recall", "recall"),
          ("prec", "precision")]


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10.5, 'axes.linewidth': 0.8, 'axes.grid': True,
        'grid.alpha': 0.25, 'grid.linewidth': 0.6, 'legend.fontsize': 8.5,
        'legend.frameon': False, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'lines.linewidth': 1.8, 'lines.markersize': 5,
    })


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def score(df, fn, use_input=False):
    """Metrics at noise level fn.

    use_input=True scores the NOISY INPUT (surviving genes plus injected false
    positives, before denoising): the model-free baseline, identical across
    rungs because the corruption seed is fixed per fn.
    """
    d = df[df["fn"].astype(str) == f"{fn:g}"]
    truth = d["truth"].to_numpy().astype(bool)
    if use_input:
        pred = d["input_present"].to_numpy().astype(float) > 0.5
    else:
        pred = d["denoised_prob"].to_numpy().astype(float) > 0.5
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn_ = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    rec = tp / (tp + fn_) if (tp + fn_) else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    den = math.sqrt((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
    mcc = (tp * tn - fp * fn_) / den if den else float('nan')
    return {"MCC": mcc, "recall": rec, "prec": prec, "nFP": fp}


def metric_vec(d, key):
    """Vectorised MCC/recall/precision over the reps_*.tsv confusion counts
    (TP/FP/FN/TN, one row per noise instance)."""
    tp = d["TP"].to_numpy().astype(float); fp = d["FP"].to_numpy().astype(float)
    fn_ = d["FN"].to_numpy().astype(float); tn = d["TN"].to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if key == "recall":
            return np.where(tp + fn_ > 0, tp / (tp + fn_), np.nan)
        if key == "prec":
            return np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        den = np.sqrt((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
        return np.where(den > 0, (tp * tn - fp * fn_) / den, np.nan)


def hpd(x, q):
    """Highest-density interval covering fraction q of a 1-D sample: the
    narrowest window holding ceil(q*n) points, exact for a unimodal sample."""
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    k = max(1, int(np.ceil(q * n)))
    if k >= n:
        return (float(x[0]), float(x[-1]))
    w = x[k - 1:] - x[:n - k + 1]
    i = int(np.argmin(w))
    return (float(x[i]), float(x[i + k - 1]))


def fn_slice(repdf, fn):
    return repdf[repdf["fn"].astype(str) == f"{fn:g}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis/ecoli_zoom")
    ap.add_argument("--repdir", default="analysis/ecoli_zoom",
                    help="dir with reps_ecoli_ecolizoom_<rung>_ho3_fn468.tsv "
                         "(noise-instance replicates -> the HPD band)")
    ap.add_argument("--out", default="analysis/figures/fig_ecoli_timeladder.pdf")
    A = ap.parse_args()
    setup_style()

    data, reps = {}, {}
    for rk, *_ in RUNGS:
        p = Path(A.indir) / f"recover_ecoli_ecolizoom_{rk}_ho3_fn468.tsv"
        if p.exists():
            data[rk] = pd.read_csv(p, sep="\t")
        rp = Path(A.repdir) / f"reps_ecoli_ecolizoom_{rk}_ho3_fn468.tsv"
        if rp.exists():
            reps[rk] = pd.read_csv(rp, sep="\t")
    have = [r for r in RUNGS if r[0] in data]
    print(f"rungs present: {[r[0] for r in have]}; "
          f"with replicates: {[r for r in reps]}")
    n_rep = max((len(d) // 3 for d in reps.values()), default=0)  # rows span 3 fn levels

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.0), constrained_layout=True)
    panel_tag = "ABC"
    for j, (ax, (key, ylab)) in enumerate(zip(axes, PANELS)):
        # Tint the shallow region whose rungs are literature-dated rather than
        # dated on the Davin tree.
        ax.axvspan(-80, 700, color='0.92', alpha=0.6, lw=0, zorder=0)
        def pt_y(rk, fn, key):
            """Median over the noise-instance replicates, i.e. the centre of the
            band; falls back to the single-instance recover TSV only where
            replicates are absent."""
            if rk in reps:
                return float(np.nanmedian(metric_vec(fn_slice(reps[rk], fn), key)))
            return score(data[rk], fn)[key]

        for fn, col, mk in FN_STYLE:
            # Nested 95% + 68% HPD over noise instances, behind the curve; the
            # markers are the per-(rung, fn) median of the SAME draws.
            band = [r for r in have if r[0] in reps]
            if band:
                bx = np.array([DIV_FACTOR * r[2] for r in band])
                vals = {r[0]: metric_vec(fn_slice(reps[r[0]], fn), key) for r in band}
                for qlev, al in ((0.95, 0.10), (0.68, 0.18)):
                    lo = np.array([hpd(vals[r[0]], qlev)[0] for r in band])
                    hi = np.array([hpd(vals[r[0]], qlev)[1] for r in band])
                    ax.fill_between(bx, lo, hi, color=col, alpha=al, lw=0, zorder=2)
            pts = [(DIV_FACTOR * r[2], pt_y(r[0], fn, key), r[4]) for r in have]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.8,
                    label=f"$f_N = {fn:g}$", zorder=3)
            # filled marker = Davin-dated node; open marker = literature estimate
            for x, y, davin in pts:
                ax.plot(x, y, marker=mk, ms=6, ls='none', zorder=4,
                        markerfacecolor=(col if davin else 'white'),
                        markeredgecolor=col, markeredgewidth=1.4)
            # Published date ranges, drawn once (on the fn=0.8 curve) to avoid
            # three coincident bars per rung.
            if fn == 0.8:
                for r in have:
                    lo, hi = r[3]
                    if hi > lo:
                        y = pt_y(r[0], fn, key)
                        ax.plot([DIV_FACTOR * lo, DIV_FACTOR * hi], [y, y],
                                color=col, lw=1.0, alpha=0.5, zorder=2)
        # Model-free baseline per noise level; constant across rungs.
        ref = data[have[0][0]]
        for fn, col, mk in FN_STYLE:
            b = score(ref, fn, use_input=True)[key]
            ax.axhline(b, color=col, ls='--', lw=1.1, alpha=0.75, zorder=1)
        ax.set_xlim(-80, 5800)                       # Ma, linear; recent left -> deep right
        ax.set_ylim(0, 1.0)
        despine(ax)
        ax.set_xlabel("divergence time from nearest\ntraining relative (Ma)")
        ax.set_ylabel(ylab)
        ax.text(0.0, 1.24, f"({panel_tag[j]})", transform=ax.transAxes,
                fontweight='bold', fontsize=12, va='bottom', ha='left')
        if key == "MCC":
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            h, l = ax.get_legend_handles_labels()
            h += [Line2D([0], [0], color='0.4', ls='--', lw=1.1),
                  Line2D([0], [0], color='0.4', marker='o', ls='none',
                         markerfacecolor='0.4', markeredgecolor='0.4'),
                  Line2D([0], [0], color='0.4', marker='o', ls='none',
                         markerfacecolor='white', markeredgecolor='0.4', markeredgewidth=1.4)]
            l += ['noisy input (baseline)', 'Davin 2025 (filled)', 'literature (open)']
            if reps:
                h += [Patch(facecolor='0.4', alpha=0.18, lw=0)]
                l += [f'68/95% HPD ({n_rep} noise draws)']
            ax.legend(h, l, loc='lower left', fontsize=6.8, ncol=1, handletextpad=0.5)
        # Staggered so the near-coincident deep pair (Proteobacteria ~2.75 Ga,
        # Gammaproteobacteria ~2.53 Ga) stays legible.
        for i, r in enumerate(have):
            x = DIV_FACTOR * r[2]
            ax.annotate(r[1], xy=(x, 1.0), xytext=(x, 1.012 + 0.075 * (i % 2)),
                        xycoords=('data', 'axes fraction'),
                        textcoords=('data', 'axes fraction'),
                        rotation=18, ha='center', va='bottom', fontsize=6.3,
                        color='0.25', linespacing=0.9,
                        arrowprops=dict(arrowstyle='-', color='0.78', lw=0.5))

    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(A.out)
    fig.savefig(str(A.out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"wrote {A.out} (+ .png)")


if __name__ == "__main__":
    main()
