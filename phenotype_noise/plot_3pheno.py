#!/usr/bin/env python3
"""Combined 3-phenotype figure, "Denoising rescues phenotype prediction under gene
loss" (Fig 4), in the main-text Figure-1 house style.

Noise-robust predictor only, 2 rows x 3 columns keyed by the GRID table below:
  row 1  recovery -- MCC (oxygen use), MCC (monoderm/diderm), RMSE in C (OGT).
  row 2  calibration -- ECE, ECE, calibration error in C.
Each panel draws one curve per false-positive rate f_P in {0, 0.01, 0.05} against
the gene-loss rate f_N: solid with markers = Ising-denoised reconstruction, dashed
= raw noisy input, each behind a low-alpha +/-1 s.d. band over the 10 whole-phylum
splits.

Inputs are the three pickled metric caches written by the denoise-then-predict
runs, each a dict keyed 'robust_noisy'/'robust_denoised' -> {(f_N, f_P): {metric:
(mean, s.d.)}}; a missing cache or a missing metric draws a placeholder panel
rather than failing.

  python3 phenotype_noise/plot_3pheno.py \
    --aerob results/aerob_cache.pkl --ogt results/ogt_cache.pkl \
    --diderm results/diderm_cache.pkl --out analysis/figures/fig_phenotype_denoise_3pheno
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---- Figure-1 house style: Okabe-Ito colour-blind-safe palette
OKABE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
         "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10.5, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5,
})
ANC_LO, ANC_HI = 0.75, 0.90          # shaded band: the deep-ancestral gene-loss regime


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def shade_anc(ax):
    ax.axvspan(ANC_LO, ANC_HI, color="0.85", alpha=0.45, lw=0, zorder=0)


# (f_P, colour, marker) for the curves drawn in every panel
FP_STYLE = [(0.0, OKABE["blue"], "o"), (0.01, OKABE["orange"], "s"), (0.05, OKABE["green"], "^")]
DROP_FP = 0.15   # f_P present in the caches but not plotted

# Columns are phenotypes, rows are recovery then calibration.  Each cell is
# (pheno_key, column_title [row 0 only], metric, y-label, better, y-max, panel-letter).
GRID = [
    [("aerob", "Oxygen use", "mcc", "MCC", "higher", None, "a"),
     ("diderm", "Monoderm / diderm", "mcc", "MCC", "higher", None, "b"),
     ("ogt", "Optimal growth temp.", "rmse", "RMSE (°C)", "lower", 15, "c")],
    [("aerob", None, "ece", "ECE", "lower", None, "d"),
     ("diderm", None, "ece", "ECE", "lower", None, "e"),
     ("ogt", None, "calib_c", "calibration error (°C)", "lower", 6, "f")],
]


def ms(cell):
    if isinstance(cell, (tuple, list, np.ndarray)):
        return float(cell[0]), (float(cell[1]) if len(cell) > 1 and cell[1] is not None else 0.0)
    return float(cell), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aerob"); ap.add_argument("--ogt"); ap.add_argument("--diderm")
    ap.add_argument("--out", default="analysis/figures/fig_phenotype_denoise_3pheno")
    ap.add_argument("--no-bands", action="store_true", help="hide the +/-1 s.d. split bands")
    a = ap.parse_args()
    loaded = {k: pickle.load(open(v, "rb")) for k, v in
              {"aerob": a.aerob, "ogt": a.ogt, "diderm": a.diderm}.items() if v and os.path.exists(v)}
    bands = not a.no_bands

    fig, axes = plt.subplots(len(GRID), 3, figsize=(13.5, 7.6), sharex=True, squeeze=False,
                             constrained_layout=True)
    for ri, rowcells in enumerate(GRID):
        for ci, (key, ctitle, met, ylab, better, ymax, letter) in enumerate(rowcells):
            ax = axes[ri, ci]; cache = loaded.get(key)
            shade_anc(ax); despine(ax)
            if cache is None:
                ax.text(0.5, 0.5, f"{key}\n(pending)", ha="center", va="center",
                        transform=ax.transAxes, color="0.6"); continue
            na, da = "robust_noisy", "robust_denoised"
            # A cache built before this metric existed simply lacks the key.
            sample = next(iter(cache[na].values()), {})
            if met not in sample:
                ax.text(0.5, 0.5, f"{met}\n(pending)", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
                ax.set_title(f"({letter})", loc="left", fontweight="bold"); continue
            fn_all = sorted({f for f, _ in cache[na]})
            for fp, col, mk in FP_STYLE:
                xr = np.array([f for f in fn_all if (f, fp) in cache[na]])
                if not len(xr):
                    continue
                m_r = np.array([ms(cache[na][(f, fp)][met])[0] for f in xr])
                m_d = np.array([ms(cache[da][(f, fp)][met])[0] for f in xr])
                if bands:
                    s_r = np.array([ms(cache[na][(f, fp)][met])[1] for f in xr])
                    s_d = np.array([ms(cache[da][(f, fp)][met])[1] for f in xr])
                    ax.fill_between(xr, m_d - s_d, m_d + s_d, color=col, alpha=0.18, lw=0)
                    ax.fill_between(xr, m_r - s_r, m_r + s_r, color=col, alpha=0.09, lw=0)
                ax.plot(xr, m_r, ls="--", color=col, lw=1.5, alpha=0.9)
                ax.plot(xr, m_d, ls="-", color=col, marker=mk, ms=3.6, lw=2.0,
                        label=fr"$f_P={fp:g}$")
            ax.set_xlim(0, 1)
            if met == "mcc":
                ax.set_ylim(-0.05, 1.0)
            if ymax is not None:
                ax.set_ylim(0, ymax)
            ax.set_ylabel(f"{ylab}  ({'↑' if better == 'higher' else '↓'})")
            arrow = "higher better" if better == "higher" else "lower better"
            ttl = f"({letter})  {ctitle}  ({arrow})" if (ri == 0 and ctitle) else f"({letter})  ({arrow})"
            ax.set_title(ttl, loc="left", fontweight="bold")
            if ri == len(GRID) - 1:
                ax.set_xlabel(r"false-negative noise  $f_N$")
            if ri == 0 and ci == 0:
                ax.legend(loc="lower left", fontsize=8.5, handlelength=1.6,
                          title=r"false-positive rate $f_P$", title_fontsize=8.0)

    handles = [
        Line2D([0], [0], color=OKABE["black"], lw=2.0, ls="-", marker="o", ms=3.6,
               label="Ising-denoised reconstruction"),
        Line2D([0], [0], color=OKABE["black"], lw=1.5, ls="--",
               label="raw noisy input"),
        Line2D([0], [0], marker="o", ls="none", ms=6, mfc=OKABE["blue"], mec="none",
               label=r"colour $=f_P$ (0 / 0.01 / 0.05)"),
        Patch(facecolor=OKABE["black"], alpha=0.18, lw=0,
              label=r"$\pm1$ s.d. over 10 whole-phylum splits"),
        Patch(facecolor="0.85", alpha=0.9, lw=0,
              label=r"deep-ancestral regime ($f_N\!=\!0.75$–$0.90$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), columnspacing=1.6, handlelength=1.9)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}.pdf/.png  ({sorted(loaded)})")


if __name__ == "__main__":
    main()
