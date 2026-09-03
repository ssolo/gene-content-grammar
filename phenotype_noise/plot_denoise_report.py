#!/usr/bin/env python3
"""Report figure for the phenotype-prediction-under-gene-loss test.

2x2 grid:
    rows = MCC (accuracy)  /  ECE (calibration)
    cols = base predictor  /  noise-robust predictor

Each panel draws two curves against the gene-loss rate r_FN: the raw (noisy)
input (grey dashed) and the denoised reconstruction (blue solid), each averaged
over the r_FP values present in the cache (the pipeline default is
{0, 0.1, 0.2}) and banded by +/-1 s.d. across the ten phylum-holdout splits.

Out: analysis/figures/fig_phenotype_denoise.{pdf,png}

Usage:
    python3 phenotype_noise/plot_denoise_report.py \
        [--pkl phenotype_noise/results/denoise_then_predict_cache.pkl]

The cache comes from denoise_then_predict.py --cache. Each entry is
    d[arm][(fn, fp)][metric] = (mean_over_splits, sd_over_splits, ...)
for arm in {base,robust}x{noisy,denoised}, metric in {mcc,ece}.
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PH)
FIGDIR = os.path.join(REPO, "analysis", "figures")

RAW = "0.45"        # grey level for the raw (noisy) input
DEN = "#2166ac"     # denoised reconstruction


def _mean_sd(cell):
    """(mean, sd) from a cache cell, which may be a tuple, list, array or scalar;
    sd is 0.0 when the cell carries no second entry."""
    if isinstance(cell, (tuple, list, np.ndarray)):
        m = float(cell[0])
        s = float(cell[1]) if len(cell) > 1 and cell[1] is not None else 0.0
        return m, s
    return float(cell), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(PH, "results", "denoise_then_predict_cache.pkl"))
    ap.add_argument("--out", default=os.path.join(FIGDIR, "fig_phenotype_denoise"))
    A = ap.parse_args()
    d = pickle.load(open(A.pkl, "rb"))
    fn_grid = sorted({fn for fn, _ in d["base_noisy"]})
    fp_grid = sorted({fp for _, fp in d["base_noisy"]})
    fn = np.array(fn_grid)

    def fpmean(arm, met):
        """Mean (and mean s.d.) over r_FP, per gene-loss rate."""
        means, sds = [], []
        for f in fn_grid:
            ms = [_mean_sd(d[arm][(f, fp)][met]) for fp in fp_grid]
            means.append(np.mean([m for m, _ in ms]))
            sds.append(np.mean([s for _, s in ms]))
        return np.array(means), np.array(sds)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), sharex=True)
    cols = [("base", "Base predictor"), ("robust", "Noise-robust predictor")]
    rows = [("mcc", "MCC (accuracy)", (-0.05, 1.0)),
            ("ece", "Expected calibration error", (0.0, 0.52))]

    for ci, (who, ctitle) in enumerate(cols):
        for ri, (met, ylab, ylim) in enumerate(rows):
            ax = axes[ri, ci]
            m_r, s_r = fpmean(f"{who}_noisy", met)
            m_d, s_d = fpmean(f"{who}_denoised", met)
            ax.fill_between(fn, m_r - s_r, m_r + s_r, color=RAW, alpha=0.12, lw=0)
            ax.fill_between(fn, m_d - s_d, m_d + s_d, color=DEN, alpha=0.15, lw=0)
            ax.plot(fn, m_r, "--", color=RAW, lw=1.8, label="raw (noisy) input")
            ax.plot(fn, m_d, "-", color=DEN, lw=2.6, label="denoised reconstruction")
            ax.set_ylim(*ylim)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.25, lw=0.6)
            if ri == 0:
                ax.set_title(ctitle, fontsize=13)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=12)
            if ri == 1:
                ax.set_xlabel(r"gene-loss rate $r_{\mathrm{FN}}$ (fraction of genes deleted)",
                              fontsize=11)

    axb = axes[0, 0]
    m_rb, _ = fpmean("base_noisy", "mcc")
    m_db, _ = fpmean("base_denoised", "mcc")
    if 0.8 in fn_grid:
        j = fn_grid.index(0.8)
        axb.annotate(rf"$+{m_db[j]-m_rb[j]:.2f}$ MCC at $80\%$ loss",
                     xy=(0.8, m_db[j]), xytext=(0.06, 0.30), fontsize=10,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1,
                                     connectionstyle="arc3,rad=-0.2"))
    handles = [Line2D([0], [0], color=DEN, lw=2.6, ls="-", label="denoised reconstruction"),
               Line2D([0], [0], color=RAW, lw=1.8, ls="--", label="raw (noisy) input")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Denoising rescues phenotype (aerobicity) prediction under gene loss",
                 fontsize=13.5, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(A.out + ".pdf", bbox_inches="tight")
    fig.savefig(A.out + ".png", dpi=200, bbox_inches="tight")
    print(f"wrote {A.out}.pdf and {A.out}.png  (fp-mean over {fp_grid}, {len(fn_grid)} loss points)")
    if 0.8 in fn_grid:
        print(f"@80% loss (fp-mean): base raw {m_rb[j]:.3f} -> denoised {m_db[j]:.3f}")


if __name__ == "__main__":
    main()
