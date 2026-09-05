#!/usr/bin/env python3
"""Single-panel divergence-time ladder at one noise level.

Recall and precision share a y-axis (both are percentages), plotted against
E. coli's divergence from its nearest training relative. Divergence time is
2x the MRCA node age, since E. coli and the relative each evolve for T after
the split. Defaults to the deepest corruption level, fn = 0.8.

  python3 scripts/plot_ecoli_ladder_single.py --out analysis/figures/fig_ladder_single.pdf
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OKABE = {'sky': '#56B4E9', 'green': '#009E73', 'verm': '#D55E00', 'blue': '#0072B2'}
DIV_FACTOR = 2.0

# rung key -> (clade label, MRCA node age T [Ma])
RUNGS = [("species", "Escherichia", 25), ("family", "Enterobacteriaceae", 300),
         ("order", "Enterobacterales", 929), ("intermediate", "Gammaproteobacteria sub-clade", 1563),
         ("class", "Gammaproteobacteria", 2526), ("phylum", "Pseudomonadota", 2749)]


def metrics(df, fn):
    s = df[abs(df['fn'] - fn) < 1e-9]
    pred = (s['denoised_prob'] > 0.5).to_numpy()
    truth = (s['truth'] == 1).to_numpy()
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn_ = int((~pred & truth).sum())
    return (100.0 * tp / (tp + fn_) if tp + fn_ else math.nan,
            100.0 * tp / (tp + fp) if tp + fp else math.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='analysis/ecoli_zoom')
    ap.add_argument('--fn', type=float, default=0.8)
    ap.add_argument('--out', default='analysis/figures/fig_ladder_single.pdf')
    a = ap.parse_args()

    plt.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': 11, 'axes.labelsize': 11.5, 'axes.titlesize': 12.5,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    data = {}
    for rk, _, _ in RUNGS:
        p = Path(a.indir) / f'recover_ecoli_ecolizoom_{rk}_ho3_fn468.tsv'
        if p.exists():
            data[rk] = pd.read_csv(p, sep='\t')
    have = [r for r in RUNGS if r[0] in data]
    if not have:
        raise SystemExit(f'no rung files under {a.indir}')

    xs = [DIV_FACTOR * r[2] / 1000.0 for r in have]   # Ga of divergence
    rec = [metrics(data[r[0]], a.fn)[0] for r in have]
    pre = [metrics(data[r[0]], a.fn)[1] for r in have]

    fig, ax = plt.subplots(figsize=(9.6, 6.4), constrained_layout=True)

    ax.axvspan(1.8, max(xs) * 1.06, color='#f4f4f4', zorder=0)
    ax.text(3.7, 97.5, 'beyond ~1.8 billion years: flat', fontsize=9.5,
            color='0.45', ha='center', style='italic', zorder=1)

    ax.plot(xs, pre, ls='--', color=OKABE['blue'], lw=2.2, marker='o', ms=6.5,
            mfc='white', mew=1.8, zorder=3)
    ax.plot(xs, rec, ls='-', color=OKABE['verm'], lw=2.6, marker='o', ms=6.5, zorder=3)

    ax.annotate(f'{pre[-1]:.0f}% correctly recovered', xy=(xs[-1], pre[-1]),
                xytext=(10, 2), textcoords='offset points', color=OKABE['blue'],
                fontsize=10.5, fontweight='bold', va='center', annotation_clip=False)
    ax.annotate(f'{rec[-1]:.0f}% recovered', xy=(xs[-1], rec[-1]),
                xytext=(10, 0), textcoords='offset points', color=OKABE['verm'],
                fontsize=10.5, fontweight='bold', va='center', annotation_clip=False)

    for x in xs:
        ax.plot([x, x], [55, 100], color='0.94', lw=0.7, zorder=0)


    ax.set_xlim(-0.25, max(xs) + 1.9)
    ax.set_ylim(55, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_ylabel('percent of the true gene set')
    ax.set_xlabel('divergence from the nearest training relative\n(billions of years of separate evolution)')
    ax.set_box_aspect(0.70)         # landscape plotting area
    ax.grid(axis='y', color='0.92', lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Rungs closer than 15% in time share one tick: the class and phylum nodes
    # differ by <10% and their labels would otherwise overlap.
    ticks, labels = [], []
    for (rk, lbl, T) in have:
        x = DIV_FACTOR * T / 1000.0
        if ticks and x - ticks[-1] < 0.50:
            labels[-1] = labels[-1] + ' / ' + lbl
            ticks[-1] = 0.5 * (ticks[-1] + x)
        else:
            ticks.append(x)
            labels.append(lbl)
    top = ax.secondary_xaxis('top')
    top.set_xticks(ticks)
    top.set_xticklabels(labels, rotation=30, ha='left', fontsize=8.6, color='0.42')
    top.tick_params(axis='x', length=3, color='0.75', pad=2)
    top.spines['top'].set_visible(False)


    ax.plot([], [], color=OKABE['verm'], ls='-', lw=2.4,
            label='recall  (of the true genes, how many recovered)')
    ax.plot([], [], color=OKABE['blue'], ls='--', lw=2.0,
            label='precision  (of the genes recovered, how many correct)')
    ax.legend(loc='lower left', frameon=False, fontsize=9.8, handlelength=2.6,
              bbox_to_anchor=(0.0, 0.015))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'))
    print(f'wrote {out}  (recall {rec[-1]:.1f}%, precision {pre[-1]:.1f}% at the deepest rung)')


if __name__ == '__main__':
    main()
