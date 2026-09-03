#!/usr/bin/env python3
"""Coherent-FP rejection frontier figure.

recall (truly-present genes recovered) vs coherent-FP-removed (grafted
foreign-organism module genes silenced), swept over the decision threshold, for
the default / uniform-FP / realistic-FP models on the bacterial (bac val) and
mixed (mixed val) lines. Cross-split SD as a shaded band. The realistic-FP
curriculum's frontier sits ABOVE the others -- at matched recall it rejects
coherent contamination markedly better, while the uniform-FP curriculum barely
moves it. Data: scripts/coherent_fp_eval.py (data/coherent_fp_*.tsv)."""
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'default': '#999999', 'uniform': '#E69F00', 'real': '#0072B2', 'real_hq': '#009E73',
       'marginal_hq': '#D55E00', 'marginal_cons': '#CC79A7'}
LAB = {'default': 'default (no FP curriculum)', 'uniform': 'uniform-FP',
       'real': 'realistic (coherent) FP', 'real_hq': 'realistic FP, HQ-trained',
       'marginal_hq': 'marginal-FP, HQ-trained',
       'marginal_cons': 'marginal-FP + consistency'}
GROUPS = {'bac': {'default': 'bacFT', 'uniform': 'bacFTfp-unif', 'real': 'bacFTfp-real', 'real_hq': 'bacFTfp-realhq',
                  'marginal_hq': 'bacFTfp-marginalhq', 'marginal_cons': 'bacFTfp-marginalcons'},
          'mix': {'default': 'mixFT', 'uniform': 'mixFTfp-unif', 'real': 'mixFTfp-real', 'real_hq': 'mixFTfp-realhq',
                  'marginal_hq': 'mixFTfp-marginalhq', 'marginal_cons': 'mixFTfp-marginalcons'}}
KEYS = ['default', 'uniform', 'real', 'real_hq', 'marginal_hq', 'marginal_cons']
TITLE = {'bac': '(A)  bacterial specialists (bac val)',
         'mix': '(B)  mixed-domain models (mixed val)'}


def frontier(df, model, fn):
    s = df[(df.model == model) & (np.isclose(df.fn, fn))]
    rows = []
    for tau in sorted(s.tau.unique()):
        t = s[s.tau == tau]
        rec = t.n_recovered.sum() / t.n_present.sum() * 100
        rem = t.n_fp_removed.sum() / t.n_grafted.sum() * 100
        per = t.assign(r=t.n_fp_removed / t.n_grafted * 100).groupby('split').r.std(ddof=0)
        sd = float(t.assign(r=t.n_fp_removed / t.n_grafted * 100)
                    .groupby('split').r.mean().std())
        rows.append((rec, rem, sd if np.isfinite(sd) else 0.0))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fn', type=float, default=0.9)
    ap.add_argument('--glob', default='data/coherent_fp_*.tsv')
    ap.add_argument('--out', default='analysis/figures/coherent_fp_frontier.pdf')
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(a.glob)], ignore_index=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight', 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True, sharey=True)
    for ax, grp in zip(axes, ['bac', 'mix']):
        for key in KEYS:
            lbl = GROUPS[grp].get(key)        # marginal_cons is mix-only -> None for bac
            if lbl is None:
                continue
            fr = frontier(df, lbl, a.fn)
            if fr.size == 0:  # model not yet evaluated (e.g. HQ data pending) -> skip its line
                continue
            ax.plot(fr[:, 0], fr[:, 1], '-o', color=COL[key], label=LAB[key], ms=4, lw=1.7, zorder=3)
            ax.fill_between(fr[:, 0], fr[:, 1] - fr[:, 2], fr[:, 1] + fr[:, 2],
                            color=COL[key], alpha=0.15, lw=0)
        ax.set_xlabel('recall of truly-present genes (%)')
        ax.set_title(TITLE[grp], fontsize=10, loc='left', fontweight='bold')
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].set_ylabel('coherent FP removed (%)')
    axes[1].legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.suptitle(f'Coherent false-positive rejection frontier '
                 f'(deep-ancestral $\\mathrm{{fn}}={a.fn}$; a foreign organism\'s '
                 f'modules grafted in)', fontsize=10.5)
    fig.savefig(a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
