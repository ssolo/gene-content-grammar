#!/usr/bin/env python3
"""
plot_extant_marginal_noise.py -- extant recovery under marginal-type (common-gene)
contamination: the un-fine-tuned generalist (HO-T20) against the
contamination-trained marginal model, on the six held-out test genomes.

Reads data/interactome/fpcompare/recover_<tag>_<arm>_mfp.tsv (arm in {gen,marg}),
produced by scripts/interactome/run_fp_compare_marginal_noise.sh, and writes two
grouped-bar panels at fn=0.9, the deep-loss regime: (A) recall of truly-present
genes, (B) fraction of the injected contamination removed. Higher is better in both.

Usage: python scripts/plot_extant_marginal_noise.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = os.path.join(HERE, 'data/interactome/fpcompare')
C_GEN, C_MARG = '#999999', '#D55E00'
GEN = [('ecoli', 'E. coli'), ('medbac', 'median\nbacterium'),
       ('medbac_hq', 'median bac\n(HQ)'), ('arch', 'M. jannaschii'),
       ('medarc', 'median\narchaeon'), ('medarc_hq', 'median arc\n(HQ)')]


# Both returns are percentages at the 0.5 posterior call threshold: recall over
# the truly-present genes, and removal over the injected false positives only.
def rfp(tag, arm, fn):
    p = f'{FP}/recover_{tag}_{arm}_mfp.tsv'
    if not os.path.exists(p):
        return np.nan, np.nan
    d = pd.read_csv(p, sep='\t'); d = d[np.isclose(d.fn, fn)]
    pres = d[d.truth == 1]
    rec = (pres.denoised_prob > 0.5).mean() if len(pres) else np.nan
    inj = d[(d.truth == 0) & (d.input_present == 1)]
    fpr = (inj.denoised_prob <= 0.5).mean() if len(inj) else np.nan
    return rec * 100, fpr * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fn', type=float, default=0.9)
    ap.add_argument('--out', default=os.path.join(HERE, 'analysis/figures/fig_extant_marginal_noise.pdf'))
    a = ap.parse_args()

    rec_g, rec_m, fpr_g, fpr_m = [], [], [], []
    for tag, _ in GEN:
        rg, fg = rfp(tag, 'gen', a.fn); rm, fm = rfp(tag, 'marg', a.fn)
        rec_g.append(rg); rec_m.append(rm); fpr_g.append(fg); fpr_m.append(fm)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight', 'figure.dpi': 120})
    fig, (axR, axF) = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    x = np.arange(len(GEN)); w = 0.38
    for ax, g, m, title in [(axR, rec_g, rec_m, '(A)  recall of truly-present genes'),
                            (axF, fpr_g, fpr_m, '(B)  contamination removed')]:
        ax.bar(x - w/2, g, w, color=C_GEN, label='generalist (HO-T20)')
        ax.bar(x + w/2, m, w, color=C_MARG, label='marginal (fine-tuned)')
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in GEN], fontsize=8)
        ax.set_ylabel('%'); ax.set_ylim(0, 100)
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25, lw=0.5)
    axR.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.suptitle(f'Extant recovery under realistic deep-ancestral noise '
                 f'(marginal-type contamination, $\\mathrm{{fp}}=0.01$, '
                 f'$\\mathrm{{fn}}={a.fn}$)', fontsize=10.5)
    fig.savefig(a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
