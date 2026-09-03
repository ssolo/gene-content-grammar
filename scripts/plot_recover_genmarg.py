#!/usr/bin/env python3
"""
plot_recover_genmarg.py -- per-genome extant recovery, generalist (HO-T20) vs the
fine-tuned marginal model, under realistic deep-ancestral (marginal-type) noise.

One figure per held-out test genome, two panels, generalist (grey) vs marginal
(vermillion), across the false-negative sweep fn = 0.5/0.75/0.9 at marginal-type
fp = 0.01: (A) recall of truly-present genes, (B) fraction of the common-gene
contamination removed. Reads
data/interactome/fpcompare/recover_<tag>_<arm>_mfp.tsv (arm in {gen, marg}),
written by scripts/interactome/run_fp_compare_marginal_noise.sh.

Usage: python scripts/plot_recover_genmarg.py
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
OUT = os.path.join(HERE, 'analysis/figures')
C_GEN, C_MARG = '#999999', '#D55E00'
FNS = [0.5, 0.75, 0.9]
# tag -> (species label, output stem, marginal-model label)
GENOMES = {
    'ecoli':     ('Escherichia coli',               'ecoli',     'bac-FT-fp-marginal'),
    'medbac':    ('Kryptonium thompsonii (median bacterium)', 'medbac', 'bac-FT-fp-marginal'),
    'medbac_hq': ('UBA7675 (HQ median bacterium)',   'medbac_hq', 'bac-FT-fp-marginal'),
    'arch':      ('Methanocaldococcus jannaschii',   'archaeon',  'mix-FT-fp-marginal'),
    'medarc':    ('SM1-50 (median archaeon)',        'medarc',    'mix-FT-fp-marginal'),
    'medarc_hq': ('Methanocorpusculum (HQ median archaeon)', 'medarc_hq', 'mix-FT-fp-marginal'),
}


def metrics(tag, arm):
    p = f'{FP}/recover_{tag}_{arm}_mfp.tsv'
    d = pd.read_csv(p, sep='\t')
    rec, fpr = [], []
    for fn in FNS:
        s = d[np.isclose(d.fn, fn)]
        pres = s[s.truth == 1]
        rec.append((pres.denoised_prob > 0.5).mean() * 100 if len(pres) else np.nan)
        inj = s[(s.truth == 0) & (s.input_present == 1)]
        fpr.append((inj.denoised_prob <= 0.5).mean() * 100 if len(inj) else np.nan)
    return rec, fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='one tag, else all')
    a = ap.parse_args()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight', 'figure.dpi': 120})
    x = np.arange(len(FNS)); w = 0.38
    tags = [a.only] if a.only else list(GENOMES)
    for tag in tags:
        sp, stem, mlab = GENOMES[tag]
        rg, fg = metrics(tag, 'gen'); rm, fm = metrics(tag, 'marg')
        fig, (axR, axF) = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
        for ax, g, m, ttl in [(axR, rg, rm, '(A)  recall of truly-present genes'),
                              (axF, fg, fm, '(B)  contamination removed')]:
            ax.bar(x - w/2, g, w, color=C_GEN, label='generalist (HO-T20)')
            ax.bar(x + w/2, m, w, color=C_MARG, label=mlab)
            for xi, (vg, vm) in enumerate(zip(g, m)):
                ax.text(xi - w/2, vg + 1.5, f'{vg:.0f}', ha='center', fontsize=7.5, color='#555')
                ax.text(xi + w/2, vm + 1.5, f'{vm:.0f}', ha='center', fontsize=7.5, color=C_MARG)
            ax.set_xticks(x); ax.set_xticklabels([f'fn={f}' for f in FNS], fontsize=9)
            ax.set_ylabel('%'); ax.set_ylim(0, 105)
            ax.set_title(ttl, loc='left', fontsize=9.5, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.25, lw=0.5)
        axR.legend(frameon=False, fontsize=8.2, loc='lower left')
        fig.suptitle(f'{sp} -- recovery under marginal-type contamination '
                     f'($\\mathrm{{fp}}=0.01$)', fontsize=10, fontweight='bold')
        out = f'{OUT}/fig_recover_{stem}_genmarg.pdf'
        fig.savefig(out); plt.close(fig)
        print(f'wrote {out}  (recall gen->marg @fn0.9 {rg[-1]:.0f}->{rm[-1]:.0f}; '
              f'FP-removed {fg[-1]:.0f}->{fm[-1]:.0f})')


if __name__ == '__main__':
    main()
