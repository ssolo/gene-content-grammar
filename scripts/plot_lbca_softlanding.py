#!/usr/bin/env python3
"""
plot_lbca_softlanding.py -- two-panel LBCA builder-vs-pruner figure.

Panel A reads data/lbca_comparison_summary.json (written by
scripts/compare_lbca_reconstructions.py); panel B reads the per-family
marginal tables marg_sparse.tsv / marg_soft.tsv / marg_soft_min4.tsv, whose
columns input_prob (the reconciliation's support p) and mean_actual (the
denoised posterior) are compared at mean_actual <= 0.5 to call a family
declined.

(A) For each LBCA reconstruction (sparse reconciliation x=p, softlanding min1,
    softlanding min4) the denoised output split into genes kept from the input
    and genes added (rescued), with genes removed (silenced) below the axis.
    The dense softlanding inputs are net-pruned; the sparse reconciliation is
    net-built.
(B) Of the families an input supports at >= p, the fraction the model declines.
    The sparse build declines only its weak tail; the dense softlanding inputs
    prune roughly uniformly across support.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_KEPT, C_ADD, C_REM = '#0072B2', '#009E73', '#D55E00'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default=os.path.join(HERE, 'data/lbca_comparison_summary.json'))
    ap.add_argument('--removal', default=os.path.join(HERE, 'data/lbca_reconciliation_removal_by_confidence.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'analysis/figures/fig_lbca_softlanding.pdf'))
    a = ap.parse_args()

    d = json.load(open(a.summary))
    r = d['recons']
    sp = r['sparse-rescue']
    # For the sparse reconciliation "kept" means retained at non-zero support and
    # "added" means called present where the input had zero support; the dense
    # softlanding inputs are binary, so their kept/added come straight through.
    sp_kept = sp['removal_by_confidence'][0]['kept']      # kept at p >= 0.01, of the 493-family support
    sp_add = sp['added_zero_support']
    sp_rem = sp['removal_by_confidence'][0]['removed']
    labels = ['sparse\nreconciliation\n(x=p)', 'GLD\nmin1', 'GLD\nmin4']
    kept = [sp_kept, r['soft-trim-min1']['kept'], r['soft-trim-min4']['kept']]
    add = [sp_add, r['soft-trim-min1']['added'], r['soft-trim-min4']['added']]
    rem = [sp_rem, r['soft-trim-min1']['removed'], r['soft-trim-min4']['removed']]
    out = [r['sparse-rescue']['output'], r['soft-trim-min1']['output'], r['soft-trim-min4']['output']]

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight', 'figure.dpi': 120})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)

    x = np.arange(3)
    axA.bar(x, kept, color=C_KEPT, label='kept from input')
    axA.bar(x, add, bottom=kept, color=C_ADD, label='added (rescued)')
    axA.bar(x, [-v for v in rem], color=C_REM, label='removed (silenced)')
    axA.axhline(0, color='k', lw=0.8)
    for i, o in enumerate(out):
        axA.text(i, kept[i] + add[i] + 30, f'{o:,}', ha='center', fontsize=9, fontweight='bold')
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel('families')
    axA.set_ylim(top=max(k + a for k, a in zip(kept, add)) * 1.28)  # headroom for the in-axes legend
    axA.set_title('(A)  builder vs pruner: input -> denoised LBCA', loc='left',
                  fontsize=9.5, fontweight='bold')
    axA.legend(frameon=False, fontsize=8.3, loc='upper left', ncol=3,
               columnspacing=1.0, handletextpad=0.4)

    LBCA_IN = [('sparse', 'marg_sparse.tsv', C_KEPT),
               ('GLD min-1', 'marg_soft.tsv', C_REM),
               ('GLD min-4', 'marg_soft_min4.tsv', '#E69F00')]

    def load_marg(fn):
        ip, mo = [], []
        for r in csv.DictReader(open(os.path.join(HERE, fn)), delimiter='\t'):
            ip.append(float(r['input_prob'])); mo.append(float(r['mean_actual']))
        return np.array(ip), np.array(mo)

    THR = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    xb = np.arange(len(THR)); nin = len(LBCA_IN); w = 0.8 / nin
    for i, (short, fn, col) in enumerate(LBCA_IN):
        ip, mo = load_marg(fn)
        fr = [(((ip >= t) & (mo <= 0.5)).sum()) / max((ip >= t).sum(), 1) for t in THR]
        axB.bar(xb + (i - (nin - 1) / 2) * w, fr, w, color=col, label=short)
    axB.set_xticks(xb); axB.set_xticklabels([f'{t:g}' for t in THR])
    axB.set_xlabel('input support threshold $p$ (families with support $\\geq p$)')
    axB.set_ylabel('fraction of supported families declined')
    axB.set_title('(B)  declined fraction by support, all three inputs',
                  loc='left', fontsize=9.5, fontweight='bold')
    axB.legend(frameon=False, fontsize=8.2)

    fig.savefig(a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
