#!/usr/bin/env python3
"""
plot_laca_softlanding.py -- the LACA analogue of the builder-vs-pruner figure,
parallel to plot_lbca_softlanding.py but computed directly from the per-COG
prediction TSVs.

(A) For each of the five LACA inputs -- three sparse reconciliations (raw x=p)
    and two dense GLD reconstructions -- the denoised output decomposed into
    genes KEPT from the input plus ADDED (output present, input absent), with
    REMOVED (silenced) below the axis. The sparse reconciliations are built up;
    the dense GLD min-1 is pruned.
(B) Of the families an input supports at >= p, the fraction the model declines,
    across six support thresholds and the same five inputs.

Panel A calls a family present at input_prob > 0.5; panel B sweeps the threshold.
Writes analysis/figures/fig_laca_softlanding.pdf.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(x): return os.path.join(HERE, x)
C_KEPT, C_ADD, C_REM = '#0072B2', '#009E73', '#D55E00'
# per-input colours for the grouped panel B (Okabe-Ito)
IN_COL = ['#CC79A7', '#0072B2', '#56B4E9', '#D55E00', '#E69F00']

INPUTS = [  # (panel-A label, short label, file, expected-blue?)
    # The two uniform-origination roots are diffuse -- almost nothing clears 0.5
    # -- so a >0.5 count badly understates a graded input. Their blue bar is
    # instead the SUM OF EXPECTED input genes among the present set,
    # sum(input_prob) over output > 0.5. The dense/ML roots are sharp and the
    # binary count is used for them.
    ('COG-based\n(uniform\norig.)', 'COG-based', 'LACA_merged_pred.tsv', True),
    ('Euryroot\n(ML orig.)', 'Eury ML', 'LACA_euryroot_pred.tsv', False),
    ('Euryroot\n(uniform\norig.)', 'Eury unif', 'LACA_euryroot_uniform_pred.tsv', True),
    ('recount\nmin-1', 'min-1', 'LACA_gld_min1_pred.tsv', False),
    ('recount\nmin-4', 'min-4', 'LACA_gld_min4_pred.tsv', False),
]


def load(f):
    ip, mo = [], []
    with open(P(f)) as fh:
        h = fh.readline().rstrip('\n').split('\t')
        ii = h.index('input_prob'); mi = h.index('mean_actual')
        for ln in fh:
            c = ln.rstrip('\n').split('\t')
            if len(c) > max(ii, mi):
                try:
                    ip.append(float(c[ii])); mo.append(float(c[mi]))
                except ValueError:
                    pass
    return np.array(ip), np.array(mo)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight'})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.3), constrained_layout=True)

    kept, add, rem, out, labels, expflag = [], [], [], [], [], []
    perinput = []  # (short, ip, mo) for the grouped panel B
    for label, short, f, expected in INPUTS:
        ip, mo = load(f)
        present = mo > 0.5
        cnt = int(present.sum())
        if expected:   # blue = expected input genes among the present set
            blue = float(ip[present].sum())
            red_v = float(ip[~present].sum())     # expected input genes silenced
            green = cnt - blue
        else:          # binary >0.5 decomposition, for the sharp inputs
            blue = float((present & (ip > 0.5)).sum())
            green = float((present & (ip <= 0.5)).sum())
            red_v = float((~present & (ip > 0.5)).sum())
        kept.append(blue); add.append(green); rem.append(red_v); out.append(cnt)
        labels.append(label); expflag.append(expected); perinput.append((short, ip, mo))

    x = np.arange(len(labels))
    axA.bar(x, kept, color=C_KEPT, label='kept from input')
    axA.bar(x, add, bottom=kept, color=C_ADD, label='added (rescued)')
    axA.bar(x, [-v for v in rem], color=C_REM, label='removed (silenced)')
    axA.axhline(0, color='k', lw=0.8)
    for i, o in enumerate(out):
        axA.text(i, kept[i] + add[i] + 25, f'{o:,}', ha='center', fontsize=9,
                 fontweight='bold')
    axA.set_xticks(x)
    axA.set_xticklabels([f'{L}$^{{*}}$' if e else L for L, e in zip(labels, expflag)],
                        fontsize=8.4)
    axA.set_ylabel('families  ($^{*}$blue = expected input genes)')
    axA.set_ylim(top=max(k + a for k, a in zip(kept, add)) * 1.46)
    axA.set_title('(A)  builder vs pruner: input $\\to$ denoised LACA',
                  loc='left', fontsize=9.5, fontweight='bold')
    axA.legend(frameon=False, fontsize=8.0, loc='upper left')

    # Panel B reads as builder vs pruner too: builders decline a small fraction
    # of what their input supports at any threshold, pruners a large one.
    THR = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    xb = np.arange(len(THR))
    nin = len(perinput); w = 0.8 / nin
    for i, (short, ip, mo) in enumerate(perinput):
        fr = [(((ip >= t) & (mo <= 0.5)).sum()) / max((ip >= t).sum(), 1) for t in THR]
        axB.bar(xb + (i - (nin - 1) / 2) * w, fr, w, color=IN_COL[i], label=short)
    axB.set_xticks(xb); axB.set_xticklabels([f'{t:g}' for t in THR])
    axB.set_xlabel('input support threshold $p$ (families with support $\\geq p$)')
    axB.set_ylabel('fraction of supported families declined')
    axB.set_title('(B)  declined fraction by support, all five inputs',
                  loc='left', fontsize=9.5, fontweight='bold')
    axB.legend(frameon=False, fontsize=7.3, ncol=2, loc='upper right')

    o = P('analysis/figures/fig_laca_softlanding.pdf')
    fig.savefig(o)
    print('wrote', o)
    print('panel A (uniform roots: expected-gene blue):')
    for L, k, a, r, ot, e in zip(labels, kept, add, rem, out, expflag):
        print('  %-22s kept=%.0f added=%.0f removed=%.0f out=%d %s'
              % (L.replace('\n', ' '), k, a, r, ot, '(expected)' if e else ''))


if __name__ == '__main__':
    main()
