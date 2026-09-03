#!/usr/bin/env python3
"""Regenerate fig:fpvarying for the REALISTIC false-positive-tolerant mix
specialist (mix-FT-fp, trained on coherent organism-module contamination), with
its high-completeness retrain (mix-FT-fp HQ) overlaid as dashed lines.

Reads the committed cross-split summary (data/spectra_fpreal_summary.csv), which
already carries the per-(fn,fp) mean and standard deviation across the 10
phylum-holdout splits -- so no per-split parquet is needed. This replaces the
earlier figure, which was built from the uniform-FP model (mixFP) and therefore
understated the recall the realistic model trades for coherent-FP rejection."""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_results import OKABE, setup_style, despine, shade_anc

FP_LIST = (0.01, 0.025, 0.05, 0.1, 0.2)


def series(rows, fam, fpv, mean_col, std_col):
    pts = [(float(r['fn']), float(r[mean_col]), float(r.get(std_col) or 0.0))
           for r in rows
           if r['family'] == fam and abs(float(r['fp']) - fpv) < 1e-9]
    if not pts:
        return None
    pts.sort()
    x = np.array([p[0] for p in pts])
    m = np.array([p[1] for p in pts])
    s = np.array([p[2] for p in pts])
    return x, m, s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', default='data/spectra_fpreal_summary.csv')
    p.add_argument('--out', default='analysis/figures/spectra_fp_varying.pdf')
    p.add_argument('--fam', default='nohidden_HO_T20_mixFTfpreal',
                   help='realistic FP-tolerant family (solid curves)')
    p.add_argument('--fam-hq', default='nohidden_HO_T20_mixFTfprealhq',
                   help='high-completeness retrain (dashed overlay); '
                        'pass empty to omit')
    args = p.parse_args(argv)

    rows = list(csv.DictReader(open(args.summary)))
    fams = {r['family'] for r in rows}
    if args.fam not in fams:
        raise SystemExit(f'[err] family {args.fam} absent from {args.summary}')
    avail = sorted({float(r['fp']) for r in rows if r['family'] == args.fam})
    fps = [f for f in FP_LIST if any(np.isclose(avail, f))]
    has_hq = bool(args.fam_hq) and args.fam_hq in fams

    setup_style()
    cols = [OKABE['sky'], OKABE['green'], OKABE['blue'], OKABE['orange'],
            OKABE['verm'], OKABE['purple']]
    fig, (axM, axE) = plt.subplots(1, 2, figsize=(10.4, 4.4),
                                   constrained_layout=True)
    band_lab = r'$\pm1$ SD (10 splits)'
    for ax, mean_col, std_col, ylab in (
            (axM, 'MCC_mean', 'MCC_std', 'MCC  (higher = better)'),
            (axE, 'ece_mean', 'ece_std', 'ECE  (lower = better)')):
        shade_anc(ax)
        for i, fpv in enumerate(fps):
            col = cols[i % len(cols)]
            real = series(rows, args.fam, fpv, mean_col, std_col)
            if real is None:
                continue
            x, m, s = real
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0,
                            label=band_lab if i == 0 else None)
            ax.plot(x, m, color=col, marker='o', ms=4, label=f'FP = {fpv:g}')
            if has_hq:
                hq = series(rows, args.fam_hq, fpv, mean_col, std_col)
                if hq is not None:
                    ax.plot(hq[0], hq[1], color=col, ls='--', lw=1.1,
                            alpha=0.85)
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylab)
    axM.set_ylim(0, 1)
    if has_hq:
        axM.plot([], [], color='0.35', ls='--', lw=1.1,
                 label='HQ retrain (dashed)')
    axM.set_title('(A)  MCC across FP  (higher better)', loc='left',
                  fontweight='bold')
    axE.set_title('(B)  ECE across FP  (lower better)', loc='left',
                  fontweight='bold')
    axM.legend(fontsize=7, loc='upper right', framealpha=0.9)
    fig.savefig(args.out)
    print(f'wrote {args.out}  (fam={args.fam}, hq={has_hq}, fps={fps})')


if __name__ == '__main__':
    main()
