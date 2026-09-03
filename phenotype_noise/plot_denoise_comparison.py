#!/usr/bin/env python3
"""Plot aerobicity prediction across noise, with and without the denoiser front-end.

Reads phenotype_noise/results/pheno_denoise_results.pkl (written by
denoise_phenotype_extension.py) and, for each false-positive rate, draws a
4-panel (MCC / F1 / Brier / ECE) figure of metric vs false-negative rate, one
line per arm:
    orig_noisy       original predictor, noisy input          (grey)
    robust_noisy     noise-robust predictor, noisy input      (blue)
    orig_denoised    original predictor, DENOISED input       (red)
    robust_denoised  noise-robust predictor, DENOISED input   (green)

Then prints the mean MCC gain from denoising per false-negative rate.

Usage: python3 phenotype_noise/plot_denoise_comparison.py
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

PH = os.path.dirname(os.path.abspath(__file__))
ARMS = [('orig_noisy', 'orig / noisy', 'grey', '--'),
        ('robust_noisy', 'robust / noisy', 'tab:blue', '-'),
        ('orig_denoised', 'orig / DENOISED', 'tab:red', '-'),
        ('robust_denoised', 'robust / DENOISED', 'tab:green', '-')]
METRICS = ['mcc', 'f1', 'brier', 'ece']
TITLES = {'mcc': 'MCC', 'f1': 'F1', 'brier': 'Brier', 'ece': 'ECE'}


def main():
    with open(os.path.join(PH, 'results/pheno_denoise_results.pkl'), 'rb') as fh:
        D = pickle.load(fh)
    res, fn_grid, fp_grid = D['results'], D['fn_grid'], D['fp_grid']
    os.makedirs(os.path.join(PH, 'results/plots'), exist_ok=True)

    for fp in fp_grid:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        for ax, met in zip(axes, METRICS):
            for arm, lab, col, ls in ARMS:
                xs, ys, es = [], [], []
                for fn in fn_grid:
                    cell = res.get(arm, {}).get((fn, fp))
                    if cell and met in cell:
                        xs.append(fn); ys.append(cell[met][0]); es.append(cell[met][1])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, label=lab, color=col, linestyle=ls,
                                marker='o', ms=4, capsize=2, alpha=0.9)
            ax.set_xlabel(r'$r_{FN}$'); ax.set_title(TITLES[met])
            ax.set_ylim([-0.1, 1.05] if met in ('mcc', 'f1') else None)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=9, loc='lower left')
        fig.suptitle(f'Aerobicity prediction across noise -- denoiser front-end '
                     f'(HQ-marginal consistent ensemble),  $r_{{FP}}={fp}$,  '
                     f"{D['n_splits']} splits", fontsize=12)
        fig.tight_layout()
        out = os.path.join(PH, f'results/plots/denoise_compare_fp_{fp}.pdf')
        fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
        print('wrote', out)

    print('\n=== MCC: denoised - noisy (positive = denoiser helps) ===')
    print('fn      orig:(den-noisy)   robust:(den-noisy)')
    for fn in fn_grid:
        do = dr = no = nr = no_n = nr_n = 0.0; co = cr = 0
        for fp in fp_grid:
            for fam, dn, nz in [('o', 'orig_denoised', 'orig_noisy'), ('r', 'robust_denoised', 'robust_noisy')]:
                cd = res.get(dn, {}).get((fn, fp)); cn = res.get(nz, {}).get((fn, fp))
                if cd and cn:
                    if fam == 'o':
                        do += cd['mcc'][0] - cn['mcc'][0]; co += 1
                    else:
                        dr += cd['mcc'][0] - cn['mcc'][0]; cr += 1
        oo = do / co if co else float('nan'); rr = dr / cr if cr else float('nan')
        print(f'{fn:<6} {oo:+.3f}            {rr:+.3f}')


if __name__ == '__main__':
    main()
