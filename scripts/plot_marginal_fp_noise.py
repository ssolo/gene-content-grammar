#!/usr/bin/env python3
"""
plot_marginal_fp_noise.py -- illustrate the marginal-frequency dense-FP noise
model used to fine-tune the marginal-FP HQ denoisers.

The marginal-FP curriculum (ising_denoiser/data.py, the prune branch) injects
DENSE false positives into a clean genome by drawing absent COGs in proportion
to their per-COG cross-genome marginal frequency p_c, the fraction of training
genomes that carry COG c (self.marginal = (clean == 1).mean(0)). The injected
contamination is therefore dominated by COMMON genes: a broad, module-incoherent
over-reconstruction that mimics an over-rich Count/softlanding ancestor, which the
model must learn to TRIM back to the clean target. This differs from uniform-FP
(every absent COG equally likely, so mostly rare junk) and from coherent-FP
(whole foreign modules grafted in).

Panel A: the marginal-frequency weights p_c, sorted.
Panel B: the frequency composition of the injected FP. Marginal sampling (weight
         proportional to p_c) concentrates on common genes; uniform sampling
         spreads across the rare-dominated COG pool.

Reads the training feather to recompute p_c; writes a two-panel PDF.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL_MARG = '#D55E00'   # marginal-FP (matches the Pareto colour)
COL_UNIF = '#E69F00'   # uniform-FP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feather', default=os.path.join(HERE, 'data/COG_train1_phylum.feather'))
    ap.add_argument('--out', default=os.path.join(HERE, 'analysis/figures/fig_marginal_fp_noise.pdf'))
    a = ap.parse_args()

    cols = [c for c in pd.read_feather(a.feather).columns if c.startswith('COG')]
    df = pd.read_feather(a.feather, columns=cols)
    present = (df.values > 0)
    p = present.mean(axis=0)                  # per-COG marginal frequency p_c
    n_gen, n_cog = present.shape
    core = int((p >= 0.9).sum()); rare = int((p < 0.1).sum())

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight', 'figure.dpi': 120})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    ps = np.sort(p)[::-1]
    axA.plot(np.arange(1, n_cog + 1), ps, color=COL_MARG, lw=1.8)
    axA.fill_between(np.arange(1, n_cog + 1), ps, color=COL_MARG, alpha=0.15, lw=0)
    axA.axhline(0.9, ls=':', color='#555555', lw=0.9)
    axA.set_xlabel('COG rank (by frequency)')
    axA.set_ylabel(r'marginal frequency $p_c$')
    axA.set_title(f'(A)  per-COG weights  ($n={n_cog}$ COGs, {n_gen:,} genomes)',
                  loc='left', fontsize=9.5, fontweight='bold')
    axA.text(0.97, 0.92, f'{core} COGs $\\geq 0.9$ (near-universal core)\n{rare} COGs $< 0.1$ (rare tail)',
             transform=axA.transAxes, ha='right', va='top', fontsize=8.2)

    # The composition of the injected FP is the distribution of p_c over the drawn
    # COGs, so it is a histogram of p_c weighted by the sampling weight: flat 1/N
    # under uniform sampling, proportional to p_c under marginal sampling.
    bins = np.linspace(0, 1, 21)
    w_unif = np.ones_like(p) / len(p)
    w_marg = p / p.sum()
    axB.hist(p, bins=bins, weights=w_unif, color=COL_UNIF, alpha=0.65,
             label='uniform-FP (flat)')
    axB.hist(p, bins=bins, weights=w_marg, color=COL_MARG, alpha=0.65,
             label='marginal-FP ($\\propto p_c$)')
    axB.set_xlabel(r'marginal frequency $p_c$ of injected COG')
    axB.set_ylabel('share of injected false positives')
    axB.set_title('(B)  what gets injected as dense FP', loc='left',
                  fontsize=9.5, fontweight='bold')
    axB.legend(frameon=False, fontsize=8.5, loc='upper center')
    mean_unif = float((p * w_unif).sum()); mean_marg = float((p * w_marg).sum())
    axB.text(0.97, 0.55, f'mean $p_c$ injected:\n uniform {mean_unif:.2f}\n marginal {mean_marg:.2f}',
             transform=axB.transAxes, ha='right', va='top', fontsize=8.2)

    fig.savefig(a.out)
    print(f'wrote {a.out}')
    print(f'  n_cog={n_cog} n_gen={n_gen} core(>=0.9)={core} rare(<0.1)={rare}')
    print(f'  mean p_c injected: uniform={mean_unif:.3f} marginal={mean_marg:.3f}')


if __name__ == '__main__':
    main()
