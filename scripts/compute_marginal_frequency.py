#!/usr/bin/env python3
"""
compute_marginal_frequency.py -- per-COG cross-genome marginal frequency p_c.

p_c = fraction of training genomes that carry COG c (presence = copy number > 0),
the prevalence prior the marginal-FP curriculum samples contamination from
(ising_denoiser/data.py: self.marginal). Committing it as a TSV lets the
extant-recovery corruption (reconstruct_extant.py --fp-mode marginal) inject
common-gene contamination without re-reading the 90k-genome feather at recon time.

Usage: python scripts/compute_marginal_frequency.py
       [--feather data/COG_train1_phylum.feather] [--out data/cog_marginal_frequency.tsv]
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feather', default=os.path.join(HERE, 'data/COG_train1_phylum.feather'))
    ap.add_argument('--out', default=os.path.join(HERE, 'data/cog_marginal_frequency.tsv'))
    a = ap.parse_args()
    cols = [c for c in pd.read_feather(a.feather).columns if c.startswith('COG')]
    df = pd.read_feather(a.feather, columns=cols)
    p = (df.values > 0).mean(axis=0)
    out = pd.DataFrame({'COG_ID': cols, 'p_c': p.round(6)})
    out.to_csv(a.out, sep='\t', index=False)
    print(f'wrote {a.out}: {len(out)} COGs, {int((p>=0.9).sum())} with p_c>=0.9, '
          f'{int((p<0.1).sum())} with p_c<0.1, from {df.shape[0]:,} genomes')


if __name__ == '__main__':
    main()
