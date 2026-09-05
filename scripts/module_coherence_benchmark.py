#!/usr/bin/env python3
"""Module coherence of a reconstruction against present-day genomes of similar size.

Coherence is the fraction of KEGG modules, among those with at least --min-members
member families, whose completeness (fraction of members present) falls in the
fragmentary band [--lo, --hi]. A coherent genome leaves few modules there; a set of
independently scored genes leaves many.

  python3 scripts/module_coherence_benchmark.py \
      --pred data/lbca_cog_lists.tsv --present-col sparse_out_post \
      --val-glob '<dir>/COG_val*_phylum.feather'
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def frag_fraction(present, M, min_members, lo, hi):
    """present: (N,) bool over COGs; M: (N, K) module membership."""
    size = M.sum(0).astype(np.float64)
    keep = size >= min_members
    if keep.sum() == 0:
        return np.nan, 0
    comp = (present @ M)[keep] / size[keep]
    return float(((comp >= lo) & (comp <= hi)).mean()), int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='data/lbca_cog_lists.tsv')
    ap.add_argument('--present-col', default='sparse_out_post')
    ap.add_argument('--module-pt', default='data/module_matrix_kegg.pt')
    ap.add_argument('--val-glob', required=True)
    ap.add_argument('--min-members', type=int, default=4)
    ap.add_argument('--lo', type=float, default=0.25)
    ap.add_argument('--hi', type=float, default=0.74)
    ap.add_argument('--band', type=int, nargs=2, default=[1300, 1500])
    a = ap.parse_args()

    mm = torch.load(a.module_pt, map_location='cpu', weights_only=False)
    M = mm['M_kegg']
    M = (M.numpy() if hasattr(M, 'numpy') else np.asarray(M)).astype(np.float64)
    cogs = list(mm['cog_names'])
    idx = {c: i for i, c in enumerate(cogs)}

    rows = list(csv.DictReader(open(a.pred), delimiter='\t'))
    pres = np.zeros(len(cogs), dtype=float)
    for r in rows:
        i = idx.get(r['COG_ID'])
        if i is not None and float(r[a.present_col] or 0) > 0.5:
            pres[i] = 1.0
    n_genes = int(pres.sum())
    frag, n_mod = frag_fraction(pres, M, a.min_members, a.lo, a.hi)
    print(f"reconstruction: {n_genes} families present, {n_mod} modules with "
          f">= {a.min_members} members")
    print(f"  fragmentary ({a.lo:.0%}-{a.hi:.0%}): {100 * frag:.1f}%")

    vals = []
    for f in sorted(glob.glob(a.val_glob)):
        d = pd.read_feather(f)
        cols = [c for c in d.columns if c in idx]
        order = np.array([idx[c] for c in cols])
        X = (d[cols].to_numpy() > 0)
        n = X.sum(1)
        sel = (n >= a.band[0]) & (n <= a.band[1])
        for row in X[sel]:
            v = np.zeros(len(cogs), dtype=np.float64); v[order] = row
            fr, _ = frag_fraction(v, M, a.min_members, a.lo, a.hi)
            if not np.isnan(fr):
                vals.append(fr)
    vals = np.array(vals)
    print(f"present-day genomes with {a.band[0]}-{a.band[1]} families: n = {len(vals)}")
    print(f"  fragmentary: {100 * vals.mean():.1f}% +/- {100 * vals.std(ddof=1):.1f}%")


if __name__ == '__main__':
    main()
