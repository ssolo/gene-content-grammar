#!/usr/bin/env python3
"""Rebuild data/spectra_fpreal_summary.csv from the per-(family, split) parquets
in spectra_fpreal/, written by sweep_spectra_calibration.py.

For each (family, fn, fp) cell, EVERY (split, annealing-step) row is pooled and
the mean and sample std (ddof=1) of MCC and ECE are taken; n is the number of
distinct splits. The metric is therefore averaged over the whole internal
annealing trajectory, not read off the converged final step alone.

Before overwriting, every (family, fn, fp) row already present in the committed
summary must be reproduced within tolerance; a mismatch exits non-zero and
writes nothing. Adding a model means dropping its
spectra_fpreal/<family>_split*.parquet files in place and re-running: the new
family appears and the existing rows are re-verified.

    python scripts/build_spectra_fpreal_summary.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOB = os.path.join(HERE, 'spectra_fpreal', '*.parquet')
DEFAULT_OUT = os.path.join(HERE, 'data', 'spectra_fpreal_summary.csv')
TOL = 1.5e-3            # the committed CSV is stored to 4 dp


def build(parquet_glob):
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise SystemExit(f'[err] no parquets match {parquet_glob!r}')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for col in ('family', 'split', 'fn', 'fp', 'MCC', 'ece'):
        if col not in df.columns:
            raise SystemExit(f'[err] parquet missing column {col!r}')
    g = (df.groupby(['family', 'fn', 'fp'])
           .agg(MCC_mean=('MCC', 'mean'),
                MCC_std=('MCC', lambda x: x.std(ddof=1)),
                ece_mean=('ece', 'mean'),
                ece_std=('ece', lambda x: x.std(ddof=1)),
                n=('split', 'nunique'))
           .reset_index())
    for c in ('MCC_mean', 'MCC_std', 'ece_mean', 'ece_std'):
        g[c] = g[c].round(4)
    return g.sort_values(['family', 'fn', 'fp']).reset_index(drop=True), files


def verify_against_committed(new, committed_csv):
    """True when every (family, fn, fp) row of the committed CSV is reproduced within TOL."""
    if not os.path.exists(committed_csv):
        print('[skip] no committed summary to verify against')
        return True
    old = pd.read_csv(committed_csv)
    key = ['family', 'fn', 'fp']
    merged = old.merge(new, on=key, suffixes=('_old', '_new'), how='left')
    missing = merged[merged['MCC_mean_new'].isna()]
    bad = []
    for col in ('MCC_mean', 'MCC_std', 'ece_mean', 'ece_std'):
        d = (merged[f'{col}_old'] - merged[f'{col}_new']).abs()
        bad.append(merged[d > TOL])
    n_bad = sum(len(b) for b in bad)
    ok = (len(missing) == 0 and n_bad == 0)
    print(f'   [{"PASS" if ok else "FAIL"}] reproduce committed summary: '
          f'{len(old)} existing rows, {len(missing)} missing, {n_bad} value mismatches (tol {TOL})')
    if len(missing):
        print('     missing rows:\n', missing[key].to_string(index=False))
    for b in bad:
        if len(b):
            print('     mismatched:\n', b[key].to_string(index=False))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glob', default=DEFAULT_GLOB)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--check-only', action='store_true',
                    help='verify reproduction but do not overwrite')
    a = ap.parse_args()

    new, files = build(a.glob)
    fams = sorted(new.family.unique())
    print(f'built summary from {len(files)} parquets, {len(new)} rows, families:')
    for f in fams:
        print(f'   {f}  ({new[new.family==f].n.iloc[0]} splits, '
              f'{len(new[new.family==f])} fn x fp cells)')

    ok = verify_against_committed(new, a.out)
    if not ok:
        print('\nFAILED verification -- not writing.')
        sys.exit(1)
    if a.check_only:
        print('\n[check-only] verification passed; not writing.')
        return
    new.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
