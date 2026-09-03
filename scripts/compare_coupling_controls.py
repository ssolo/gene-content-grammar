#!/usr/bin/env python3
"""Compare coherent-FP frontiers across the inference-time coupling controls.

Reads the per-config frontier TSVs written by coherent_fp_eval.py,
data/cohfp_controls_<fam>_<cfg>.tsv, with columns model, split, fn, tau,
n_present, n_grafted, n_recovered, n_fp_removed. The four configs are the
baseline, support shrinkage, genome-mass normalisation, and both together, all
applied post-hoc to a frozen model.

For each family and noise level the splits are pooled, the (recall, FP-removed)
frontier is built over the tau grid, and the coherent FPs removed at matched
recall are reported for each control against the baseline. This is the test of
whether disciplining J lifts the frontier without costing recall.

Usage:  python3 scripts/compare_coupling_controls.py [--glob 'data/cohfp_controls_*.tsv']
"""
import argparse
import glob

import pandas as pd

CFG_ORDER = ['baseline', 'shrink', 'norm', 'shrnorm']
RECALL_TARGETS = [0.60, 0.70, 0.75, 0.80]


def frontier(df):
    """Pool splits into a (recall, fp_removed) frontier over the tau grid.

    Sorted by ascending recall so np.interp sees a monotone x-axis.
    """
    g = df.groupby('tau').agg(
        rec=('n_recovered', 'sum'), pres=('n_present', 'sum'),
        rem=('n_fp_removed', 'sum'), graft=('n_grafted', 'sum')).reset_index()
    g['recall'] = g['rec'] / g['pres']
    g['fp_removed'] = g['rem'] / g['graft']
    return g.sort_values('recall')


def interp_fp_at_recall(fr, r):
    """FP-removed at a target recall; NaN at or outside the frontier's endpoints."""
    x = fr['recall'].to_numpy()
    y = fr['fp_removed'].to_numpy()
    if r <= x.min() or r >= x.max():
        return float('nan')
    import numpy as np
    return float(np.interp(r, x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='data/cohfp_controls_*.tsv')
    ap.add_argument('--fn', type=float, default=0.9, help='noise level to report (deep-loss=0.9)')
    a = ap.parse_args()

    frames = {}
    for path in sorted(glob.glob(a.glob)):
        df = pd.read_csv(path, sep='\t')
        lab = df['model'].iloc[0]              # '<fam>-<cfg>'
        frames[lab] = df

    fams = sorted({lab.split('-')[0] for lab in frames})
    for fam in fams:
        print(f'\n===== {fam}  (fn={a.fn}, coherent-FP removed at matched recall) =====')
        base = None
        rows = []
        for cfg in CFG_ORDER:
            lab = f'{fam}-{cfg}'
            if lab not in frames:
                continue
            d = frames[lab]
            d = d[d['fn'] == a.fn]
            if d.empty:
                continue
            fr = frontier(d)
            vals = {r: interp_fp_at_recall(fr, r) for r in RECALL_TARGETS}
            rows.append((cfg, vals))
            if cfg == 'baseline':
                base = vals
        hdr = 'config'.ljust(10) + ''.join(f'  R={r:.2f}' for r in RECALL_TARGETS) \
            + '     (delta vs baseline)'
        print(hdr)
        for cfg, vals in rows:
            cells = '  '.join(f'{vals[r]:.3f}' if vals[r] == vals[r] else '  -  '
                              for r in RECALL_TARGETS)
            if base and cfg != 'baseline':
                dlt = '  '.join(f'{vals[r]-base[r]:+.3f}'
                                if (vals[r] == vals[r] and base[r] == base[r]) else '  -  '
                                for r in RECALL_TARGETS)
                print(f'{cfg.ljust(10)}  {cells}     {dlt}')
            else:
                print(f'{cfg.ljust(10)}  {cells}')
        print('higher FP-removed at matched recall = better contamination rejection;'
              ' positive delta = the control beats baseline.')


if __name__ == '__main__':
    main()
