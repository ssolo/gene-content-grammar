#!/usr/bin/env python3
"""Merge raw per-stratum accumulators from split-sharded stratified_eval.py runs.

Each shard (one CV split, one GPU) writes data/strat_raw_<split>.pkl via
stratified_eval.py --raw-out: a dict keyed by (fn, dim, stratum) with value
[TP, FP, TN, FN, hist_n(NB), hist_truth(NB), hist_conf(NB), n_genomes]. MCC and
ECE are NOT additive across shards, so the raw counts and the reliability
histograms are summed first and MCC/ECE computed on the totals; the result is
identical to a single unsharded pooled run.

Imports neither torch nor pyarrow, so it runs wherever pandas does.

    python scripts/merge_strat_raw.py data/strat_raw_*.pkl --out data/stratified_eval.tsv
"""
import argparse
import pickle

import numpy as np
import pandas as pd

NB = 20                                  # ECE reliability bins; must match stratified_eval.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parts', nargs='+', help='strat_raw_*.pkl shard files')
    ap.add_argument('--out', default='data/stratified_eval.tsv')
    a = ap.parse_args()

    tot = {}
    for p in a.parts:
        with open(p, 'rb') as f:
            acc = pickle.load(f)
        for k, v in acc.items():
            if k not in tot:
                tot[k] = [0, 0, 0, 0, np.zeros(NB), np.zeros(NB), np.zeros(NB), 0]
            t = tot[k]
            for i in (0, 1, 2, 3, 7):
                t[i] += v[i]
            for i in (4, 5, 6):
                t[i] = t[i] + np.asarray(v[i], dtype=float)

    rows = []
    for (fn, dim, label), acc in tot.items():
        TP, FP, TN, FN = (float(acc[i]) for i in range(4))
        den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = (TP * TN - FP * FN) / den if den > 0 else 0.0
        n, st, sc = acc[4], acc[5], acc[6]
        tn = n.sum()
        ece = float(sum((n[i] / tn) * abs(st[i] / n[i] - sc[i] / n[i])
                        for i in range(NB) if n[i] > 0)) if tn > 0 else 0.0
        rows.append((dim, label, fn, acc[7], round(mcc, 4), round(ece, 4)))

    out = (pd.DataFrame(rows, columns=['dim', 'stratum', 'fn', 'n_genomes', 'MCC', 'ECE'])
           .sort_values(['dim', 'stratum', 'fn']).reset_index(drop=True))
    out.to_csv(a.out, sep='\t', index=False)
    print(f'merged {len(a.parts)} shards -> {a.out}: {len(out)} rows '
          f'({out.stratum[out.dim=="phylum"].nunique()} phyla, '
          f'{out.stratum[out.dim=="completeness"].nunique()} completeness bins)')


if __name__ == '__main__':
    main()
