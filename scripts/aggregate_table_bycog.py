#!/usr/bin/env python3
"""Aggregate a Davin et al. (2025) ancestral-reconstruction table
(sub-family x node) to a per-COG presence-probability matrix (bare COG x node),
for all nodes at once.

Each output cell = min(sum of that COG's sub-family copy numbers at the node,
1.0), rounded to 2 decimals: sub-families (COGxxxx_0, COGxxxx_1, COGxxxx_X, ...)
are summed to the bare COG, non-numeric and negative cells count as 0, and the
total is capped at 1.0 to read as a presence probability. This is the whole-table
form of extract_node_probs() in scripts/analyze_ancestral_node.py; the two must
agree cell for cell.

Usage:
    python scripts/aggregate_table_bycog.py --in data/TableAncestralRoot1.tsv \
                                            --out data/TableAncestralRoot1_aggregated_byCOG.tsv
"""
import argparse
import pandas as pd

SUFFIX = r'^(COG\d+)(?:_(?:\d+|X))?$'   # COG1414_0 / COG3013_X / COG1414 -> COG1414


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', default='data/TableAncestralRoot1.tsv',
                    help='input sub-family x node table (col 0 = COG id with suffix)')
    ap.add_argument('--out', default='data/TableAncestralRoot1_aggregated_byCOG.tsv',
                    help='output bare-COG x node presence-probability matrix')
    a = ap.parse_args()

    df = pd.read_csv(a.inp, sep='\t', dtype={'COG': str})
    nodes = [c for c in df.columns if c != 'COG']
    df['COG'] = df['COG'].str.extract(SUFFIX)[0]
    df = df.dropna(subset=['COG'])                               # ids not matching SUFFIX extract to NaN
    df[nodes] = df[nodes].apply(pd.to_numeric, errors='coerce').clip(lower=0).fillna(0.0)
    agg = df.groupby('COG', sort=True)[nodes].sum().clip(upper=1.0).round(2)
    agg.to_csv(a.out, sep='\t')
    print(f'{a.inp} -> {a.out}: {agg.shape[0]} COGs x {agg.shape[1]} nodes')


if __name__ == '__main__':
    main()
