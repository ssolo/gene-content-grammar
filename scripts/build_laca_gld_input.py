#!/usr/bin/env python3
"""
build_laca_gld_input.py -- LACA GLD (Count) root inputs from the archaeal-root
arCOG table (arc269_root_cogs_brownian_both_omin.tsv).

Maps arCOG -> COG via the COG_xref column (normalised COG00577 -> COG0577) and
aggregates the arCOGs mapping to one COG onto the COG vocabulary
(data/module_matrix_kegg.pt), capped at 1.0, for the two observation thresholds
min1 (P_root_omin1) and min4 (P_root_omin4). min4 entries flagged
'filtered(sum<4)', and any other non-numeric entry, are treated as absent.

Output: LACA_GLD_{min1,min4}_input.tsv (COG<TAB>prob), the dense GLD LACA inputs
for the marginal denoiser (mix-FT-fp-marginal-hq).

Usage: python scripts/build_laca_gld_input.py [--table <arc269...tsv>]
"""
import argparse
import csv
import os

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm_cog(x):
    x = x.strip()
    if not x.startswith('COG'):
        return None
    d = x[3:]
    return f"COG{int(d):04d}" if d.isdigit() else None


def fval(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=os.path.join(HERE, 'data/arc269_root_cogs_brownian_both_omin.tsv'),
                    help='archaeal-root arCOG GLD table (slim copy in data/: '
                         'family_id, COG_xref, P_root_omin1, P_root_omin4). The full '
                         'table is the recount output arc269_root_cogs_brownian_both_'
                         'omin.tsv from validation/build_arc269_root_cogs.py.')
    ap.add_argument('--module-matrix', default=os.path.join(HERE, 'data/module_matrix_kegg.pt'))
    ap.add_argument('--agg', default='noisyor', choices=['noisyor', 'max'],
                    help='combine the posterior presences of arCOG families that map '
                         'to one COG: noisyor = 1-prod(1-p) ("present if at least one '
                         'arCOG is"; canonical, matches extract_laca_arcog_node.py) or '
                         'max. They differ by <=7 families -- every COG is dominated '
                         'by one arCOG -- so the choice is immaterial here.')
    a = ap.parse_args()
    vocab = set(torch.load(a.module_matrix, map_location='cpu', weights_only=False)['cog_names'])
    # Accumulator convention: noisyor holds a running product of (1 - p) per COG
    # and is inverted at write time; max holds a running max.
    acc = {'min1': {}, 'min4': {}}
    col = {'min1': 'P_root_omin1', 'min4': 'P_root_omin4'}
    with open(a.table) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            c = norm_cog(r.get('COG_xref', ''))
            if c is None or c not in vocab:
                continue
            for cut in ('min1', 'min4'):
                p = fval(r[col[cut]])
                if p != p:                 # P_root_omin4 is NaN below the Omin=4 cut -> absent
                    continue
                p = min(max(p, 0.0), 1.0)
                if a.agg == 'max':
                    acc[cut][c] = max(acc[cut].get(c, 0.0), p)
                else:
                    acc[cut][c] = acc[cut].get(c, 1.0) * (1.0 - p)
    for cut in ('min1', 'min4'):
        agg = {c: (v if a.agg == 'max' else 1.0 - v) for c, v in acc[cut].items()}
        out = os.path.join(HERE, f'LACA_GLD_{cut}_input.tsv')
        with open(out, 'w') as f:
            f.write(f"COG\tLACA_GLD_{cut}\n")
            for c in sorted(agg, key=lambda x: int(x[3:])):
                if agg[c] > 0:
                    f.write(f"{c}\t{agg[c]:.4f}\n")
        n = sum(1 for v in agg.values() if v >= 0.5)
        print(f"wrote {out}: {len(agg)} vocab COGs, {n} present (>=0.5), agg={a.agg}")


if __name__ == '__main__':
    main()
