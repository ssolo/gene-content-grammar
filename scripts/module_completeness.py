#!/usr/bin/env python3
"""module_completeness.py -- KEGG-module / COG-pathway completeness audit
for a denoised ancestral-node prediction.

For each KEGG module and COG pathway, reports the fraction of member COGs
called present.  Mass concentrated at 0% and 100% indicates a coherent
gene set; a heavy 25-74% middle indicates genes called independently of
their pathway context.

Usage:
  python3 scripts/module_completeness.py --pred LBCA_alt_pred.tsv \\
      [--module-matrix data/module_matrix_kegg.pt] \\
      [--present-col mean_>0] [--cutoff 0.5] [--label LBCA_alt]

--present-col picks which column of the pred TSV defines "present":
  input_prob   -- the raw reconstruction input set
  mean_>0      -- denoised, all-input-present variant (default)
  mean_actual  -- denoised, graded-input variant
"""
import argparse
from collections import Counter

import torch


def load_pred(path, present_col, cutoff):
    """Return {COG_ID: present > cutoff} for the chosen pred column."""
    rows = {}
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        if present_col not in header:
            raise SystemExit(f'no column {present_col!r}; have {header}')
        ci = header.index(present_col)
        c0 = header.index('COG_ID')
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) <= max(ci, c0):
                continue
            rows[p[c0]] = float(p[ci]) > cutoff
    return rows


def main():
    pa = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--pred', required=True, help='pred.tsv from analyze_ancestral_node.py')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    pa.add_argument('--present-col', default='mean_>0')
    pa.add_argument('--cutoff', type=float, default=0.5)
    pa.add_argument('--label', default=None)
    pa.add_argument('--min-module-size', type=int, default=4,
                    help='Ignore modules with fewer member COGs than this.')
    A = pa.parse_args()
    label = A.label or A.pred

    present = load_pred(A.pred, A.present_col, A.cutoff)
    n_present = sum(present.values())
    print(f'=== {label} ===')
    print(f'present set: {n_present} COGs '
          f'(column {A.present_col!r}, cutoff {A.cutoff})')

    m = torch.load(A.module_matrix, weights_only=False)
    cog_names = list(m['cog_names'])
    idx = {c: i for i, c in enumerate(cog_names)}

    def completeness(M, names, descs, kind):
        """For each module column, fraction of member COGs that are present."""
        rows = []
        M = M.cpu().numpy() if hasattr(M, 'cpu') else M
        for j, name in enumerate(names):
            members = [cog_names[i] for i in range(len(cog_names)) if M[i, j] > 0]
            # COGs outside the denoiser vocabulary carry no call at all.
            members = [c for c in members if c in present]
            if len(members) < A.min_module_size:
                continue
            n_in = sum(present[c] for c in members)
            frac = n_in / len(members)
            rows.append((name, descs[j] if descs else '', len(members), n_in, frac))
        return rows

    for kind, Mk, names, descs in [
        ('KEGG module', m['M_kegg'], m['kegg_names'], m.get('kegg_descs')),
        ('COG pathway', m['M_path'], m['path_names'], None),
    ]:
        rows = completeness(Mk, names, descs, kind)
        buckets = Counter()
        for *_, frac in rows:
            if   frac == 0.0:        buckets['  0%'] += 1
            elif frac < 0.25:        buckets['  1-24%'] += 1
            elif frac < 0.50:        buckets[' 25-49%'] += 1
            elif frac < 0.75:        buckets[' 50-74%'] += 1
            elif frac < 1.0:         buckets[' 75-99%'] += 1
            else:                    buckets[' 100%'] += 1
        print(f'\n--- {kind} completeness  '
              f'({len(rows)} modules with >= {A.min_module_size} member COGs) ---')
        for b in ['  0%', '  1-24%', ' 25-49%', ' 50-74%', ' 75-99%', ' 100%']:
            n = buckets.get(b, 0)
            bar = '#' * n
            print(f'  {b:>9s} : {n:3d}  {bar}')
        partial = sum(buckets.get(b, 0) for b in [' 25-49%', ' 50-74%'])
        present_mods = sum(buckets.get(b, 0) for b in
                           [' 25-49%', ' 50-74%', ' 75-99%', ' 100%', '  1-24%'])
        print(f'  --> {partial} modules at 25-74% completeness '
              f'(the "fragmentary" / incoherent middle)')

        frag = sorted([r for r in rows if 0.25 <= r[4] < 0.75],
                      key=lambda r: -r[2])
        if frag:
            print(f'  fragmentary {kind}s (25-74% complete, sorted by size):')
            for name, desc, ntot, nin, frac in frag[:25]:
                d = (desc or '')[:52]
                print(f'    {name:9s} {nin:3d}/{ntot:<3d} {frac:5.0%}  {d}')


if __name__ == '__main__':
    main()
