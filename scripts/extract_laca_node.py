#!/usr/bin/env python3
"""
extract_laca_node.py - convert a long-format archaeal per-species
                       event-count reconciliation TSV (Davin/Astral LACA pipeline) into the
                       wide format analyze_ancestral_node.py reads.

Input layout (long; ~928 k rows for the Eury/MHH archaeal tables):

  species_label	speciations	duplications	losses	transfers	presence	origination	Family	cluster	arCOG
  Node_GCA-000006805.1_AP024487.1_0	0.57	0.27	0.0	0.76	0.3	0.3	COG2768_01	COG2768_01	COG2768
  ...

One row per Family (e.g. COG2768_01) is extracted for the requested node, with
the 'presence' column as the value - the same quantity as one column of the
bacterial TableAncestralRoot1.tsv.

Output: wide TSV with a single node column

  COG	<node_label>
  COG0001_01	0.0123
  COG0001_02	0.0456
  ...

The sub-family _NN suffixes are left in place; analyze_ancestral_node.py's
strip_cog_suffix (COGxxxx_NN -> COGxxxx) collapses them and sums the values.
This script therefore assumes the input's family ids are already in COG space.
For a reconciliation in native arCOG space use extract_laca_arcog_node.py,
which translates arCOG -> COG before aggregating.

Usage:
  python3 scripts/extract_laca_node.py \\
      --in  /path/to/perspecies_eventcount_per_family_summary_Euryroot.tsv \\
      --node Node_GCA-000006805.1_AP024487.1_0 \\
      --out data/LACA_Euryroot_table.tsv
"""
import argparse
import sys
from pathlib import Path


def main():
    pa = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--in', dest='inp', required=True,
                    help='Long-format per-species event-count TSV '
                         '(may be .tsv or .tsv.gz).')
    pa.add_argument('--node', required=True,
                    help='species_label value to extract '
                         '(e.g. Node_GCA-000006805.1_AP024487.1_0).')
    pa.add_argument('--out', required=True,
                    help='Output wide-format TSV (2 columns: '
                         'COG\\t<node_label>).')
    pa.add_argument('--value-col', default='presence',
                    help='Which column to extract per row (default: '
                         '"presence" -- expected presence count from '
                         'the recon).  Alternatives: speciations, '
                         'duplications, losses, transfers, origination.')
    pa.add_argument('--family-col', default='Family',
                    help='Which column gives the per-row COG / sub-family '
                         'identifier (default "Family", which has the '
                         '_NN sub-family suffix; alternative "arCOG" '
                         'gives bare COG IDs).')
    A = pa.parse_args()

    if A.inp.endswith('.gz'):
        import gzip
        fh = gzip.open(A.inp, 'rt')
    else:
        fh = open(A.inp, 'rt')

    header = fh.readline().rstrip('\n').split('\t')
    try:
        i_species = header.index('species_label')
        i_value = header.index(A.value_col)
        i_family = header.index(A.family_col)
    except ValueError as e:
        fh.close()
        sys.exit(f'ERROR: {e} -- header was: {header}')

    rows = []
    n_seen = 0
    n_node = 0
    for line in fh:
        n_seen += 1
        parts = line.rstrip('\n').split('\t')
        if len(parts) <= max(i_species, i_value, i_family):
            continue
        if parts[i_species] != A.node:
            continue
        n_node += 1
        rows.append((parts[i_family], parts[i_value]))
    fh.close()

    if not rows:
        sys.exit(f'ERROR: no rows matched species_label={A.node!r}.  '
                 f'Scanned {n_seen} data rows.')

    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    with open(A.out, 'w') as fout:
        fout.write(f'COG\t{A.node}\n')
        for fam, val in rows:
            fout.write(f'{fam}\t{val}\n')

    try:
        nz_count = sum(1 for _, v in rows if float(v) > 0)
    except ValueError:
        nz_count = -1
    print(f'  wrote {len(rows)} rows  ({nz_count} with {A.value_col} > 0)',
          file=sys.stderr)
    print(f'  scanned {n_seen} data rows from {A.inp}', file=sys.stderr)
    print(f'  node:   {A.node}', file=sys.stderr)
    print(f'  output: {A.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
