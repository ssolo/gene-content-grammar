#!/usr/bin/env python3
"""
extract_laca_arcog_node.py - extract one ancestral node from an arCOG-based
                             per-species event-count reconciliation and
                             translate it to COG content for the denoiser.

Companion to extract_laca_node.py, which handles reconciliations whose family
ids are already COG ids.  The Euryarchaeota-root reconciliation (Zenodo
18701544) is in native arCOG space: families are arCOG sub-clusters (e.g.
arCOG04792_01_default) with a bare-arCOG column (arCOG04792).  The denoiser
vocabulary is COG2020, so the arCOGs must be translated to COGs first.

Translation and aggregation
---------------------------
1. For the requested node, take the per-family 'presence' value of every row.
2. Map each row's arCOG -> COG via the arCOG database definition table
   (ar14.arCOGdef.tab, column "COG/Supercluster"): a value like COG01695 is a
   5-digit zero-padded COG id and is normalised to the denoiser's COG1695.
   arCOGs whose COG/Supercluster cell is empty (archaea-specific) or holds a
   supercluster (SC.*) carry no COG and are dropped - they are not in the
   denoiser vocabulary.
3. Aggregate every contribution landing on the same COG.  This collapses both
   the arCOG sub-clusters _01/_02/... and distinct arCOGs sharing a COG:
     - noisyor (default): 1 - prod(1 - p_i), i.e. the COG is present if at
       least one contributing arCOG is.  Matches build_laca_gld_input.py.
     - sumcap: min(1, sum p_i).
     - max:    max p_i.

Output: a 2-column wide TSV (COG <tab> <node_label>), one row per COG with
aggregated presence > 0, ready for analyze_ancestral_node.py --table/--node
(whose strip_cog_suffix and sum-aggregation are no-ops on these already-bare,
already-unique COG ids).

Usage:
  python3 scripts/extract_laca_arcog_node.py \\
      --in   /path/to/zen_18701544/euryroot/perspecies_eventcount_per_family_summary.tsv \\
      --node Node_GCA-000006805.1_AP024487.1_0 \\
      --arcog-map data/ar14.arCOGdef.tab \\
      --vocab data/cog-20.def.tab \\
      --out  data/LACA_Euryroot_arcog_table.tsv
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

COG_RE = re.compile(r'^COG0*([0-9]+)$')   # COG01695 -> group(1)=1695


def _open(path):
    return gzip.open(path, 'rt', errors='replace') if str(path).endswith('.gz') \
        else open(path, 'rt', errors='replace')


def build_arcog2cog(map_path):
    """Parse ar14.arCOGdef.tab -> {arCOG: COG}.  Column 1 is the arCOG id and
    column 5 ("COG/Supercluster") holds either COG0#### (-> COG####), a
    supercluster (SC.*), or nothing.  Only real COG ids are kept."""
    a2c = {}
    n_arcog = n_cog = n_sc = n_empty = 0
    with _open(map_path) as fh:
        header = fh.readline()  # CLU FUNC_ONE Gene Annotation COG/Supercluster ...
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 5:
                continue
            arcog = f[0].strip()
            if not arcog.startswith('arCOG'):
                continue
            n_arcog += 1
            cell = f[4].strip()
            if not cell:
                n_empty += 1
                continue
            m = COG_RE.match(cell)
            if m:
                a2c[arcog] = 'COG%04d' % int(m.group(1))
                n_cog += 1
            else:
                n_sc += 1  # supercluster or other, not a COG
    print('  arCOG map: %d arCOGs ; %d -> COG ; %d supercluster/other ; %d empty'
          % (n_arcog, n_cog, n_sc, n_empty), file=sys.stderr)
    return a2c


def load_vocab(def_path):
    vocab = set()
    with _open(def_path) as fh:
        for line in fh:
            tok = line.split('\t', 1)[0].strip()
            if re.match(r'^COG[0-9]{4}', tok):
                vocab.add(tok[:7])
    return vocab


def main():
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--in', dest='inp', required=True,
                    help='Long per-species event-count TSV (.tsv or .tsv.gz).')
    pa.add_argument('--node', required=True, help='species_label to extract.')
    pa.add_argument('--arcog-map', required=True,
                    help='arCOG database definition table (ar14.arCOGdef.tab).')
    pa.add_argument('--out', required=True, help='Output COG<tab>node TSV.')
    pa.add_argument('--vocab', default=None,
                    help='COG definition table (cog-20.def.tab) for in-vocab '
                         'coverage reporting (optional).')
    pa.add_argument('--value-col', default='presence',
                    help='Per-row value column (default: presence).')
    pa.add_argument('--arcog-col', default='arCOG',
                    help='Column holding the bare arCOG id (default: arCOG).')
    pa.add_argument('--agg', default='noisyor',
                    choices=['noisyor', 'sumcap', 'max'],
                    help='How to combine the posterior presences of arCOG '
                         'sub-families mapping to the same COG. noisyor (default) = '
                         '1-prod(1-p), "the COG is present if at least one arCOG is" '
                         '(the principled choice; matches build_laca_gld_input.py). '
                         'sumcap = min(1,sum p); max. The three differ by <=15 '
                         'families -- every COG is dominated by one arCOG.')
    A = pa.parse_args()

    a2c = build_arcog2cog(A.arcog_map)
    vocab = load_vocab(A.vocab) if A.vocab else None

    fh = _open(A.inp)
    header = fh.readline().rstrip('\n').split('\t')
    try:
        i_sp = header.index('species_label')
        i_val = header.index(A.value_col)
        i_ac = header.index(A.arcog_col)
    except ValueError as e:
        fh.close()
        sys.exit('ERROR: %s -- header was: %s' % (e, header))

    # Accumulator convention: sumcap/max hold a running presence, noisyor holds
    # the running product of (1 - p) and is inverted below.
    agg = {}
    n_node = n_rows = 0
    n_mapped = n_unmapped = 0
    arcogs_seen, arcogs_unmapped = set(), set()
    for line in fh:
        p = line.rstrip('\n').split('\t')
        if len(p) <= i_sp or p[i_sp] != A.node:
            continue
        n_node += 1
        arcog = p[i_ac].strip()
        arcogs_seen.add(arcog)
        try:
            v = float(p[i_val])
        except ValueError:
            continue
        cog = a2c.get(arcog)
        if cog is None:
            n_unmapped += 1
            arcogs_unmapped.add(arcog)
            continue
        n_mapped += 1
        if A.agg == 'sumcap':
            agg[cog] = agg.get(cog, 0.0) + v
        elif A.agg == 'max':
            agg[cog] = max(agg.get(cog, 0.0), v)
        else:  # noisyor
            agg[cog] = agg.get(cog, 1.0) * (1.0 - v)
    fh.close()

    out = {}
    n_capped = 0
    for cog, x in agg.items():
        if A.agg == 'sumcap':
            if x > 1.0:
                n_capped += 1
            out[cog] = min(x, 1.0)
        elif A.agg == 'max':
            out[cog] = x
        else:  # noisyor
            out[cog] = 1.0 - x
    out = {c: v for c, v in out.items() if v > 0}

    if not out:
        sys.exit('ERROR: no COGs produced for node %r (saw %d rows).'
                 % (A.node, n_node))

    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    with open(A.out, 'w') as fo:
        fo.write('COG\t%s\n' % A.node)
        for cog in sorted(out, key=lambda c: int(c[3:])):
            fo.write('%s\t%.6g\n' % (cog, out[cog]))

    present = sum(1 for v in out.values() if v > 0.5)
    ssum = sum(out.values())
    in_vocab = sum(1 for c in out if c in vocab) if vocab else None
    print('  node:             %s' % A.node, file=sys.stderr)
    print('  family rows:      %d  (%d arCOG sub-families)' % (n_node, len(arcogs_seen)),
          file=sys.stderr)
    print('  rows mapped->COG: %d ; rows dropped (no COG): %d' % (n_mapped, n_unmapped),
          file=sys.stderr)
    print('  arCOGs unmapped:  %d distinct (archaea-specific / supercluster)'
          % len(arcogs_unmapped), file=sys.stderr)
    print('  COGs out:         %d  (agg=%s ; %d had sum>1 capped)'
          % (len(out), A.agg, n_capped), file=sys.stderr)
    if vocab is not None:
        print('  in denoiser vocab:%d / %d  (%d COGs not in cog-20.def.tab)'
              % (in_vocab, len(out), len(out) - in_vocab), file=sys.stderr)
    print('  present (>0.5):   %d ; sum(presence): %.1f' % (present, ssum),
          file=sys.stderr)
    print('  output:           %s' % A.out, file=sys.stderr)


if __name__ == '__main__':
    main()
