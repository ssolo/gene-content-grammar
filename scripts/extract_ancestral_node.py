#!/usr/bin/env python3
"""
extract_ancestral_node.py - convert Davin et al. ancestral-reconstruction
                            posterior tables into denoise_genome.py input.

Input table layout (e.g. data/TableAncestralRoot1.tsv):
  - Column 0: COG identifier with a sub-cluster suffix, e.g. COG1414_0,
              COG1414_1, COG2981_1, COG3013_X.  The suffix denotes an
              orthologous sub-family within the COG.
  - Columns 1..N: ancestral and extant nodes of the Davin et al. tree of
              bacteria.  Integer headers (e.g. '2012') are ancestral nodes,
              parenthesised headers (e.g. 'AABM5X1(0)') are extant tips.
  - Cells: posterior mean copy number for that (sub-family, node) pair, in
              [0, large].  At an ancestral root most values are 0 and the rest
              are <= 1.

The denoiser's vocabulary is bare COGs (no _N suffix), so sub-family rows are
collapsed to one value per COG.  Aggregation is SUM by default (the total
expected copy count, each sub-family contributing independently); the
alternatives are --aggregate noisy_or (1 - prod(1 - p_i), treating each value as
an independent presence probability) and --aggregate max.  The aggregate is
capped at 1.0 and read as a probability of presence.  COGs aggregating to
exactly 0 are dropped: the denoiser treats a missing entry as absent anyway.

Output: 2-column TSV (COG_ID, probability) for denoise_genome.py --input.

Examples:
  python3 scripts/extract_ancestral_node.py \\
      --table data/TableAncestralRoot1.tsv --node 2012 \\
      --output node2012_input.tsv

  python3 scripts/extract_ancestral_node.py \\
      --table data/TableAncestralRoot1.tsv --all-nodes --outdir node_inputs/

  python3 scripts/denoise_genome.py \\
      --model gsd_results_higher_order_nohidden_T20_bac_hard_split1/model_ho3.pth \\
      --input node2012_input.tsv --device cpu --top 50 > node2012_denoised.tsv
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


COG_SUFFIX_RE = re.compile(r'^(COG\d+)(?:_(?:\d+|X))?$')


def strip_cog_suffix(s):
    """COG1414_0 -> COG1414; COG3013_X -> COG3013; COG0001 -> COG0001."""
    m = COG_SUFFIX_RE.match(s.strip())
    if m is None:
        return None
    return m.group(1)


def aggregate(values, mode):
    """Combine a COG's sub-family copy numbers into one probability of presence."""
    if not values:
        return 0.0
    if mode == 'sum':
        # Each value is an expected copy contribution, so a total of >= 1 copy
        # means the COG is present: the cap turns the sum into a probability.
        return min(sum(values), 1.0)
    if mode == 'max':
        return min(max(values), 1.0)
    if mode == 'noisy_or':
        p = 1.0
        for v in values:
            v_c = min(max(v, 0.0), 1.0)
            p *= (1.0 - v_c)
        return 1.0 - p
    raise ValueError(f'Unknown aggregation mode: {mode}')


def load_table(path):
    """Yield (cog_subfamily, values) for each data row.

    The first non-blank line is consumed as the header.  Stdlib only, so the
    script carries no pandas import."""
    headers = None
    with open(path) as f:
        for line in f:
            if line.endswith('\n'):
                line = line[:-1]
            if not line:
                continue
            parts = line.split('\t')
            if headers is None:
                headers = parts
                continue
            yield parts[0], parts[1:]
    return headers


def resolve_node_columns(headers, node_args, all_nodes):
    """Map --node / --all-nodes to column indices in the data rows, i.e.
    relative to row[1:], since row[0] is the COG name.

    Numeric ancestral node IDs (no parens) are the typical target; extant nodes
    carry parens like 'AABM5X1(0)'.  Either form is accepted.

    Returns a list of (column_index_in_data, display_name).
    """
    col_headers = headers[1:]
    if all_nodes:
        return [(i, h) for i, h in enumerate(col_headers)]
    selected = []
    for tag in node_args:
        if tag in col_headers:
            i = col_headers.index(tag)
            selected.append((i, tag))
            continue
        match_idx = None
        for i, h in enumerate(col_headers):
            base = h.split('(')[0]
            if base == tag:
                match_idx = i
                break
        if match_idx is not None:
            selected.append((match_idx, col_headers[match_idx]))
            continue
        sys.exit(f"ERROR: node {tag!r} not found in table headers.  "
                 f"First 10 headers: {col_headers[:10]}")
    return selected


def extract_node(path, node_col_idx, aggregate_mode):
    """Build a dict {COG_ID: probability} for one node column."""
    accumulator = defaultdict(list)
    n_rows = 0
    n_unparseable = 0
    n_bad_value = 0
    headers = None
    with open(path) as f:
        for line in f:
            if line.endswith('\n'):
                line = line[:-1]
            if not line:
                continue
            parts = line.split('\t')
            if headers is None:
                headers = parts
                continue
            cog_subfam = parts[0]
            cog = strip_cog_suffix(cog_subfam)
            if cog is None:
                n_unparseable += 1
                continue
            cell = parts[1 + node_col_idx]
            try:
                v = float(cell)
            except ValueError:
                n_bad_value += 1
                continue
            if v == 0.0:
                # A zero changes neither the sum nor the noisy-or, so it is not
                # accumulated.  The key is still created, so a COG whose
                # sub-families are all zero counts towards rows_seen and is then
                # filtered out of the output below.
                accumulator[cog]  # defaultdict touch: register the key
                continue
            accumulator[cog].append(v)
            n_rows += 1

    out = {}
    for cog, vals in accumulator.items():
        p = aggregate(vals, aggregate_mode)
        if p > 0.0:
            out[cog] = p

    return out, dict(
        rows_seen=sum(1 for _ in accumulator),
        rows_nonzero=n_rows,
        unparseable_cog_ids=n_unparseable,
        bad_values=n_bad_value,
    )


def main():
    pa = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--table', required=True,
                    help='Path to TableAncestralRoot*.tsv.')
    pa.add_argument('--node', action='append', default=None,
                    help='Node label to extract (e.g. 2012).  Repeatable. '
                         'Mutually exclusive with --all-nodes.')
    pa.add_argument('--all-nodes', action='store_true',
                    help='Extract every node in the table.  Requires --outdir.')
    pa.add_argument('--output', default=None,
                    help='Output file for a single node (default: stdout).')
    pa.add_argument('--outdir', default=None,
                    help='Directory for multi-node output.  One file per '
                         'node, named node_<label>.tsv.')
    pa.add_argument('--aggregate', default='sum',
                    choices=['sum', 'max', 'noisy_or'],
                    help='How to combine sub-family copy numbers for the '
                         'same COG.  See script docstring.')
    pa.add_argument('--list-nodes', action='store_true',
                    help='Print available node labels and exit.')
    A = pa.parse_args()

    with open(A.table) as f:
        first = next(f).rstrip('\n').split('\t')
    headers = first

    if A.list_nodes:
        for h in headers[1:]:
            print(h)
        return

    if not A.node and not A.all_nodes:
        sys.exit('ERROR: pass --node <label> (repeatable) or --all-nodes.')
    if A.node and A.all_nodes:
        sys.exit('ERROR: --node and --all-nodes are mutually exclusive.')

    cols = resolve_node_columns(headers, A.node or [], A.all_nodes)
    print(f'# table: {A.table}', file=sys.stderr)
    print(f'# nodes selected: {len(cols)}', file=sys.stderr)
    print(f'# aggregate: {A.aggregate}', file=sys.stderr)

    if A.all_nodes or len(cols) > 1:
        if not A.outdir:
            sys.exit('ERROR: --outdir is required when extracting multiple '
                     'nodes.  Pass --outdir DIR.')
        outdir = Path(A.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for col_idx, label in cols:
            probs, stats = extract_node(A.table, col_idx, A.aggregate)
            # Extant labels carry parens ('AABM5X1(0)'), so sanitise for a filename.
            fname = 'node_' + re.sub(r'[^A-Za-z0-9_.-]', '_', label) + '.tsv'
            out_path = outdir / fname
            with open(out_path, 'w') as fh:
                fh.write(f'# extracted from {A.table}\n')
                fh.write(f'# node: {label}\n')
                fh.write(f'# aggregate: {A.aggregate}\n')
                fh.write(f'# n_present_cogs: {len(probs)}\n')
                for cog in sorted(probs):
                    fh.write(f'{cog}\t{probs[cog]:.4f}\n')
            print(f'  {label:>20s} -> {out_path}  ({len(probs)} present COGs)',
                  file=sys.stderr)
    else:
        col_idx, label = cols[0]
        probs, stats = extract_node(A.table, col_idx, A.aggregate)
        out_fh = open(A.output, 'w') if A.output else sys.stdout
        out_fh.write(f'# extracted from {A.table}\n')
        out_fh.write(f'# node: {label}\n')
        out_fh.write(f'# aggregate: {A.aggregate}\n')
        out_fh.write(f'# n_present_cogs: {len(probs)}\n')
        for cog in sorted(probs):
            out_fh.write(f'{cog}\t{probs[cog]:.4f}\n')
        if A.output:
            out_fh.close()
            print(f'# wrote {len(probs)} present COGs (of total found) '
                  f'to {A.output}', file=sys.stderr)
        else:
            print(f'# wrote {len(probs)} present COGs to stdout',
                  file=sys.stderr)


if __name__ == '__main__':
    main()
