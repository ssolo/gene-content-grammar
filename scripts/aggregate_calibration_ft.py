#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_calibration_ft.py -- Aggregate the per-split calibration outputs of
the fine-tune calibration run into a single pre-FT vs mix-FT comparison.

One output dir per cross-validation split is expected:

    calibration_T20_ft_split{N}/predictions/global_metrics.parquet
    calibration_T20_ft_split{N}/predictions/cat_metrics.parquet
    calibration_T20_ft_split{N}/predictions/module_metrics.parquet
    ... (cog, comp)

In each parquet the 'model' column carries two repurposed slot keys rather
than display labels:
    'hidden'    <- the pre-fine-tune T20 checkpoint of split N
    'nohidden'  <- the mix-fine-tuned T20 checkpoint of split N
The mapping to labels is applied here; override with --label-pre/--label-post.

ECE / Brier / NLL / MCE are reported per synthetic false-negative noise level
fn, as the across-split mean +/- std for both models plus the post-minus-pre
delta.  Lower is better, so a negative delta means the fine-tune helped.

Usage:
  python3 scripts/aggregate_calibration_ft.py \\
      [--results-glob 'calibration_T20_ft_split*'] \\
      [--cat-fn 0.25] [--outdir calibration_T20_ft_aggregate]

Outputs (under --outdir):
  global_aggregate.csv    long-form per (split, model, fn) global metrics
  global_summary.csv      per (model, fn) mean/std across splits
  category_summary.csv    per (category, model) mean/std at --cat-fn
  SUMMARY.txt             the tables above as text
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# The two repurposed 'model' slot keys written into the prediction parquets.
PRE_KEY = 'hidden'
POST_KEY = 'nohidden'


def _split_num(path):
    m = re.search(r'split(\d+)', str(path))
    return int(m.group(1)) if m else -1


def load_metric(results_dirs, fname):
    """Concatenate <dir>/predictions/<fname> across split dirs, tagging split."""
    frames = []
    found = []
    for d in results_dirs:
        p = Path(d) / 'predictions' / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df['split'] = _split_num(d)
        frames.append(df)
        found.append(_split_num(d))
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), sorted(found)


def fmt_mean_std(m, s):
    return f"{m:.4f}+/-{s:.4f}"


def aggregate_global(gdf, lab_pre, lab_post):
    """Return (summary_df, lines) for the headline per-fn comparison."""
    metrics = ['ece', 'brier', 'nll', 'mce']
    metrics = [m for m in metrics if m in gdf.columns]

    agg = (gdf.groupby(['model', 'fn'])[metrics]
              .agg(['mean', 'std', 'count'])
              .reset_index())
    agg.columns = ['model', 'fn'] + [f'{m}_{stat}' for m in metrics
                                     for stat in ('mean', 'std', 'count')]

    lines = []
    fns = sorted(gdf['fn'].unique())
    label = {PRE_KEY: lab_pre, POST_KEY: lab_post}

    def get(model, fn, col):
        row = agg[(agg['model'] == model) & (agg['fn'] == fn)]
        return float(row[col].iloc[0]) if len(row) else float('nan')

    n_splits = int(gdf['split'].nunique())
    lines.append(f"Splits aggregated: {n_splits}  "
                 f"({', '.join(str(s) for s in sorted(gdf['split'].unique()))})")
    lines.append(f"  PRE  ('{PRE_KEY}')  = {lab_pre}")
    lines.append(f"  POST ('{POST_KEY}') = {lab_post}")
    lines.append("  (lower ECE/Brier/NLL/MCE = better calibration; "
                 "delta = POST - PRE, negative = FT improved)")
    lines.append("")

    for met in metrics:
        lines.append(f"=== {met.upper()} vs false-negative noise (mean +/- std across splits) ===")
        header = (f"{'fn':>6} | {lab_pre:>20} | {lab_post:>20} | "
                  f"{'delta(post-pre)':>16} | {'better':>7}")
        lines.append(header)
        lines.append("-" * len(header))
        for fn in fns:
            pm, ps = get(PRE_KEY, fn, f'{met}_mean'), get(PRE_KEY, fn, f'{met}_std')
            qm, qs = get(POST_KEY, fn, f'{met}_mean'), get(POST_KEY, fn, f'{met}_std')
            delta = qm - pm
            better = 'POST' if delta < 0 else ('PRE' if delta > 0 else 'tie')
            lines.append(f"{fn:>6.2f} | {fmt_mean_std(pm, ps):>20} | "
                         f"{fmt_mean_std(qm, qs):>20} | {delta:>+16.4f} | {better:>7}")
        # Headline row: the mean over all fn levels, unweighted.
        pre_all = gdf[gdf['model'] == PRE_KEY][met].mean()
        post_all = gdf[gdf['model'] == POST_KEY][met].mean()
        lines.append(f"{'ALL':>6} | {pre_all:>20.4f} | {post_all:>20.4f} | "
                     f"{post_all - pre_all:>+16.4f} | "
                     f"{'POST' if post_all < pre_all else 'PRE':>7}")
        lines.append("")

    return agg, lines


def aggregate_category(cdf, cat_fn, lab_pre, lab_post):
    """Per-category ECE at a chosen fn, sorted by post-minus-pre delta."""
    if cdf is None:
        return None, ["(no cat_metrics.parquet found -- skipping category breakdown)"]
    fns = sorted(cdf['fn'].unique())
    # --cat-fn is snapped to the nearest fn actually present in the parquet.
    fn = min(fns, key=lambda x: abs(x - cat_fn))
    sub = cdf[cdf['fn'] == fn].copy()

    agg = (sub.groupby(['category', 'category_desc', 'model'])['ece']
              .agg(['mean', 'std']).reset_index())
    piv = agg.pivot_table(index=['category', 'category_desc'],
                          columns='model', values='mean')
    for k in (PRE_KEY, POST_KEY):
        if k not in piv.columns:
            piv[k] = float('nan')
    piv['delta'] = piv[POST_KEY] - piv[PRE_KEY]
    piv = piv.sort_values('delta')  # most-improved (negative) first

    lines = [f"=== Per-category ECE at fn={fn:.2f} "
             f"(mean across splits; delta = POST - PRE) ===",
             f"{'cat':>4} {'description':<34} {lab_pre:>12} {lab_post:>12} {'delta':>10}"]
    lines.append("-" * 76)
    for (cat, desc), row in piv.iterrows():
        d = desc if isinstance(desc, str) else ''
        lines.append(f"{cat:>4} {d[:34]:<34} {row[PRE_KEY]:>12.4f} "
                     f"{row[POST_KEY]:>12.4f} {row['delta']:>+10.4f}")
    return piv.reset_index(), lines


def main():
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--results-glob', default='calibration_T20_ft_split*',
                    help="Glob for the per-split output dirs.")
    pa.add_argument('--outdir', default='calibration_T20_ft_aggregate')
    pa.add_argument('--cat-fn', type=float, default=0.25,
                    help="Noise level for the per-category breakdown.")
    pa.add_argument('--label-pre', default='T20 pre-FT')
    pa.add_argument('--label-post', default='T20 mix-FT')
    pa.add_argument('--eval-desc',
                    default='original split-N held-out, archaea only '
                            '(--val-domain d__Archaea)',
                    help='One-line eval-set description for the SUMMARY '
                         'header; set this for the global both-domain pass.')
    A = pa.parse_args()

    dirs = sorted(glob.glob(A.results_glob), key=_split_num)
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit(f"No result dirs match {A.results_glob!r}")

    gdf, gfound = load_metric(dirs, 'global_metrics.parquet')
    if gdf is None:
        raise SystemExit("No global_metrics.parquet found in any split dir")
    cdf, cfound = load_metric(dirs, 'cat_metrics.parquet')

    outdir = Path(A.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gdf.to_csv(outdir / 'global_aggregate.csv', index=False)
    gsum, glines = aggregate_global(gdf, A.label_pre, A.label_post)
    gsum.to_csv(outdir / 'global_summary.csv', index=False)
    cpiv, clines = aggregate_category(cdf, A.cat_fn, A.label_pre, A.label_post)
    if cpiv is not None:
        cpiv.to_csv(outdir / 'category_summary.csv', index=False)

    report = []
    report.append("=" * 76)
    report.append("Calibration comparison: T20 NoHidden HO3  PRE vs MIX fine-tune")
    report.append(f"Eval set: {A.eval_desc}")
    report.append("=" * 76)
    report.append(f"global_metrics found in splits: {gfound}")
    report.append(f"cat_metrics found in splits:    {cfound}")
    report.append("")
    report.extend(glines)
    report.append("")
    report.extend(clines)
    text = "\n".join(report)
    (outdir / 'SUMMARY.txt').write_text(text + "\n")
    print(text)
    print(f"\nWrote: {outdir}/SUMMARY.txt, global_summary.csv, "
          f"global_aggregate.csv, category_summary.csv")


if __name__ == '__main__':
    main()
