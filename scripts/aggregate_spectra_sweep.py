#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_spectra_sweep.py -- concatenate the per-(family, split) parquets
written by sweep_spectra_calibration.py into one tidy plotting dataset, plus an
across-split mean/std summary.

Each input parquet (one per scored checkpoint) has one row per
(fn, fp, step) cell with columns:
    model_tag family split cls_kind T onsager n_genomes
    fn fp step  MCC F1 prec rec fp_removed base_rate  ece mce brier nll

`step` is the model's INTERNAL annealing iteration: model rows carry
step = 1..T, where step T is the converged output and is byte-identical to the
single-pass prediction, and the model-free null baseline carries step = 0.
Single-pass parquets have no `step` column at all; those rows are given the
sentinel step = -1 and treated as already final.

The long-form output keeps every step so that per-step trajectory plots remain
possible; the headline tables are taken at each family's final step.

Outputs:
    <out>                       long-form per-(family, split, fn, fp, step) parquet
    <out stem>_summary.csv      per-(family, fn, fp, step) mean/std/n across splits
    <out stem>_SUMMARY.txt      headline tables at the final step

Usage:
  python3 scripts/aggregate_spectra_sweep.py \\
      [--results-glob 'spectra_traj/*.parquet'] \\
      [--out data/spectra_calibration_all.parquet]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

SPECTRA = ['MCC', 'F1', 'prec', 'rec', 'fp_removed']
CALIB = ['ece', 'mce', 'brier', 'nll']


def main():
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--results-glob', default='spectra_traj/*.parquet')
    pa.add_argument('--out', default='data/spectra_calibration_all.parquet')
    pa.add_argument('--headline-fp', type=float, default=0.01,
                    help='FP level used for the printed headline tables.')
    A = pa.parse_args()

    files = sorted(glob.glob(A.results_glob))
    if not files:
        raise SystemExit(f"No parquets match {A.results_glob!r}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Single-pass parquets carry no `step` column, so in a mixed glob concat
    # leaves their entries NaN; fill with the sentinel -1 before the integer cast.
    has_step = 'step' in df.columns
    if not has_step:
        df['step'] = -1
    df['step'] = df['step'].fillna(-1).astype(int)
    # A null baseline scored under several splits is legitimate; two rows sharing
    # (family, split, fn, fp, step) are a duplicated scoring run.
    df = df.drop_duplicates(subset=['family', 'split', 'fn', 'fp', 'step'])

    out = Path(A.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    metrics = [m for m in SPECTRA + CALIB if m in df.columns]
    summary = (df.groupby(['family', 'fn', 'fp', 'step'])[metrics]
                 .agg(['mean', 'std', 'count']).reset_index())
    summary.columns = ['family', 'fn', 'fp', 'step'] + [
        f'{m}_{s}' for m in metrics for s in ('mean', 'std', 'count')]
    sum_csv = out.with_name(out.stem + '_summary.csv')
    summary.to_csv(sum_csv, index=False)

    # Final annealing step per family: T for a model, 0 for the null baseline.
    # A step of -1 is already converged, so it survives the filter even when a
    # sibling split of the same family contributed a higher trajectory step.
    smax = df.groupby('family')['step'].transform('max')
    df_final = df[(df['step'] == smax) | (df['step'] == -1)]

    fams = sorted(f for f in df_final['family'].unique() if f != 'noisy')
    fns = sorted(df_final['fn'].unique())
    fps = sorted(df_final['fp'].unique())
    fp_h = min(fps, key=lambda x: abs(x - A.headline_fp))
    tfam = {f: int(df[df.family == f]['step'].max()) for f in fams}

    L = []
    L.append('=' * 78)
    L.append('Denoising-spectra + calibration sweep -- aggregate')
    L.append('=' * 78)
    L.append(f"inputs: {len(files)} parquets, {len(df)} rows "
             f"({'per-step trajectory' if has_step else 'single-pass'})")
    L.append(f"families: {fams}")
    L.append(f"FN grid ({len(fns)}): {fns}")
    L.append(f"FP grid: {fps}")
    nsplit = {f: int(df[df.family == f]['split'].nunique()) for f in fams}
    L.append(f"splits per family: {nsplit}")
    if has_step:
        L.append(f"final step per family (T): {tfam}")
        L.append("(headline tables below are at each family's FINAL step)")
    L.append('')

    def table(metric, lo_is_better):
        L.append(f"=== {metric.upper()} vs FN at FP={fp_h:.2f} "
                 f"(final-step mean across splits; "
                 f"{'lower' if lo_is_better else 'higher'} = better) ===")
        hdr = f"{'family':<26}" + ''.join(f"{fn:>8.2f}" for fn in fns)
        L.append(hdr)
        L.append('-' * len(hdr))
        for fam in fams:
            sub = df_final[(df_final.family == fam)
                           & (np.abs(df_final.fp - fp_h) < 1e-9)]
            cells = []
            for fn in fns:
                v = sub[np.abs(sub.fn - fn) < 1e-9][metric]
                cells.append(f"{v.mean():>8.4f}" if len(v) else f"{'--':>8}")
            L.append(f"{fam:<26}" + ''.join(cells))
        L.append('')

    for met in ['MCC', 'F1']:
        if met in df.columns:
            table(met, lo_is_better=False)
    for met in ['ece', 'brier', 'nll']:
        if met in df.columns:
            table(met, lo_is_better=True)

    txt = '\n'.join(L)
    sum_txt = out.with_name(out.stem + '_SUMMARY.txt')
    sum_txt.write_text(txt + '\n')
    print(txt)
    print(f"\nWrote:\n  {out}\n  {sum_csv}\n  {sum_txt}")


if __name__ == '__main__':
    main()
