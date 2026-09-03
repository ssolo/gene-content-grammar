#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py -- figures for the denoising-spectra and calibration study.

Two independent evaluation axes:

  Calibration fine-tune (pre-FT vs mix-FT), from the CSVs in
  data/calibration_ft/ (scripts/aggregate_calibration_ft.py).

  Spectra + cross-family calibration, from
  data/spectra_calibration_all.parquet (scripts/aggregate_spectra_sweep.py),
  drawn only if that parquet is present.

Each figure goes to --outdir; main() lists the output names and the fig_*
docstrings describe their content.

Figures use mathtext only, so no system LaTeX is required.

In data/calibration_ft the parquet/CSV 'model' column carries repurposed
slot keys: 'hidden' = T20 pre-FT, 'nohidden' = T20 mix-FT (see
aggregate_calibration_ft.py); LAB maps them back for display.

Usage:
  python3 scripts/plot_results.py
  python3 scripts/plot_results.py --calib-dir data/calibration_ft \\
      --spectra data/spectra_calibration_all.parquet --outdir analysis/figures
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- Palette + style
OKABE = {
    'black':  '#000000',
    'orange': '#E69F00',
    'sky':    '#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'verm':   '#D55E00',
    'purple': '#CC79A7',
}

C_PRE = OKABE['blue']
C_MIX = OKABE['verm']

# slot-key -> display label
LAB = {'hidden': 'T20 pre-FT', 'nohidden': 'T20 mix-FT'}

# cross-family display order, label and colour
FAMILY_STYLE = [
    ('nohidden_HO_T8',       'NoHidden HO, T=8',   OKABE['sky'],    'o'),
    ('nohidden_HO_T12',      'NoHidden HO, T=12',  OKABE['green'],  's'),
    ('nohidden_HO_T16',      'NoHidden HO, T=16',  OKABE['blue'],   '^'),
    ('nohidden_HO_T20',      'NoHidden HO, T=20',  OKABE['black'],  'D'),
    ('nohidden_HO_T20_mixFT','T20 mix-FT',         OKABE['verm'],   'v'),
    ('nohidden_HO_T20_arcFT','T20 arc-FT',         OKABE['orange'], 'P'),
    ('hidden_HO_T8',         'Hidden HO, T=8',     OKABE['purple'], 'X'),
]
NULL_TAGS = {'noisy', 'null'}

METRICS = ['ece', 'mce', 'brier', 'nll']
MET_LABEL = {'ece': 'ECE', 'mce': 'MCE', 'brier': 'Brier score',
             'nll': 'NLL'}

# Ancestral (LBCA/LUCA) operating range in false-negative noise.
ANC_LO, ANC_HI = 0.75, 0.90


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10.5,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
    })


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def shade_anc(ax):
    ax.axvspan(ANC_LO, ANC_HI, color='0.85', alpha=0.45, lw=0, zorder=0)


# ---- Calibration fine-tune figures
def fig_cal_ft(summary_csv, title, out_pdf):
    """2x2 [ECE,MCE,Brier,NLL] vs FN: pre-FT vs mix-FT, +/- std bands."""
    df = pd.read_csv(summary_csv)
    fns = sorted(df['fn'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    for ax, met in zip(axes.flat, METRICS):
        shade_anc(ax)
        for key, col in (('hidden', C_PRE), ('nohidden', C_MIX)):
            sub = df[df['model'] == key].sort_values('fn')
            m = sub[f'{met}_mean'].values
            s = sub[f'{met}_std'].values
            x = sub['fn'].values
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.15, lw=0)
            ax.plot(x, m, color=col, marker='o', label=LAB[key])
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(MET_LABEL[met] + r'  (lower = better)')
        ax.set_title(MET_LABEL[met])
        ax.set_xlim(min(fns) - 0.02, max(fns) + 0.02)
        ax.margins(y=0.08)
    axes.flat[0].legend(loc='upper left')
    fig.suptitle(title, fontsize=12.5, y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_cal_delta(global_agg_csv, arc_agg_csv, out_pdf):
    """Paired across-split mix-minus-pre delta vs FN, all four metrics.

    Splits are paired within each (model, fn) cell; the band is the
    across-split SE of the paired delta.  Negative = the fine-tune improved.
    """
    def paired_delta(agg_csv):
        df = pd.read_csv(agg_csv)
        out = {}
        for met in METRICS:
            piv = df.pivot_table(index=['split', 'fn'], columns='model',
                                 values=met).reset_index()
            piv['delta'] = piv['nohidden'] - piv['hidden']
            g = (piv.groupby('fn')['delta']
                    .agg(['mean', 'std', 'count']).reset_index())
            out[met] = g
        return out

    sets = [('global (both domains)', paired_delta(global_agg_csv), OKABE['black']),
            ('archaea-only',          paired_delta(arc_agg_csv),    OKABE['verm'])]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    for ax, met in zip(axes.flat, METRICS):
        shade_anc(ax)
        ax.axhline(0.0, color='0.4', lw=1.0, ls='--', zorder=1)
        for label, dd, col in sets:
            g = dd[met].sort_values('fn')
            x = g['fn'].values
            m = g['mean'].values
            se = (g['std'] / np.sqrt(g['count'].clip(lower=1))).values
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
            ax.plot(x, m, color=col, marker='o', label=label)
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(r'$\Delta$ ' + MET_LABEL[met] + '  (mix $-$ pre)')
        ax.set_title(r'$\Delta$ ' + MET_LABEL[met])
        ax.margins(y=0.1)
    axes.flat[0].legend(loc='lower left', title='held-out set')
    fig.suptitle('Mix fine-tune minus pre-FT calibration delta '
                 '(negative = improved; band = across-split SE)',
                 fontsize=12.5, y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_cal_ft_bac_delta(hard_agg_csv, vhard_agg_csv, out_pdf):
    """Paired (FT - pre-FT) calibration delta vs FN for the two bacterial
    fine-tuning curricula, all four metrics.

    Both arms are fine-tuned from the same depth-20 generalist and share the
    pre-FT baseline ('hidden'); held-out splits are bacteria-only, with phyla
    unseen in training.  Splits are paired within each (model, fn) cell and
    the band is the across-split SE of the paired delta, since the raw
    across-split std (~0.012) dwarfs the deltas (~0.001).  Negative = the
    fine-tune improved calibration."""
    def paired_delta(agg_csv):
        df = pd.read_csv(agg_csv)
        out = {}
        for met in METRICS:
            piv = df.pivot_table(index=['split', 'fn'], columns='model',
                                 values=met).reset_index()
            piv['delta'] = piv['nohidden'] - piv['hidden']
            g = (piv.groupby('fn')['delta']
                    .agg(['mean', 'std', 'count']).reset_index())
            out[met] = g
        return out

    sets = [('bac-hard-FT',  paired_delta(hard_agg_csv),  OKABE['green']),
            ('bac-vhard-FT', paired_delta(vhard_agg_csv), OKABE['orange'])]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    for ax, met in zip(axes.flat, METRICS):
        shade_anc(ax)
        ax.axhline(0.0, color='0.4', lw=1.0, ls='--', zorder=1)
        for label, dd, col in sets:
            g = dd[met].sort_values('fn')
            x = g['fn'].values
            m = g['mean'].values
            se = (g['std'] / np.sqrt(g['count'].clip(lower=1))).values
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
            ax.plot(x, m, color=col, marker='o', label=label)
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(r'$\Delta$ ' + MET_LABEL[met] + r'  (FT $-$ pre)')
        ax.set_title(r'$\Delta$ ' + MET_LABEL[met])
        ax.margins(y=0.1)
    axes.flat[0].legend(loc='upper left', title='curriculum')
    fig.suptitle('Bacterial fine-tune minus pre-FT calibration delta, '
                 'bacteria-only held-out (unseen phyla)\n'
                 '(negative = improved; band = across-split SE; '
                 'shaded = ancestral range)',
                 fontsize=11.5, y=1.05)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_cal_category(global_cat_csv, arc_cat_csv, out_pdf):
    """Per-COG-category ECE delta at fn=0.25, global and archaea-only."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 7.2), constrained_layout=True,
                             sharex=True)
    for ax, csv, title in ((axes[0], global_cat_csv, 'global (both domains)'),
                           (axes[1], arc_cat_csv, 'archaea-only')):
        df = pd.read_csv(csv).sort_values('delta')
        labels = [f"{r.category}  {str(r.category_desc)[:26]}"
                  for r in df.itertuples()]
        y = np.arange(len(df))
        colors = [C_MIX if d < 0 else OKABE['sky'] for d in df['delta']]
        ax.barh(y, df['delta'].values, color=colors, height=0.72)
        ax.axvline(0.0, color='0.3', lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.0)
        ax.set_ylim(-0.6, len(df) - 0.4)
        ax.set_xlabel(r'$\Delta$ ECE  (mix $-$ pre)')
        ax.set_title(title)
        despine(ax)
        ax.grid(axis='y', visible=False)
    fig.suptitle(r'Per-category ECE change at $f_N=0.25$ '
                 '(negative = mix-FT better calibrated)',
                 fontsize=12.5, y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ---- Spectra + cross-family figures (gated on the sweep parquet)
def _present_families(df):
    have = set(df['family'].unique())
    return [(f, lab, c, mk) for (f, lab, c, mk) in FAMILY_STYLE if f in have]


def final_step_df(df):
    """Rows at each family's final internal annealing step (step == max).

    A stepless parquet is returned unchanged.  Required by every figure that
    is a function of (fn, fp) only: otherwise groupby('fn').mean() silently
    averages over the whole annealing trajectory."""
    if 'step' not in df.columns:
        return df
    smax = df.groupby('family')['step'].transform('max')
    # step == -1 marks a single-pass row (no trajectory recorded); it counts as
    # final even when another split of the same family reached a higher step.
    return df[(df['step'] == smax) | (df['step'] == -1)]


def _null_curve(df, value, fp):
    n = df[(df['family'].isin(NULL_TAGS)) & (np.isclose(df['fp'], fp))]
    if n.empty:
        return None, None
    g = n.groupby('fn')[value].mean().reset_index().sort_values('fn')
    return g['fn'].values, g[value].values


def fig_spectra_metric_grid(df, value, ylabel, out_pdf, title):
    """value vs FN per family, one panel per FP in the grid."""
    fps = sorted(df['fp'].unique())
    fams = _present_families(df)
    ncol = 2
    nrow = int(np.ceil(len(fps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0, 3.4 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, fp in zip(axes.flat, fps):
        shade_anc(ax)
        for fam, lab, col, mk in fams:
            sub = (df[(df['family'] == fam) & (np.isclose(df['fp'], fp))]
                   .groupby('fn')[value].mean().reset_index().sort_values('fn'))
            if sub.empty:
                continue
            ax.plot(sub['fn'], sub[value], color=col, marker=mk,
                    markersize=4, label=lab)
        nx, ny = _null_curve(df, value, fp)
        if nx is not None:
            ax.plot(nx, ny, color='0.5', ls=':', lw=1.5, label='noisy input')
        despine(ax)
        ax.set_title(f'FP = {fp:g}')
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylabel)
        if value == 'MCC':
            ax.set_ylim(0, 1)
    for ax in axes.flat[len(fps):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc='best', ncol=1)
    fig.suptitle(title, fontsize=12.5, y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_spectra_prec_rec(df, out_pdf, fp=0.01):
    """F1 / precision / recall vs FN at a single FP."""
    fams = _present_families(df)
    panels = [('F1', 'F1'), ('prec', 'precision'), ('rec', 'recall')]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7),
                             constrained_layout=True)
    for ax, (col_, name) in zip(axes, panels):
        shade_anc(ax)
        for fam, lab, col, mk in fams:
            sub = (df[(df['family'] == fam) & (np.isclose(df['fp'], fp))]
                   .groupby('fn')[col_].mean().reset_index().sort_values('fn'))
            if sub.empty:
                continue
            ax.plot(sub['fn'], sub[col_], color=col, marker=mk,
                    markersize=4, label=lab)
        despine(ax)
        ax.set_title(name)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(name)
    axes[0].legend(loc='best')
    fig.suptitle(f'Recovery quality vs noise at FP = {fp:g}',
                 fontsize=12.5, y=1.04)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_mcc_heatmaps(df, out_pdf):
    """MCC over the (FN,FP) grid, one panel per family."""
    fams = _present_families(df)
    fns = sorted(df['fn'].unique())
    fps = sorted(df['fp'].unique())
    ncol = 4
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow),
                             constrained_layout=True, squeeze=False)
    vmin = max(0.0, float(df[df['family'].isin([f for f, *_ in fams])]['MCC'].min()))
    im = None
    for ax, (fam, lab, _c, _m) in zip(axes.flat, fams):
        sub = df[df['family'] == fam]
        grid = (sub.groupby(['fp', 'fn'])['MCC'].mean()
                   .reset_index()
                   .pivot(index='fp', columns='fn', values='MCC'))
        grid = grid.reindex(index=fps, columns=fns)
        im = ax.imshow(grid.values, origin='lower', aspect='auto',
                       cmap='cividis', vmin=vmin, vmax=1.0,
                       extent=[min(fns), max(fns), 0, len(fps)])
        ax.set_yticks(np.arange(len(fps)) + 0.5)
        ax.set_yticklabels([f'{v:g}' for v in fps], fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel('$f_N$')
        ax.set_ylabel('FP')
        ax.grid(False)
    for ax in axes.flat[len(fams):]:
        ax.set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                     label='MCC')
    fig.suptitle('Denoising MCC over the (FN, FP) noise grid',
                 fontsize=12.5, y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_xfamily_calibration(df, out_pdf, fp=0.01):
    """ECE vs FN across families at a single FP."""
    fams = _present_families(df)
    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    shade_anc(ax)
    for fam, lab, col, mk in fams:
        sub = (df[(df['family'] == fam) & (np.isclose(df['fp'], fp))]
               .groupby('fn')['ece'].mean().reset_index().sort_values('fn'))
        if sub.empty:
            continue
        ax.plot(sub['fn'], sub['ece'], color=col, marker=mk,
                markersize=4, label=lab)
    despine(ax)
    ax.set_xlabel('false-negative noise  $f_N$')
    ax.set_ylabel('ECE  (lower = better)')
    ax.set_title(f'Cross-family calibration at FP = {fp:g}')
    ax.legend(loc='best')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_mcc_ece_tradeoff(df, out_pdf, fp=0.01):
    """Accuracy vs calibration in the ancestral FN operating range."""
    fams = _present_families(df)
    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    anc = df[(df['fn'] >= ANC_LO) & (df['fn'] <= ANC_HI)
             & (np.isclose(df['fp'], fp))]
    for fam, lab, col, mk in fams:
        sub = anc[anc['family'] == fam]
        if sub.empty:
            continue
        g = sub.groupby('fn')[['MCC', 'ece']].mean().reset_index()
        ax.plot(g['ece'], g['MCC'], color=col, marker=mk, ms=7, ls='-',
                lw=1.0, label=lab)
    despine(ax)
    ax.set_xlabel('ECE  (lower = better calibrated)')
    ax.set_ylabel('MCC  (higher = better recovery)')
    ax.set_title(r'Accuracy vs calibration, ancestral range '
                 f'($f_N$ {ANC_LO:g}-{ANC_HI:g}), FP = {fp:g}')
    ax.legend(loc='best')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_metric_vs_step(df, value, ylabel, out_pdf, title,
                       fns_show=(0.10, 0.50, 0.75, 0.90), fp=0.01):
    """`value` vs internal annealing step t=1..T, one curve per family, one
    panel per false-negative level, at a single FP.  The null baseline is
    step-independent and is drawn as a horizontal reference."""
    fams = _present_families(df)
    avail = sorted(df['fn'].unique())
    fns_show = [f for f in fns_show if any(np.isclose(avail, f))]
    ncol = 2
    nrow = int(np.ceil(len(fns_show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0, 3.4 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, fn in zip(axes.flat, fns_show):
        for fam, lab, col, mk in fams:
            sub = (df[(df['family'] == fam) & np.isclose(df['fp'], fp)
                      & np.isclose(df['fn'], fn) & (df['step'] >= 1)]
                   .groupby('step')[value].mean().reset_index()
                   .sort_values('step'))
            if sub.empty:
                continue
            ax.plot(sub['step'], sub[value], color=col, marker=mk,
                    markersize=3.5, label=lab)
        nv = df[(df['family'].isin(NULL_TAGS)) & np.isclose(df['fp'], fp)
                & np.isclose(df['fn'], fn)][value]
        if len(nv):
            ax.axhline(nv.mean(), color='0.5', ls=':', lw=1.5,
                       label='noisy input')
        despine(ax)
        ax.set_title(f'$f_N$ = {fn:g}')
        ax.set_xlabel('internal annealing step  $t$')
        ax.set_ylabel(ylabel)
        ax.xaxis.get_major_locator().set_params(integer=True)
    for ax in axes.flat[len(fns_show):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc='best', ncol=1, fontsize=7)
    fig.suptitle(f'{title}  (FP = {fp:g})', fontsize=12.5, y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ---- Higher-order vs plain pairwise at matched relaxation depth.
# HO = NoHiddenDenoiser plus the attention-parameterised higher-order field;
# plain = the same model without it.  Colour encodes depth T.
PLAIN_VS_HO_DEPTHS = [(8, 'sky'), (12, 'green'), (16, 'blue'), (20, 'black')]


def _fam_fn_curve(df, fam, value, fp):
    sub = (df[(df['family'] == fam) & np.isclose(df['fp'], fp)]
           .groupby('fn')[value].mean().reset_index().sort_values('fn'))
    return (sub['fn'].values, sub[value].values) if not sub.empty else (None, None)


def fig_ho_vs_plain(df, out_pdf, fp=0.01):
    """The higher-order ablation. Top row: absolute MCC (A) and ECE (B) vs FN.
    Bottom row: HO minus plain at matched depth T, in MCC (C) and ECE (D).
    Pass df reduced to the final step."""
    fams = set(df['family'].unique())
    depths = [(d, c) for (d, c) in PLAIN_VS_HO_DEPTHS
              if f'nohidden_HO_T{d}' in fams and f'nohidden_plain_T{d}' in fams]
    if not depths:
        print('[skip] ho-vs-plain: no matched HO/plain depths present')
        return
    fig, ((axMa, axEa), (axMd, axEd)) = plt.subplots(
        2, 2, figsize=(10.4, 7.8), constrained_layout=True)
    TOP = [('nohidden_HO_T20', 'HO T=20', OKABE['black'], 'D', '-'),
           ('nohidden_HO_T20_mixFT', 'mix-FT (best, Fig. 1)', OKABE['verm'],
            'o', '-')]
    for ax, value, ylab in ((axMa, 'MCC', 'MCC  (higher = better)'),
                            (axEa, 'ece', 'ECE  (lower = better)')):
        shade_anc(ax)
        for fam, lab, col, mk, ls in TOP:
            s = (df[(df['family'] == fam) & np.isclose(df['fp'], fp)]
                 .groupby('fn')[value].agg(['mean', 'std']).reset_index().sort_values('fn'))
            if s.empty:
                continue
            xs, ys = s['fn'].values, s['mean'].values
            sd = s['std'].fillna(0.0).values
            ax.fill_between(xs, ys - sd, ys + sd, color=col, alpha=0.15, lw=0)
            ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=4, label=lab)
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylab)
    axMa.set_ylim(0, 1)
    axMa.set_title('(A)  MCC: HO T=20 vs mix-FT  (higher better)', loc='left',
                   fontweight='bold')
    axEa.set_title('(B)  ECE: HO T=20 vs mix-FT  (lower better)', loc='left',
                   fontweight='bold')
    axMa.legend(loc='lower left', fontsize=8.0)
    for ax, value, ylab in (
            (axMd, 'MCC', r'$\Delta$MCC  (HO $-$ plain)'),
            (axEd, 'ece', r'$\Delta$ECE  (HO $-$ plain)')):
        shade_anc(ax)
        ax.axhline(0, color='0.4', lw=1.0, zorder=1)
        for d, c in depths:
            hf = df[(df['family'] == f'nohidden_HO_T{d}') & np.isclose(df['fp'], fp)][['split', 'fn', value]]
            pf = (df[(df['family'] == f'nohidden_plain_T{d}') & np.isclose(df['fp'], fp)]
                  [['split', 'fn', value]].rename(columns={value: '_p'}))
            mm = hf.merge(pf, on=['split', 'fn'])
            if mm.empty:
                continue
            mm['_d'] = mm[value] - mm['_p']
            g = mm.groupby('fn')['_d'].agg(['mean', 'std']).reset_index().sort_values('fn')
            gx, gm = g['fn'].values, g['mean'].values
            gs = g['std'].fillna(0.0).values
            ax.fill_between(gx, gm - gs, gm + gs, color=OKABE[c], alpha=0.12, lw=0)
            ax.plot(gx, gm, color=OKABE[c], marker='o', ms=3.5, label=f'T={d}')
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylab)
    axMd.set_title(r'(C)  $\Delta$MCC, HO $-$ plain  (higher better)',
                   loc='left', fontweight='bold')
    axEd.set_title(r'(D)  $\Delta$ECE, HO $-$ plain  (lower better)',
                   loc='left', fontweight='bold')
    axMd.legend(loc='best', ncol=2, fontsize=7.0)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_spectra_best_delta(df, out_pdf, best='nohidden_HO_T20_mixFT', fp=0.01):
    """Top row: the best model's absolute MCC (left) and ECE (right) vs FN,
    with the noisy-input baseline. Bottom row: every other model's delta to
    the best model in MCC (left) and ECE (right). Pass df already reduced to
    the final annealing step."""
    fams = set(df['family'].unique())
    if best not in fams:
        print(f'[skip] best-delta: best model {best} absent')
        return
    COMPARE = [
        ('nohidden_HO_T8',        'HO T8',        OKABE['sky'],   'o', '-'),
        ('nohidden_HO_T12',       'HO T12',       OKABE['green'], 's', '-'),
        ('nohidden_HO_T16',       'HO T16',       OKABE['blue'],  '^', '-'),
        ('nohidden_HO_T20',       'HO T20',       OKABE['black'], 'D', '-'),
        ('nohidden_plain_T20',    'Pairwise T20', '#777777',      'v', '--'),
        ('hidden_HO_T8',          'hidden+HO T8', OKABE['purple'],'X', '-'),
        ('nohidden_HO_T20_arcFT', 'arc-FT',       OKABE['orange'],'P', '-'),
    ]
    COMPARE = [c for c in COMPARE if c[0] in fams]

    def curve(fam, val):
        sub = (df[(df['family'] == fam) & np.isclose(df['fp'], fp)]
               .groupby('fn')[val].mean().reset_index().sort_values('fn'))
        return sub['fn'].values, sub[val].values

    fig, ((axMa, axEa), (axMd, axEd)) = plt.subplots(
        2, 2, figsize=(10.4, 7.8), constrained_layout=True)
    for ax, val, ylab in ((axMa, 'MCC', 'MCC  (higher = better)'),
                          (axEa, 'ece', 'ECE  (lower = better)')):
        shade_anc(ax)
        bsub = (df[(df['family'] == best) & np.isclose(df['fp'], fp)]
                .groupby('fn')[val].agg(['mean', 'std']).reset_index().sort_values('fn'))
        bx, by = bsub['fn'].values, bsub['mean'].values
        bs = bsub['std'].fillna(0.0).values
        ax.fill_between(bx, by - bs, by + bs, color=OKABE['verm'], alpha=0.18,
                        lw=0, label=r'$\pm1$ SD (10 splits)')
        ax.plot(bx, by, color=OKABE['verm'], marker='o', ms=4, lw=2.0,
                label='mix-FT (best)')
        nx, ny = _null_curve(df, val, fp)
        if nx is not None:
            ax.plot(nx, ny, color='0.45', ls=':', lw=1.7, label='noisy input')
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylab)
    axMa.set_ylim(0, 1)
    axMa.set_title('(A)  MCC, best model  (higher better)', loc='left',
                   fontweight='bold')
    axEa.set_title('(B)  ECE, best model  (lower better)', loc='left',
                   fontweight='bold')
    axMa.legend(loc='lower left', fontsize=8.5)
    bxM, byM = curve(best, 'MCC'); bestM = dict(zip(np.round(bxM, 3), byM))
    bxE, byE = curve(best, 'ece'); bestE = dict(zip(np.round(bxE, 3), byE))
    for ax, val, bmap, ylab in (
            (axMd, 'MCC', bestM, r'$\Delta$MCC  (model $-$ mix-FT)'),
            (axEd, 'ece', bestE, r'$\Delta$ECE  (model $-$ mix-FT)')):
        shade_anc(ax)
        ax.axhline(0, color=OKABE['verm'], lw=1.4, zorder=1)
        for fam, lab, col, mk, ls in COMPARE:
            # Per-split paired delta to the best model; band = across-split SD.
            sf = df[(df['family'] == fam) & np.isclose(df['fp'], fp)][['split', 'fn', val]]
            sb = (df[(df['family'] == best) & np.isclose(df['fp'], fp)]
                  [['split', 'fn', val]].rename(columns={val: '_b'}))
            mm = sf.merge(sb, on=['split', 'fn'])
            if mm.empty:
                continue
            mm['_d'] = mm[val] - mm['_b']
            g = (mm.groupby('fn')['_d'].agg(['mean', 'std']).reset_index().sort_values('fn'))
            gx, gm = g['fn'].values, g['mean'].values
            gs = g['std'].fillna(0.0).values
            ax.fill_between(gx, gm - gs, gm + gs, color=col, alpha=0.12, lw=0)
            ax.plot(gx, gm, color=col, marker=mk, ms=3.5, ls=ls, label=lab)
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylab)
    axMd.set_title(r'(C)  $\Delta$MCC to best  (higher better)', loc='left',
                   fontweight='bold')
    axEd.set_title(r'(D)  $\Delta$ECE to best  (lower better)', loc='left',
                   fontweight='bold')
    axEd.legend(loc='best', fontsize=7.3, ncol=2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'wrote {out_pdf}')


def fig_spectra_fp_varying(df, out_pdf, fam='nohidden_HO_T20_mixFP',
                           fp_list=(0.01, 0.025, 0.05, 0.1, 0.2)):
    """MCC (A) and ECE (B) vs FN for `fam`, one curve per false-positive
    rate. Pass df reduced to the final step."""
    fams = set(df['family'].unique())
    if fam not in fams:
        print(f'[skip] fp-varying: family {fam} absent')
        return
    avail = sorted(df[df['family'] == fam]['fp'].unique())
    fps = [f for f in fp_list if any(np.isclose(avail, f))]
    if not fps:
        print('[skip] fp-varying: none of the requested FPs present')
        return
    cols = [OKABE['sky'], OKABE['green'], OKABE['blue'], OKABE['orange'],
            OKABE['verm'], OKABE['purple']]
    fig, (axM, axE) = plt.subplots(1, 2, figsize=(10.4, 4.4),
                                   constrained_layout=True)
    band_lab = r'$\pm1$ SD (10 splits)'
    for ax, val, ylab in ((axM, 'MCC', 'MCC  (higher = better)'),
                          (axE, 'ece', 'ECE  (lower = better)')):
        shade_anc(ax)
        for i, fpv in enumerate(fps):
            sub = (df[(df['family'] == fam) & np.isclose(df['fp'], fpv)]
                   .groupby('fn')[val].agg(['mean', 'std'])
                   .reset_index().sort_values('fn'))
            if sub.empty:
                continue
            x, m = sub['fn'].values, sub['mean'].values
            s = sub['std'].fillna(0.0).values
            col = cols[i % len(cols)]
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0,
                            label=band_lab if i == 0 else None)
            ax.plot(x, m, color=col, marker='o', ms=4, label=f'FP = {fpv:g}')
        despine(ax)
        ax.set_xlabel('false-negative noise  $f_N$')
        ax.set_ylabel(ylab)
    axM.set_ylim(0, 1)
    axM.set_title('(A)  MCC across FP  (higher better)', loc='left',
                  fontweight='bold')
    axE.set_title('(B)  ECE across FP  (lower better)', loc='left',
                  fontweight='bold')
    axM.legend(loc='lower left', fontsize=8, title='false-positive rate')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def print_ho_vs_plain_table(df, fp=0.01, fns=(0.10, 0.50, 0.75, 0.85, 0.90)):
    """Print matched-depth HO vs plain MCC and ECE at representative FN."""
    for value, lab in (('MCC', 'MCC (higher=better)'),
                       ('ece', 'ECE (lower=better)')):
        print(f"\n  {lab}  at FP={fp:g}   (HO / plain)")
        hdr = f"  {'fn':>5}"
        for d, _ in PLAIN_VS_HO_DEPTHS:
            hdr += f"{'T='+str(d):>16}"
        print(hdr)
        for fn in fns:
            row = f"  {fn:5.2f}"
            for d, _ in PLAIN_VS_HO_DEPTHS:
                cell = ''
                for kind in ('HO', 'plain'):
                    sub = df[(df['family'] == f'nohidden_{kind}_T{d}')
                             & np.isclose(df['fp'], fp) & np.isclose(df['fn'], fn)]
                    cell += ('  --' if sub.empty else f"{sub[value].mean():6.3f}")
                    cell += '' if kind == 'plain' else '/'
                row += f"{cell:>16}"
            print(row)


def main():
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--calib-dir', default='data/calibration_ft')
    pa.add_argument('--bac-hard-dir', default='calibration_T20_bacft_hard_aggregate',
                    help='bac-hard-FT calibration aggregate dir (global_aggregate.csv)')
    pa.add_argument('--bac-vhard-dir', default='calibration_T20_bacft_vhard_aggregate',
                    help='bac-vhard-FT calibration aggregate dir (global_aggregate.csv)')
    pa.add_argument('--spectra', default='data/spectra_calibration_all.parquet')
    pa.add_argument('--outdir', default='analysis/figures')
    A = pa.parse_args()

    setup_style()
    out = Path(A.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(A.calib_dir)

    if (cdir / 'global_summary.csv').exists():
        fig_cal_ft(cdir / 'global_summary.csv',
                   'Calibration vs noise: T20 pre-FT vs mix-FT '
                   '(global held-out, 10 splits)',
                   out / 'cal_ft_global.pdf')
        fig_cal_ft(cdir / 'arc_summary.csv',
                   'Calibration vs noise: T20 pre-FT vs mix-FT '
                   '(archaea-only held-out, 9 splits)',
                   out / 'cal_ft_arc.pdf')
        fig_cal_delta(cdir / 'global_aggregate.csv',
                      cdir / 'arc_aggregate.csv',
                      out / 'cal_ft_delta.pdf')
        fig_cal_category(cdir / 'global_category.csv',
                         cdir / 'arc_category.csv',
                         out / 'cal_ft_category.pdf')
    else:
        print(f"[skip] calibration: {cdir}/global_summary.csv not found")

    hard_agg = Path(A.bac_hard_dir) / 'global_aggregate.csv'
    vhard_agg = Path(A.bac_vhard_dir) / 'global_aggregate.csv'
    if hard_agg.exists() and vhard_agg.exists():
        fig_cal_ft_bac_delta(hard_agg, vhard_agg, out / 'cal_ft_bac_delta.pdf')
    else:
        print(f"[skip] bac-FT calibration: {hard_agg} or {vhard_agg} not found")

    if os.path.exists(A.spectra):
        df = pd.read_parquet(A.spectra)
        has_step = 'step' in df.columns
        dff = final_step_df(df)
        mode = ('per-step trajectory; (fn,fp) figures at final step'
                if has_step else 'single-pass')
        print(f"loaded spectra: {df.shape} from {A.spectra} ({mode})")
        fig_spectra_metric_grid(dff, 'MCC', 'MCC  (higher = better)',
                                out / 'spectra_mcc_vs_fn.pdf',
                                'Denoising MCC vs false-negative noise')
        fig_spectra_prec_rec(dff, out / 'spectra_f1_prec_rec.pdf')
        fig_spectra_metric_grid(dff, 'fp_removed',
                                'fraction injected FP removed',
                                out / 'spectra_fp_removed.pdf',
                                'False-positive removal vs noise')
        fig_mcc_heatmaps(dff, out / 'spectra_mcc_heatmap.pdf')
        fig_xfamily_calibration(dff, out / 'xfamily_calibration.pdf')
        fig_mcc_ece_tradeoff(dff, out / 'mcc_ece_tradeoff.pdf')
        fig_spectra_best_delta(dff, out / 'spectra_best_delta.pdf')
        fig_spectra_fp_varying(dff, out / 'spectra_fp_varying.pdf')
        fig_ho_vs_plain(dff, out / 'spectra_ho_vs_plain.pdf')
        print_ho_vs_plain_table(dff)
        if has_step:
            fig_metric_vs_step(df, 'MCC', 'MCC  (higher = better)',
                               out / 'spectra_mcc_vs_step.pdf',
                               'Denoising MCC across internal annealing steps')
            fig_metric_vs_step(df, 'ece', 'ECE  (lower = better)',
                               out / 'spectra_ece_vs_step.pdf',
                               'Calibration (ECE) across internal annealing steps')
    else:
        print(f"[gate] spectra parquet {A.spectra} not present yet -- "
              "skipping spectra figures (re-run after the sweep lands).")

    print(f"\nfigures in {out}/")


if __name__ == '__main__':
    main()
