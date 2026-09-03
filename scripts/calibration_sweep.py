#!/usr/bin/env python3
"""Fine-grained FN/FP calibration sweep, with figures and tables.

Sweeps FN in [0, 0.9] step 0.025 (37 values) x FP in {0, 0.01, 0.025, 0.05, 0.1}
for the hidden and no-hidden denoisers plus a noisy-input null.  Each condition
yields global and per-COG classification metrics (MCC, F1, precision, recall),
their aggregates by COG category and by KEGG module, and the error of KEGG
module completeness aggregated from the COG predictions.

Metrics are written as parquet, so replotting needs no re-inference:

  plots/mcc_vs_fn.pdf                     MCC vs FN, colour = FP, style = model
  plots/prec_rec_vs_fn.pdf                Precision and recall side by side
  plots/module_mae_vs_fn.pdf              Module-completeness MAE
  plots/per_category/per_category_all.pdf One page per COG category
  tables/*.csv                            Best/worst COG, category, module and
                                          completeness rankings at focal FNs

Runs under torch.distributed.run across the available GPUs.
"""

import argparse, json, time, sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from ising_denoiser.data import load_feathers
from ising_denoiser.models import ModuleConditionedDenoiser, NoHiddenDenoiser
from ising_denoiser.training import (
    setup_dist, cleanup, is_main, barrier, log)


# ----  Configuration

FN_DEFAULT_GRID = [round(i * 0.025, 4) for i in range(37)]  # 0.000 .. 0.900
FP_DEFAULT_GRID = [0.0, 0.01, 0.025, 0.05, 0.1]
FOCAL_FNS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
FOCAL_FP = 0.01

MODEL_COLOURS = {
    'hidden':   '#1f77b4',
    'nohidden': '#ff7f0e',
    'noisy':    '#7f7f7f',
}
MODEL_LABELS = {
    'hidden':   'Hidden-state denoiser',
    'nohidden': 'No-hidden denoiser',
    'noisy':    'Noisy input (null)',
}


# ----  Noise

def apply_noise(val_t, fn, fp):
    """Deterministic noise with seed keyed on (fn, fp)."""
    seed = 42 + int(fn * 1000) + int(fp * 10000) * 997
    torch.manual_seed(seed)
    ny = val_t.clone()
    r = torch.rand_like(ny)
    ny[(val_t == 1) & (r < fn)] = -1
    r2 = torch.rand_like(ny)
    ny[(val_t == -1) & (r2 < fp)] = 1
    return ny


# ----  Inference

@torch.no_grad()
def predict_distributed(model, noisy_t, M_mod, M_sizes, dev, batch_size):
    """Shard noisy_t across ranks, run batched inference, all-gather to rank 0.

    Returns a (B, N) float32 prediction tensor on rank 0, None elsewhere.
    """
    model.eval()
    use_ddp = dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world = dist.get_world_size() if use_ddp else 1

    B_full = noisy_t.size(0)
    local_idx = list(range(rank, B_full, world))
    local_noisy = noisy_t[local_idx].to(dev)

    parts = []
    with autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, local_noisy.size(0), batch_size):
            batch = local_noisy[i:i + batch_size]
            if M_mod is not None:
                out, _ = model(batch, M_mod, M_sizes)
            else:
                out, _ = model(batch)
            prob = (out.float() + 1.0) / 2.0
            parts.append(prob.cpu())
    local_pred = torch.cat(parts, dim=0)

    if not use_ddp:
        return local_pred

    max_local = (B_full + world - 1) // world
    padded = torch.zeros(max_local, local_pred.size(1))
    padded[:local_pred.size(0)] = local_pred
    padded_gpu = padded.to(dev)
    gathered_gpu = [torch.zeros_like(padded_gpu) for _ in range(world)]
    dist.all_gather(gathered_gpu, padded_gpu)
    gathered = [g.cpu() for g in gathered_gpu]

    if rank == 0:
        full = torch.zeros(B_full, local_pred.size(1))
        for r in range(world):
            idx = list(range(r, B_full, world))
            actual = len(idx)
            full[idx] = gathered[r][:actual]
        return full
    return None


# ----  Metrics (rank 0 only)

def confusion_counts_np(y_pred_bin, y_true_bin):
    """y_{pred,true}_bin: (B, N) int/bool arrays.  Returns TP,FP,TN,FN (N,)."""
    pred = y_pred_bin.astype(np.int32)
    true = y_true_bin.astype(np.int32)
    TP = ((pred == 1) & (true == 1)).sum(axis=0)
    FP = ((pred == 1) & (true == 0)).sum(axis=0)
    TN = ((pred == 0) & (true == 0)).sum(axis=0)
    FN = ((pred == 0) & (true == 1)).sum(axis=0)
    return TP, FP, TN, FN


def metrics_from_counts(TP, FP, TN, FN, eps=1e-12):
    """Per-COG MCC/F1/prec/rec from per-COG confusion counts."""
    TP = TP.astype(np.float64); FP = FP.astype(np.float64)
    TN = TN.astype(np.float64); FN = FN.astype(np.float64)
    num = TP * TN - FP * FN
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps)
    mcc = num / den
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return mcc, f1, prec, rec


def global_metrics_from_counts(TP, FP, TN, FN, eps=1e-12):
    """Pooled MCC/F1/prec/rec over all genes."""
    tp, fp, tn, fn = float(TP.sum()), float(FP.sum()), float(TN.sum()), float(FN.sum())
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = num / den
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return dict(MCC=mcc, F1=f1, prec=prec, rec=rec)


def module_completeness(y_01, M_mod, M_sizes):
    """Per-genome per-module completeness, (B, n_mod) in [0, 1].

    y_01: (B, N), either 0/1 truth or a [0, 1] probability.
    """
    raw = y_01 @ M_mod
    comp = raw / np.maximum(M_sizes, 1)
    return comp


# ----  Phase 1: full sweep

def sweep_one_condition(model_name, model, val_t, val_01, fn, fp, M_mod,
                        M_sizes, dev, batch_size,
                        cat_info, kegg_info, category_of_cog, lf):
    """Run one (model, fn, fp) condition; rank 0 returns metrics rows."""
    noisy = apply_noise(val_t, fn, fp)

    if model_name == 'noisy':
        # Null model: the noisy input itself, read as a probability.
        if is_main():
            pred = ((noisy.float() + 1.0) / 2.0).numpy()
        else:
            return None
    else:
        pred_t = predict_distributed(model, noisy, M_mod, M_sizes, dev, batch_size)
        if not is_main():
            return None
        pred = pred_t.numpy()

    y_true = val_01.numpy()
    y_pred_bin = (pred > 0.5)

    TP, FP, TN, FN = confusion_counts_np(y_pred_bin, y_true)
    cog_mcc, cog_f1, cog_prec, cog_rec = metrics_from_counts(TP, FP, TN, FN)

    glob = global_metrics_from_counts(TP, FP, TN, FN)

    cat_rows = []
    M_cat, cat_names, cat_descs = cat_info
    for ci, cn in enumerate(cat_names):
        mask = M_cat[:, ci] > 0
        n = int(mask.sum())
        if n == 0: continue
        cat_rows.append(dict(
            model=model_name, fn=fn, fp=fp, category=cn,
            category_desc=cat_descs[ci],
            MCC=float(cog_mcc[mask].mean()),
            F1=float(cog_f1[mask].mean()),
            prec=float(cog_prec[mask].mean()),
            rec=float(cog_rec[mask].mean()),
            n_cogs=n))

    # KEGG tier only; the category tier is aggregated above.
    mod_rows = []
    M_kegg, kegg_names = kegg_info
    for mi, mn in enumerate(kegg_names):
        mask = M_kegg[:, mi] > 0
        n = int(mask.sum())
        if n == 0: continue
        mod_rows.append(dict(
            model=model_name, fn=fn, fp=fp, module=mn,
            MCC=float(cog_mcc[mask].mean()),
            F1=float(cog_f1[mask].mean()),
            prec=float(cog_prec[mask].mean()),
            rec=float(cog_rec[mask].mean()),
            n_cogs=n))

    # One row per COG, so materialised only at the focal (FN, FP) cells to
    # bound the parquet size.
    cog_rows = []
    is_focal_fp = abs(fp - FOCAL_FP) < 1e-6
    is_focal_fn = any(abs(fn - f) < 1e-4 for f in FOCAL_FNS)
    if is_focal_fp and is_focal_fn:
        for ci in range(len(cog_mcc)):
            cog_rows.append(dict(
                model=model_name, fn=fn, fp=fp, cog_idx=ci,
                category=category_of_cog[ci],
                MCC=float(cog_mcc[ci]), F1=float(cog_f1[ci]),
                prec=float(cog_prec[ci]), rec=float(cog_rec[ci])))

    # Predicted vs true module completeness, KEGG tier only.
    kegg_sizes = M_kegg.sum(axis=0).astype(np.float32)
    true_comp = module_completeness(
        y_true.astype(np.float32), M_kegg.astype(np.float32), kegg_sizes)
    pred_comp = module_completeness(
        pred.astype(np.float32), M_kegg.astype(np.float32), kegg_sizes)
    comp_rows = []
    for mi, mn in enumerate(kegg_names):
        if M_kegg[:, mi].sum() == 0: continue
        t = true_comp[:, mi]; p = pred_comp[:, mi]
        mae = float(np.abs(p - t).mean())
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        if t.std() > 1e-8 and p.std() > 1e-8:
            corr = float(np.corrcoef(t, p)[0, 1])
        else:
            corr = float('nan')
        # Module called present at >= 0.5 completeness.
        tb = (t >= 0.5).astype(np.int32)
        pb = (p >= 0.5).astype(np.int32)
        tp = int(((tb == 1) & (pb == 1)).sum())
        fpp = int(((tb == 0) & (pb == 1)).sum())
        tn = int(((tb == 0) & (pb == 0)).sum())
        fn_ = int(((tb == 1) & (pb == 0)).sum())
        gm = global_metrics_from_counts(
            np.array([tp]), np.array([fpp]), np.array([tn]), np.array([fn_]))
        comp_rows.append(dict(
            model=model_name, fn=fn, fp=fp, module=mn,
            mae=mae, rmse=rmse, corr=corr,
            presence_MCC=gm['MCC'], presence_F1=gm['F1']))

    return dict(
        glob=[dict(model=model_name, fn=fn, fp=fp, **glob)],
        cat=cat_rows,
        mod=mod_rows,
        cog=cog_rows,
        comp=comp_rows,
    )


def run_sweep(models, val_t, val_01, fn_grid, fp_grid, M_mod, M_sizes,
              cat_info, kegg_info, category_of_cog, dev, batch_size, lf):
    """Iterate all (model, fn, fp) and collect metrics on rank 0."""
    all_glob, all_cat, all_mod, all_cog, all_comp = [], [], [], [], []

    total = (len(models) + 1) * len(fn_grid) * len(fp_grid)  # +1 for noisy null
    done = 0
    t_start = time.time()

    all_models = [('noisy', None)] + list(models.items())

    for mn, model in all_models:
        for fp in fp_grid:
            for fn in fn_grid:
                t0 = time.time()
                res = sweep_one_condition(
                    mn, model, val_t, val_01, fn, fp, M_mod, M_sizes,
                    dev, batch_size, cat_info, kegg_info, category_of_cog, lf)
                done += 1
                if is_main() and res is not None:
                    all_glob += res['glob']
                    all_cat += res['cat']
                    all_mod += res['mod']
                    all_cog += res['cog']
                    all_comp += res['comp']
                    dt = time.time() - t0
                    eta = (time.time() - t_start) / done * (total - done)
                    log(f"  [{done:>3}/{total}] {mn:<8} fn={fn:.3f} "
                        f"fp={fp:.3f}  {dt:.1f}s  (eta {eta/60:.1f}m)", lf)
                barrier()

    if is_main():
        return (pd.DataFrame(all_glob), pd.DataFrame(all_cat),
                pd.DataFrame(all_mod), pd.DataFrame(all_cog),
                pd.DataFrame(all_comp))
    return None


# ----  Phase 2: figures

def _apply_pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'lines.linewidth': 1.6,
        'figure.dpi': 120,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,  # editable text in PDF
    })


def _fp_cmap(fp_grid):
    """Viridis colours for the FP levels; darker = lower FP."""
    import matplotlib.cm as cm
    n = len(fp_grid)
    return {fp: cm.viridis(0.15 + 0.7 * i / max(n - 1, 1))
            for i, fp in enumerate(sorted(fp_grid))}


def plot_metric_vs_fn(df, metric, ax, fp_grid, fp_colours,
                      models_to_show=('hidden', 'nohidden', 'noisy'),
                      ylabel=None, title=None):
    """Plot metric vs FN, one line per (model, fp): model = line style,
    FP level = colour."""
    style_by_model = {'hidden': '-', 'nohidden': '--', 'noisy': ':'}
    for mn in models_to_show:
        for fp in sorted(fp_grid):
            sub = df[(df['model'] == mn) & (df['fp'] == fp)].sort_values('fn')
            if sub.empty: continue
            ax.plot(sub['fn'], sub[metric],
                    color=fp_colours[fp], linestyle=style_by_model[mn],
                    label=f"{MODEL_LABELS[mn]}, FP={fp}")
    ax.set_xlabel('False-negative rate (FN)')
    ax.set_ylabel(ylabel or metric)
    ax.set_xlim(0, 0.9)
    if title:
        ax.set_title(title)


def plot_mcc(df, fp_grid, outpath, title_suffix=''):
    _apply_pub_style()
    fp_colours = _fp_cmap(fp_grid)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_metric_vs_fn(df, 'MCC', ax, fp_grid, fp_colours,
                      ylabel="Matthews correlation coefficient",
                      title=f"MCC vs FN rate{title_suffix}")
    ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    fp_handles = [Line2D([0], [0], color=fp_colours[fp], lw=2,
                         label=f"FP={fp}") for fp in sorted(fp_grid)]
    style_handles = [Line2D([0], [0], color='black',
                            linestyle={'hidden': '-', 'nohidden': '--',
                                       'noisy': ':'}[mn],
                            lw=2, label=MODEL_LABELS[mn])
                     for mn in ['hidden', 'nohidden', 'noisy']]
    leg1 = ax.legend(handles=fp_handles, loc='upper right',
                     title='False positive', frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='center right', frameon=False)
    fig.savefig(outpath)
    plt.close(fig)


def plot_prec_rec(df, fp_grid, outpath, title_suffix=''):
    _apply_pub_style()
    fp_colours = _fp_cmap(fp_grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    plot_metric_vs_fn(df, 'prec', axes[0], fp_grid, fp_colours,
                      ylabel='Precision',
                      title=f"Precision vs FN{title_suffix}")
    plot_metric_vs_fn(df, 'rec', axes[1], fp_grid, fp_colours,
                      ylabel='Recall',
                      title=f"Recall vs FN{title_suffix}")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    from matplotlib.lines import Line2D
    fp_handles = [Line2D([0], [0], color=fp_colours[fp], lw=2,
                         label=f"FP={fp}") for fp in sorted(fp_grid)]
    style_handles = [Line2D([0], [0], color='black',
                            linestyle={'hidden': '-', 'nohidden': '--',
                                       'noisy': ':'}[mn],
                            lw=2, label=MODEL_LABELS[mn])
                     for mn in ['hidden', 'nohidden', 'noisy']]
    axes[1].legend(handles=fp_handles + style_handles,
                   loc='center left', bbox_to_anchor=(1.01, 0.5),
                   frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_module_mae(comp_df, fp_grid, outpath):
    """Module-completeness MAE vs FN, hidden vs no-hidden."""
    _apply_pub_style()
    fp_colours = _fp_cmap(fp_grid)
    fig, ax = plt.subplots(figsize=(7, 5))
    agg = comp_df.groupby(['model', 'fn', 'fp'])['mae'].mean().reset_index()
    style_by_model = {'hidden': '-', 'nohidden': '--'}
    for mn in ['hidden', 'nohidden']:
        for fp in sorted(fp_grid):
            sub = agg[(agg['model'] == mn) & (agg['fp'] == fp)].sort_values('fn')
            if sub.empty: continue
            ax.plot(sub['fn'], sub['mae'],
                    color=fp_colours[fp], linestyle=style_by_model[mn],
                    label=f"{MODEL_LABELS[mn]}, FP={fp}")
    ax.set_xlabel('False-negative rate (FN)')
    ax.set_ylabel('Module completeness MAE')
    ax.set_title('KEGG module-completeness inference error')
    ax.set_xlim(0, 0.9)
    ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    fp_handles = [Line2D([0], [0], color=fp_colours[fp], lw=2,
                         label=f"FP={fp}") for fp in sorted(fp_grid)]
    style_handles = [Line2D([0], [0], color='black',
                            linestyle=style_by_model[mn], lw=2,
                            label=MODEL_LABELS[mn])
                     for mn in ['hidden', 'nohidden']]
    leg1 = ax.legend(handles=fp_handles, loc='upper left',
                     title='False positive', frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='upper right', frameon=False)
    fig.savefig(outpath)
    plt.close(fig)


def plot_per_category(cat_df, fp_grid, outdir):
    """One multi-page PDF, MCC/precision/recall subplots per COG category."""
    _apply_pub_style()
    fp_colours = _fp_cmap(fp_grid)
    outdir.mkdir(parents=True, exist_ok=True)
    cats = sorted(cat_df['category'].unique())
    pdf_path = outdir / 'per_category_all.pdf'
    with PdfPages(pdf_path) as pdf:
        for cat in cats:
            sub = cat_df[cat_df['category'] == cat]
            desc = sub['category_desc'].iloc[0] if 'category_desc' in sub else cat
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
            for ax, metric, ylbl in zip(
                axes, ['MCC', 'prec', 'rec'],
                ['MCC', 'Precision', 'Recall']):
                plot_metric_vs_fn(sub, metric, ax, fp_grid, fp_colours,
                                  ylabel=ylbl,
                                  title=f"{ylbl}")
                ax.set_ylim(0, 1)
            fig.suptitle(f"Category {cat}: {desc}", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            indiv = outdir / f"cat_{cat}.pdf"
    return pdf_path


# ----  Phase 3: summary tables

def best_worst_summary(cog_df, cat_df, mod_df, comp_df, focal_fns, fp,
                       outdir, top_k=20):
    """Write best/worst CSV rankings at each focal FN, at the given FP."""
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for mn in ['hidden', 'nohidden']:
        for fn in focal_fns:
            if not cog_df.empty:
                c = cog_df[(cog_df.model == mn) & (cog_df.fn == fn) &
                           (cog_df.fp == fp)].copy()
                if not c.empty:
                    c.sort_values('MCC', inplace=True, ascending=False)
                    best = c.head(top_k); worst = c.tail(top_k)
                    best.to_csv(outdir / f"best_cogs_{mn}_fn{fn:.2f}.csv",
                                index=False)
                    worst.to_csv(outdir / f"worst_cogs_{mn}_fn{fn:.2f}.csv",
                                 index=False)
            c = cat_df[(cat_df.model == mn) & (cat_df.fn == fn) &
                       (cat_df.fp == fp)].copy()
            if not c.empty:
                c.sort_values('MCC', inplace=True, ascending=False)
                c.to_csv(outdir / f"categories_{mn}_fn{fn:.2f}.csv",
                         index=False)
            c = mod_df[(mod_df.model == mn) & (mod_df.fn == fn) &
                       (mod_df.fp == fp)].copy()
            if not c.empty:
                c.sort_values('MCC', inplace=True, ascending=False)
                best = c.head(top_k); worst = c.tail(top_k)
                best.to_csv(outdir / f"best_modules_{mn}_fn{fn:.2f}.csv",
                            index=False)
                worst.to_csv(outdir / f"worst_modules_{mn}_fn{fn:.2f}.csv",
                             index=False)
            c = comp_df[(comp_df.model == mn) & (comp_df.fn == fn) &
                        (comp_df.fp == fp)].copy()
            if not c.empty:
                c.sort_values('mae', inplace=True, ascending=True)
                c.head(top_k).to_csv(
                    outdir / f"best_comp_{mn}_fn{fn:.2f}.csv", index=False)
                c.tail(top_k).to_csv(
                    outdir / f"worst_comp_{mn}_fn{fn:.2f}.csv", index=False)


def main():
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--ckpt-hidden', default=None,
                    help='Hidden-state model checkpoint (model_s3e.pth or model_f3.pth). '
                         'Required unless --phase plots.')
    pa.add_argument('--ckpt-nohidden', default=None,
                    help='No-hidden model checkpoint.  Required unless --phase plots.')
    pa.add_argument('--val-feather', default='data/COG_val1_phylum.feather')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    pa.add_argument('--outdir', default='calibration_sweep')
    pa.add_argument('--H', type=int, default=1000)
    pa.add_argument('--T', type=int, default=8,
                    help='T iterations of model (must match checkpoint)')
    pa.add_argument('--skip-rank', type=int, default=32)
    pa.add_argument('--field-rank', type=int, default=16)
    pa.add_argument('--onsager', default='full',
                    choices=['none', 'within', 'full', 'tied'],
                    help='Onsager mode (must match training)')
    pa.add_argument('--fn-min', type=float, default=0.0)
    pa.add_argument('--fn-max', type=float, default=0.9)
    pa.add_argument('--fn-step', type=float, default=0.025)
    pa.add_argument('--fp-grid', type=float, nargs='+',
                    default=FP_DEFAULT_GRID)
    pa.add_argument('--eval-batch', type=int, default=256)
    pa.add_argument('--phase', default='all',
                    choices=['all', 'sweep', 'plots'],
                    help='sweep=phase 1 only; plots=phase 2/3 only (reuses saved parquets); all=both')
    A = pa.parse_args()

    rank, ws, lr_rank, dev = setup_dist()
    outdir = Path(A.outdir)
    if is_main():
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'plots').mkdir(exist_ok=True)
        (outdir / 'plots' / 'per_category').mkdir(exist_ok=True)
        (outdir / 'tables').mkdir(exist_ok=True)
        (outdir / 'data').mkdir(exist_ok=True)
    barrier()
    lf = outdir / 'calibration_sweep.log'

    log(f"\n{'='*72}", lf)
    log(f"  Calibration sweep — {time.strftime('%Y-%m-%d %H:%M')}", lf)
    log(f"  {ws} GPU × {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", lf)
    log(f"  Outdir: {outdir.resolve()}", lf)
    log(f"{'='*72}", lf)

    eps = 1e-6
    fn_grid = []
    x = A.fn_min
    while x <= A.fn_max + eps:
        fn_grid.append(round(x, 4))
        x += A.fn_step
    fp_grid = sorted(set(A.fp_grid))
    log(f"  FN grid ({len(fn_grid)} values): {fn_grid[0]:.3f} … {fn_grid[-1]:.3f} step {A.fn_step}", lf)
    log(f"  FP grid ({len(fp_grid)} values): {fp_grid}", lf)

    if A.phase in ('all', 'sweep'):
        if not A.ckpt_hidden or not A.ckpt_nohidden:
            raise SystemExit(
                "--ckpt-hidden and --ckpt-nohidden required unless --phase plots")
        # load_feathers returns (train, val, vocab); only val is needed here.
        _, val_t, _ = load_feathers(A.val_feather, A.val_feather)
        # val_t uses the {-1, +1} spin convention; the metrics take {0, 1}.
        val_01 = ((val_t + 1) / 2).to(torch.int8)
        N = val_t.size(1)
        log(f"  Val: {val_t.shape} — N={N}", lf)

        mod_data = torch.load(A.module_matrix, map_location='cpu',
                              weights_only=False)
        # Files without a combined 'M' store the M_cat / M_path / M_kegg
        # tiers separately; concatenate them in that order.
        M_mod = mod_data.get('M')
        M_sizes = mod_data.get('sizes')
        if M_mod is None:
            M_mod = torch.cat([mod_data['M_cat'], mod_data['M_path'],
                               mod_data['M_kegg']], dim=1).float()
            M_sizes = M_mod.sum(dim=0)
        M_mod_dev = M_mod.to(dev)
        M_sizes_dev = M_sizes.to(dev).clamp(min=1)
        n_mod_full = M_mod.shape[1]
        log(f"  Modules: {n_mod_full} ({M_mod.shape})", lf)

        # Tier splits for the per-category and per-KEGG-module aggregations.
        M_cat = mod_data['M_cat'].numpy().astype(np.int8)
        M_kegg = mod_data['M_kegg'].numpy().astype(np.int8)
        cat_names = mod_data['cat_names']
        cat_descs = mod_data.get('cat_descs', cat_names)
        kegg_names = mod_data['kegg_names']

        # Primary category of a COG = the first category it belongs to.
        category_of_cog = []
        M_cat_np = M_cat
        for ci in range(M_cat_np.shape[0]):
            hits = np.where(M_cat_np[ci] > 0)[0]
            category_of_cog.append(cat_names[hits[0]] if len(hits) else 'None')

        cat_info = (M_cat, cat_names, cat_descs)
        kegg_info = (M_kegg, kegg_names)

        log(f"  Loading models …", lf)
        n_mod = M_mod.shape[1]
        m_hidden = ModuleConditionedDenoiser(
            N=N, H=A.H, T=A.T, skip_rank=A.skip_rank,
            field_rank=A.field_rank, n_modules=n_mod, adaptive_temp=True,
            onsager=A.onsager if A.onsager != 'none' else False).to(dev)
        sd = torch.load(A.ckpt_hidden, map_location=dev, weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        m_hidden.load_state_dict(sd, strict=False)
        m_hidden.eval()

        m_nohid = NoHiddenDenoiser(
            N=N, T=A.T, skip_rank=A.skip_rank, field_rank=A.field_rank,
            n_modules=n_mod, adaptive_temp=True,
            onsager=bool(A.onsager != 'none')).to(dev)
        sd = torch.load(A.ckpt_nohidden, map_location=dev, weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        m_nohid.load_state_dict(sd, strict=False)
        m_nohid.eval()

        models = OrderedDict(hidden=m_hidden, nohidden=m_nohid)
        log(f"  Models ready.", lf)

        log(f"\n── Phase 1: FN/FP sweep ──", lf)
        t1 = time.time()
        dfs = run_sweep(
            models, val_t, val_01, fn_grid, fp_grid,
            M_mod_dev, M_sizes_dev, cat_info, kegg_info,
            category_of_cog, dev, A.eval_batch, lf)
        log(f"  Phase 1 done: {(time.time() - t1)/60:.1f} min", lf)

        if is_main():
            glob_df, cat_df, mod_df, cog_df, comp_df = dfs
            glob_df.to_parquet(outdir / 'data' / 'global.parquet')
            cat_df.to_parquet(outdir / 'data' / 'category.parquet')
            mod_df.to_parquet(outdir / 'data' / 'kegg_module.parquet')
            cog_df.to_parquet(outdir / 'data' / 'per_cog_focal.parquet')
            comp_df.to_parquet(outdir / 'data' / 'module_completeness.parquet')
            log(f"  Saved parquets to {outdir/'data'}", lf)

        # Tear down DDP before the rank-0-only CPU work below.
        if dist.is_initialized():
            cleanup()
        if not is_main():
            return

    # Phases 2 and 3, rank 0 only.
    if A.phase in ('all', 'plots'):
        data_dir = outdir / 'data'
        glob_df = pd.read_parquet(data_dir / 'global.parquet')
        cat_df = pd.read_parquet(data_dir / 'category.parquet')
        mod_df = pd.read_parquet(data_dir / 'kegg_module.parquet')
        cog_df = pd.read_parquet(data_dir / 'per_cog_focal.parquet')
        comp_df = pd.read_parquet(data_dir / 'module_completeness.parquet')

        log(f"\n── Phase 2: plots ──", lf)
        plot_mcc(glob_df, fp_grid, outdir / 'plots' / 'mcc_vs_fn.pdf')
        plot_prec_rec(glob_df, fp_grid, outdir / 'plots' / 'prec_rec_vs_fn.pdf')
        plot_module_mae(comp_df, fp_grid,
                        outdir / 'plots' / 'module_mae_vs_fn.pdf')
        plot_per_category(cat_df, fp_grid,
                          outdir / 'plots' / 'per_category')
        log(f"  Plots saved.", lf)

        log(f"\n── Phase 3: summary tables ──", lf)
        best_worst_summary(cog_df, cat_df, mod_df, comp_df,
                           FOCAL_FNS, FOCAL_FP,
                           outdir / 'tables')
        log(f"  Tables saved.", lf)

    log(f"\nDone. Output: {outdir.resolve()}", lf)


if __name__ == '__main__':
    main()
