#!/usr/bin/env python3
"""calibration_analysis.py -- Multi-GPU posterior calibration analysis.

Measures ECE/MCE/Brier/NLL of Ising denoisers (Hidden+ELBO vs NoHidden+Onsager)
across biological aggregation levels (per-site, per-module, module-completeness).

Three phases:
  1. DDP prediction generation -- shard val set, batched inference, all_gather.
  2. Metric computation (rank 0) -- per-column ECE, percentile-bootstrap CIs.
  3. Figures + LaTeX tables (rank 0) -- 7 PDFs (Cal1--Cal7) + 3 tables.
"""

import argparse, json, time, os, warnings, math
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from ising_denoiser.data import load_feathers
from ising_denoiser.models import ModuleConditionedDenoiser, NoHiddenDenoiser
from ising_denoiser.training import (setup_dist, cleanup, is_main, barrier, log,
                                     unwrap, strip_compile_prefix, load_ckpt_cfg)

# Lazy, optional scipy import; CI code below uses the percentile bootstrap.
_scipy_norm = None
def _get_scipy_norm():
    global _scipy_norm
    if _scipy_norm is None:
        try:
            from scipy.stats import norm
            _scipy_norm = norm
        except ImportError:
            _scipy_norm = False
            warnings.warn("scipy not available; falling back to percentile bootstrap CIs")
    return _scipy_norm if _scipy_norm is not False else None


# ---- Lazy matplotlib
_plt = None
_mpl = None
def _import_mpl():
    global _plt, _mpl
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _mpl = matplotlib
        _plt = plt
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 9,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        })
    return _plt


# ---- Colourblind-safe Wong palette
C_HIDDEN   = '#CC79A7'   # reddish-purple
C_NOHIDDEN = '#0072B2'   # blue
C_NOISY    = '#777777'   # grey

MODEL_LABELS = OrderedDict([
    ('hidden', 'Hidden+ELBO'),
    ('nohidden', 'NoHidden+Onsager'),
])
MODEL_COLOURS = {'hidden': C_HIDDEN, 'nohidden': C_NOHIDDEN}


# ----  Checkpoint loading, with model class auto-detected
#
#  Fixing the class per CLI slot mis-loads an attention-bearing checkpoint:
#  the attn.* weights land in 'unexpected' under strict=False and are dropped,
#  leaving a randomly initialised attention block.  Missing/unexpected counts
#  are logged below so a dirty load is not silent.

def detect_class(sd):
    """Auto-detect model class from state-dict keys.

    Returns one of 'higher_order', 'nohidden_higher_order', 'hidden',
    'nohidden'.  Attention sits on either a hidden-state base
    (HigherOrderDenoiser, which also carries the A/W hidden matrices) or a
    no-hidden base (NoHiddenHigherOrderDenoiser), so attn.* alone does not
    identify the class; has_hidden splits the two.
    """
    keys = set(sd.keys())
    has_attn = any(k.startswith('attn.') for k in keys)
    has_hidden = 'A' in keys and 'W' in keys
    if has_attn and has_hidden:
        return 'higher_order'
    if has_attn:
        return 'nohidden_higher_order'
    if has_hidden:
        return 'hidden'
    return 'nohidden'


def build_model_from_ckpt(ckpt_path, dev, N, n_modules, A, lf):
    """Load a checkpoint into the detected model class.

    T and onsager come from the model_<name>.cfg.json sidecar when present,
    otherwise from CLI --T and the per-class onsager default.
    Returns (model, cls_kind, T).
    """
    sd = torch.load(ckpt_path, map_location=dev, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = strip_compile_prefix(sd)

    cfg = load_ckpt_cfg(ckpt_path)
    onsager = cfg.get('onsager')
    T = int(cfg.get('T', A.T))
    cls_kind = detect_class(sd)

    if cls_kind == 'higher_order':
        from ising_denoiser.models import HigherOrderDenoiser
        onsager_val = onsager if onsager and onsager != 'none' else 'tied'
        model = HigherOrderDenoiser(
            N=N, H=A.H, T=T, n_modules=n_modules,
            adaptive_temp=True, onsager=onsager_val,
            attn_d_model=128, attn_nhead=4, attn_n_layers=1,
            attn_dim_feedforward=512, attn_use_checkpoint=False)
    elif cls_kind == 'nohidden_higher_order':
        from ising_denoiser.models import NoHiddenHigherOrderDenoiser
        onsager_bool = (onsager is None) or (onsager and onsager != 'none')
        model = NoHiddenHigherOrderDenoiser(
            N=N, T=T, n_modules=n_modules,
            adaptive_temp=True, onsager=onsager_bool,
            attn_d_model=128, attn_nhead=4, attn_n_layers=1,
            attn_dim_feedforward=512, attn_use_checkpoint=False)
    elif cls_kind == 'hidden':
        onsager_val = onsager if onsager and onsager != 'none' else 'full'
        model = ModuleConditionedDenoiser(
            N, H=A.H, T=T, skip_rank=A.skip_rank, field_rank=A.field_rank,
            n_modules=n_modules, onsager=onsager_val)
    else:  # nohidden
        onsager_bool = (onsager is None) or (onsager and onsager != 'none')
        model = NoHiddenDenoiser(
            N, T=T, skip_rank=A.skip_rank, field_rank=A.field_rank,
            n_modules=n_modules, onsager=onsager_bool)

    model = model.to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tag = Path(ckpt_path).parent.name or Path(ckpt_path).name
    log(f"  [{tag}] class={cls_kind} T={T} onsager={onsager or 'auto'}"
        f"  | load: {len(missing)} missing, {len(unexpected)} unexpected", lf)
    if missing:
        log(f"    missing[:6]: {list(missing)[:6]}", lf)
    if unexpected:
        log(f"    unexpected[:6]: {list(unexpected)[:6]}", lf)
    model.eval()
    return model, cls_kind, T


# ----  Phase 1: prediction generation

def apply_noise(val_t, fn, fp):
    """Deterministic noise matching eval_spectra: seed = 42 + int(fn*1000).

    fp is absent from the seed because this script holds it constant for a
    whole run; calibration_sweep.py sweeps fp and seeds on it too.
    """
    torch.manual_seed(42 + int(fn * 1000))
    ny = val_t.clone()
    r = torch.rand_like(ny)
    ny[(val_t == 1) & (r < fn)] = -1
    r2 = torch.rand_like(ny)
    ny[(val_t == -1) & (r2 < fp)] = 1
    return ny


def sdpa_ctx():
    """Restrict SDPA to the flash and memory-efficient kernels.

    The math fallback materialises the full (B, nhead, N, N) score tensor; at
    N ~ 4789 that is ~175 GiB and the forward segfaults.  Degrades to the
    legacy toggle on older torch, and to a no-op on CPU.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION])
    except Exception:  # noqa: BLE001 -- older torch: try the legacy toggle
        try:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=True, enable_mem_efficient=True,
                              enable_math=False)
        except Exception:  # noqa: BLE001 -- no SDPA controls (e.g. CPU)
            from contextlib import nullcontext
            return nullcontext()


@torch.no_grad()
def predict_distributed(model, noisy_t, M_mod, M_sizes, dev, amp, batch_size):
    """Shard noisy_t across ranks, batched inference, all_gather to rank 0.

    Returns the (B, N) prediction tensor on rank 0, None elsewhere; works
    unchanged on a single GPU.
    """
    model.eval()
    use_ddp = dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world = dist.get_world_size() if use_ddp else 1

    B_full = noisy_t.size(0)
    local_idx = list(range(rank, B_full, world))
    local_noisy = noisy_t[local_idx].to(dev)

    # no_grad is load-bearing: retaining the graph over T mean-field
    # iterations at this N makes inference intractable.
    parts = []
    with torch.no_grad():
        for i in range(0, local_noisy.size(0), batch_size):
            batch = local_noisy[i:i + batch_size]
            with sdpa_ctx(), autocast('cuda', enabled=amp):
                if M_mod is not None:
                    out, _ = model(batch, M_mod, M_sizes)
                else:
                    out, _ = model(batch)
            prob = (out + 1.0) / 2.0   # spin mean -> P(gene present)
            parts.append(prob.float().cpu())
    local_pred = torch.cat(parts, dim=0)

    if not use_ddp:
        return local_pred

    # all_gather needs equal-length buffers, so the tail rank is padded.
    max_local = (B_full + world - 1) // world
    padded = torch.zeros(max_local, local_pred.size(1))
    padded[:local_pred.size(0)] = local_pred

    gathered = [torch.zeros_like(padded) for _ in range(world)]
    padded_gpu = padded.to(dev)
    gathered_gpu = [torch.zeros_like(padded_gpu) for _ in range(world)]
    dist.all_gather(gathered_gpu, padded_gpu)
    gathered = [g.cpu() for g in gathered_gpu]

    if rank == 0:
        # Undo the shard stride, restoring the original row order.
        full = torch.zeros(B_full, local_pred.size(1))
        for r in range(world):
            idx = list(range(r, B_full, world))
            actual = len(idx)
            full[idx] = gathered[r][:actual]
        return full
    return None


def generate_predictions(models, val_t, M_mod, M_sizes, fn_grid, fp,
                         dev, amp, batch_size, pred_dir, force, lf):
    """Generate and cache predictions for all (model, fn) pairs."""
    for model_name, model in models.items():
        mdir = pred_dir / model_name
        if is_main():
            mdir.mkdir(parents=True, exist_ok=True)
        barrier()

        for fn in fn_grid:
            parquet_path = mdir / f'fn_{fn:.2f}.parquet'
            noisy_path = mdir / f'fn_{fn:.2f}_noisy.parquet'

            if parquet_path.exists() and not force:
                log(f"  [{model_name}] fn={fn:.2f} cached, skipping", lf)
                continue

            t0 = time.time()
            noisy_t = apply_noise(val_t, fn, fp)
            pred = predict_distributed(model, noisy_t, M_mod, M_sizes,
                                       dev, amp, batch_size)

            if is_main() and pred is not None:
                df = pd.DataFrame(pred.numpy())
                df.to_parquet(parquet_path, engine='pyarrow')
                ny_01 = ((noisy_t + 1.0) / 2.0).numpy()
                pd.DataFrame(ny_01).to_parquet(noisy_path, engine='pyarrow')
                elapsed = time.time() - t0
                log(f"  [{model_name}] fn={fn:.2f}  {pred.shape}  {elapsed:.1f}s", lf)

            barrier()


# ----  Phase 2: metrics (vectorised, rank 0 only)

def reliability_bins(y_true, p_pred, n_bins=20):
    """Equal-frequency reliability diagram: returns (mean_pred, frac_pos, counts)."""
    order = np.argsort(p_pred)
    sorted_p = p_pred[order]
    sorted_y = y_true[order]
    B = len(y_true)
    bin_size = B // n_bins
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        s = b * bin_size
        e = s + bin_size if b < n_bins - 1 else B
        mean_pred[b] = sorted_p[s:e].mean()
        frac_pos[b] = sorted_y[s:e].mean()
        counts[b] = e - s
    return mean_pred, frac_pos, counts


def ece(y_true, p_pred, n_bins=20):
    """Expected Calibration Error = sum_b (n_b/N)|frac_b - pred_b|."""
    mp, fp, c = reliability_bins(y_true, p_pred, n_bins)
    return np.sum(c / c.sum() * np.abs(fp - mp))


def mce(y_true, p_pred, n_bins=20):
    """Maximum Calibration Error = max_b |frac_b - pred_b|."""
    mp, fp, _ = reliability_bins(y_true, p_pred, n_bins)
    return np.max(np.abs(fp - mp))


def brier(y_true, p_pred):
    """Brier score = mean((y - p)^2)."""
    return np.mean((y_true - p_pred) ** 2)


def nll(y_true, p_pred, eps=1e-7):
    """Binary cross-entropy = -mean(y log p + (1-y) log(1-p))."""
    p = np.clip(p_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def ece_per_column(y_2d, p_2d, n_bins=20):
    """Equal-frequency ECE for every column of a (B, N) pair.  Returns (N,)."""
    B, N = y_2d.shape
    order = np.argsort(p_2d, axis=0)
    sorted_p = np.take_along_axis(p_2d, order, axis=0)
    sorted_y = np.take_along_axis(y_2d, order, axis=0)
    bin_size = B // n_bins
    eces = np.zeros(N)
    for b in range(n_bins):
        s = b * bin_size
        e = s + bin_size if b < n_bins - 1 else B
        mp = sorted_p[s:e].mean(axis=0)
        my = sorted_y[s:e].mean(axis=0)
        w = (e - s) / B
        eces += w * np.abs(my - mp)
    return eces


def brier_per_column(y_2d, p_2d):
    """Per-column Brier score. Returns (N,)."""
    return np.mean((y_2d - p_2d) ** 2, axis=0)


def _per_genome_ece(y_2d, p_2d, n_bins=20):
    """Per-genome ECE, (B,): equal-frequency ECE of each genome's own N
    predictions.  Reduces the bootstrap below to resampling B scalars.
    """
    B, N = y_2d.shape
    order = np.argsort(p_2d, axis=1)
    sp = np.take_along_axis(p_2d, order, axis=1)
    sy = np.take_along_axis(y_2d, order, axis=1)
    bin_size = N // n_bins
    ece_arr = np.zeros(B)
    for b in range(n_bins):
        s = b * bin_size
        e = s + bin_size if b < n_bins - 1 else N
        mp = sp[:, s:e].mean(axis=1)
        my = sy[:, s:e].mean(axis=1)
        w = (e - s) / N
        ece_arr += w * np.abs(my - mp)
    return ece_arr


def _per_genome_brier(y_2d, p_2d):
    """Brier score per genome: (B,) array."""
    return np.mean((y_2d - p_2d) ** 2, axis=1)


def bootstrap_percentile_ci(per_genome_vals, n_boot=200, alpha=0.05, seed=42):
    """Percentile bootstrap on a (B,) array of per-genome scalars.
    Returns (point, lo, hi)."""
    rng = np.random.RandomState(seed)
    B = len(per_genome_vals)
    point = per_genome_vals.mean()
    boot_idx = rng.randint(0, B, (n_boot, B))
    boot_means = per_genome_vals[boot_idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return point, lo, hi


def module_completeness_predictions(y_2d, p_2d, M_tier, tier_sizes):
    """C_true = y @ M / sizes, C_pred = p @ M / sizes. Returns (B, M_tier)."""
    sz = np.maximum(tier_sizes, 1)
    c_true = (y_2d @ M_tier) / sz[np.newaxis, :]
    c_pred = (p_2d @ M_tier) / sz[np.newaxis, :]
    return c_true, c_pred


def _load_pred(pred_dir, model_name, fn):
    """Load cached (B, N) prediction array."""
    path = pred_dir / model_name / f'fn_{fn:.2f}.parquet'
    return pd.read_parquet(path).values.astype(np.float32)


def _load_noisy(pred_dir, model_name, fn):
    path = pred_dir / model_name / f'fn_{fn:.2f}_noisy.parquet'
    return pd.read_parquet(path).values.astype(np.float32)


def compute_all_metrics(pred_dir, val_01, module_data, fn_grid, n_bins,
                        n_bootstrap, ci_alpha, boot_seed, lf):
    """Compute and save all metric DataFrames (rank 0 only).

    Each (model, fn) prediction is loaded once and its per-column and
    per-genome ECE/Brier arrays cached; the COG, category and module sections
    all read the same arrays.
    """
    cog_names = module_data['cog_names']
    N = len(cog_names)
    model_names = list(MODEL_LABELS.keys())
    base_rates = val_01.mean(axis=0)

    t_cache = time.time()
    cache = {}
    for mn in model_names:
        for fn in fn_grid:
            t0 = time.time()
            log(f"  Caching {mn} fn={fn:.2f}...", lf)
            p = _load_pred(pred_dir, mn, fn)
            log(f"    loaded parquet ({p.shape}) {time.time()-t0:.1f}s", lf)
            ec = ece_per_column(val_01, p, n_bins)
            log(f"    ece_per_column {time.time()-t0:.1f}s", lf)
            bc = brier_per_column(val_01, p)
            pge = _per_genome_ece(val_01, p, n_bins)
            log(f"    per_genome_ece {time.time()-t0:.1f}s", lf)
            pgb = _per_genome_brier(val_01, p)
            cache[(mn, fn)] = {
                'p': p, 'ece_col': ec, 'brier_col': bc,
                'pg_ece': pge, 'pg_brier': pgb,
            }
            log(f"    done {time.time()-t0:.1f}s", lf)
    log(f"  Cached {len(cache)} prediction sets in {time.time()-t_cache:.1f}s", lf)

    # ---- Global metrics with bootstrap CIs
    t_global = time.time()
    global_rows = []
    for mn in model_names:
        for fn in fn_grid:
            c = cache[(mn, fn)]
            yf = val_01.ravel()
            pf = c['p'].ravel()

            # Mixed aggregation: ece and brier are per-genome means (CIs over
            # the per-genome spread); mce and nll are pooled over all entries.
            e_pt = c['pg_ece'].mean()
            m_pt = mce(yf, pf, n_bins)
            b_pt = c['pg_brier'].mean()
            n_pt = nll(yf, pf)

            e_pt2, e_lo, e_hi = bootstrap_percentile_ci(
                c['pg_ece'], n_bootstrap, ci_alpha, boot_seed)
            b_pt2, b_lo, b_hi = bootstrap_percentile_ci(
                c['pg_brier'], n_bootstrap, ci_alpha, boot_seed)

            global_rows.append(dict(
                model=mn, fn=fn,
                ece=e_pt, ece_lo=e_lo, ece_hi=e_hi,
                mce=m_pt,
                brier=b_pt, brier_lo=b_lo, brier_hi=b_hi,
                nll=n_pt))
            log(f"  {mn} fn={fn:.2f}  ECE_pg={e_pt:.4f}[{e_lo:.4f},{e_hi:.4f}]"
                f"  Brier_pg={b_pt:.4f}  NLL={n_pt:.4f}  "
                f"(_pg = per-genome mean)", lf)

    global_df = pd.DataFrame(global_rows)
    global_df.to_parquet(pred_dir / 'global_metrics.parquet')
    log(f"  Global metrics: {time.time()-t_global:.1f}s", lf)

    # ---- Per-COG metrics
    t_cog = time.time()
    cog_rows = []
    for mn in model_names:
        for fn in fn_grid:
            c = cache[(mn, fn)]
            for j in range(N):
                cog_rows.append(dict(
                    model=mn, fn=fn, cog=cog_names[j],
                    ece=c['ece_col'][j], brier=c['brier_col'][j],
                    base_rate=base_rates[j]))
    cog_df = pd.DataFrame(cog_rows)
    cog_df.to_parquet(pred_dir / 'cog_metrics.parquet')
    log(f"  Per-COG metrics: {time.time()-t_cog:.1f}s", lf)

    # ---- Per-category metrics
    t_cat = time.time()
    M_cat = module_data['M_cat']
    cat_names = module_data['cat_names']
    cat_descs = module_data['cat_descs']
    cat_rows = []
    for mn in model_names:
        for fn in fn_grid:
            c = cache[(mn, fn)]
            for ci, cn in enumerate(cat_names):
                mask = M_cat[:, ci] > 0
                if mask.sum() == 0:
                    continue
                cat_rows.append(dict(
                    model=mn, fn=fn, category=cn,
                    category_desc=cat_descs[ci],
                    ece=c['ece_col'][mask].mean(),
                    brier=c['brier_col'][mask].mean(),
                    n_cogs=int(mask.sum())))
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_parquet(pred_dir / 'cat_metrics.parquet')
    log(f"  Per-category metrics: {time.time()-t_cat:.1f}s", lf)

    # ---- Per-module and completeness metrics, over the 3 module tiers
    t_mod = time.time()
    tiers = [
        ('cat', module_data['M_cat'], module_data['cat_sizes'],
         module_data['cat_names']),
        ('path', module_data['M_path'], module_data['path_sizes'],
         module_data['path_names']),
        ('kegg', module_data['M_kegg'], module_data['kegg_sizes'],
         module_data['kegg_names']),
    ]
    mod_rows = []
    comp_rows = []
    for tier_name, M_tier, tier_sizes, tier_names in tiers:
        for mn in model_names:
            for fn in fn_grid:
                c = cache[(mn, fn)]

                # Site-level ECE: per-gene Bernoulli ECE averaged over the
                # module's member COGs.
                for mi, mname in enumerate(tier_names):
                    mask = M_tier[:, mi] > 0
                    if mask.sum() == 0:
                        continue
                    mod_rows.append(dict(
                        tier=tier_name, model=mn, fn=fn,
                        module=mname, ece=c['ece_col'][mask].mean(),
                        n_cogs=int(mask.sum())))

                # Completeness-level ECE: calibration of predicted against
                # true module completeness (fraction of member COGs present)
                # across genomes.  A continuous predictor, not a Bernoulli
                # posterior -- a different quantity from the site-level ECE,
                # sharing the 'ece' column name in comp_metrics.parquet.
                c_true, c_pred = module_completeness_predictions(
                    val_01, c['p'], M_tier, tier_sizes)
                comp_eces = ece_per_column(c_true, c_pred, min(n_bins, c_true.shape[0] // 5))
                for mi, mname in enumerate(tier_names):
                    if tier_sizes[mi] < 1:
                        continue
                    comp_rows.append(dict(
                        tier=tier_name, model=mn, fn=fn,
                        module=mname,
                        ece=comp_eces[mi],
                        n_cogs=int(tier_sizes[mi])))

    module_df = pd.DataFrame(mod_rows)
    comp_df = pd.DataFrame(comp_rows)
    module_df.to_parquet(pred_dir / 'module_metrics.parquet')
    comp_df.to_parquet(pred_dir / 'comp_metrics.parquet')
    log(f"  Module metrics: {time.time()-t_mod:.1f}s", lf)

    del cache
    log("  All metrics saved.", lf)
    return global_df, cat_df, module_df, comp_df, cog_df


# ----  Phase 3: figures

def _fn_color(fn, fn_grid):
    """Map fn to a colour from the coolwarm colormap."""
    plt = _import_mpl()
    import matplotlib.cm as cm
    norm = plt.Normalize(vmin=min(fn_grid), vmax=max(fn_grid))
    return cm.coolwarm(norm(fn))


def plot_cal1_reliability(global_df, pred_dir, val_01, fn_grid, n_bins, fig_dir, cog_names):
    """Cal1: Global reliability diagrams, 2-panel + ECE inset."""
    plt = _import_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    model_names = list(MODEL_LABELS.keys())

    for ax_i, mn in enumerate(model_names):
        ax = axes[ax_i]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=0.8)
        ece_vals = []
        for fn in fn_grid:
            p = _load_pred(pred_dir, mn, fn)
            y = val_01
            mp, fp, _ = reliability_bins(y.ravel(), p.ravel(), n_bins)
            c = _fn_color(fn, fn_grid)
            ax.plot(mp, fp, '-o', color=c, ms=3, lw=1.2,
                    label=f'FN={fn:.2f}')
            ece_vals.append(ece(y.ravel(), p.ravel(), n_bins))

        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.set_title(MODEL_LABELS[mn])
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        inset = ax.inset_axes([0.05, 0.62, 0.35, 0.32])
        colours = [_fn_color(fn, fn_grid) for fn in fn_grid]
        inset.bar(range(len(fn_grid)), ece_vals, color=colours, width=0.7)
        inset.set_xticks(range(len(fn_grid)))
        inset.set_xticklabels([f'{fn:.1f}' for fn in fn_grid], fontsize=5)
        inset.set_ylabel('ECE', fontsize=6)
        inset.tick_params(labelsize=5)
        inset.set_title('ECE by FN', fontsize=6)

    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal1_reliability.pdf')
    plt.close(fig)


def plot_cal2_heatmap(cat_df, fn_grid, fig_dir):
    """Cal2: ECE heatmap by COG category x FN, 2-panel."""
    plt = _import_mpl()
    model_names = list(MODEL_LABELS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax_i, mn in enumerate(model_names):
        ax = axes[ax_i]
        sub = cat_df[cat_df.model == mn]
        pivot = sub.pivot_table(index='category_desc', columns='fn',
                                values='ece', aggfunc='mean')
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r')
        ax.set_xticks(range(len(fn_grid)))
        ax.set_xticklabels([f'{fn:.2f}' for fn in fn_grid], fontsize=7)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([d[:40] for d in pivot.index], fontsize=6)
        ax.set_xlabel('FN rate')
        ax.set_title(MODEL_LABELS[mn])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=5, color='white' if v > pivot.values.mean() else 'black')
        fig.colorbar(im, ax=ax, shrink=0.6, label='ECE')

    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal2_heatmap.pdf')
    plt.close(fig)


def plot_cal3_modules(pred_dir, val_01, fn_grid, module_data, n_bins, fig_dir):
    """Cal3: Module calibration at 3 tiers -- 2x3 grid."""
    plt = _import_mpl()
    model_names = list(MODEL_LABELS.keys())
    tiers = [
        ('cat', module_data['M_cat'], module_data['cat_sizes'], 'COG category'),
        ('path', module_data['M_path'], module_data['path_sizes'], 'COG pathway'),
        ('kegg', module_data['M_kegg'], module_data['kegg_sizes'], 'KEGG module'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    fn_site = 0.5
    fn_comp = 0.75

    for col, (tname, M_tier, tier_sizes, label) in enumerate(tiers):
        # Top row: site-level reliability at fn=0.5
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=0.8)
        for mn in model_names:
            p = _load_pred(pred_dir, mn, fn_site)
            y = val_01
            # Pool every COG belonging to this tier
            tier_mask = M_tier.sum(axis=1) > 0
            yf = y[:, tier_mask].ravel()
            pf = p[:, tier_mask].ravel()
            mp, fp, _ = reliability_bins(yf, pf, n_bins)
            ax.plot(mp, fp, '-o', color=MODEL_COLOURS[mn], ms=3, lw=1.2,
                    label=MODEL_LABELS[mn])
        ax.set_title(f'{label} (site, FN={fn_site})', fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Bottom row: completeness-level at fn=0.75
        ax = axes[1, col]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=0.8)
        for mn in model_names:
            p = _load_pred(pred_dir, mn, fn_comp)
            y = val_01
            c_true, c_pred = module_completeness_predictions(y, p, M_tier, tier_sizes)
            mp, fp, _ = reliability_bins(c_true.ravel(), c_pred.ravel(), n_bins)
            ax.plot(mp, fp, '-o', color=MODEL_COLOURS[mn], ms=3, lw=1.2,
                    label=MODEL_LABELS[mn])
        ax.set_title(f'{label} (completeness, FN={fn_comp})', fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    axes[1, 0].set_xlabel('Predicted')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 2].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Observed')
    axes[1, 0].set_ylabel('Observed')
    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal3_modules.pdf')
    plt.close(fig)


def plot_cal4_top_miscal(cog_df, fn_grid, module_data, fig_dir):
    """Cal4: Top miscalibrated COGs -- scatter + bar chart."""
    plt = _import_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sub = cog_df[(cog_df.model == 'hidden') & (cog_df.fn == 0.5)].copy()

    M_cat = module_data['M_cat']
    cat_names = module_data['cat_names']
    cog_names = module_data['cog_names']
    cog_to_cat = {}
    for j, cn in enumerate(cog_names):
        cats = np.where(M_cat[j] > 0)[0]
        cog_to_cat[cn] = cat_names[cats[0]] if len(cats) > 0 else 'S'

    sub['category'] = sub['cog'].map(cog_to_cat)

    ax = axes[0]
    cats_unique = sorted(sub['category'].unique())
    cmap = plt.colormaps['tab20'].resampled(len(cats_unique))
    for ci, cat in enumerate(cats_unique):
        mask = sub['category'] == cat
        ax.scatter(sub.loc[mask, 'base_rate'],
                   sub.loc[mask, 'ece'],
                   c=[cmap(ci)], s=8, alpha=0.6, label=cat)
    ax.set_xlabel('Base rate')
    ax.set_ylabel('ECE')
    ax.set_title('Per-COG ECE vs base rate (FN=0.5, Hidden)')
    ax.legend(fontsize=5, ncol=3, markerscale=0.8, loc='upper right')

    ax = axes[1]
    sub['brier_excess'] = sub['brier'] - sub['base_rate'] * (1 - sub['base_rate'])
    top20 = sub.nlargest(20, 'brier_excess')
    colours = [cmap(cats_unique.index(c)) if c in cats_unique else '#999'
               for c in top20['category']]
    ax.barh(range(20), top20['brier_excess'].values, color=colours)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20['cog'].values, fontsize=6)
    ax.set_xlabel('Brier excess')
    ax.set_title('Top-20 miscalibrated COGs')
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal4_miscalibrated.pdf')
    plt.close(fig)


def plot_cal5_comparison(global_df, comp_df, pred_dir, val_01, fn_grid, n_bins, fig_dir):
    """Cal5: Hidden vs NoHidden -- 3-panel comparison."""
    plt = _import_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    model_names = list(MODEL_LABELS.keys())

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=0.8)
    for mn in model_names:
        p = _load_pred(pred_dir, mn, 0.5)
        y = val_01
        mp, fp, _ = reliability_bins(y.ravel(), p.ravel(), n_bins)
        ax.plot(mp, fp, '-o', color=MODEL_COLOURS[mn], ms=3, lw=1.5,
                label=MODEL_LABELS[mn])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
    ax.set_title('(a) Reliability at FN=0.5')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')

    ax = axes[1]
    for mn in model_names:
        sub = global_df[global_df.model == mn].sort_values('fn')
        ax.plot(sub.fn, sub.ece, '-o', color=MODEL_COLOURS[mn], ms=4, lw=1.5,
                label=MODEL_LABELS[mn])
        ax.fill_between(sub.fn, sub.ece_lo, sub.ece_hi,
                         color=MODEL_COLOURS[mn], alpha=0.15)
    ax.set_xlabel('FN rate'); ax.set_ylabel('ECE')
    ax.set_title('(b) ECE vs FN')
    ax.legend(fontsize=7)

    ax = axes[2]
    tiers = ['cat', 'path', 'kegg']
    tier_labels = ['Category', 'Pathway', 'KEGG']
    positions = []
    labels = []
    bp_data = []
    colours_bp = []
    pos = 0
    for ti, tier in enumerate(tiers):
        for mn in model_names:
            sub = comp_df[(comp_df.tier == tier) & (comp_df.model == mn) &
                          (comp_df.fn == 0.75)]
            bp_data.append(sub['ece'].values)
            colours_bp.append(MODEL_COLOURS[mn])
            positions.append(pos)
            labels.append(f'{tier_labels[ti]}\n{MODEL_LABELS[mn][:6]}')
            pos += 1
        pos += 0.5

    bps = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True)
    for patch, c in zip(bps['boxes'], colours_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6, rotation=30, ha='right')
    ax.set_ylabel('Completeness ECE')
    ax.set_title('(c) Module completeness (FN=0.75)')

    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal5_comparison.pdf')
    plt.close(fig)


def plot_cal6_kegg_map(cog_df, module_data, fig_dir, outdir):
    """Cal6: KEGG metabolic map painted by per-entry ECE.

    Downloads KGML and base PNG from KEGG REST, parses entry rectangles,
    maps KO->COG, colours by ECE. Caches downloads.
    """
    plt = _import_mpl()
    import urllib.request
    import xml.etree.ElementTree as ET

    cache_dir = outdir / 'kegg_cache'
    cache_dir.mkdir(exist_ok=True)
    pathway_id = 'map01100'  # Global metabolic overview

    kgml_path = cache_dir / f'{pathway_id}.xml'
    png_path = cache_dir / f'{pathway_id}.png'

    def _download(url, dest):
        if dest.exists():
            return True
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Python/research'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(dest, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            warnings.warn(f"KEGG download failed ({url}): {e}")
            return False

    ok1 = _download(f'https://rest.kegg.jp/get/ko{pathway_id[3:]}/kgml', kgml_path)
    ok2 = _download(f'https://rest.kegg.jp/get/ko{pathway_id[3:]}/image', png_path)

    if not ok1 or not ok2:
        warnings.warn("Skipping Cal6 (KEGG download failed)")
        return

    tree = ET.parse(str(kgml_path))
    root = tree.getroot()

    ko_cog_file = Path('ko_cog_cache.tsv')
    ko2cog = {}
    if ko_cog_file.exists():
        with open(ko_cog_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ko2cog[parts[0]] = parts[1].split(',')

    cog_names = module_data['cog_names']
    sub = cog_df[(cog_df.model == 'hidden') & (cog_df.fn == 0.75)]
    cog_ece = dict(zip(sub['cog'], sub['ece']))

    from PIL import Image
    base_img = Image.open(str(png_path)).convert('RGBA')
    img_w, img_h = base_img.size

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ['Hidden+ELBO', 'NoHidden+Onsager']
    model_keys = ['hidden', 'nohidden']

    for ax_i, (mn, title) in enumerate(zip(model_keys, titles)):
        ax = axes[ax_i]
        sub_m = cog_df[(cog_df.model == mn) & (cog_df.fn == 0.75)]
        cog_ece_m = dict(zip(sub_m['cog'], sub_m['ece']))

        ax.imshow(base_img, extent=[0, img_w, img_h, 0])

        cmap = plt.colormaps['RdYlGn_r']
        norm = plt.Normalize(vmin=0, vmax=0.15)
        n_painted = 0
        for entry in root.findall('.//entry'):
            enames = entry.get('name', '').split()
            graphics = entry.find('graphics')
            if graphics is None:
                continue
            if graphics.get('type') != 'rectangle':
                continue
            try:
                x = float(graphics.get('x', 0))
                y_pos = float(graphics.get('y', 0))
                w = float(graphics.get('width', 0))
                h = float(graphics.get('height', 0))
            except (ValueError, TypeError):
                continue

            ece_vals = []
            for name in enames:
                ko = name.replace('ko:', '')
                for cog in ko2cog.get(ko, []):
                    if cog in cog_ece_m:
                        ece_vals.append(cog_ece_m[cog])
            if not ece_vals:
                continue
            mean_ece = np.mean(ece_vals)
            colour = cmap(norm(mean_ece))
            rect = plt.Rectangle((x - w/2, y_pos - h/2), w, h,
                                  facecolor=colour, edgecolor='none', alpha=0.7)
            ax.add_patch(rect)
            n_painted += 1

        ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)
        ax.set_title(f'{title} (FN=0.75, {n_painted} entries)')
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 0.15))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, label='ECE')

    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal6_kegg_map.pdf')
    plt.close(fig)


def plot_cal7_funnel(cog_df, module_df, comp_df, fn_grid, fig_dir):
    """Cal7: Calibration funnel -- violin plots at 3 aggregation levels."""
    plt = _import_mpl()
    model_names = list(MODEL_LABELS.keys())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fn_target = 0.75

    for ax_i, mn in enumerate(model_names):
        ax = axes[ax_i]
        levels = []
        level_names = ['Per-site', 'Per-module-site', 'Module-completeness']

        sub_cog = cog_df[(cog_df.model == mn) & (cog_df.fn == fn_target)]
        levels.append(sub_cog['ece'].values)

        sub_mod = module_df[(module_df.model == mn) & (module_df.fn == fn_target)]
        levels.append(sub_mod['ece'].values)

        sub_comp = comp_df[(comp_df.model == mn) & (comp_df.fn == fn_target)]
        levels.append(sub_comp['ece'].values)

        vp = ax.violinplot(levels, positions=[0, 1, 2], showmedians=True,
                            showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(MODEL_COLOURS[mn])
            body.set_alpha(0.5)
        vp['cmedians'].set_color(MODEL_COLOURS[mn])

        medians = [np.median(l) for l in levels]
        q25 = [np.percentile(l, 25) for l in levels]
        q75 = [np.percentile(l, 75) for l in levels]
        ax.fill_between([0, 1, 2], q25, q75, color=MODEL_COLOURS[mn], alpha=0.1)
        ax.plot([0, 1, 2], medians, '-o', color=MODEL_COLOURS[mn], lw=1.5, ms=5)

        ax.set_ylabel('ECE')
        ax.set_title(f'{MODEL_LABELS[mn]} (FN={fn_target})')

    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['Per-site', 'Per-module-site', 'Module-completeness'])
    fig.tight_layout()
    fig.savefig(fig_dir / 'Cal7_funnel.pdf')
    plt.close(fig)


# ----  Phase 4: tables

def _bold_min(vals, fmt='.4f'):
    """Return list of formatted strings with min bolded (LaTeX)."""
    arr = np.array(vals, dtype=float)
    best = arr.argmin()
    out = []
    for i, v in enumerate(vals):
        s = f'{v:{fmt}}'
        out.append(f'\\textbf{{{s}}}' if i == best else s)
    return out


def write_cal1_table(global_df, fn_grid, tab_dir):
    """Cal1 table: global metrics, bolded better model per FN."""
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Global calibration metrics (bolded = better). ECE and Brier '
        r'are per-genome means (each genome contributes one ECE/Brier computed '
        r'on its $N{=}4789$ predictions); MCE and NLL are pooled across all '
        r'(genome, gene) entries.}',
        r'\label{tab:cal1}',
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'FN & Model & ECE & MCE & Brier & NLL \\',
        r'\midrule',
    ]
    for fn in fn_grid:
        sub = global_df[global_df.fn == fn]
        ece_vals = [sub[sub.model == m]['ece'].values[0]
                    for m in MODEL_LABELS]
        mce_vals = [sub[sub.model == m]['mce'].values[0]
                    for m in MODEL_LABELS]
        brier_vals = [sub[sub.model == m]['brier'].values[0]
                      for m in MODEL_LABELS]
        nll_vals = [sub[sub.model == m]['nll'].values[0]
                    for m in MODEL_LABELS]
        ece_s = _bold_min(ece_vals)
        mce_s = _bold_min(mce_vals)
        brier_s = _bold_min(brier_vals)
        nll_s = _bold_min(nll_vals)
        for mi, mn in enumerate(MODEL_LABELS):
            fn_str = f'{fn:.2f}' if mi == 0 else ''
            lines.append(
                f'  {fn_str} & {MODEL_LABELS[mn]} & '
                f'{ece_s[mi]} & {mce_s[mi]} & {brier_s[mi]} & {nll_s[mi]} \\\\')
        if fn != fn_grid[-1]:
            lines.append(r'  \addlinespace')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    (tab_dir / 'Cal1_global.tex').write_text('\n'.join(lines))


def write_cal2_table(cat_df, fn_grid, tab_dir):
    """Cal2 table: per-category calibration summary."""
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Per-category ECE (Hidden model, selected FN rates).}',
        r'\label{tab:cal2}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
    ]
    fn_sub = [f for f in fn_grid if f in [0.0, 0.25, 0.5, 0.75, 0.9]]
    header = 'Category & ' + ' & '.join([f'FN={fn:.2f}' for fn in fn_sub]) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    sub = cat_df[cat_df.model == 'hidden']
    cats = sub.groupby('category_desc')['ece'].mean().sort_values(ascending=False).index
    for cat in cats:
        vals = []
        for fn in fn_sub:
            row = sub[(sub.category_desc == cat) & (sub.fn == fn)]
            vals.append(f'{row.ece.values[0]:.3f}' if len(row) > 0 else '--')
        cat_short = cat[:35]
        lines.append(f'  {cat_short} & ' + ' & '.join(vals) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    (tab_dir / 'Cal2_category.tex').write_text('\n'.join(lines))


def write_cal3_table(module_df, comp_df, fn_grid, tab_dir):
    """Cal3 table: three-tier module calibration summary."""
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Module calibration summary (median ECE across modules, FN=0.75).}',
        r'\label{tab:cal3}',
        r'\begin{tabular}{llcc}',
        r'\toprule',
        r'Tier & Metric & Hidden & NoHidden \\',
        r'\midrule',
    ]

    fn_t = 0.75
    for tier, tier_label in [('cat', 'Category'), ('path', 'Pathway'), ('kegg', 'KEGG')]:
        vals_site = []
        for mn in MODEL_LABELS:
            sub = module_df[(module_df.tier == tier) & (module_df.model == mn) &
                            (module_df.fn == fn_t)]
            vals_site.append(sub['ece'].median())
        site_s = _bold_min(vals_site, '.4f')
        lines.append(f'  {tier_label} & Site-level ECE & {site_s[0]} & {site_s[1]} \\\\')

        vals_comp = []
        for mn in MODEL_LABELS:
            sub = comp_df[(comp_df.tier == tier) & (comp_df.model == mn) &
                          (comp_df.fn == fn_t)]
            vals_comp.append(sub['ece'].median())
        comp_s = _bold_min(vals_comp, '.4f')
        lines.append(f'  & Completeness ECE & {comp_s[0]} & {comp_s[1]} \\\\')

        if tier != 'kegg':
            lines.append(r'  \addlinespace')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    (tab_dir / 'Cal3_module.tex').write_text('\n'.join(lines))



def main():
    pa = argparse.ArgumentParser(
        description='Multi-GPU calibration analysis of Ising denoisers.')
    pa.add_argument('--ckpt-hidden', required=True,
                    help='Checkpoint for ModuleConditionedDenoiser')
    pa.add_argument('--ckpt-nohidden', required=True,
                    help='Checkpoint for NoHiddenDenoiser')
    pa.add_argument('--train-feather', default='data/COG_train1_phylum.feather')
    pa.add_argument('--val-feather', default='data/COG_val1_phylum.feather')
    pa.add_argument('--val-domain', default=None,
                    help="If set (e.g. 'd__Archaea'), restrict the val set to "
                         "rows whose 'domain' column equals this value.  Reads "
                         "the domain column from --val-feather in file order "
                         "(matching load_feathers, frac=1.0) and row-masks "
                         "val_t.  Used to evaluate on the archaea-only subset "
                         "of an original held-out split.")
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    pa.add_argument('--outdir', default='calibration_output')
    pa.add_argument('--T', type=int, default=8)
    pa.add_argument('--H', type=int, default=1000)
    pa.add_argument('--skip-rank', type=int, default=32)
    pa.add_argument('--field-rank', type=int, default=16)
    pa.add_argument('--fn-grid', type=float, nargs='+',
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    pa.add_argument('--fp', type=float, default=0.01)
    pa.add_argument('--n-eval', type=int, default=None,
                    help='Number of val genomes (None = all)')
    pa.add_argument('--n-bins', type=int, default=20)
    pa.add_argument('--n-bootstrap', type=int, default=200,
                    help='Bootstrap replicates for CIs (default: 200; '
                         '1000 for publication, but 200 is fast and accurate '
                         'enough for exploratory runs).')
    pa.add_argument('--eval-batch', type=int, default=512)
    pa.add_argument('--force', action='store_true',
                    help='Regenerate predictions even if cached')
    pa.add_argument('--ci-alpha', type=float, default=0.05)
    pa.add_argument('--boot-seed', type=int, default=42)
    pa.add_argument('--label-hidden', default=None,
                    help="Display label for the --ckpt-hidden slot in all "
                         "figures/tables (e.g. 'T20 pre-FT'). The two slots "
                         "are just 'model A vs model B'; relabel them when "
                         "comparing two same-class checkpoints. "
                         "Default: 'Hidden+ELBO'.")
    pa.add_argument('--label-nohidden', default=None,
                    help="Display label for the --ckpt-nohidden slot "
                         "(e.g. 'T20 mix-FT'). Default: 'NoHidden+Onsager'.")
    A = pa.parse_args()

    # Only the displayed labels change; the dict keys stay 'hidden' and
    # 'nohidden', so plotting and table code is untouched.
    if A.label_hidden:
        MODEL_LABELS['hidden'] = A.label_hidden
    if A.label_nohidden:
        MODEL_LABELS['nohidden'] = A.label_nohidden

    outdir = Path(A.outdir)
    pred_dir = outdir / 'predictions'
    fig_dir = outdir / 'figures'
    tab_dir = outdir / 'tables'
    lf = str(outdir / 'calibration.log')

    use_ddp = 'RANK' in os.environ
    if use_ddp:
        rank, world, local_rank, dev = setup_dist()
    else:
        rank, world = 0, 1
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'

    if is_main():
        outdir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)

    barrier()
    log(f"Calibration analysis  GPUs={world}  dev={dev}", lf)
    t_total = time.time()

    log("Loading data...", lf)
    _, val_t, cog_names = load_feathers(A.train_feather, A.val_feather)
    # load_feathers reads rows in file order with frac=1.0 (no shuffle), so an
    # independent read of the 'domain' column aligns row-for-row with val_t.
    if A.val_domain is not None:
        dom = pd.read_feather(A.val_feather, columns=['domain'])['domain']
        mask = (dom.to_numpy() == A.val_domain)
        if mask.shape[0] != val_t.size(0):
            raise RuntimeError(
                f"domain column length {mask.shape[0]} != val rows "
                f"{val_t.size(0)}; cannot align domain mask")
        n_keep = int(mask.sum())
        log(f"  --val-domain {A.val_domain}: keeping {n_keep}/{mask.shape[0]} "
            f"val rows", lf)
        if n_keep == 0:
            raise RuntimeError(
                f"--val-domain {A.val_domain} matched 0 rows in {A.val_feather}")
        val_t = val_t[torch.from_numpy(mask)]
    if A.n_eval is not None:
        val_t = val_t[:A.n_eval]
    N = val_t.size(1)
    log(f"  val: {val_t.shape}, N={N}", lf)

    val_01 = ((val_t + 1.0) / 2.0).numpy()  # spins -> {0, 1}

    log("Loading module matrix...", lf)
    mm = torch.load(A.module_matrix, weights_only=False)
    M_full = mm['M'].to(dev)
    sizes_full = mm['sizes'].to(dev)
    n_modules = M_full.shape[1]

    # Metrics work on numpy, models on torch.
    M_cat_np = mm['M_cat'].numpy()
    M_path_np = mm['M_path'].numpy()
    M_kegg_np = mm['M_kegg'].numpy()
    module_data = {
        'M_cat': M_cat_np,
        'M_path': M_path_np,
        'M_kegg': M_kegg_np,
        'M_full': mm['M'].numpy(),
        'sizes_full': mm['sizes'].numpy(),
        'cat_sizes': mm['cat_sizes'].numpy(),
        'path_sizes': mm['path_sizes'].numpy(),
        'kegg_sizes': mm['kegg_sizes'].numpy(),
        'cat_names': mm['cat_names'],
        'cat_descs': mm['cat_descs'],
        'path_names': mm['path_names'],
        'kegg_names': mm['kegg_names'],
        'kegg_descs': mm.get('kegg_descs', mm['kegg_names']),
        'cog_names': cog_names,
    }
    log(f"  Modules: {n_modules} ({M_cat_np.shape[1]} cat + "
        f"{M_path_np.shape[1]} path + {M_kegg_np.shape[1]} kegg)", lf)

    log("Loading models...", lf)
    model_hidden, cls_h, T_h = build_model_from_ckpt(
        A.ckpt_hidden, dev, N, n_modules, A, lf)
    model_nohidden, cls_nh, T_nh = build_model_from_ckpt(
        A.ckpt_nohidden, dev, N, n_modules, A, lf)

    models = OrderedDict([
        ('hidden', model_hidden),
        ('nohidden', model_nohidden),
    ])
    log(f"  slot 'hidden'   <- {A.ckpt_hidden}  [{cls_h}, T={T_h}]  "
        f"label='{MODEL_LABELS['hidden']}'", lf)
    log(f"  slot 'nohidden' <- {A.ckpt_nohidden}  [{cls_nh}, T={T_nh}]  "
        f"label='{MODEL_LABELS['nohidden']}'", lf)
    log("  Models loaded.", lf)

    # ----  Phase 1: DDP prediction generation
    log("\n=== Phase 1: Prediction generation ===", lf)
    t1 = time.time()
    generate_predictions(models, val_t, M_full, sizes_full,
                         A.fn_grid, A.fp, dev, amp, A.eval_batch,
                         pred_dir, A.force, lf)
    barrier()
    log(f"Phase 1 done: {time.time()-t1:.1f}s", lf)

    # Phases 2-3 are CPU-only work on rank 0; idle ranks would hit the NCCL
    # watchdog timeout, so the group is destroyed and they exit here.
    rank_0 = is_main()
    if use_ddp:
        cleanup()
    if not rank_0:
        return

    del models, model_hidden, model_nohidden
    torch.cuda.empty_cache()

    # ----  Phase 2: metrics (rank 0 only)
    log("\n=== Phase 2: Metric computation ===", lf)
    t2 = time.time()
    global_df, cat_df, module_df, comp_df, cog_df = compute_all_metrics(
        pred_dir, val_01, module_data, A.fn_grid, A.n_bins,
        A.n_bootstrap, A.ci_alpha, A.boot_seed, lf)
    log(f"Phase 2 done: {time.time()-t2:.1f}s", lf)

    # ----  Phases 3 and 4: figures and tables
    log("\n=== Phase 3: Figures + Tables ===", lf)
    t3 = time.time()

    log("  Cal1: reliability diagrams...", lf)
    plot_cal1_reliability(global_df, pred_dir, val_01, A.fn_grid,
                          A.n_bins, fig_dir, cog_names)

    log("  Cal2: category heatmap...", lf)
    plot_cal2_heatmap(cat_df, A.fn_grid, fig_dir)

    log("  Cal3: module tiers...", lf)
    plot_cal3_modules(pred_dir, val_01, A.fn_grid, module_data,
                      A.n_bins, fig_dir)

    log("  Cal4: top miscalibrated COGs...", lf)
    plot_cal4_top_miscal(cog_df, A.fn_grid, module_data, fig_dir)

    log("  Cal5: model comparison...", lf)
    plot_cal5_comparison(global_df, comp_df, pred_dir, val_01,
                         A.fn_grid, A.n_bins, fig_dir)

    log("  Cal6: KEGG metabolic map...", lf)
    try:
        plot_cal6_kegg_map(cog_df, module_data, fig_dir, outdir)
    except Exception as e:
        log(f"  Cal6 skipped: {e}", lf)

    log("  Cal7: calibration funnel...", lf)
    plot_cal7_funnel(cog_df, module_df, comp_df, A.fn_grid, fig_dir)

    log("  Writing tables...", lf)
    write_cal1_table(global_df, A.fn_grid, tab_dir)
    write_cal2_table(cat_df, A.fn_grid, tab_dir)
    write_cal3_table(module_df, comp_df, A.fn_grid, tab_dir)

    log(f"Phase 3 done: {time.time()-t3:.1f}s", lf)
    log(f"\nTotal: {time.time()-t_total:.1f}s", lf)
    log(f"Output: {outdir.resolve()}", lf)


if __name__ == '__main__':
    main()
