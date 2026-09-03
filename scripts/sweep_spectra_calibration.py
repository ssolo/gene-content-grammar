#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_spectra_calibration.py -- per-step denoising-spectra + calibration
sweep for one denoiser checkpoint, across a 2D (FN, FP) noise grid.

Each (false-negative, false-positive) cell runs one forward pass over a fixed
(seeded) sample of validation genomes and keeps the whole annealing
trajectory: the checkpoint runs T shared-weight learned TAP mean-field
iterations (input-dependent gates, optionally adaptive temperature), and the
visible magnetisation after iteration t, m^t = tanh(pre-activation), is that
step's genome estimate.  Every step of the (T, B, N) trajectory is scored for
both spectra (MCC, F1, precision, recall, fp_removed; sign-thresholded at
prob 0.5, pooled over all genes) and calibration (ECE, MCE, Brier, NLL), so
the two come from the same predictions.  The `step` column is 1..T for model
rows (step T is the final single-pass output) and 0 for the model-free null
baseline.  One tidy parquet per checkpoint; ``aggregate_spectra_sweep.py``
concatenates them across model families and cross-validation splits into the
committed plotting dataset.

Unlike calibration_sweep.py, this driver is family-agnostic: it reuses
build_model_for_ckpt, which auto-detects nohidden / hidden / higher_order /
nohidden_higher_order, so the attention-bearing checkpoints load correctly.

Execution modes.  Default is single-GPU (plain `python`, no torchrun) and
forward-only: one gradient-free forward per cell, after which the (T, B, N)
trajectory moves to the host and is scored with the numpy metrics.
--gpu-metrics scores the trajectory on-device via metrics_gpu(); --compile
re-enables torch.compile.  A DDP mode (under torch.distributed.run) gathers to
rank 0 for the numpy metrics and is bit-for-bit comparable: noise is drawn on
the full seeded tensor before sharding, the model couples genes only within a
genome, and metrics_gpu matches the numpy bin semantics.

Noise model (matches ising_denoiser.metrics.eval_spectra_2d so spectra are
directly comparable to the training-time 2D scan and scan_2d_spectra.py):
  per cell  seed = 42 + int(fn*1000) + 7919*int(fp*1000)
  flip present->absent with prob FN, then absent->present with prob FP.
Noise is drawn on the full sampled tensor before DDP sharding, so the result
is independent of the number of GPUs.

Calibration binning: fixed-width reliability bins (O(N), the Guo et al.
definition), whereas calibration_analysis.py uses equal-frequency bins with
per-genome bootstrap CIs; the two ECE numbers differ slightly by construction.
Brier and NLL are binning-independent and match exactly.

Usage (one checkpoint):
  python3 -m torch.distributed.run --standalone --nproc_per_node=gpu \\
      scripts/sweep_spectra_calibration.py \\
      --ckpt        gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth \\
      --val-feather data/COG_val1_phylum.feather \\
      --module-matrix data/module_matrix_kegg.pt \\
      --model-tag   nohidden_T20 --family nohidden_T20 --split 1 \\
      --out         spectra_sweep/nohidden_T20_split1.parquet

Usage (noisy-input null baseline, no model -- run once per val set/grid):
  python3 scripts/sweep_spectra_calibration.py --null-only \\
      --val-feather data/COG_val1_phylum.feather \\
      --out spectra_sweep/_null_split1.parquet
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast

# Repo + scripts on path, for the checkpoint loader and distributed helpers.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

from ising_denoiser.data import load_feathers
from ising_denoiser.modules import load_module_matrix
from ising_denoiser.training import (
    setup_dist, cleanup, is_main, barrier, log, strip_compile_prefix)
# Auto-detecting loader; onsager is resolved from the model_*.cfg.json sidecar.
from analyze_ancestral_node import build_model_for_ckpt


# ----  Noise, with seeding identical to ising_denoiser.metrics.eval_spectra_2d
def make_noisy(vc, fn, fp):
    """Return (noisy_pm1, injected_fp_mask) for one (fn, fp) cell.

    vc: (B, N) truth in {-1, +1} on CPU.  The mask marks the genes flipped
    -1 -> +1, and is what fp_removed is scored against.
    """
    torch.manual_seed(42 + int(fn * 1000) + 7919 * int(fp * 1000))
    ny = vc.clone()
    r = torch.rand_like(ny)
    ny[(vc == 1) & (r < fn)] = -1
    r2 = torch.rand_like(ny)
    inj = (vc == -1) & (r2 < fp)
    ny[inj] = 1
    return ny, inj


# ----  DDP-sharded inference -> rank-0 full probability trajectory
def _gather_BN(local_BN, B_full, world, rank, dev):
    """Gather one stride-sharded (B_local, N) tensor to (B_full, N) on rank 0.

    Returns None off rank 0.  Called once per annealing step: a single
    (T, B, N) all_gather is T-fold larger and OOMs at T=20.
    """
    N = local_BN.size(1)
    max_local = (B_full + world - 1) // world
    padded = torch.zeros(max_local, N)
    padded[:local_BN.size(0)] = local_BN
    padded_gpu = padded.to(dev)
    gathered = [torch.zeros_like(padded_gpu) for _ in range(world)]
    dist.all_gather(gathered, padded_gpu)
    if rank != 0:
        return None
    full = torch.zeros(B_full, N)
    for r in range(world):
        idx = list(range(r, B_full, world))
        full[idx] = gathered[r].cpu()[:len(idx)]
    return full


def resolve_amp_dtype(choice, dev):
    """Autocast dtype for the forward: a torch dtype, or None for fp32.

    'auto' gives bf16 on sm_80 and later (the precision the calibration tables
    use) and fp16 below.  fp16 is mandatory on sm_70, which has no native bf16
    and where the only usable SDPA kernel (see sdpa_ctx) rejects bf16.
    """
    if choice == 'fp32':
        return None
    if choice == 'bf16':
        return torch.bfloat16
    if choice == 'fp16':
        return torch.float16
    if dev.type != 'cuda':
        return None
    major = torch.cuda.get_device_capability(dev)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def sdpa_ctx():
    """Context manager restricting SDPA to the flash + memory-efficient
    kernels, never the math fallback.

    The math backend materialises the full (B, nhead, N, N) score tensor,
    ~175 GiB at N ~ 4789; with math excluded, SDPA raises instead of
    attempting that allocation.  Degrades to the legacy toggle on older torch
    and to a no-op on CPU.
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
def predict_trajectory_distributed(model, noisy_t, M_mod, M_sizes, dev,
                                   batch_size, amp_dtype=torch.bfloat16,
                                   gpu_resident=False):
    """Shard noisy_t by stride, run one autocast forward with
    return_trajectory, gather the per-step visible magnetisations.

    Returns a (T, B, N) float32 probability tensor on rank 0 (None elsewhere),
    slice t being ((m^{t+1} + 1) / 2), the same sign->prob map as the
    single-pass path.

    gpu_resident is single-GPU only: False (default) moves each chunk's
    trajectory to host as soon as its forward returns, so the GPU holds only
    the forward; True keeps the trajectory on device for metrics_gpu.  The DDP
    path always moves to host, for the per-step all_gather.
    """
    model.eval()
    use_ddp = dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world = dist.get_world_size() if use_ddp else 1
    T = int(model.T)
    B_full = noisy_t.size(0)
    N = noisy_t.size(1)

    local_idx = list(range(rank, B_full, world))
    local_noisy = noisy_t[local_idx].to(dev)

    parts = []
    use_amp = (dev.type == 'cuda') and (amp_dtype is not None)
    with sdpa_ctx(), autocast('cuda', dtype=(amp_dtype or torch.float16),
                              enabled=use_amp):
        for i in range(0, local_noisy.size(0), batch_size):
            batch = local_noisy[i:i + batch_size]
            if M_mod is not None:
                _, _, traj = model(batch, M_mod, M_sizes,
                                   return_trajectory=True)
            else:
                _, _, traj = model(batch, return_trajectory=True)
            prob = (traj.float() + 1.0) / 2.0
            keep = (not use_ddp) and gpu_resident
            parts.append(prob if keep else prob.cpu())

    if not use_ddp:
        empty = torch.zeros(T, 0, N, device=(dev if gpu_resident else 'cpu'))
        return torch.cat(parts, dim=1) if parts else empty

    local_traj = (torch.cat(parts, dim=1) if parts
                  else torch.zeros(T, 0, N))
    full = torch.zeros(T, B_full, N) if rank == 0 else None
    for t in range(T):
        g = _gather_BN(local_traj[t], B_full, world, rank, dev)
        if rank == 0:
            full[t] = g
    return full


# ----  Metrics (rank 0, numpy)
def spectra_metrics(y_pm, prob, inj):
    """Pooled (over all genes) sign-threshold metrics + fp_removed.

    y_pm: (B, N) truth in {-1,+1}; prob: (B, N) in [0,1]; inj: bool mask of the
    injected false positives.  Matches ising_denoiser.metrics.metrics().
    """
    pred_pos = prob > 0.5
    true_pos = y_pm > 0
    tp = float(np.sum(pred_pos & true_pos))
    fp = float(np.sum(pred_pos & ~true_pos))
    fn = float(np.sum(~pred_pos & true_pos))
    tn = float(np.sum(~pred_pos & ~true_pos))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else float('nan')
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    n_inj = int(np.sum(inj))
    fp_removed = (float(np.mean(~pred_pos[inj])) if n_inj > 0 else float('nan'))
    return dict(MCC=mcc, F1=f1, prec=prec, rec=rec,
                fp_removed=fp_removed, base_rate=float(np.mean(true_pos)))


def calib_metrics(y01, prob, n_bins=20, eps=1e-7):
    """Pooled calibration: fixed-width-bin ECE/MCE + Brier + NLL.

    y01: (B*N,) in {0,1}; prob: (B*N,) in [0,1].
    """
    p = np.clip(prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    psum = np.bincount(idx, weights=p, minlength=n_bins)
    ysum = np.bincount(idx, weights=y01.astype(np.float64), minlength=n_bins)
    nz = cnt > 0
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    conf[nz] = psum[nz] / cnt[nz]
    acc[nz] = ysum[nz] / cnt[nz]
    gap = np.abs(acc - conf)
    w = cnt / cnt.sum()
    ece = float(np.sum(w * gap))
    mce = float(np.max(gap[nz])) if nz.any() else float('nan')
    brier = float(np.mean((y01 - p) ** 2))
    pc = np.clip(p, eps, 1 - eps)
    nll = float(-np.mean(y01 * np.log(pc) + (1 - y01) * np.log(1 - pc)))
    return dict(ece=ece, mce=mce, brier=brier, nll=nll)


@torch.no_grad()
def metrics_gpu(traj_prob, y_pm_g, inj_g, n_bins=20, eps=1e-7):
    """GPU-vectorised per-step spectra + calibration for the single-GPU path.

    All three inputs are on device: traj_prob the unsharded (T, B, N)
    trajectory in [0,1], y_pm_g the (B, N) truth in {-1,+1}, inj_g the (B, N)
    injected-FP mask.  Returns a list of T (spectra_dict, calib_dict) pairs for
    steps 1..T, following spectra_metrics()/calib_metrics() but batched over
    steps, with the (T,B,N) reductions on the GPU and the final scalar
    arithmetic in float64 on the host.
    """
    T = traj_prob.shape[0]
    true_pos = (y_pm_g > 0)
    base_rate = float(true_pos.float().mean().item())
    n_inj = int(inj_g.sum().item())
    pos1 = true_pos.unsqueeze(0)
    pred_pos = traj_prob > 0.5

    # Confusion counts per step: int64 sums on GPU, float64 on the host.
    def _cnt(mask):
        return mask.sum(dim=(1, 2)).double().cpu().numpy()
    tp = _cnt(pred_pos & pos1)
    fp = _cnt(pred_pos & ~pos1)
    fn = _cnt(~pred_pos & pos1)
    tn = _cnt(~pred_pos & ~pos1)
    if n_inj > 0:
        fpr = _cnt(~pred_pos & inj_g.unsqueeze(0)) / float(n_inj)
    else:
        fpr = np.full(T, np.nan)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    with np.errstate(invalid='ignore', divide='ignore'):
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.sqrt(denom),
                       np.nan)
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

    # Calibration: Brier and NLL vectorised; ECE/MCE via per-step bincount.
    p = traj_prob.clamp(0.0, 1.0)
    y01 = true_pos.float()
    y01b = y01.unsqueeze(0)
    brier = ((y01b - p) ** 2).mean(dim=(1, 2)).cpu().numpy()
    pc = p.clamp(eps, 1.0 - eps)
    nll = (-(y01b * pc.log() + (1.0 - y01b) * (1.0 - pc).log())
           ).mean(dim=(1, 2)).cpu().numpy()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    # torch.bucketize's `right` is inverted relative to np.digitize: the numpy
    # path's digitize(..., right=False) is bucketize(..., right=True).  With
    # right=False, exactly-on-edge probabilities (p = 0.5 from a bf16 m == 0)
    # land in the wrong bin and ECE/MCE diverge from the numpy path.
    idx = torch.clamp(torch.bucketize(p, edges[1:-1], right=True),
                      0, n_bins - 1)
    idxT = idx.view(T, -1)
    pT = p.view(T, -1).double()
    yflat = y01.view(-1).double()
    ece = np.empty(T)
    mce = np.empty(T)
    for t in range(T):
        it = idxT[t]
        cnt = torch.bincount(it, minlength=n_bins).double().cpu().numpy()
        psum = torch.bincount(it, weights=pT[t],
                              minlength=n_bins).cpu().numpy()
        ysum = torch.bincount(it, weights=yflat,
                              minlength=n_bins).cpu().numpy()
        nz = cnt > 0
        conf = np.zeros(n_bins)
        acc = np.zeros(n_bins)
        conf[nz] = psum[nz] / cnt[nz]
        acc[nz] = ysum[nz] / cnt[nz]
        gap = np.abs(acc - conf)
        w = cnt / cnt.sum()
        ece[t] = float(np.sum(w * gap))
        mce[t] = float(np.max(gap[nz])) if nz.any() else float('nan')

    out = []
    for t in range(T):
        out.append((
            dict(MCC=float(mcc[t]), F1=float(f1[t]), prec=float(prec[t]),
                 rec=float(rec[t]), fp_removed=float(fpr[t]),
                 base_rate=base_rate),
            dict(ece=float(ece[t]), mce=float(mce[t]),
                 brier=float(brier[t]), nll=float(nll[t]))))
    return out


# ----  Grid
def build_fn_grid(fn_min, fn_max, fn_step):
    eps = 1e-6
    grid = []
    x = fn_min
    while x <= fn_max + eps:
        grid.append(round(x, 4))
        x += fn_step
    return grid


def main():
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--ckpt', default=None,
                    help='Checkpoint to score (model_ho3.pth). '
                         'Omit only with --null-only.')
    pa.add_argument('--null-only', action='store_true',
                    help='Score the noisy-input null baseline (no model).')
    pa.add_argument('--val-feather', required=True)
    pa.add_argument('--val-domain', default=None,
                    help="If set (e.g. 'd__Archaea'), restrict the val set to "
                         "rows whose 'domain' column equals this value "
                         "(applied BEFORE the genome sample).")
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    pa.add_argument('--out', required=True,
                    help='Output parquet path for this checkpoint.')
    # Identity columns, passed through into the parquet for aggregation.
    pa.add_argument('--model-tag', default=None)
    pa.add_argument('--family', default=None)
    pa.add_argument('--split', type=int, default=-1)
    pa.add_argument('--fn-min', type=float, default=0.0)
    pa.add_argument('--fn-max', type=float, default=0.9)
    pa.add_argument('--fn-step', type=float, default=0.05)
    pa.add_argument('--fp-grid', type=float, nargs='+',
                    default=[0.01, 0.02, 0.05, 0.1])
    # Named --n-genomes, not --n: under torch.distributed.run, torchrun's own
    # argparse reads a bare --n as an ambiguous abbreviation of --nnodes /
    # --nproc-per-node / --node-rank and aborts before forwarding anything.
    pa.add_argument('--n-genomes', dest='n', type=int, default=4096,
                    help='Validation genomes to score (seeded sample). '
                         '0 = use all.')
    pa.add_argument('--sample-seed', type=int, default=0,
                    help='Seed for the genome sample (same genomes for '
                         'every checkpoint -> comparable spectra).')
    pa.add_argument('--n-bins', type=int, default=20)
    pa.add_argument('--eval-batch', type=int, default=256)
    pa.add_argument('--H', type=int, default=1000)
    pa.add_argument('--compile', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='DEFAULT OFF.  The spectra noise scan is a plain '
                         'gradient-free forward: load the weights and run the '
                         'forward to measure spectra -- no torch.compile, no '
                         'Triton codegen.  Pass --compile to opt back into '
                         'torch.compile (Triton) if you ever want it.')
    pa.add_argument('--amp-dtype', choices=['auto', 'bf16', 'fp16', 'fp32'],
                    default='auto',
                    help="Autocast dtype for the forward.  'auto' (default): "
                         "bf16 on Ampere+ (sm_80+, e.g. an 80 GB GPU) -- the headline "
                         "precision -- and fp16 on pre-Ampere (sm_70 a 16 GB GPU).  "
                         "a 16 GB GPU has no native bf16, and the ONLY SDPA kernel "
                         "that avoids the O(N^2) (B,H,N,N) attention "
                         "materialisation there (flash needs sm_80+) is the "
                         "memory-efficient one, which rejects bf16 -- so a "
                         "a 16 GB GPU run MUST be fp16.  'fp32' disables autocast.")
    pa.add_argument('--gpu-metrics', action='store_true',
                    help='Score the (T,B,N) trajectory ON the GPU via '
                         'metrics_gpu instead of moving it to host and using '
                         'the numpy spectra/calib metrics.  DEFAULT OFF: for '
                         'the noise scan the GPU runs ONLY the forward, then '
                         'the trajectory is moved to host and scored on CPU '
                         '(the numpy metrics are the reference path, and the '
                         'GPU footprint stays just the forward -- which fits '
                         'a 16 GB a 16 GB GPU).  Opt in on a big an 80 GB GPU for speed.')
    A = pa.parse_args()

    if not A.null_only and not A.ckpt:
        raise SystemExit('--ckpt is required unless --null-only is given.')

    # DDP only when launched under torchrun (RANK set in the environment).
    if os.environ.get('RANK') is not None:
        rank, ws, lr_rank, dev = setup_dist()
    else:
        rank, ws, lr_rank = 0, 1, 0
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_path = Path(A.out)
    if is_main():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    barrier()

    tag = A.model_tag or (Path(A.ckpt).parent.name if A.ckpt else 'noisy')
    family = A.family or tag

    log(f"\n{'='*72}", None)
    log(f"  sweep_spectra_calibration  {time.strftime('%Y-%m-%d %H:%M')}", None)
    log(f"  tag={tag} family={family} split={A.split}  "
        f"{ws} rank(s) dev={dev}", None)

    # Seeded sample: the same genomes for every checkpoint.
    _, val_t, vocab = load_feathers(A.val_feather, A.val_feather, frac=1.0)
    N = len(vocab)
    if A.val_domain:
        dom = pd.read_feather(A.val_feather, columns=['domain'])['domain'].to_numpy()
        if dom.shape[0] != val_t.size(0):
            raise RuntimeError(f"domain column length {dom.shape[0]} != val "
                               f"rows {val_t.size(0)}; cannot align mask")
        mask = (dom == A.val_domain)
        n_keep = int(mask.sum())
        log(f"  --val-domain {A.val_domain}: keeping {n_keep}/{mask.shape[0]} "
            f"val rows", None)
        if n_keep == 0:
            raise RuntimeError(f"--val-domain {A.val_domain} matched 0 rows")
        val_t = val_t[torch.from_numpy(mask)]
    if A.n and A.n < val_t.size(0):
        g = torch.Generator().manual_seed(A.sample_seed)
        idx = torch.randperm(val_t.size(0), generator=g)[:A.n]
        val_t = val_t[idx].contiguous()
    B = val_t.size(0)
    log(f"  val: {B} genomes x {N} COGs  (from {A.val_feather})", None)

    # Module matrix: the conditioning input for the forward.
    _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, H=A.H)
    n_mod = M_mod.shape[1]

    model = cls_kind = T = onsager = None
    if not A.null_only:
        sd = torch.load(A.ckpt, map_location=dev, weights_only=True)
        sd = strip_compile_prefix({k.replace('module.', ''): v
                                   for k, v in sd.items()})
        model, cls_kind, T, onsager = build_model_for_ckpt(
            sd, dev, N, n_mod, A.ckpt, H=A.H)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()
        log(f"  model: {cls_kind} T={T} onsager={onsager}  "
            f"(missing={len(missing)} unexpected={len(unexpected)})", None)
        if len(missing) or len(unexpected):
            log(f"  WARNING missing={list(missing)[:6]} "
                f"unexpected={list(unexpected)[:6]}", None)
        if A.compile and dev.type == 'cuda':
            try:
                model = torch.compile(model)
                log("  torch.compile: ON (first cell pays warmup; the rest "
                    "reuse the graph)", None)
            except Exception as e:  # noqa: BLE001 -- never let compile abort eval
                log(f"  torch.compile FAILED ({e}); falling back to eager",
                    None)

    amp_dtype = resolve_amp_dtype(A.amp_dtype, dev)
    _dt = ('fp32' if amp_dtype is None
           else {torch.bfloat16: 'bf16', torch.float16: 'fp16'}[amp_dtype])
    if dev.type == 'cuda':
        _cap = torch.cuda.get_device_capability(dev)
        log(f"  device: {torch.cuda.get_device_name(dev)} "
            f"sm_{_cap[0]}{_cap[1]}  amp={_dt} (--amp-dtype {A.amp_dtype})  "
            f"SDPA=flash+mem_efficient (math excluded)", None)
    else:
        log(f"  device: CPU  amp={_dt}", None)

    fn_grid = build_fn_grid(A.fn_min, A.fn_max, A.fn_step)
    fp_grid = sorted(set(A.fp_grid))
    log(f"  FN grid ({len(fn_grid)}): {fn_grid}", None)
    log(f"  FP grid ({len(fp_grid)}): {fp_grid}", None)

    use_ddp = dist.is_initialized()
    # DDP and the null baseline always take the host (numpy) metric path.
    gpu_metrics = (A.gpu_metrics and (not use_ddp) and (dev.type == 'cuda')
                   and (not A.null_only))
    val_pm_gpu = val_t.to(dev) if gpu_metrics else None
    val_pm_np = val_t.numpy() if (is_main() and not gpu_metrics) else None
    y01_flat = (((val_pm_np + 1) // 2).ravel()
                if val_pm_np is not None else None)

    def _row(fn, fp, step, sm, cm):
        return dict(model_tag=tag, family=family, split=A.split,
                    cls_kind=(cls_kind or 'noisy'),
                    T=(int(T) if T is not None else -1),
                    onsager=str(onsager), n_genomes=B,
                    fn=fn, fp=fp, step=step, **sm, **cm)

    rows = []
    t_start = time.time()
    total = len(fn_grid) * len(fp_grid)
    done = 0
    for fn in fn_grid:
        for fp in fp_grid:
            noisy, inj = make_noisy(val_t, fn, fp)
            sm = cm = None
            n_steps = 0
            if gpu_metrics:
                traj = predict_trajectory_distributed(
                    model, noisy, M_mod, M_sizes, dev, A.eval_batch, amp_dtype,
                    gpu_resident=True)
                per_step = metrics_gpu(traj, val_pm_gpu, inj.to(dev),
                                       n_bins=A.n_bins)
                del traj
                for t, (sm, cm) in enumerate(per_step):
                    rows.append(_row(fn, fp, t + 1, sm, cm))
                n_steps = len(per_step)
            elif A.null_only:
                if is_main():
                    pred = ((noisy.float() + 1.0) / 2.0).numpy()
                    sm = spectra_metrics(val_pm_np, pred, inj.numpy())
                    cm = calib_metrics(y01_flat, pred.ravel(),
                                       n_bins=A.n_bins)
                    rows.append(_row(fn, fp, 0, sm, cm))
                    n_steps = 1
            else:  # host-scored model path; DDP gathers to rank 0
                traj_t = predict_trajectory_distributed(
                    model, noisy, M_mod, M_sizes, dev, A.eval_batch, amp_dtype)
                if is_main():
                    inj_np = inj.numpy()
                    for t in range(traj_t.size(0)):
                        pred = traj_t[t].numpy()
                        sm = spectra_metrics(val_pm_np, pred, inj_np)
                        cm = calib_metrics(y01_flat, pred.ravel(),
                                           n_bins=A.n_bins)
                        rows.append(_row(fn, fp, t + 1, sm, cm))
                    n_steps = int(traj_t.size(0))
            done += 1
            if is_main() and sm is not None:
                dt = time.time() - t_start
                eta = dt / done * (total - done)
                # sm/cm hold the last step scored: T for the model, 0 for the
                # null baseline.
                log(f"  [{done:>3}/{total}] fn={fn:.2f} fp={fp:.2f}  "
                    f"final MCC={sm['MCC']:.4f} F1={sm['F1']:.4f} "
                    f"ECE={cm['ece']:.4f} Brier={cm['brier']:.4f}  "
                    f"({n_steps} steps, eta {eta/60:.1f}m)", None)
            barrier()

    # Write before tearing down the process group: is_main() is
    # `not dist.is_initialized() or rank == 0`, so once cleanup() destroys the
    # group every rank is_main(), and ranks 1..n-1 (empty `rows`) would clobber
    # rank 0's parquet with a 0-row file.
    if is_main():
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        log(f"\n  wrote {out_path}  ({len(df)} rows)", None)
        log(f"  total: {(time.time() - t_start)/60:.1f} min", None)
    barrier()
    if dist.is_initialized():
        cleanup()


if __name__ == '__main__':
    main()
