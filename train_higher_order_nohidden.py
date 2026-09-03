#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_higher_order_nohidden.py — Add an attention-parameterised
                                  higher-order field Δ to a
                                  NoHiddenDenoiser checkpoint.

Visible-only higher-order path: the attention reads (x, t) alone, with no
hidden state z.  Module membership M_mod feeds only the inherited
module_cond MLP, which emits the per-iteration (γ_skip, γ_field) gates;
the attention itself never sees M_mod.  No aux head is built here, so the
completeness loss is never applied.  Loss is the pairwise trainer's ELBO,
CE(visible) + pl_alpha·PL(visible), plus the optional Δ-norm and
cross-input consistency penalties in HO2/HO3 and the coupling
regularisers in HO3.

Three stages, each resumable from its own checkpoint in --outdir
(defaults shown):

  HO1   attention warmup, everything else frozen
        beta_high, 20 ep, lr 1e-4
  HO2   attn + gates + cond, J frozen
        beta_hard, 40 ep, lr 5e-5
  HO3   full joint (J at j_lr_frac × base_lr)
        beta_hard, 20 ep, lr 2e-5

Writes model_ho{1,2,3}.pth, a report/ directory, and done.flag.

Usage:
    python -m torch.distributed.run --standalone --nproc_per_node=gpu \\
        train_higher_order_nohidden.py \\
            --init-from gsd_results_nohidden_finetune_chain_T8to20_split1/model_T8to12_f3.pth \\
            --new-T 12 \\
            --module-matrix data/module_matrix_kegg.pt \\
            --outdir gsd_results_higher_order_nohidden_T12_split1
"""
import argparse, json, math, time
from pathlib import Path
from collections import OrderedDict

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ising_denoiser.models import (
    NoHiddenDenoiser, NoHiddenHigherOrderDenoiser,
)
from ising_denoiser.modules import load_module_matrix
from ising_denoiser.data import ReconciliationNoiseDataset, load_feathers
from ising_denoiser.metrics import (
    eval_spectra, eval_spectra_multistep, FN_GRID,
)
from ising_denoiser.training import (
    setup_dist, cleanup, is_main, barrier, log, unwrap,
    load_and_extend, load_ckpt_cfg,
    save_ckpt, load_progress, broadcast_params, strip_compile_prefix,
)
from ising_denoiser.report import eval_null, generate_report

import train_denovo
from train_denovo import (
    train_denoiser_stage,
    param_diagnostics, param_diagnostics_full,
)

# Registered into train_denovo's stage table so train_denoiser_stage can
# resolve 'ho1'/'ho2'/'ho3'.  hidden=True in ho3 is inert: NoHiddenDenoiser
# has no hidden state and set_trainable ignores the flag.
HIGHER_ORDER_STAGE_TRAINABLE = {
    'ho1': dict(ising=False, hidden=False, scalar_skip=False,
                diag=False, lrs=False, lrf=False, cond=False, attn=True),
    'ho2': dict(ising=False, hidden=False, scalar_skip=True,
                diag=True,  lrs=True,  lrf=True,  cond=True,  attn=True),
    'ho3': dict(ising=True,  hidden=True,  scalar_skip=True,
                diag=True,  lrs=True,  lrf=True,  cond=True,  attn=True),
}
train_denovo.STAGE_TRAINABLE.update(HIGHER_ORDER_STAGE_TRAINABLE)


def detect_checkpoint_type(ckpt_path):
    """Return (has_hidden, has_attn, old_T) for a checkpoint.

    old_T is read from skip_alpha in either layout: the stacked (T, ...)
    tensor, or the older per-timestep 'skip_alpha.0', 'skip_alpha.1', ...
    keys.
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    has_hidden = all(name in sd for name in ('A', 'W'))
    has_attn = any(k.startswith('attn.') for k in sd)

    if 'skip_alpha' in sd and sd['skip_alpha'].dim() >= 1 and not any(
            k.startswith('skip_alpha.') for k in sd):
        old_T = int(sd['skip_alpha'].shape[0])
    else:
        old_T = sum(1 for k in sd if k.startswith('skip_alpha.'))
    if old_T <= 0:
        raise ValueError(
            f"Could not detect old_T from checkpoint {ckpt_path}: no "
            f"'skip_alpha' key found.")
    return has_hidden, has_attn, old_T


def main():
    pa = argparse.ArgumentParser(
        description='Train NoHiddenHigherOrderDenoiser: visible-only '
                    'attention Δ on top of a NoHiddenDenoiser checkpoint.')
    pa.add_argument('--init-from', required=True,
                    help='Path to a NoHiddenDenoiser checkpoint '
                         '(typically model_f3.pth from the finetune chain).')
    pa.add_argument('--new-T', type=int, default=None,
                    help='Target T.  Defaults to source T.')
    pa.add_argument('--old-T', type=int, default=None,
                    help='Source T, auto-detected from the checkpoint.')

    # Data
    pa.add_argument('--train-feather', default='data/COG_train1_phylum.feather')
    pa.add_argument('--val-feather',   default='data/COG_val1_phylum.feather')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')

    # Model geometry (must match the source checkpoint)
    pa.add_argument('--skip-rank',  type=int, default=32)
    pa.add_argument('--field-rank', type=int, default=16)
    pa.add_argument('--onsager',
                    choices=['auto', 'none', 'within', 'full'], default='auto',
                    help='Onsager/TAP correction mode (must match source). '
                         '"auto" (default) reads it from the init checkpoint sidecar; '
                         'falls back to "full" if the sidecar is missing (legacy ckpt).')

    # Stage schedule
    pa.add_argument('--ho1-epochs', type=int,   default=20)
    pa.add_argument('--ho1-lr',     type=float, default=1e-4)
    pa.add_argument('--ho2-epochs', type=int,   default=40)
    pa.add_argument('--ho2-lr',     type=float, default=5e-5)
    pa.add_argument('--ho3-epochs', type=int,   default=20)
    pa.add_argument('--ho3-lr',     type=float, default=2e-5)

    # Attention hyperparameters
    pa.add_argument('--attn-d-model',     type=int,   default=128)
    pa.add_argument('--attn-nhead',       type=int,   default=4)
    pa.add_argument('--attn-n-layers',    type=int,   default=1)
    pa.add_argument('--attn-dim-ff',      type=int,   default=512)
    pa.add_argument('--attn-dropout',     type=float, default=0.1)
    pa.add_argument('--attn-lr-frac',     type=float, default=0.3,
                    help='Attention LR = base_lr × attn_lr_frac.')
    pa.add_argument('--no-attn-checkpoint', action='store_true',
                    help='Disable gradient checkpointing inside the attention '
                         'module (faster but more memory).')

    # Loss
    pa.add_argument('--loss', type=str, default='elbo', choices=['wmse', 'elbo'])
    pa.add_argument('--pl-alpha', type=float, default=0.3)
    pa.add_argument('--ce-eps',   type=float, default=1e-6)
    pa.add_argument('--aux-lambda', type=float, default=0.0,
                    help='Aux lambda (default 0 — no hidden state, aux loss '
                         'is always skipped).')
    pa.add_argument('--lambda-delta', type=float, default=0.0,
                    help='Explicit lambda_Delta * E||Delta||^2 penalty weight '
                         '(eq:loss-ho / impl-map [M13]). Default 0 = implicit '
                         'regularisation only. Applied in HO2 and HO3 (after '
                         'the HO1 attention warmup).')
    pa.add_argument('--consistency-lambda', type=float, default=0.0,
                    help='Cross-input consistency weight: penalise the variance '
                         'across the K corrupted copies of each genome so the '
                         'sparse/under- and dense/over-corrupted copies map to the '
                         'SAME denoised output (input-invariance -> collapses the '
                         'sparse-build vs dense-trim gap). Applied in HO2 and HO3.')

    # Noise / augmentation
    pa.add_argument('--fn-max', type=float, default=0.9)
    pa.add_argument('--ho-fn-curriculum', type=str, default='beta_hard',
                    choices=['uniform', 'beta_low', 'beta_mid',
                             'beta_high', 'beta_hard', 'beta_vhard'],
                    help='FN curriculum for HO2/HO3 (default beta_hard, the '
                         'proven mix-FT recipe). Use beta_vhard with '
                         '--fn-max 0.95 to concentrate training mass at the '
                         'FN 0.85-0.95 deep-ancestor (LBCA/LACA) point in the '
                         'fine-tune A/B. HO1 always uses beta_high (attention '
                         'warmup).')
    pa.add_argument('--fp',     type=float, default=0.01)
    pa.add_argument('--fp-max', type=float, default=None,
                    help='If set, FP is no longer fixed at --fp: each training '
                         'copy draws its own FP from [0, fp-max] via --fp-dist '
                         '(the FP-tolerant curriculum). HO2/HO3 only; HO1 '
                         'attention warmup keeps the fixed --fp.')
    pa.add_argument('--fp-dist', type=str, default='beta_low',
                    choices=['uniform', 'beta_low', 'beta_high'],
                    help='FP sampling distribution when --fp-max is set. '
                         'beta_low = Beta(1,5)*fp_max (mass near 0, gentle tail '
                         'to fp_max); uniform = U[0,fp_max]; beta_high = '
                         'Beta(2,1)*fp_max (mass near fp_max).')
    # ---- Coherent false-positive injection (HO2/HO3)
    # Independent per-COG false positives (--fp-max) are easy for the couplings
    # to reject; real reconciliation errors arrive in functionally coherent
    # blocks.  These knobs add whole-module events on top of the per-COG
    # curriculum and require --module-matrix.
    pa.add_argument('--mod-inject-rate', type=float, default=0.0,
                    help='per training copy, prob. of injecting one absent '
                         'functional module entire (coherent FP).')
    pa.add_argument('--mod-swap-rate', type=float, default=0.0,
                    help='per training copy, prob. of grafting up to '
                         '--mod-swap-max modules from a RANDOM OTHER training '
                         'genome (a foreign organism\'s signature in an '
                         'incompatible context -- the realistic FP failure mode).')
    pa.add_argument('--mod-delete-rate', type=float, default=0.0,
                    help='per training copy, prob. of deleting one present '
                         'module entire (coherent FN).')
    pa.add_argument('--mod-swap-max', type=int, default=3,
                    help='cap on modules grafted per swap event.')
    pa.add_argument('--marginal-fp-rate', type=float, default=0.0,
                    help='per training copy, prob. of a MARGINAL-PRUNE example: '
                         'low FN + dense FP drawn by per-COG cross-genome marginal '
                         'frequency (broad, module-incoherent over-reconstruction '
                         'to be trimmed -- the Count/Brownian marginal signature).')
    pa.add_argument('--marginal-fp-max', type=float, default=0.0,
                    help='max FP excess fraction (of true gene count) added on a '
                         'marginal-prune copy; drawn U[0.1, this].')
    pa.add_argument('--marginal-fp-fn', type=float, default=0.1,
                    help='max FN on a marginal-prune copy (drawn U[0, this]); kept '
                         'low so the example tests trimming, not rescue.')
    pa.add_argument('--marginal-freq-feather', default=None,
                    help='OPTIONAL feather whose per-COG prevalence overrides the '
                         'marginal-FP contamination prior (aligned to the training '
                         'COG order). E.g. data/COG_arc_train1_phylum.feather to '
                         'train on the mix set but draw contamination from how '
                         'common each COG is in archaea. Default: prevalence in the '
                         'training set itself.')
    pa.add_argument('--pos-w',  type=float, default=3.0)
    pa.add_argument('--K',          type=int, default=4)
    pa.add_argument('--clean-frac', type=float, default=0.125)
    pa.add_argument('--batch-per-gpu', type=int, default=384)
    pa.add_argument('--num-workers',   type=int, default=4)

    # Optimisation
    pa.add_argument('--j-lr-frac',   type=float, default=0.1)
    pa.add_argument('--weight-decay', type=float, default=1e-4)
    pa.add_argument('--max-norm',    type=float, default=50.0)
    pa.add_argument('--ms-steps',    type=int,   default=3)

    pa.add_argument('--compile', action='store_true',
                    help='torch.compile the model (max-autotune-no-cudagraphs).')
    pa.add_argument('--ckpt-every', type=int, default=5,
                    help='Save mid-stage checkpoint every N epochs.')
    pa.add_argument('--outdir', default='gsd_results_higher_order_nohidden')
    # ---- Coupling-discipline controls on J~
    pa.add_argument('--genome-mass-norm', action='store_true',
                    help='Bake in sqrt(kbar/k_n) coupling-drive normalisation.')
    pa.add_argument('--gmass-freq-tsv', default='data/cog_marginal_frequency.tsv')
    pa.add_argument('--lowrank-lambda', type=float, default=0.0,
                    help='Tail-energy penalty on J~ (toward transferable low rank).')
    pa.add_argument('--lowrank-rank', type=int, default=16,
                    help='Target rank for the low-rank tail penalty.')
    pa.add_argument('--zeromean-lambda', type=float, default=0.0,
                    help='Zero-mean coupling penalty (field carries base rates).')
    pa.add_argument('--per-state-cal', action='store_true',
                    help='Trainable per-state gain/loss output calibration head.')

    A = pa.parse_args()

    # No RANK/WORLD_SIZE in the environment means no DDP, so the same file
    # serves either a torchrun job or a single-GPU job array.
    if 'RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        rank, ws, lr_rank, dev = setup_dist()
    else:
        rank, ws, lr_rank = 0, 1, 0
        torch.cuda.set_device(0)
        dev = torch.device('cuda:0')
    amp = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    od = Path(A.outdir)
    if is_main():
        od.mkdir(exist_ok=True)
    barrier()
    lf = od / 'train.log'

    prog = load_progress(od) if is_main() else {'completed': []}
    if dist.is_initialized():
        ol = [prog]; dist.broadcast_object_list(ol, src=0); prog = ol[0]
    done = set(prog['completed'])
    t0g = time.time()

    log(f"\n{'='*72}", lf)
    log(f"  HIGHER ORDER NO-HIDDEN (visible-only Δ) — "
        f"{time.strftime('%Y-%m-%d %H:%M')}", lf)
    log(f"  init_from={A.init_from}", lf)
    log(f"  onsager={A.onsager}, loss={A.loss}, compile={A.compile}", lf)
    log(f"  attn: d={A.attn_d_model} heads={A.attn_nhead} "
        f"L={A.attn_n_layers} ff={A.attn_dim_ff} dropout={A.attn_dropout} "
        f"lr_frac={A.attn_lr_frac}", lf)
    log(f"  {ws} x {torch.cuda.get_device_name(0)}", lf)
    log(f"{'='*72}", lf)

    # ---- Data
    train_t, val_t, vocab = load_feathers(A.train_feather, A.val_feather)
    N = len(vocab)
    log(f"\n  Train: {train_t.shape[0]:,} x {N}  Val: {val_t.shape[0]:,} x {N}", lf)

    # Optional external contamination prior p_c for marginal-FP draws; the
    # override is realigned to the training COG order, not the donor's.
    marginal_freq = None
    if A.marginal_freq_feather:
        import pandas as pd
        mdf = pd.read_feather(A.marginal_freq_feather)
        mcols = [c for c in mdf.columns if c.startswith('COG')]
        pcm = dict(zip(mcols, (mdf[mcols].values > 0).mean(axis=0)))
        marginal_freq = torch.tensor([float(pcm.get(c, 0.0)) for c in vocab],
                                     dtype=torch.float32)
        log(f"  marginal-freq override: {A.marginal_freq_feather} "
            f"({mdf.shape[0]:,} genomes; {int((marginal_freq >= 0.9).sum())} COGs "
            f"p_c>=0.9, {int((marginal_freq > 0).sum())} >0)", lf)

    # ---- Module matrix, for the module_cond gates only
    M_mod = None; M_sizes = None; n_mod = 1
    aux_head = None
    if Path(A.module_matrix).exists():
        _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, 1)
        n_mod = M_mod.shape[1]
        log(f"  Module matrix: {n_mod} targets (gates only, no aux head)", lf)
    else:
        log(f"  WARNING: module matrix {A.module_matrix} not found — "
            f"training without conditioning.", lf)

    ref_sp = None
    barrier()

    # ---- Detect source model type and T
    if not Path(A.init_from).exists():
        raise FileNotFoundError(f"--init-from not found: {A.init_from}")

    has_hidden_detected, has_attn_detected, old_T_detected = \
        detect_checkpoint_type(A.init_from)
    if has_hidden_detected:
        raise ValueError(
            "train_higher_order_nohidden.py requires a NoHiddenDenoiser "
            "source checkpoint (this path supports only the no-hidden "
            "pairwise base).")

    old_T = A.old_T if A.old_T is not None else old_T_detected
    if A.new_T is None:
        A.new_T = old_T
        log(f"  --new-T unspecified, defaulting to old_T={old_T}", lf)

    src_kind = ('NoHiddenHigherOrderDenoiser' if has_attn_detected
                else 'NoHiddenDenoiser')
    log(f"\n  Source checkpoint: T={old_T}, {src_kind}", lf)

    saved_cfg = load_ckpt_cfg(A.init_from)
    saved_onsager = saved_cfg.get('onsager')
    if A.onsager == 'auto':
        if saved_onsager is not None:
            A.onsager = saved_onsager
            log(f"  onsager={A.onsager} (from {Path(A.init_from).name} sidecar)", lf)
        else:
            A.onsager = 'full'
            log(f"  onsager={A.onsager} (default; sidecar missing — legacy ckpt, "
                f"may not match training)", lf)
    elif saved_onsager is not None and A.onsager != saved_onsager:
        raise SystemExit(
            f"--onsager={A.onsager} conflicts with checkpoint sidecar "
            f"onsager={saved_onsager}. Omit --onsager to use the saved value, "
            f"or pass {saved_onsager} explicitly.")

    log(f"  Target: T={A.new_T}, onsager={A.onsager}", lf)

    if A.new_T < old_T:
        raise ValueError(f"--new-T ({A.new_T}) must be >= old_T ({old_T}).")

    onsager_val = A.onsager != 'none'

    attn_kwargs = dict(
        attn_d_model=A.attn_d_model,
        attn_nhead=A.attn_nhead,
        attn_n_layers=A.attn_n_layers,
        attn_dim_feedforward=A.attn_dim_ff,
        attn_dropout=A.attn_dropout,
        attn_use_checkpoint=not A.no_attn_checkpoint,
    )

    in_prog = prog.get('in_progress')

    if not done and not in_prog:
        raw = load_and_extend(
            A.init_from, N, None, old_T, A.new_T,
            A.skip_rank, A.field_rank,
            NoHiddenHigherOrderDenoiser, dev,
            n_modules=n_mod, adaptive_temp=True, onsager=onsager_val,
            **attn_kwargs,
        )
        log(f"\n  Built: {raw.config_str()}", lf)
    else:
        raw = NoHiddenHigherOrderDenoiser(
            N, A.new_T, A.skip_rank, A.field_rank,
            n_modules=n_mod, adaptive_temp=True, onsager=onsager_val,
            **attn_kwargs,
        ).to(dev)

        last = in_prog['stage'] if in_prog else prog['completed'][-1]
        ck = od / f'model_{last}.pth'
        if ck.exists():
            sd = strip_compile_prefix(
                torch.load(ck, map_location=dev, weights_only=True))
            raw.load_state_dict(sd, strict=False)
            tag = (f'mid-stage ep{in_prog["epoch"]}' if in_prog
                   else 'completed')
            log(f"\n  Resumed from: {ck} ({tag})", lf)

    raw.attn_lr_frac = A.attn_lr_frac

    if A.compile:
        log(f"  torch.compile(mode='max-autotune-no-cudagraphs')", lf)
        raw = torch.compile(raw, mode='max-autotune-no-cudagraphs')

    broadcast_params(raw)
    if dist.is_initialized():
        model = DDP(raw, device_ids=[lr_rank], find_unused_parameters=True)
    else:
        model = raw
    base = unwrap(model)

    if A.genome_mass_norm:
        import pandas as _pd
        _p = _pd.read_csv(A.gmass_freq_tsv, sep='\t')['p_c'].to_numpy()
        base.gmass_kbar = torch.tensor(float(_p.sum()), device=dev)
        log(f"  genome-mass norm ON: kbar={float(_p.sum()):.1f}", lf)
    if A.per_state_cal:
        base.use_gainloss_cal = True
        base.gainloss_cal.requires_grad_(True)
        log("  per-state (gain/loss) calibration head ON", lf)
    if A.lowrank_lambda > 0 or A.zeromean_lambda > 0:
        log(f"  coupling regularisers: lowrank(nuc)={A.lowrank_lambda} "
            f"zeromean={A.zeromean_lambda}", lf)

    # From a pairwise source Δ must start at zero, so the extended model
    # reproduces its parent before HO1 trains the attention.
    if not done and is_main():
        base.eval()
        sp0 = eval_spectra(base, val_t, dev, amp, fp=A.fp,
                           M_mod=M_mod, M_sizes=M_sizes)
        post_label = ('trained attn preserved' if has_attn_detected
                      else 'attention zero-init')
        log(f"\n  Post-extend spectrum ({post_label}):", lf)
        for r in sp0:
            log(f"    FN={r['fn']:.2f}: MCC={r['MCC']:.4f} F1={r['F1']:.4f}", lf)

        nprobe = min(8, val_t.size(0))
        x_probe = val_t[:nprobe].to(dev).float()
        dn = base.delta_norm(x_probe)
        if has_attn_detected:
            log(f"\n  Δ norm at init: {dn:.3e}  "
                f"(HO source — trained attn preserved)", lf)
            assert math.isfinite(dn) and dn < 100.0, (
                f"Δ norm not finite or absurdly large: {dn}")
        else:
            log(f"\n  Δ norm at init: {dn:.3e}  (must be < 1e-6)", lf)
            assert dn < 1e-6, (
                f"HigherOrderModule output head not zero-init: {dn}")

    barrier()

    def hdr(name, desc, ep):
        log(f"\n{'='*72}\n  {name}: {desc} ({ep} ep) — "
            f"{time.strftime('%H:%M')} ({(time.time()-t0g)/60:.0f}m)\n{'='*72}", lf)

    def stage_eval(name):
        if not is_main():
            return
        sp = eval_spectra(base, val_t, dev, amp, fp=A.fp,
                          M_mod=M_mod, M_sizes=M_sizes)
        log(f"\n    {name}: spectra", lf)
        for r in sp:
            log(f"      FN={r['fn']:.2f}: MCC={r['MCC']:.4f} F1={r['F1']:.4f}", lf)
        param_diagnostics_full(model, lf)

    def _start_ep(stage):
        if in_prog and in_prog['stage'] == stage:
            return int(in_prog['epoch']) + 1
        return 1

    # ---- HO1
    if 'ho1' not in done and A.ho1_epochs > 0:
        hdr('HO1', 'beta_high — attention warmup (pairwise frozen)',
            A.ho1_epochs)
        ds = ReconciliationNoiseDataset(train_t, A.fn_max, A.fp, A.K,
                                        A.clean_frac, 'beta_high')
        train_denoiser_stage(
            model, ds, val_t, dev, 'ho1',
            ne=A.ho1_epochs, lr=A.ho1_lr, wd=A.weight_decay,
            amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
            j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
            pos_w=A.pos_w, fp=A.fp,
            aux_head=aux_head, aux_lam=A.aux_lambda,
            M_mod=M_mod, M_sizes=M_sizes,
            ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
            loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
            start_ep=_start_ep('ho1'), ckpt_every=A.ckpt_every,
            od=od, prog=prog,
        )
        stage_eval('HO1'); save_ckpt(model, od, 'ho1', prog, lf); barrier()
    else:
        log(f"\n  Skip HO1", lf)

    # ---- HO2
    if 'ho2' not in done and A.ho2_epochs > 0:
        hdr('HO2', f'{A.ho_fn_curriculum}: attn + gates + cond (J frozen)',
            A.ho2_epochs)
        ds = ReconciliationNoiseDataset(train_t, A.fn_max, A.fp, A.K,
                                        A.clean_frac, A.ho_fn_curriculum,
                                        fp_max=A.fp_max, fp_dist=A.fp_dist,
                                        module_membership=(M_mod.cpu() if M_mod is not None else None),
                                        module_sizes=(M_sizes.cpu() if M_sizes is not None else None),
                                        module_inject_rate=A.mod_inject_rate,
                                        module_swap_rate=A.mod_swap_rate,
                                        module_delete_rate=A.mod_delete_rate,
                                        module_swap_max=A.mod_swap_max,
                                        marginal_fp_rate=A.marginal_fp_rate,
                                        marginal_fp_max=A.marginal_fp_max,
                                        marginal_fp_fn=A.marginal_fp_fn,
                                        marginal_freq=marginal_freq)
        train_denoiser_stage(
            model, ds, val_t, dev, 'ho2',
            ne=A.ho2_epochs, lr=A.ho2_lr, wd=A.weight_decay,
            amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
            j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
            pos_w=A.pos_w, fp=A.fp,
            aux_head=aux_head, aux_lam=A.aux_lambda,
            M_mod=M_mod, M_sizes=M_sizes,
            ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
            loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
            delta_lam=A.lambda_delta, consistency_lambda=A.consistency_lambda,
            start_ep=_start_ep('ho2'), ckpt_every=A.ckpt_every,
            od=od, prog=prog,
        )
        stage_eval('HO2'); save_ckpt(model, od, 'ho2', prog, lf); barrier()
    else:
        log(f"\n  Skip HO2", lf)

    # ---- HO3
    if 'ho3' not in done and A.ho3_epochs > 0:
        hdr('HO3', f'{A.ho_fn_curriculum}: full joint (J protected by j_lr_frac)',
            A.ho3_epochs)
        ds = ReconciliationNoiseDataset(train_t, A.fn_max, A.fp, A.K,
                                        A.clean_frac, A.ho_fn_curriculum,
                                        fp_max=A.fp_max, fp_dist=A.fp_dist,
                                        module_membership=(M_mod.cpu() if M_mod is not None else None),
                                        module_sizes=(M_sizes.cpu() if M_sizes is not None else None),
                                        module_inject_rate=A.mod_inject_rate,
                                        module_swap_rate=A.mod_swap_rate,
                                        module_delete_rate=A.mod_delete_rate,
                                        module_swap_max=A.mod_swap_max,
                                        marginal_fp_rate=A.marginal_fp_rate,
                                        marginal_fp_max=A.marginal_fp_max,
                                        marginal_fp_fn=A.marginal_fp_fn,
                                        marginal_freq=marginal_freq)
        train_denoiser_stage(
            model, ds, val_t, dev, 'ho3',
            ne=A.ho3_epochs, lr=A.ho3_lr, wd=A.weight_decay,
            amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
            j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
            pos_w=A.pos_w, fp=A.fp,
            aux_head=aux_head, aux_lam=A.aux_lambda,
            M_mod=M_mod, M_sizes=M_sizes,
            ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
            loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
            delta_lam=A.lambda_delta, consistency_lambda=A.consistency_lambda,
            lowrank_lambda=A.lowrank_lambda, zeromean_lambda=A.zeromean_lambda,
            lowrank_rank=A.lowrank_rank,
            start_ep=_start_ep('ho3'), ckpt_every=A.ckpt_every,
            od=od, prog=prog,
        )
        stage_eval('HO3'); save_ckpt(model, od, 'ho3', prog, lf); barrier()
    else:
        log(f"\n  Skip HO3", lf)

    # ---- Final report
    if is_main():
        e = (time.time() - t0g) / 60
        log(f"\n{'='*72}\n  FINAL — {e:.0f}m\n{'='*72}", lf)
        param_diagnostics_full(model, lf)

        nprobe = min(64, val_t.size(0))
        x_probe = val_t[:nprobe].to(dev).float()
        dn_final = base.delta_norm(x_probe)
        log(f"\n  Δ norm at end of training: {dn_final:.4e}", lf)

        fn_grid = FN_GRID
        spectra = OrderedDict()
        spectra['Noisy input'] = eval_null(val_t, dev, fn_grid, fp=A.fp)

        sp_1 = eval_spectra(base, val_t, dev, amp, fp=A.fp,
                            M_mod=M_mod, M_sizes=M_sizes, fn_grid=fn_grid)
        spectra[f'NoHidden HO T={A.new_T} (1-step)'] = sp_1

        if M_mod is not None:
            for ms in (2, 3, 5):
                sp_ms = eval_spectra_multistep(
                    base, val_t, dev, amp, M_mod, M_sizes, ms,
                    fn_grid=fn_grid, fp=A.fp)
                spectra[f'NoHidden HO T={A.new_T} ({ms}-step)'] = sp_ms

        log(f"\n  Final spectra:", lf)
        for name, rows in spectra.items():
            log(f"    {name}:", lf)
            for r in rows:
                log(f"      FN={r['fn']:.2f}: MCC={r['MCC']:.4f} "
                    f"F1={r['F1']:.4f} P={r['prec']:.3f} R={r['rec']:.3f}", lf)

        report_dir = od / 'report'
        try:
            generate_report(spectra, fn_grid, report_dir)
            log(f"\n  Report: {report_dir}/", lf)
        except Exception as ex:
            log(f"\n  Report generation failed: {ex}", lf)

        (od / 'done.flag').touch()
        log(f"\n  Wrote {od}/done.flag — training fully complete", lf)

    barrier(); cleanup()


if __name__ == '__main__':
    main()
