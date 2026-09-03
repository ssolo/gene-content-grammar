#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_add_hidden.py -- add a hidden state to a trained NoHidden HO3
                       checkpoint, from a transparent start and under a
                       module-coherent corruption curriculum.

The source checkpoint has J, low-rank gates, module conditioning and the
attention Delta, but no hidden spins.  This trainer adds the hidden
machinery (A, U, W) and trains the hidden state under a module-coherent
corruption curriculum.

Target model: HigherOrderDenoiser with attention tied across the T mean-field
steps, the ELBO loss (ce + pl_alpha * pseudo-likelihood) and, by default,
onsager='full' (TAP corrections both visible-visible and hidden-cross).

Zeroing the visible-side coupling A after the strict=False state-dict load
makes step 0 bit-identical to the source model: with A == 0 both the hidden
term A @ z and the full-TAP cross term in A^2 drop out of the visible
updates.  The hidden state still evolves from x through U, so the aux loss
has informative input from step 1.

Stages:

AH1   Hidden only (A, U, W); J, gates, attention and conditioning frozen.
      Heavy module-coherent corruption (inject + delete + swap) and a
      boosted aux_lambda.

AH2   Hidden + gates + conditioning + attention, J still frozen.

AH3   Full joint with J at --j-lr-frac.

Writes into --outdir: model_ah{1,2,3}.pth and their .cfg.json sidecars,
progress.json (resume state), spectra_2d.tsv, report/, done.flag.

Usage:
  python -m torch.distributed.run --standalone --nproc_per_node=gpu \\
      train_add_hidden.py \\
          --init-from <nohidden-HO3-run>/model_ho3.pth \\
          --train-feather data/COG_train1_phylum.feather \\
          --val-feather   data/COG_val1_phylum.feather \\
          --module-matrix data/module_matrix_kegg.pt \\
          --outdir gsd_results_addhidden_T20_split1
"""
import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ising_denoiser.models import HigherOrderDenoiser
from ising_denoiser.modules import load_module_matrix
from ising_denoiser.data import ReconciliationNoiseDataset
from ising_denoiser.metrics import (
    eval_spectra, eval_spectra_multistep, eval_spectra_2d, FN_GRID, FP_GRID,
)
from ising_denoiser.training import (
    setup_dist, cleanup, is_main, barrier, log, unwrap,
    strip_compile_prefix, load_progress, save_ckpt,
)
from ising_denoiser.report import eval_null, generate_report

import train_denovo
from train_denovo import (
    train_denoiser_stage,
    param_diagnostics, param_diagnostics_full,
)


# ---- Add-Hidden stage schedule, registered into train_denovo.STAGE_TRAINABLE
AH_STAGE_TRAINABLE = {
    # Adaptive-temperature parameters are trainable in every stage: the base
    # set_trainable() hardcodes them and these masks cannot switch them off.
    'ah1': dict(ising=False, hidden=True,  scalar_skip=False,
                diag=False,  lrs=False,    lrf=False,
                cond=False,  attn=False),
    # HO2 mask plus the hidden machinery.
    'ah2': dict(ising=False, hidden=True,  scalar_skip=True,
                diag=True,   lrs=True,     lrf=True,
                cond=True,   attn=True),
    # HO3 mask.
    'ah3': dict(ising=True,  hidden=True,  scalar_skip=True,
                diag=True,   lrs=True,     lrf=True,
                cond=True,   attn=True),
}
train_denovo.STAGE_TRAINABLE.update(AH_STAGE_TRAINABLE)


# ---- Source-checkpoint surgery

def detect_T_from_sd(sd):
    """Read T off the skip_alpha entry of a source state dict.

    Accepts both layouts: a stacked tensor whose first axis is T, and
    per-timestep keys 'skip_alpha.0', 'skip_alpha.1', ...
    """
    if 'skip_alpha' in sd and sd['skip_alpha'].dim() >= 1:
        return int(sd['skip_alpha'].shape[0])
    n = sum(1 for k in sd if k.startswith('skip_alpha.'))
    if n > 0:
        return n
    raise ValueError("Could not detect T from source state dict.")


def main():
    pa = argparse.ArgumentParser(
        description="Add hidden on top of NoHidden HO3 with transparent "
                    "start + module-coherent corruption curriculum.")
    pa.add_argument('--init-from', required=True,
                    help='NoHidden HO3 model_ho3.pth (e.g. '
                         'gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth)')
    pa.add_argument('--new-T', type=int, default=None,
                    help='Target T.  Defaults to source T (no T extension here).')

    # Data
    pa.add_argument('--train-feather', default='data/COG_train1_phylum.feather')
    pa.add_argument('--val-feather',   default='data/COG_val1_phylum.feather')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')

    # Model geometry (must match the source checkpoint)
    pa.add_argument('--H',          type=int, default=1000)
    pa.add_argument('--skip-rank',  type=int, default=32)
    pa.add_argument('--field-rank', type=int, default=16)
    pa.add_argument('--onsager', default='full', choices=['none', 'within', 'full', 'tied'],
                    help="TAP/Onsager mode; 'tied' (U=A^T) is a clean single-matrix EBM "
                         "so the ELBO is a proper bound -- use for the tied-ELBO run.")
    pa.add_argument('--a-init-noise', type=float, default=0.0,
                    help="If >0, init the hidden->visible coupling A with N(0,this) "
                         "instead of zero -- breaks the A=0 deadlock (essential for "
                         "onsager=tied where U=A^T; near-transparent for small values).")

    # Attention (must match the source's attention)
    pa.add_argument('--attn-d-model',     type=int,   default=128)
    pa.add_argument('--attn-nhead',       type=int,   default=4)
    pa.add_argument('--attn-n-layers',    type=int,   default=1)
    pa.add_argument('--attn-dim-ff',      type=int,   default=512)
    pa.add_argument('--attn-dropout',     type=float, default=0.1)
    pa.add_argument('--attn-lr-frac',     type=float, default=0.3)

    # Stage schedule
    pa.add_argument('--ah1-epochs', type=int,   default=12)
    pa.add_argument('--ah1-lr',     type=float, default=1e-4)
    pa.add_argument('--ah2-epochs', type=int,   default=15)
    pa.add_argument('--ah2-lr',     type=float, default=5e-5)
    pa.add_argument('--ah3-epochs', type=int,   default=8)
    pa.add_argument('--ah3-lr',     type=float, default=2e-5)

    # Loss
    pa.add_argument('--loss',     default='elbo', choices=['wmse', 'elbo'])
    pa.add_argument('--pl-alpha', type=float, default=0.3)
    pa.add_argument('--ce-eps',   type=float, default=1e-6)
    pa.add_argument('--aux-lambda', type=float, default=0.3,
                    help='Base aux_lambda for module-aux loss.  AH1 boosts '
                         'this to ah1-aux-mult * aux_lambda to emphasise '
                         'module-signature learning during hidden warmup.')
    pa.add_argument('--ah1-aux-mult', type=float, default=3.0)

    # Standard per-COG noise
    pa.add_argument('--fn-max', type=float, default=0.9)
    pa.add_argument('--fp',     type=float, default=0.01)
    pa.add_argument('--pos-w',  type=float, default=3.0)
    pa.add_argument('--K',          type=int, default=4)
    pa.add_argument('--clean-frac', type=float, default=0.125)
    pa.add_argument('--batch-per-gpu', type=int, default=128)
    pa.add_argument('--num-workers',   type=int, default=4)

    # Module-coherent corruption rates (per-stage)
    pa.add_argument('--ah1-mod-inject', type=float, default=0.30)
    pa.add_argument('--ah1-mod-delete', type=float, default=0.15)
    pa.add_argument('--ah1-mod-swap',   type=float, default=0.55)
    pa.add_argument('--ah23-mod-inject', type=float, default=0.20)
    pa.add_argument('--ah23-mod-delete', type=float, default=0.10)
    pa.add_argument('--ah23-mod-swap',   type=float, default=0.30)

    # Marginal-FP prune curriculum
    pa.add_argument('--marginal-fp-rate', type=float, default=0.0,
                    help='per copy, prob. of a marginal-PRUNE example (low FN + dense '
                         'FP by per-COG marginal frequency -- tasks the model to trim).')
    pa.add_argument('--marginal-fp-max', type=float, default=0.0)
    pa.add_argument('--marginal-fp-fn',  type=float, default=0.1)
    pa.add_argument('--marginal-freq-feather', default=None,
                    help='optional feather whose per-COG prevalence overrides the '
                         'marginal-FP prior (aligned to the training vocab).')

    # Optimisation
    pa.add_argument('--j-lr-frac',   type=float, default=0.1)
    pa.add_argument('--weight-decay', type=float, default=1e-4)
    pa.add_argument('--max-norm',    type=float, default=50.0)
    pa.add_argument('--ms-steps',    type=int,   default=3)
    pa.add_argument('--aux-mix-mod', type=float, default=1.0,
                    help='Aux-loss scale.  Used as the multiplier on the '
                         'existing module-aux loss term; the per-stage '
                         'aux_lambda is also adjusted.')

    pa.add_argument('--compile', action='store_true')
    pa.add_argument('--ckpt-every', type=int, default=5)
    pa.add_argument('--outdir', default='gsd_results_addhidden')
    A = pa.parse_args()

    # Fall back to a single GPU when not launched under torchrun: the model is
    # communication-bound, so a sweep runs faster as N one-GPU jobs.
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
        od.mkdir(exist_ok=True, parents=True)
    barrier()
    lf = od / 'train.log'

    prog = load_progress(od) if is_main() else {'completed': []}
    if dist.is_initialized():
        ol = [prog]; dist.broadcast_object_list(ol, src=0); prog = ol[0]
    done = set(prog.get('completed', []))
    in_prog = prog.get('in_progress', None)

    t0g = time.time()
    log(f"\n{'='*72}\n  ADD HIDDEN (transparent start, full TAP, ELBO, tied attn)"
        f"\n{'='*72}", lf)
    log(f"  init-from : {A.init_from}", lf)
    log(f"  outdir    : {od}", lf)

    # ---- Data
    from ising_denoiser.data import load_feathers
    log("\n  Loading train/val feathers...", lf)
    train_t, val_t, vocab = load_feathers(A.train_feather, A.val_feather, frac=1.0)
    N = len(vocab)

    # Optional external contamination prior (e.g. an archaea- or bacteria-only
    # p_c) overriding the marginal-FP prune prior.  Re-indexed onto the
    # training COG order: the source feather need not carry the same COG set.
    marginal_freq = None
    if A.marginal_freq_feather:
        import pandas as pd
        mdf = pd.read_feather(A.marginal_freq_feather)
        mcols = [c for c in mdf.columns if c.startswith('COG')]
        pcm = dict(zip(mcols, (mdf[mcols].values > 0).mean(axis=0)))
        marginal_freq = torch.tensor([float(pcm.get(c, 0.0)) for c in vocab],
                                     dtype=torch.float32)
        log(f"  marginal-freq override: {A.marginal_freq_feather} "
            f"({mdf.shape[0]:,} genomes; {int((marginal_freq > 0).sum())} COGs >0)", lf)
    log(f"  train: {train_t.shape[0]:,} genomes x {N} COGs", lf)
    log(f"  val:   {val_t.shape[0]:,} genomes",  lf)

    # ---- Module matrix + aux head
    aux_head, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, A.H)
    n_mod = M_mod.shape[1] if M_mod is not None else 1
    log(f"  modules: {n_mod} (combined)", lf)

    # ---- KEGG-modules block for the noise dataset
    # Module-coherent corruption uses the KEGG modules alone: COG categories
    # are too broad and COG pathways too narrow to stand in for a coherently
    # gained or lost function.
    mm = torch.load(A.module_matrix, weights_only=False)
    M_kegg = mm['M_kegg'].cpu().float()
    M_kegg_sizes = mm['kegg_sizes'].cpu().float()
    log(f"  KEGG modules for corruption: {M_kegg.shape[1]}", lf)

    # ---- Source checkpoint
    log(f"\n  Loading source NoHidden HO3 checkpoint: {A.init_from}", lf)
    src_sd = torch.load(A.init_from, map_location='cpu', weights_only=True)
    src_sd = strip_compile_prefix(src_sd)
    src_T = detect_T_from_sd(src_sd)
    new_T = A.new_T if A.new_T is not None else src_T
    log(f"  source T = {src_T}; target T = {new_T}", lf)
    if new_T != src_T:
        raise SystemExit(
            f"T extension not supported in train_add_hidden "
            f"(source T={src_T}, target T={new_T}).  Use train_denovo (chain "
            f"mode) to extend T first, then run add_hidden at the extended T.")

    # ---- Build the target HigherOrderDenoiser (tied attention)
    log(f"\n  Building HigherOrderDenoiser shell (onsager={A.onsager}, tied attn)...", lf)
    model = HigherOrderDenoiser(
        N=N, H=A.H, T=new_T, n_modules=n_mod,
        skip_rank=A.skip_rank, field_rank=A.field_rank,
        adaptive_temp=True, onsager=A.onsager,
        attn_d_model=A.attn_d_model, attn_nhead=A.attn_nhead,
        attn_n_layers=A.attn_n_layers, attn_dim_feedforward=A.attn_dim_ff,
        attn_dropout=A.attn_dropout, attn_use_checkpoint=True,
    ).to(dev)

    base = model
    log(f"  built: {base.config_str()}" if hasattr(base, 'config_str')
        else f"  built HigherOrderDenoiser T={new_T} H={A.H}", lf)

    # Non-strict: the hidden machinery (A, U, W) has no counterpart in the
    # source checkpoint and stays at its fresh init.
    missing, unexpected = base.load_state_dict(src_sd, strict=False)
    log(f"\n  Loaded source -> HigherOrderDenoiser (strict=False):", lf)
    log(f"    missing keys ({len(missing)}): expected for the hidden "
        f"machinery (A, U, W) and possibly h_z if present.", lf)
    if missing:
        log(f"    {missing[:8]}{'...' if len(missing) > 8 else ''}", lf)
    if unexpected:
        log(f"    UNEXPECTED keys in source ({len(unexpected)}): "
            f"{unexpected[:6]}{'...' if len(unexpected) > 6 else ''}", lf)
        log(f"    (these are silently dropped; check if any should map to "
            f"existing target keys)", lf)

    # A starts at zero unless --a-init-noise is given.  Under onsager='tied'
    # (U = A^T) A = 0 decouples hidden from visible entirely: z receives no
    # signal, A's gradient is ~0, and A never grows.  Small noise breaks that
    # deadlock while staying near-transparent.
    with torch.no_grad():
        if A.a_init_noise > 0:
            base.A.normal_(0.0, A.a_init_noise)
            log(f"\n  A init: noise N(0, {A.a_init_noise}) -> A.abs().max() = "
                f"{base.A.abs().max().item():.2e}  (near-transparent; breaks the "
                f"A=0 deadlock so the hidden can engage).", lf)
        else:
            base.A.zero_()
            assert base.A.abs().max().item() < 1e-9, \
                "Transparency invariant broken: A is not zero after zero_()."
            log("\n  Transparency: A zeroed (strict transparent start).", lf)
    log("  At AH1 step 0 the model's visible output = source NoHidden HO3 "
        "output.  U keeps its default xavier init -> z evolves from x, so "
        "the aux loss has informative input.  A grows organically during "
        "AH1 from gradients.", lf)

    if aux_head is not None:
        log("  Aux head: ModuleCompletenessPredictor (z -> module fractions). "
            "Fresh, near-zero-init per its docstring -> initial sigmoid ~0.5.",
            lf)

    if dist.is_initialized():
        model = DDP(model, device_ids=[lr_rank], find_unused_parameters=False)
    if A.compile:
        log("\n  torch.compile(mode='max-autotune-no-cudagraphs')...", lf)
        model = torch.compile(model, mode='max-autotune-no-cudagraphs')

    def _start_ep(stage):
        if in_prog and in_prog.get('stage') == stage:
            return int(in_prog['epoch']) + 1
        return 1

    def hdr(name, sub, n_ep):
        e = (time.time() - t0g) / 60
        log(f"\n{'='*72}\n  {name}  --  {sub}  --  {n_ep} ep  (t={e:.0f}m)"
            f"\n{'='*72}", lf)

    def stage_eval(label):
        if not is_main():
            return
        base_m = unwrap(model)
        base_m.eval()
        sp = eval_spectra(base_m, val_t, dev, amp, fp=A.fp,
                          M_mod=M_mod, M_sizes=M_sizes, fn_grid=FN_GRID)
        log(f"\n    {label}: spectra (fixed fp={A.fp})", lf)
        for r in sp:
            log(f"      FN={r['fn']:.2f}: MCC={r['MCC']:.4f} F1={r['F1']:.4f}", lf)

    def make_ds(stage):
        if stage == 'ah1':
            inj, dele, swap = A.ah1_mod_inject, A.ah1_mod_delete, A.ah1_mod_swap
            fn_dist = 'beta_high'
        else:
            inj, dele, swap = (A.ah23_mod_inject, A.ah23_mod_delete,
                               A.ah23_mod_swap)
            fn_dist = 'beta_hard'
        return ReconciliationNoiseDataset(
            train_t, A.fn_max, A.fp, A.K, A.clean_frac, fn_dist,
            module_membership=M_kegg, module_sizes=M_kegg_sizes,
            module_inject_rate=inj, module_delete_rate=dele,
            module_swap_rate=swap,
            marginal_fp_rate=A.marginal_fp_rate, marginal_fp_max=A.marginal_fp_max,
            marginal_fp_fn=A.marginal_fp_fn, marginal_freq=marginal_freq,
        )

    def aux_lam_for(stage):
        if stage == 'ah1':
            return A.aux_lambda * A.ah1_aux_mult
        return A.aux_lambda

    # ---- Transparency cross-check (rank 0, no train)
    if is_main():
        with torch.no_grad():
            base_m = unwrap(model); base_m.eval()
            xt = val_t[:8].to(dev).float()
            out_a, _ = base_m(xt, M_mod, M_sizes)
            # The source model is not rebuilt for comparison, so this checks
            # only that the augmented forward is finite and in range.
            assert torch.isfinite(out_a).all(), "non-finite output from augmented model"
            log(f"\n  Transparency sanity: forward on 8 val genomes finite, "
                f"output range [{out_a.min().item():.3f}, {out_a.max().item():.3f}]",
                lf)
            base_m.train()

    # ---- AH1: hidden-only warmup
    if 'ah1' not in done:
        hdr('AH1', f'beta_high + heavy module corruption '
                   f'(inj/del/swap={A.ah1_mod_inject}/{A.ah1_mod_delete}/'
                   f'{A.ah1_mod_swap}); aux_lambda x {A.ah1_aux_mult}',
            A.ah1_epochs)
        ds = make_ds('ah1')
        train_denoiser_stage(
            model, ds, val_t, dev, 'ah1',
            ne=A.ah1_epochs, lr=A.ah1_lr, wd=A.weight_decay,
            amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
            j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
            pos_w=A.pos_w, fp=A.fp,
            aux_head=aux_head, aux_lam=aux_lam_for('ah1'),
            M_mod=M_mod, M_sizes=M_sizes,
            ref_sp=None, eval_every=50, ms_steps=A.ms_steps,
            loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
            start_ep=_start_ep('ah1'), ckpt_every=A.ckpt_every,
            od=od, prog=prog,
        )
        stage_eval('AH1')
        save_ckpt(model, od, 'ah1', prog, lf, aux_head); barrier()
    else:
        log("\n  Skip AH1 (already done)", lf)

    # ---- AH2: hidden + gates + attn + cond, J frozen
    if 'ah2' not in done:
        hdr('AH2', f'beta_hard + module corruption (rates '
                   f'{A.ah23_mod_inject}/{A.ah23_mod_delete}/{A.ah23_mod_swap}); '
                   f'J FROZEN', A.ah2_epochs)
        ds = make_ds('ah2')
        train_denoiser_stage(
            model, ds, val_t, dev, 'ah2',
            ne=A.ah2_epochs, lr=A.ah2_lr, wd=A.weight_decay,
            amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
            j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
            pos_w=A.pos_w, fp=A.fp,
            aux_head=aux_head, aux_lam=aux_lam_for('ah2'),
            M_mod=M_mod, M_sizes=M_sizes,
            ref_sp=None, eval_every=50, ms_steps=A.ms_steps,
            loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
            start_ep=_start_ep('ah2'), ckpt_every=A.ckpt_every,
            od=od, prog=prog,
        )
        stage_eval('AH2')
        save_ckpt(model, od, 'ah2', prog, lf, aux_head); barrier()
    else:
        log("\n  Skip AH2 (already done)", lf)

    # ---- AH3: full joint, J at j-lr-frac
    if 'ah3' not in done:
        hdr('AH3', f'beta_hard + module corruption; full joint, '
                   f'j-lr-frac={A.j_lr_frac}', A.ah3_epochs)
        ds = make_ds('ah3')
        train_denoiser_stage(
            model, ds, val_t, dev, 'ah3',
            ne=A.ah3_epochs, lr=A.ah3_lr, wd=A.weight_decay,
            amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
            j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
            pos_w=A.pos_w, fp=A.fp,
            aux_head=aux_head, aux_lam=aux_lam_for('ah3'),
            M_mod=M_mod, M_sizes=M_sizes,
            ref_sp=None, eval_every=50, ms_steps=A.ms_steps,
            loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
            start_ep=_start_ep('ah3'), ckpt_every=A.ckpt_every,
            od=od, prog=prog,
        )
        stage_eval('AH3')
        save_ckpt(model, od, 'ah3', prog, lf, aux_head); barrier()
    else:
        log("\n  Skip AH3 (already done)", lf)

    # ---- Final report + 2D spectra scan
    if is_main():
        e = (time.time() - t0g) / 60
        log(f"\n{'='*72}\n  FINAL -- {e:.0f}m\n{'='*72}", lf)
        try:
            param_diagnostics_full(model, lf)
        except Exception as ex:
            log(f"  (param_diagnostics_full skipped: {ex})", lf)

        base_m = unwrap(model)
        log(f"\n  A.norm() at end of training: {base_m.A.norm().item():.4f}"
            f"  (was 0.0 at AH1 step 0; growth = hidden engagement)", lf)

        fn_grid = FN_GRID
        spectra = OrderedDict()
        spectra['Noisy input'] = eval_null(val_t, dev, fn_grid, fp=A.fp)
        sp_1 = eval_spectra(base_m, val_t, dev, amp, fp=A.fp,
                            M_mod=M_mod, M_sizes=M_sizes, fn_grid=fn_grid)
        spectra[f'AddHidden T={new_T} (1-step)'] = sp_1
        if M_mod is not None:
            for ms in (2, 3, 5):
                sp_ms = eval_spectra_multistep(
                    base_m, val_t, dev, amp, M_mod, M_sizes, ms,
                    fn_grid=fn_grid, fp=A.fp)
                spectra[f'AddHidden T={new_T} ({ms}-step)'] = sp_ms

        log("\n  Final 1D spectra (fp=0.01):", lf)
        for name, rows in spectra.items():
            log(f"    {name}:", lf)
            for r in rows:
                log(f"      FN={r['fn']:.2f}: MCC={r['MCC']:.4f} "
                    f"F1={r['F1']:.4f}", lf)

        # The FN x FP plane, not the fixed-fp slice, is what separates
        # coherent-noise removal from plain FN recovery.
        log("\n  2D (FN x FP) spectrum scan:", lf)
        grid2d = eval_spectra_2d(
            base_m, val_t, dev, amp, fn_grid=FN_GRID, fp_grid=FP_GRID,
            M_mod=M_mod, M_sizes=M_sizes)
        cell = {(r['fn'], r['fp']): r for r in grid2d}
        colhdr = ''.join(f'  FP={fp:>4.2f}' for fp in FP_GRID)
        for metric in ('MCC', 'F1', 'fp_removed'):
            log(f"    {metric:<13s}{colhdr}", lf)
            for fn in FN_GRID:
                row = ''.join(f'  {cell[(fn, fp)][metric]:>7.4f}'
                              for fp in FP_GRID)
                log(f"    FN={fn:>4.2f}      {row}", lf)
        sp2d = od / 'spectra_2d.tsv'
        with open(sp2d, 'w') as fh2:
            fh2.write('fn\tfp\tMCC\tF1\tprec\trec\tfp_removed\n')
            for r in grid2d:
                fh2.write(f"{r['fn']:.3f}\t{r['fp']:.3f}\t"
                          f"{r['MCC']:.5f}\t{r['F1']:.5f}\t"
                          f"{r['prec']:.5f}\t{r['rec']:.5f}\t"
                          f"{r['fp_removed']:.5f}\n")
        log(f"  2D spectra TSV: {sp2d}", lf)

        report_dir = od / 'report'
        try:
            generate_report(spectra, fn_grid, report_dir)
            log(f"\n  Report: {report_dir}/", lf)
        except Exception as ex:
            log(f"\n  Report generation failed: {ex}", lf)

        (od / 'done.flag').touch()
        log(f"\n  Wrote {od}/done.flag -- training fully complete", lf)

    barrier(); cleanup()


if __name__ == '__main__':
    main()
