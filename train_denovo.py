#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_denovo.py — De novo tempered training for the denoiser families.

Trains a ModuleConditionedDenoiser (or, per --no-hidden/--higher-order, one of
its siblings) with adaptive_temp=True from scratch; --init-from optionally
warm-starts instead.  The curriculum tempers two axes at once, the FN
distribution and the set of trainable parameters:

  S1     MLM        h, J only                             --
  S2a    MLM        A, U, W + conditioning + aux head + τ  --
  S2b    MLM        all parameters (J at differential LR)  --
  S3a    Denoiser   ising + hidden + scalar skip + cond   beta_low   Beta(1,5)*fn_max
  S3b    Denoiser   as S3a                                beta_mid   Beta(2,3)*fn_max
  S3c    Denoiser   + diagonal gates (w, u, d)            beta_high  Beta(2,1)*fn_max
  S3d    Denoiser   + low-rank gates (V, P)               beta_hard  Beta(5,1)*fn_max
  S3e    Denoiser   gates + conditioning only, J frozen   uniform    U(0, fn_max)

Beta(2,1) (mean 0.67 fn_max) and Beta(5,1) (mean 0.83 fn_max) sit on the
LBCA/LUCA operating point; S3e sweeps the whole spectrum as a fine-tune.

--loss selects wMSE with positive-class reweighting (default) or the
ELBO-derived cross-entropy + Besag pseudo-likelihood (--pl-alpha), which
has no pos_weight to tune.

Usage:
    python -m torch.distributed.run --standalone --nproc_per_node=gpu \\
        train_denovo.py --module-matrix data/module_matrix_kegg.pt --T 8
"""
import argparse, json, time
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from ising_denoiser.models import (
    ModuleConditionedDenoiser, NoHiddenDenoiser,
    HigherOrderDenoiser, NoHiddenHigherOrderDenoiser,
    lowrank_tail_penalty, zeromean_coupling_penalty,
)
from ising_denoiser.modules import (
    load_module_matrix, module_aux_loss, compute_module_fracs
)
from ising_denoiser.data import (
    ReconciliationNoiseDataset, load_feathers, collate_rep
)
from ising_denoiser.metrics import (
    metrics, mstr, eval_spectra, eval_spectra_multistep, FN_GRID
)
from ising_denoiser.training import (
    setup_dist, cleanup, is_main, barrier, log, unwrap,
    broadcast_params,
    save_ckpt, save_ckpt_mid, load_progress, wMSE, WarmupCos, reduce_sum,
    strip_compile_prefix
)
from ising_denoiser.report import eval_null, generate_report


# ----  Loss functions

def loss_wmse(pred: torch.Tensor, target: torch.Tensor,
              pos_w: float) -> torch.Tensor:
    r"""Weighted MSE: present targets cost pos_w times more than absent ones.

    Counteracts the sparsity of +1 entries under high FN corruption.
    Duplicate of ``wMSE()`` in :mod:`ising_denoiser.training` -- keep the
    two in step.
    """
    w = torch.where(target > 0, pos_w, 1.0)
    return (w * (pred - target).pow(2)).mean()


def loss_elbo(x_denoised: torch.Tensor,
              x_clean: torch.Tensor,
              model,
              pl_alpha: float,
              ce_eps: float,
              x_input=None):
    r"""ELBO-derived loss: amortisation cross-entropy + Besag pseudo-likelihood.

        L(θ; x_clean) = L_CE(m; x_clean) + pl_alpha · L_PL(J, h; x_clean)

    with m = x_denoised ∈ (-1, 1)^N the posterior mean, mapped to a
    Bernoulli probability by p = (1 + m)/2.  ``ce_eps`` clamps m away from
    ±1 to keep the log finite.

    For E(x) = -h^T x - (1/2) x^T J x the conditional of spin i is
    σ(2·(h_i + Σ_j J_ij x_j)), the factor 2 being the energy gap
    E(x_i=-1) - E(x_i=+1), so the negative log-conditional is

        softplus(-2 · x_i · (h_i + Σ_j J̃_ij x_j))                eq. (PL.2)

    summed over i in place of the intractable log Z (Besag 1975; consistent
    for (J, h)).  The PL term is evaluated on the clean target, not the
    denoised output: it constrains the prior parameters.  J̃ is the
    symmetrised, zero-diagonal coupling from ``model._Js()``.

    Returns (total, {'ce': float, 'pl': float}).
    """
    # ---- L_CE ----
    m = x_denoised.clamp(-1.0 + ce_eps, 1.0 - ce_eps)
    # Per-state (gain/loss) recalibrated probability, conditioned on the input,
    # when that head is enabled.
    if x_input is not None and getattr(model, 'use_gainloss_cal', False):
        p_pos = model.calibrate_p(x_denoised, x_input).clamp(ce_eps, 1.0 - ce_eps)
    else:
        p_pos = 0.5 * (1.0 + m)
    y_pos = 0.5 * (1.0 + x_clean)
    loss_ce = -(y_pos * torch.log(p_pos) +
                (1.0 - y_pos) * torch.log1p(-p_pos)).mean()

    # ---- L_PL ----
    if hasattr(model, '_Js'):
        J_sym = model._Js()
    else:
        J_sym = 0.5 * (model.J + model.J.T)
        J_sym = J_sym - torch.diag(torch.diag(J_sym))
    local_field = x_clean @ J_sym + model.h.unsqueeze(0)
    loss_pl = F.softplus(-2.0 * x_clean * local_field).mean()            # eq. (PL.2)

    total = loss_ce + pl_alpha * loss_pl
    parts = {'ce': loss_ce.detach().item(),
             'pl': loss_pl.detach().item()}
    return total, parts


def loss_ce_pm1(pred: torch.Tensor, target: torch.Tensor,
                ce_eps: float) -> torch.Tensor:
    """Binary cross-entropy for predictions/targets in +/-1 convention."""
    m = pred.clamp(-1.0 + ce_eps, 1.0 - ce_eps)
    p_pos = 0.5 * (1.0 + m)
    y_pos = 0.5 * (1.0 + target)
    return -(y_pos * torch.log(p_pos) +
             (1.0 - y_pos) * torch.log1p(-p_pos)).mean()


# ----  Stage definitions

# Which parameter groups are trainable per stage.
# Keys match set_trainable() kwargs for ModuleConditionedDenoiser.
STAGE_TRAINABLE = {
    's1':  dict(ising=True,  hidden=False, scalar_skip=False, diag=False, lrs=False, lrf=False, cond=False),
    's2a': dict(ising=False, hidden=True,  scalar_skip=False, diag=False, lrs=False, lrf=False, cond=True),
    's2b': dict(ising=True,  hidden=True,  scalar_skip=False, diag=False, lrs=False, lrf=False, cond=True),
    's3a': dict(ising=True,  hidden=True,  scalar_skip=True,  diag=False, lrs=False, lrf=False, cond=True),
    's3b': dict(ising=True,  hidden=True,  scalar_skip=True,  diag=False, lrs=False, lrf=False, cond=True),
    's3c': dict(ising=True,  hidden=True,  scalar_skip=True,  diag=True,  lrs=False, lrf=False, cond=True),
    's3d': dict(ising=True,  hidden=True,  scalar_skip=True,  diag=True,  lrs=True,  lrf=True,  cond=True),
    's3e': dict(ising=False, hidden=False, scalar_skip=True,  diag=True,  lrs=True,  lrf=True,  cond=True),
}


def _effective_trainable(stage, j_lr_frac):
    # j_lr_frac == 0 freezes J outright rather than training it at lr 0: a
    # trainable J stays in the DDP autograd graph, where the (J + J.T)/2
    # symmetrisation in _Js() trips "parameter J marked ready twice" under
    # the HO attention path (checkpointing + find_unused_parameters).
    t = dict(STAGE_TRAINABLE[stage])
    if j_lr_frac == 0.0 and t.get('ising'):
        t['ising'] = False
    return t


def setup_param_groups(model, stage, base_lr, j_lr_frac, wd, aux_head=None):
    """Create optimizer with per-stage parameter groups.

    J gets j_lr_frac × base_lr when trainable, aux_head parameters base_lr,
    and higher-order attention parameters their own group via
    ``base.make_optimizer`` whenever ``STAGE_TRAINABLE[stage]['attn']``.
    """
    base = unwrap(model)
    trainable = STAGE_TRAINABLE[stage]
    has_attn = hasattr(base, 'attn_params') and trainable.get('attn', False)

    if (trainable['ising'] and j_lr_frac < 1.0) or has_attn:
        opt = base.make_optimizer(base_lr, wd, j_lr_frac)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)

    if aux_head is not None:
        opt.add_param_group({'params': list(aux_head.parameters()), 'lr': base_lr})

    return opt


# ----  MLM training stage (S1, S2a, S2b)

def train_mlm_stage(model, train_t, dev, stage, ne, lr, wd, amp, lf,
                    j_lr_frac=1.0, max_norm=50.0, mp=0.15, bs=2048,
                    aux_head=None, aux_lam=0.0, M_mod=None, M_sizes=None):
    """Single-step pseudolikelihood (MLM) pre-training for one curriculum stage.

    Masks an ``mp`` fraction of genes per genome and predicts each from the
    unmasked context through the model's one-step masked_forward(),

        logits = h + ctx J̃ + tanh(ctx Uᵀ) Aᵀ

    with no T-step iteration, initialising J/h/A/U/W before denoiser
    training.  S2a/S2b additionally train the module-completeness auxiliary
    head (return_z=True exposes the hidden state z) and the adaptive
    temperatures.
    """
    base = unwrap(model)
    base.set_trainable(**_effective_trainable(stage, j_lr_frac))
    use_aux = (aux_head is not None and aux_lam > 0 and M_mod is not None)

    opt = setup_param_groups(model, stage, lr, j_lr_frac, wd,
                             aux_head if use_aux else None)

    ds = TensorDataset(train_t)
    sampler = DistributedSampler(ds, shuffle=True) if dist.is_initialized() else None
    loader = DataLoader(ds, batch_size=bs, shuffle=(sampler is None),
                        sampler=sampler, num_workers=4, pin_memory=True,
                        drop_last=True)
    ws = dist.get_world_size() if dist.is_initialized() else 1

    lr_str = f"lr={lr}"
    if STAGE_TRAINABLE[stage]['ising']:
        if j_lr_frac == 0.0:
            lr_str += " (J frozen)"
        elif j_lr_frac < 1.0:
            lr_str += f" (J at {lr*j_lr_frac:.1e})"
    if use_aux:
        lr_str += f", aux={aux_lam}"
    log(f"\n  [{stage}] MLM (1-step): {ne} ep, {lr_str}, {base.trainable_str()}", lf)
    log(f"        bs={bs}x{ws}gpu = {bs*ws} eff, {len(loader)} steps/ep", lf)

    total = ne * len(loader)
    sched = WarmupCos(opt, int(0.05 * total), total)
    # fp16, not bf16: bf16's 8-bit mantissa accumulates error across the
    # stacked tanh + TAP iterations and costs MCC.  GradScaler covers fp16's
    # narrower dynamic range.
    scaler = GradScaler('cuda', enabled=amp)

    for ep in range(1, ne + 1):
        if sampler:
            sampler.set_epoch(ep)
        t0 = time.time()
        eloss = 0.0
        eaux = 0.0
        nm = 0
        model.train()
        if use_aux:
            aux_head.train()

        for (bx,) in loader:
            x = bx.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=amp):
                if use_aux:
                    lg, tg, n, z = base.masked_forward(x, mp, return_z=True)
                else:
                    lg, tg, n = base.masked_forward(x, mp)

                # Logistic loss on +/-1 targets, in logaddexp form.
                mlm_loss = torch.mean(torch.logaddexp(
                    torch.zeros_like(lg), -2 * tg * lg))

                if use_aux and z is not None:
                    a_loss = module_aux_loss(z, x, aux_head, M_mod, M_sizes)
                    loss = mlm_loss + aux_lam * a_loss
                    eaux += a_loss.item() * n
                else:
                    loss = mlm_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if hasattr(base, 'clamp_norms'):
                base.clamp_norms(max_norm)
            eloss += mlm_loss.item() * n
            nm += n

        st = torch.tensor([eloss, eaux, float(nm)], device=dev, dtype=torch.float64)
        st = reduce_sum(st)
        al = (st[0] / st[2]).item()
        aa = (st[1] / st[2]).item()
        dt = time.time() - t0
        aux_log = f" aux={aa:.4f}" if use_aux else ""
        if ep <= 5 or ep % 10 == 0 or ep == ne:
            log(f"    {ep:>5}/{ne} mlm={al:.4f}{aux_log} {base.norm_str()} "
                f"lr={sched.get_last_lr()[-1]:.1e} [{dt:.0f}s]", lf)

    log(f"  [{stage}] Done: {base.norm_str()}", lf)


# ----  Denoiser training stage (S3a-S3e)

def train_denoiser_stage(model, ds, val_t, dev, stage, ne, lr, wd, amp, lf,
                         bpg, nw, j_lr_frac=1.0, max_norm=50.0,
                         pos_w=3.0, gclip=1.0, fp=0.01,
                         aux_head=None, aux_lam=0.0,
                         M_mod=None, M_sizes=None,
                         ref_sp=None, eval_every=50, ms_steps=0,
                         loss_type='wmse', pl_alpha=0.3, ce_eps=1e-6,
                         delta_lam=0.0, consistency_lambda=0.0,
                         hard_fp_lambda=0.0, self_mask_lambda=0.0,
                         self_mask_prob=0.0, self_mask_value=-1.0,
                         lowrank_lambda=0.0, zeromean_lambda=0.0, lowrank_rank=16,
                         start_ep=1, ckpt_every=0, od=None, prog=None,
                         stage_key=None, val_every=1):
    # ``stage`` is the checkpoint identifier and may carry a leg prefix such
    # as 'T8to12_f1'; ``stage_key`` is the bare stage name used to index
    # STAGE_TRAINABLE, defaulting to ``stage``.
    if stage_key is None:
        stage_key = stage
    """Iterative denoiser training for one stage of the de novo curriculum.

    STAGE_TRAINABLE[stage_key] fixes the trainable parameter groups; ``ds``
    supplies the FN distribution.  Validation runs every ``val_every``
    epochs at FN = 0.50 and 0.85; a full FN spectrum (optionally multi-step,
    optionally against ref_sp) every ``eval_every`` epochs.

    Returns the best FN = 0.85 validation MCC seen in the stage.
    """
    base = unwrap(model)
    base.set_trainable(**_effective_trainable(stage_key, j_lr_frac))
    # The per-state calibration head has no STAGE_TRAINABLE category, so
    # set_trainable() may have frozen it; re-enable it when it is in use.
    if getattr(base, 'use_gainloss_cal', False) and hasattr(base, 'gainloss_cal'):
        base.gainloss_cal.requires_grad_(True)
    use_aux = (aux_lam > 0 and aux_head is not None)
    use_cond = M_mod is not None

    opt = setup_param_groups(model, stage_key, lr, j_lr_frac, wd,
                             aux_head if use_aux else None)

    aux_str = f", aux={aux_lam}" if use_aux else ""
    fn_str = f", fn_dist={ds.fn_dist}" if hasattr(ds, 'fn_dist') else ""
    lr_str = f"lr={lr}"
    if STAGE_TRAINABLE[stage_key]['ising']:
        if j_lr_frac == 0.0:
            lr_str += " (J frozen)"
        elif j_lr_frac < 1.0:
            lr_str += f" (J at {lr*j_lr_frac:.1e})"
    if loss_type == 'elbo':
        loss_desc = f"ELBO(ce+{pl_alpha}*pl)"
    else:
        loss_desc = f"wMSE_{pos_w}"
    extra_loss_desc = []
    if hard_fp_lambda > 0:
        extra_loss_desc.append(f"hardFP={hard_fp_lambda}")
    if self_mask_lambda > 0 and self_mask_prob > 0:
        extra_loss_desc.append(
            f"selfMask={self_mask_lambda}@p{self_mask_prob:g}/x{self_mask_value:g}")
    extra_loss_str = (", " + ", ".join(extra_loss_desc)) if extra_loss_desc else ""
    log(f"\n  [{stage}] {ne} ep, {lr_str}, {base.trainable_str()}, "
        f"{loss_desc}{extra_loss_str}{aux_str}{fn_str}", lf)

    sampler = DistributedSampler(ds, shuffle=True) if dist.is_initialized() else None
    loader = DataLoader(ds, batch_size=bpg, shuffle=(sampler is None), sampler=sampler,
                        num_workers=nw, pin_memory=True, collate_fn=collate_rep,
                        drop_last=False, persistent_workers=(nw > 0),
                        multiprocessing_context='spawn' if nw > 0 else None)
    total = ne * len(loader)
    sched = WarmupCos(opt, int(0.05 * total), total)
    # On resume, fast-forward the scheduler over the optimiser steps taken in
    # earlier runs of this stage, else WarmupCos re-ramps from zero and the
    # cosine decay lands out of step with the epoch number.  Stepping through
    # the public API keeps _LRScheduler's bookkeeping consistent.
    if start_ep > 1:
        for _ in range((start_ep - 1) * len(loader)):
            sched.step()
    # fp16 autocast with GradScaler — see note in train_mlm_stage.
    scaler = GradScaler('cuda', enabled=amp)

    # Fixed per-epoch validation subsets; the per-stage report still uses all
    # of val_t via eval_spectra().
    nv = min(1024, val_t.size(0))
    vc = val_t[:nv].to(dev)
    torch.manual_seed(42); vn = vc.clone()
    r = torch.rand_like(vn); vn[(vc == 1) & (r < 0.5)] = -1
    r2 = torch.rand_like(vn); vn[(vc == -1) & (r2 < fp)] = 1
    torch.manual_seed(99); vn85 = vc.clone()
    r = torch.rand_like(vn85); vn85[(vc == 1) & (r < 0.85)] = -1
    r2 = torch.rand_like(vn85); vn85[(vc == -1) & (r2 < fp)] = 1

    best_mcc85 = 0.0
    # Cached so epochs that skip validation still print a val column, marked '~'.
    last_vm = dict(F1=0, prec=0, rec=0, MCC=0)
    last_vm85 = dict(MCC=0)
    last_cf = 0.0
    if start_ep > 1:
        log(f"  [{stage}] resuming from ep {start_ep}", lf)
    for ep in range(start_ep, ne + 1):
        if sampler:
            sampler.set_epoch(ep)
        t0 = time.time()
        model.train()
        if use_aux:
            aux_head.train()
        ls = 0.0; als = 0.0; ce_s = 0.0; pl_s = 0.0; dls = 0.0; cons_s = 0.0
        hfp_s = 0.0; sm_s = 0.0; nuc_s = 0.0; zm_s = 0.0; nt = 0

        for batch in loader:
            if len(batch) == 3:
                xn, xc, fp_mask = batch
            else:
                xn, xc = batch
                fp_mask = None
            xn = xn.to(dev, non_blocking=True)
            xc = xc.to(dev, non_blocking=True)
            if fp_mask is not None:
                fp_mask = fp_mask.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=amp):
                if use_cond:
                    xd, z_final = model(xn, M_mod, M_sizes)
                else:
                    xd, z_final = model(xn)

                if loss_type == 'wmse':
                    main_loss = loss_wmse(xd, xc, pos_w)
                elif loss_type == 'elbo':
                    main_loss, elbo_parts = loss_elbo(
                        xd, xc, base, pl_alpha=pl_alpha, ce_eps=ce_eps, x_input=xn)
                    ce_s += elbo_parts['ce'] * xn.size(0)
                    pl_s += elbo_parts['pl'] * xn.size(0)
                else:
                    raise ValueError(f"Unknown loss: {loss_type}")

                # Module-completeness auxiliary loss, which needs a hidden state.
                if use_aux and z_final is not None:
                    a_loss = module_aux_loss(z_final, xc, aux_head, M_mod, M_sizes)
                    loss = main_loss + aux_lam * a_loss
                    als += a_loss.item() * xn.size(0)
                else:
                    loss = main_loss

                # Explicit lambda_Delta E||Delta||^2 penalty on the
                # attention-derived higher-order field; off by default.
                if delta_lam > 0 and hasattr(base, 'delta_sq'):
                    d_loss = base.delta_sq(xn)
                    loss = loss + delta_lam * d_loss
                    dls += d_loss.item() * xn.size(0)

                # Cross-input consistency: the K corrupted copies of one genome
                # should denoise to the same output, closing the sparse-build
                # vs dense-trim fixed-point gap.
                if consistency_lambda > 0 and getattr(ds, 'K', 1) > 1:
                    Kc = ds.K
                    xg = xd.view(-1, Kc, xd.size(-1))   # K copies contiguous per genome
                    cons = xg.var(dim=1, unbiased=False).mean()
                    loss = loss + consistency_lambda * cons
                    cons_s += cons.item() * xn.size(0)

                # Injected false positives enter at the same input strength
                # (+1) as genuinely retained genes but carry an absent clean
                # label, so supervise them directly.
                if hard_fp_lambda > 0 and fp_mask is not None and fp_mask.any():
                    hfp = loss_ce_pm1(xd[fp_mask], xc[fp_mask], ce_eps)
                    loss = loss + hard_fp_lambda * hfp
                    hfp_s += hfp.item() * xn.size(0)

            # Coupling-structure regularisers on J~, kept in fp32 outside
            # autocast because the SVD is unstable in fp16.  Zero-mean leaves
            # base rates to h and modular deviations to J.
            if lowrank_lambda > 0:
                nuc = lowrank_tail_penalty(base._Js(), rank=lowrank_rank)
                loss = loss + lowrank_lambda * nuc
                nuc_s += nuc.item() * xn.size(0)
            if zeromean_lambda > 0:
                zm = zeromean_coupling_penalty(base._Js())
                loss = loss + zeromean_lambda * zm
                zm_s += zm.item() * xn.size(0)

            scaler.scale(loss).backward()

            # Self-masked context loss: classify an input-present gene after
            # removing its own observed +1 -- the matched-strength cavity query
            # of scripts/fp_context_diagnostics.py.  This second forward must
            # follow the backward above: two live forwards feeding one backward
            # trip autograd's version check on the reused J~ buffer.  Gradients
            # accumulate across the two backwards, so the objective is unchanged.
            if self_mask_lambda > 0 and self_mask_prob > 0:
                cand = (xn == 1)
                smask = cand & (torch.rand_like(xn) < self_mask_prob)
                if smask.any():
                    xm = xn.clone()
                    xm[smask] = self_mask_value
                    with autocast('cuda', enabled=amp):
                        if use_cond:
                            xd_m, _ = model(xm, M_mod, M_sizes)
                        else:
                            xd_m, _ = model(xm)
                        sm_loss = loss_ce_pm1(xd_m[smask], xc[smask], ce_eps)
                    scaler.scale(self_mask_lambda * sm_loss).backward()
                    sm_s += sm_loss.item() * xn.size(0)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], gclip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if hasattr(base, 'clamp_norms'):
                base.clamp_norms(max_norm)
            bs = xn.size(0)
            ls += main_loss.item() * bs
            nt += bs

        st = torch.tensor([ls, als, float(nt), ce_s, pl_s, dls, cons_s,
                           hfp_s, sm_s],
                          device=dev, dtype=torch.float64)
        st = reduce_sum(st)
        al = (st[0] / st[2]).item()
        aa = (st[1] / st[2]).item()
        a_ce = (st[3] / st[2]).item()
        a_pl = (st[4] / st[2]).item()
        a_dl = (st[5] / st[2]).item()
        a_cons = (st[6] / st[2]).item()
        a_hfp = (st[7] / st[2]).item()
        a_sm = (st[8] / st[2]).item()
        dt = time.time() - t0

        do_val = (val_every <= 1 or ep == start_ep or ep == ne
                  or ep % val_every == 0)
        if is_main() and do_val:
            base.eval()
            with torch.no_grad():
                with autocast('cuda', enabled=amp):
                    if use_cond:
                        xvd, _ = base(vn, M_mod, M_sizes)
                        xvc, _ = base(vc, M_mod, M_sizes)
                        xvd85, _ = base(vn85, M_mod, M_sizes)
                    else:
                        xvd, _ = base(vn)
                        xvc, _ = base(vc)
                        xvd85, _ = base(vn85)
                last_vm = metrics(xvd, vc)
                last_cf = metrics(xvc, vc)['MCC']
                last_vm85 = metrics(xvd85, vc)
            if last_vm85['MCC'] > best_mcc85:
                best_mcc85 = last_vm85['MCC']

        vm, vm85, cf = last_vm, last_vm85, last_cf
        val_tag = '' if do_val else '~'

        star = ' *' if is_main() and do_val and vm85['MCC'] >= best_mcc85 else ''
        aux_log = f" aux={aa:.4f}" if use_aux else ""
        delta_log = f" d2={a_dl:.2e}" if delta_lam > 0 else ""
        cons_log = f" cons={a_cons:.4f}" if consistency_lambda > 0 else ""
        hfp_log = f" hfp={a_hfp:.4f}" if hard_fp_lambda > 0 else ""
        sm_log = f" sm={a_sm:.4f}" if self_mask_lambda > 0 and self_mask_prob > 0 else ""
        elbo_log = ((f" ce={a_ce:.4f} pl={a_pl:.4f}" if loss_type == 'elbo' else "")
                    + delta_log + cons_log + hfp_log + sm_log)
        if do_val:
            val_log = (f" val50[{mstr(vm)}] val85[MCC={vm85['MCC']:.4f}] "
                       f"clean={cf:.4f}")
        else:
            val_log = ""
        log(f"    {ep:>5}/{ne} L={al:.4f}{elbo_log}{aux_log}{val_log} "
            f"lr={sched.get_last_lr()[-1]:.1e} {base.norm_str()} [{dt:.0f}s]{star}", lf)

        if is_main() and eval_every and ep % eval_every == 0:
            sp = eval_spectra(base, val_t, dev, amp, fp=fp,
                              M_mod=M_mod if use_cond else None,
                              M_sizes=M_sizes if use_cond else None)
            if ms_steps > 0 and use_cond:
                sp_ms = eval_spectra_multistep(base, val_t, dev, amp,
                                               M_mod, M_sizes, ms_steps, fp=fp)
                if ref_sp:
                    log(f"\n    {stage} ep{ep}: 1-step vs {ms_steps}-step vs reference", lf)
                    log(f"      {'FN':>6} {'this':>8} {'ms':>7} {'ref':>7} "
                        f"{'d':>7} {'ms-d':>7}", lf)
                    log(f"    {'~'*55}", lf)
                    for rc, rms, rr in zip(sp, sp_ms, ref_sp):
                        log(f"      {rc['fn']:>6.2f} {rc['MCC']:>8.4f} {rms['MCC']:>7.4f} "
                            f"{rr['MCC']:>7.4f} {rc['MCC']-rr['MCC']:>+7.4f} "
                            f"{rms['MCC']-rr['MCC']:>+7.4f}", lf)
            elif ref_sp:
                log(f"\n    {stage} ep{ep}: 1-step vs reference", lf)
                log(f"      {'FN':>6} {'this':>8} {'ref':>7} {'d':>7}", lf)
                log(f"    {'~'*40}", lf)
                for rc, rr in zip(sp, ref_sp):
                    log(f"      {rc['fn']:>6.2f} {rc['MCC']:>8.4f} {rr['MCC']:>7.4f} "
                        f"{rc['MCC']-rr['MCC']:>+7.4f}", lf)
            log("", lf)

        # Mid-stage checkpoint for crash/timeout recovery; skipped on the last
        # epoch, which the caller's save_ckpt() covers.
        if ckpt_every and od is not None and prog is not None \
                and ep < ne and ep % ckpt_every == 0:
            save_ckpt_mid(model, od, stage, ep, prog, lf, aux_head)
            barrier()

    return best_mcc85


# ----  Diagnostics

def param_diagnostics(model, lf):
    """Append one line of parameter norms to the log."""
    base = unwrap(model)
    log(f"  {base.norm_str()}", lf)


def param_diagnostics_full(model, lf):
    """Parameter norms plus the per-timestep gate and temperature table."""
    base = unwrap(model)
    log(f"  {base.norm_str()}", lf)
    log(f"  Gates:\n{base.gate_diagnostics()}", lf)


def main():
    pa = argparse.ArgumentParser(description='De novo tempered training')
    pa.add_argument('--train-feather', default='data/COG_train1_phylum.feather')
    pa.add_argument('--val-feather', default='data/COG_val1_phylum.feather')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    pa.add_argument('--init-from', default=None,
                    help='WARM-START: load model weights from this checkpoint '
                         'instead of fresh init (e.g. a marginal-HQ model_ho3.pth). '
                         'Combine with --s{1,2a,2b,3a,3b,3c}-epochs 0 to skip the '
                         'MLM pretraining + early curriculum and only fine-tune the '
                         'late denoiser stages with the FP-context objectives.')
    pa.add_argument('--T', type=int, default=8)
    pa.add_argument('--H', type=int, default=1000)
    pa.add_argument('--skip-rank', type=int, default=32)
    pa.add_argument('--field-rank', type=int, default=16)
    # MLM stages
    pa.add_argument('--s1-epochs', type=int, default=100)
    pa.add_argument('--s1-lr', type=float, default=1e-3)
    pa.add_argument('--s2a-epochs', type=int, default=100)
    pa.add_argument('--s2a-lr', type=float, default=5e-4)
    pa.add_argument('--s2b-epochs', type=int, default=200)
    pa.add_argument('--s2b-lr', type=float, default=3e-4)
    # Denoiser stages
    pa.add_argument('--s3a-epochs', type=int, default=200)
    pa.add_argument('--s3a-lr', type=float, default=1e-4)
    pa.add_argument('--s3b-epochs', type=int, default=150)
    pa.add_argument('--s3b-lr', type=float, default=8e-5)
    pa.add_argument('--s3c-epochs', type=int, default=150)
    pa.add_argument('--s3c-lr', type=float, default=5e-5)
    pa.add_argument('--s3d-epochs', type=int, default=150)
    pa.add_argument('--s3d-lr', type=float, default=3e-5)
    pa.add_argument('--s3e-epochs', type=int, default=100)
    pa.add_argument('--s3e-lr', type=float, default=1e-5)
    # General
    pa.add_argument('--j-lr-frac', type=float, default=1.0,
                    help='J learning rate fraction (for denoiser stages)')
    pa.add_argument('--s2b-j-lr-frac', type=float, default=1.0,
                    help='J learning rate fraction for S2b MLM joint stage')
    pa.add_argument('--aux-lambda', type=float, default=0.3)
    pa.add_argument('--fn-max', type=float, default=0.9)
    pa.add_argument('--fp', type=float, default=0.01)
    pa.add_argument('--pos-w', type=float, default=3.0)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--clean-frac', type=float, default=0.125)
    pa.add_argument('--batch-per-gpu', type=int, default=8000)
    pa.add_argument('--mlm-batch', type=int, default=2048)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--weight-decay', type=float, default=1e-4)
    pa.add_argument('--max-norm', type=float, default=50.0)
    pa.add_argument('--ms-steps', type=int, default=3)
    pa.add_argument('--outdir', default='gsd_results_denovo')
    pa.add_argument('--loss', type=str, default='wmse',
                    choices=['wmse', 'elbo'],
                    help='Loss function. "wmse" (default) is the current '
                         'weighted MSE with positive-class reweighting. '
                         '"elbo" is the ELBO-derived cross-entropy + Besag '
                         'pseudo-likelihood loss.')
    pa.add_argument('--pl-alpha', type=float, default=0.3,
                    help='Weight of the Besag pseudo-likelihood term on J, h '
                         'when --loss elbo. Ignored otherwise.')
    pa.add_argument('--ce-eps', type=float, default=1e-6,
                    help='Clamp for tanh outputs in cross-entropy loss. '
                         'Prevents log(0). Ignored when --loss wmse.')
    pa.add_argument('--no-hidden', action='store_true',
                    help='Disable hidden spins (A, U, W) in all stages. '
                         'Module conditioning is kept.')
    pa.add_argument('--higher-order', action='store_true',
                    help='Train a HigherOrderDenoiser de novo. With '
                         '--no-hidden this builds NoHiddenHigherOrderDenoiser; '
                         'otherwise it builds HigherOrderDenoiser.')
    pa.add_argument('--attn-start-stage', default='s3c',
                    choices=['s3a', 's3b', 's3c', 's3d', 's3e'],
                    help='First denoiser stage where higher-order attention '
                         'parameters are trainable. Earlier stages keep the '
                         'zero-init Delta path frozen.')
    pa.add_argument('--attn-d-model',  type=int, default=128)
    pa.add_argument('--attn-nhead',    type=int, default=4)
    pa.add_argument('--attn-n-layers', type=int, default=1)
    pa.add_argument('--attn-dim-ff',   type=int, default=512)
    pa.add_argument('--attn-dropout',  type=float, default=0.1)
    pa.add_argument('--attn-lr-frac',  type=float, default=0.3,
                    help='Attention LR = base LR times this fraction.')
    pa.add_argument('--no-attn-checkpoint', action='store_true',
                    help='Disable gradient checkpointing inside the HO '
                         'attention block.')
    pa.add_argument('--onsager', default='none',
                    choices=['none', 'within', 'full', 'tied'],
                    help='Onsager/TAP correction: "none" (naive MF), '
                         '"within" (J²/W² only), '
                         '"full" (+ cross A²/U² with independent A, U), '
                         '"tied" (+ cross A² with U=Aᵀ; single-matrix EBM).')
    pa.add_argument('--compile', action='store_true',
                    help='torch.compile the model before DDP wrapping '
                         '(max-autotune mode; ~60s warmup, 15-25%% faster).')
    pa.add_argument('--fp-max', type=float, default=None,
                    help='If set, draw per-copy FP rates from [0, fp-max] '
                         'using --fp-dist instead of fixed --fp.')
    pa.add_argument('--fp-dist', type=str, default='uniform',
                    choices=['uniform', 'beta_low', 'beta_high'])
    pa.add_argument('--fp-mode', type=str, default='uniform',
                    choices=['uniform', 'marginal'],
                    help='How ordinary per-copy FPs are sampled. marginal '
                         'draws absent COGs proportional to per-COG training '
                         'prevalence, matching the decisive extant test.')
    pa.add_argument('--mod-inject-rate', type=float, default=0.0,
                    help='Per corrupted copy, probability of injecting one '
                         'absent functional module as coherent FP.')
    pa.add_argument('--mod-swap-rate', type=float, default=0.0,
                    help='Per corrupted copy, probability of grafting modules '
                         'from another training genome.')
    pa.add_argument('--mod-delete-rate', type=float, default=0.0,
                    help='Per corrupted copy, probability of deleting one '
                         'present module as coherent FN.')
    pa.add_argument('--mod-swap-max', type=int, default=3)
    pa.add_argument('--marginal-fp-rate', type=float, default=0.0,
                    help='Probability that a corrupted copy is a low-FN, '
                         'dense marginal-FP prune example.')
    pa.add_argument('--marginal-fp-max', type=float, default=0.0,
                    help='Max marginal FP excess fraction of true gene count.')
    pa.add_argument('--marginal-fp-fn', type=float, default=0.1,
                    help='Max FN rate on marginal-prune copies.')
    pa.add_argument('--marginal-freq-feather', default=None,
                    help='Optional feather used to compute the per-COG '
                         'marginal contamination prior.')
    pa.add_argument('--lambda-delta', type=float, default=0.0,
                    help='Optional E||Delta||^2 regulariser when the model '
                         'exposes delta_sq.')
    pa.add_argument('--consistency-lambda', type=float, default=0.0,
                    help='Cross-input consistency weight across K corrupted '
                         'copies of each clean genome.')
    pa.add_argument('--hard-fp-lambda', type=float, default=0.0,
                    help='Extra CE penalty on labelled injected-FP positions.')
    pa.add_argument('--self-mask-lambda', type=float, default=0.0,
                    help='Extra CE loss on input-present genes after masking '
                         'their own observed +1 evidence.')
    pa.add_argument('--self-mask-prob', type=float, default=0.0,
                    help='Probability of self-masking each input-present gene '
                         'when --self-mask-lambda > 0.')
    pa.add_argument('--self-mask-value', type=float, default=-1.0,
                    choices=[-1.0, 0.0],
                    help='Value used for self-masked input-present genes.')
    # Coupling-discipline controls (cross-clade transfer).
    pa.add_argument('--genome-mass-norm', action='store_true',
                    help='Bake in sqrt(kbar/k_n) coupling-drive normalisation '
                         '(kbar = sum_c p_c from --gmass-freq-tsv).')
    pa.add_argument('--gmass-freq-tsv', default='data/cog_marginal_frequency.tsv',
                    help='Per-COG presence frequency p_c for genome-mass norm.')
    pa.add_argument('--lowrank-lambda', type=float, default=0.0,
                    help='Nuclear-norm penalty on J~ (toward a transferable low rank).')
    pa.add_argument('--zeromean-lambda', type=float, default=0.0,
                    help='Penalty on each genes total coupling drive (zero-mean J~).')
    pa.add_argument('--per-state-cal', action='store_true',
                    help='Enable the trainable per-state gain/loss output calibration head.')
    A = pa.parse_args()

    if A.no_hidden:
        for stg in STAGE_TRAINABLE:
            STAGE_TRAINABLE[stg]['hidden'] = False
    if A.higher_order:
        order = ['s3a', 's3b', 's3c', 's3d', 's3e']
        start = order.index(A.attn_start_stage)
        for stg in STAGE_TRAINABLE:
            STAGE_TRAINABLE[stg]['attn'] = stg in order[start:]

    rank, ws, lr, dev = setup_dist()
    amp = True

    # TF32 for the fp32 matmul paths (e.g. the Besag PL term) that the fp16
    # autocast inside the stage loops does not cover.
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
    log(f"  DE NOVO TEMPERED TRAINING — {time.strftime('%Y-%m-%d %H:%M')}", lf)
    log(f"  T={A.T}, adaptive_temp=True, ms_steps={A.ms_steps}, loss={A.loss}, onsager={A.onsager}, compile={A.compile}", lf)
    log(f"  aux={A.aux_lambda}, module_matrix={A.module_matrix}", lf)
    log(f"  {ws} x {torch.cuda.get_device_name(0)}", lf)
    log(f"{'='*72}", lf)

    train_t, val_t, vocab = load_feathers(A.train_feather, A.val_feather)
    N = len(vocab)
    log(f"\n  Train: {train_t.shape[0]:,} x {N}  Val: {val_t.shape[0]:,} x {N}", lf)

    marginal_freq = None
    if A.marginal_freq_feather:
        import pandas as pd
        mdf = pd.read_feather(A.marginal_freq_feather)
        mcols = [c for c in mdf.columns if c.startswith('COG')]
        pcm = dict(zip(mcols, (mdf[mcols].values > 0).mean(axis=0)))
        marginal_freq = torch.tensor([float(pcm.get(c, 0.0)) for c in vocab],
                                     dtype=torch.float32)
        log(f"  marginal-freq override: {A.marginal_freq_feather} "
            f"({mdf.shape[0]:,} genomes; {int((marginal_freq >= 0.9).sum())} "
            f"COGs p_c>=0.9, {int((marginal_freq > 0).sum())} >0)", lf)

    aux_head = None; M_mod = None; M_sizes = None; n_mod = 1
    if Path(A.module_matrix).exists():
        aux_head, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, A.H)
        n_mod = M_mod.shape[1]
        log(f"  Module matrix: {n_mod} targets, "
            f"aux head: {sum(p.numel() for p in aux_head.parameters()):,} params", lf)

    ref_sp = None
    barrier()

    def make_noise_dataset(fn_dist, clean_frac=None, hard_fp=False):
        use_module_noise = (
            A.mod_inject_rate > 0
            or A.mod_swap_rate > 0
            or A.mod_delete_rate > 0
        )
        return ReconciliationNoiseDataset(
            train_t, A.fn_max, A.fp, A.K,
            A.clean_frac if clean_frac is None else clean_frac,
            fn_dist,
            fp_max=A.fp_max, fp_dist=A.fp_dist, fp_mode=A.fp_mode,
            module_membership=(M_mod.cpu() if use_module_noise and M_mod is not None else None),
            module_sizes=(M_sizes.cpu() if use_module_noise and M_sizes is not None else None),
            module_inject_rate=A.mod_inject_rate,
            module_swap_rate=A.mod_swap_rate,
            module_delete_rate=A.mod_delete_rate,
            module_swap_max=A.mod_swap_max,
            marginal_fp_rate=A.marginal_fp_rate,
            marginal_fp_max=A.marginal_fp_max,
            marginal_fp_fn=A.marginal_fp_fn,
            marginal_freq=marginal_freq,
            return_masks=hard_fp or A.hard_fp_lambda > 0,
        )

    def denoiser_loss_kwargs():
        return dict(
            delta_lam=A.lambda_delta,
            consistency_lambda=A.consistency_lambda,
            hard_fp_lambda=A.hard_fp_lambda,
            self_mask_lambda=A.self_mask_lambda,
            self_mask_prob=A.self_mask_prob,
            self_mask_value=A.self_mask_value,
            lowrank_lambda=A.lowrank_lambda,
            zeromean_lambda=A.zeromean_lambda,
        )

    onsager_val = A.onsager if A.onsager != 'none' else False
    attn_kwargs = dict(
        attn_d_model=A.attn_d_model,
        attn_nhead=A.attn_nhead,
        attn_n_layers=A.attn_n_layers,
        attn_dim_feedforward=A.attn_dim_ff,
        attn_dropout=A.attn_dropout,
        attn_use_checkpoint=not A.no_attn_checkpoint,
    )
    if A.no_hidden:
        # Without hidden units there is no U, so 'within', 'full' and 'tied'
        # all collapse to within-block TAP.
        if A.onsager == 'tied':
            log("  Warning: --onsager tied has no effect on --no-hidden "
                "(no U matrix); using within-block TAP only.", lf)
        if A.higher_order:
            _build = lambda: NoHiddenHigherOrderDenoiser(
                N, A.T, A.skip_rank, A.field_rank,
                n_modules=n_mod, adaptive_temp=True, onsager=bool(onsager_val),
                **attn_kwargs
            ).to(dev)
        else:
            _build = lambda: NoHiddenDenoiser(
                N, A.T, A.skip_rank, A.field_rank,
                n_modules=n_mod, adaptive_temp=True, onsager=bool(onsager_val)
            ).to(dev)
    else:
        if A.higher_order:
            _build = lambda: HigherOrderDenoiser(
                N, A.H, A.T, A.skip_rank, A.field_rank,
                n_modules=n_mod, adaptive_temp=True, onsager=onsager_val,
                **attn_kwargs
            ).to(dev)
        else:
            _build = lambda: ModuleConditionedDenoiser(
                N, A.H, A.T, A.skip_rank, A.field_rank,
                n_modules=n_mod, adaptive_temp=True, onsager=onsager_val
            ).to(dev)

    if not done:
        raw = _build()
        if A.init_from:
            ick = Path(A.init_from)
            if not ick.exists():
                raise FileNotFoundError(f"--init-from checkpoint not found: {ick}")
            sd = strip_compile_prefix(
                torch.load(ick, map_location=dev, weights_only=True))
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            missing, unexpected = raw.load_state_dict(sd, strict=False)
            log(f"\n  WARM-START from {ick}: {raw.config_str()}", lf)
            log(f"    load: {len(missing)} missing, {len(unexpected)} unexpected keys", lf)
            if unexpected:
                log(f"    unexpected[:6]: {list(unexpected)[:6]}", lf)
            if missing:
                log(f"    missing[:6]: {list(missing)[:6]}", lf)
        else:
            log(f"\n  Fresh model: {raw.config_str()}", lf)
    else:
        last = prog['completed'][-1]
        raw = _build()
        ck = od / f'model_{last}.pth'
        if ck.exists():
            sd = strip_compile_prefix(
                torch.load(ck, map_location=dev, weights_only=True))
            raw.load_state_dict(sd, strict=False)
            log(f"  Resumed from: {ck}", lf)
        if aux_head is not None:
            ack = od / f'aux_{last}.pth'
            if ack.exists():
                try:
                    aux_head.load_state_dict(
                        torch.load(ack, map_location=dev, weights_only=True))
                except RuntimeError:
                    pass

    if A.compile:
        log(f"  torch.compile(mode='max-autotune-no-cudagraphs') — warmup on first batch", lf)
        raw = torch.compile(raw, mode='max-autotune-no-cudagraphs')

    if A.higher_order:
        unwrap(raw).attn_lr_frac = A.attn_lr_frac
        log(f"  higher-order attention: start={A.attn_start_stage}, "
            f"d={A.attn_d_model}, heads={A.attn_nhead}, "
            f"L={A.attn_n_layers}, lr_frac={A.attn_lr_frac}", lf)

    broadcast_params(raw)
    model = DDP(raw, device_ids=[lr], find_unused_parameters=True)
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

    def hdr(name, desc, ep):
        log(f"\n{'='*72}\n  {name}: {desc} ({ep} ep) — "
            f"{time.strftime('%H:%M')} ({(time.time()-t0g)/60:.0f}m)\n{'='*72}", lf)

    def stage_eval(name):
        if is_main():
            sp = eval_spectra(base, val_t, dev, amp, fp=A.fp,
                              M_mod=M_mod, M_sizes=M_sizes)
            if A.ms_steps > 0 and M_mod is not None:
                sp_ms = eval_spectra_multistep(
                    base, val_t, dev, amp, M_mod, M_sizes, A.ms_steps, fp=A.fp)
            else:
                sp_ms = sp
            if ref_sp:
                log(f"\n    {name}: 1-step vs {A.ms_steps}-step vs reference", lf)
                log(f"      {'FN':>6} {'this':>8} {'ms':>7} {'ref':>7} "
                    f"{'d':>7} {'ms-d':>7}", lf)
                log(f"    {'~'*55}", lf)
                for rc, rms, rr in zip(sp, sp_ms, ref_sp):
                    log(f"      {rc['fn']:>6.2f} {rc['MCC']:>8.4f} {rms['MCC']:>7.4f} "
                        f"{rr['MCC']:>7.4f} {rc['MCC']-rr['MCC']:>+7.4f} "
                        f"{rms['MCC']-rr['MCC']:>+7.4f}", lf)
            else:
                log(f"\n    {name}: spectra", lf)
                for r in sp:
                    log(f"      FN={r['fn']:.2f}: MCC={r['MCC']:.4f} F1={r['F1']:.4f}", lf)
            param_diagnostics_full(model, lf)

    # ----  Stage schedule

    if 's1' not in done:
        hdr('S1', 'MLM — h, J only', A.s1_epochs)
        train_mlm_stage(model, train_t, dev, 's1',
                        ne=A.s1_epochs, lr=A.s1_lr, wd=A.weight_decay,
                        amp=amp, lf=lf, max_norm=A.max_norm, bs=A.mlm_batch)
        param_diagnostics(model, lf)
        save_ckpt(model, od, 's1', prog, lf); barrier()
    else:
        log(f"\n  Skip S1", lf)

    if 's2a' not in done:
        hdr('S2a', 'MLM — A, U, W + cond + aux + τ', A.s2a_epochs)
        train_mlm_stage(model, train_t, dev, 's2a',
                        ne=A.s2a_epochs, lr=A.s2a_lr, wd=A.weight_decay,
                        amp=amp, lf=lf, max_norm=A.max_norm, bs=A.mlm_batch,
                        aux_head=aux_head, aux_lam=A.aux_lambda,
                        M_mod=M_mod, M_sizes=M_sizes)
        param_diagnostics(model, lf)
        save_ckpt(model, od, 's2a', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S2a", lf)

    if 's2b' not in done:
        hdr('S2b', 'MLM — joint', A.s2b_epochs)
        train_mlm_stage(model, train_t, dev, 's2b',
                        ne=A.s2b_epochs, lr=A.s2b_lr, wd=A.weight_decay,
                        amp=amp, lf=lf, max_norm=A.max_norm, bs=A.mlm_batch,
                        j_lr_frac=A.s2b_j_lr_frac,
                        aux_head=aux_head, aux_lam=A.aux_lambda,
                        M_mod=M_mod, M_sizes=M_sizes)
        param_diagnostics(model, lf)
        save_ckpt(model, od, 's2b', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S2b", lf)

    if 's3a' not in done:
        hdr('S3a', 'beta_low — core + scalar skip + cond', A.s3a_epochs)
        ds = make_noise_dataset('beta_low')
        train_denoiser_stage(model, ds, val_t, dev, 's3a',
                             ne=A.s3a_epochs, lr=A.s3a_lr, wd=A.weight_decay,
                             amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
                             j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
                             pos_w=A.pos_w, fp=A.fp,
                             aux_head=aux_head, aux_lam=A.aux_lambda,
                             M_mod=M_mod, M_sizes=M_sizes,
                             ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
                             loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
                             **denoiser_loss_kwargs())
        stage_eval('S3a'); save_ckpt(model, od, 's3a', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S3a", lf)

    if 's3b' not in done:
        hdr('S3b', 'beta_mid — core + scalar skip + cond', A.s3b_epochs)
        ds = make_noise_dataset('beta_mid')
        train_denoiser_stage(model, ds, val_t, dev, 's3b',
                             ne=A.s3b_epochs, lr=A.s3b_lr, wd=A.weight_decay,
                             amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
                             j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
                             pos_w=A.pos_w, fp=A.fp,
                             aux_head=aux_head, aux_lam=A.aux_lambda,
                             M_mod=M_mod, M_sizes=M_sizes,
                             ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
                             loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
                             **denoiser_loss_kwargs())
        stage_eval('S3b'); save_ckpt(model, od, 's3b', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S3b", lf)

    if 's3c' not in done:
        hdr('S3c', 'beta_high — + diag gates', A.s3c_epochs)
        ds = make_noise_dataset('beta_high')
        train_denoiser_stage(model, ds, val_t, dev, 's3c',
                             ne=A.s3c_epochs, lr=A.s3c_lr, wd=A.weight_decay,
                             amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
                             j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
                             pos_w=A.pos_w, fp=A.fp,
                             aux_head=aux_head, aux_lam=A.aux_lambda,
                             M_mod=M_mod, M_sizes=M_sizes,
                             ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
                             loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
                             **denoiser_loss_kwargs())
        stage_eval('S3c'); save_ckpt(model, od, 's3c', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S3c", lf)

    if 's3d' not in done:
        hdr('S3d', 'beta_hard — all params', A.s3d_epochs)
        ds = make_noise_dataset('beta_hard')
        train_denoiser_stage(model, ds, val_t, dev, 's3d',
                             ne=A.s3d_epochs, lr=A.s3d_lr, wd=A.weight_decay,
                             amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
                             j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
                             pos_w=A.pos_w, fp=A.fp,
                             aux_head=aux_head, aux_lam=A.aux_lambda,
                             M_mod=M_mod, M_sizes=M_sizes,
                             ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
                             loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
                             **denoiser_loss_kwargs())
        stage_eval('S3d'); save_ckpt(model, od, 's3d', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S3d", lf)

    if 's3e' not in done:
        hdr('S3e', 'uniform — gates + cond only (finetune)', A.s3e_epochs)
        ds = make_noise_dataset('uniform', clean_frac=0.0)
        train_denoiser_stage(model, ds, val_t, dev, 's3e',
                             ne=A.s3e_epochs, lr=A.s3e_lr, wd=A.weight_decay,
                             amp=amp, lf=lf, bpg=A.batch_per_gpu, nw=A.num_workers,
                             j_lr_frac=A.j_lr_frac, max_norm=A.max_norm,
                             pos_w=A.pos_w, fp=A.fp,
                             aux_head=aux_head, aux_lam=A.aux_lambda,
                             M_mod=M_mod, M_sizes=M_sizes,
                             ref_sp=ref_sp, eval_every=50, ms_steps=A.ms_steps,
                             loss_type=A.loss, pl_alpha=A.pl_alpha, ce_eps=A.ce_eps,
                             **denoiser_loss_kwargs())
        stage_eval('S3e'); save_ckpt(model, od, 's3e', prog, lf, aux_head); barrier()
    else:
        log(f"\n  Skip S3e", lf)

    # ----  Final spectra and report
    if is_main():
        e = (time.time() - t0g) / 60
        log(f"\n{'='*72}\n  FINAL — {e:.0f}m\n{'='*72}", lf)
        param_diagnostics_full(model, lf)
        if hasattr(base, 'delta_norm'):
            nprobe = min(64, val_t.size(0))
            x_probe = val_t[:nprobe].to(dev).float()
            try:
                log(f"  Delta norm: {base.delta_norm(x_probe):.4e}", lf)
            except TypeError:
                z_probe = x_probe.new_zeros(x_probe.size(0), getattr(base, 'H', A.H))
                log(f"  Delta norm: {base.delta_norm(x_probe, z_probe):.4e}", lf)

        fn_grid = FN_GRID
        spectra = OrderedDict()

        # Null baseline: the noisy input scored unchanged.
        spectra['Noisy input'] = eval_null(val_t, dev, fn_grid, fp=A.fp)

        if ref_sp:
            spectra['Reference'] = ref_sp

        sp_1 = eval_spectra(base, val_t, dev, amp, fp=A.fp,
                            M_mod=M_mod, M_sizes=M_sizes, fn_grid=fn_grid)
        model_label = 'De novo HO' if A.higher_order else 'De novo'
        spectra[f'{model_label} (1-step)'] = sp_1

        if M_mod is not None:
            for ms in [2, 3, 5]:
                sp_ms = eval_spectra_multistep(base, val_t, dev, amp,
                                               M_mod, M_sizes, ms,
                                               fn_grid=fn_grid, fp=A.fp)
                spectra[f'{model_label} ({ms}-step)'] = sp_ms

        log(f"\n  Final spectra:", lf)
        for name, rows in spectra.items():
            log(f"    {name}:", lf)
            for r in rows:
                log(f"      FN={r['fn']:.2f}: MCC={r['MCC']:.4f} F1={r['F1']:.4f} "
                    f"P={r['prec']:.3f} R={r['rec']:.3f}", lf)

        report_dir = od / 'report'
        try:
            generate_report(spectra, fn_grid, report_dir)
            log(f"\n  Report: {report_dir}/", lf)
        except Exception as ex:
            log(f"\n  Report generation failed: {ex}", lf)

    # Report generation on rank 0 can exhaust CUDA memory and make
    # destroy_process_group fail, so free it before NCCL teardown.
    torch.cuda.empty_cache()
    barrier(); cleanup()


if __name__ == '__main__':
    main()
