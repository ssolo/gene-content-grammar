"""
Training infrastructure shared by the denoiser trainers.

  Distributed training    setup_dist(), cleanup(), barrier(), broadcast_params(),
                          reduce_sum() — NCCL multi-GPU via DDP.

  Loss helpers            wMSE — weighted MSE, present genes upweighted.  The
                          stage loops and the ELBO / Besag-pseudolikelihood
                          terms live in train_denovo.py.

  LR schedule             WarmupCos — linear warmup then cosine decay.
  EMA                     EMAModel — moving average of model weights.

  Checkpoint management   save_ckpt(), save_ckpt_mid(), load_progress() —
                          per-stage checkpoints plus progress.json for resume.

  T extension             load_and_extend() — build a model at T' >= T and
                          initialise it from a smaller-T checkpoint.
"""
import json, math, os, time, warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .data import collate_rep
from .metrics import metrics, mstr, eval_spectra, eval_spectra_multistep
from .metrics import print_three_way, print_four_way
from .modules import module_aux_loss
# ---- Distributed setup

def setup_dist():
    """Returns (rank, world_size, local_rank, device) after init_process_group."""
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(lr)
    return rank, ws, lr, torch.device(f'cuda:{lr}')


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_params(model):
    if not dist.is_initialized():
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def reduce_sum(t):
    if not dist.is_initialized():
        return t
    r = t.clone().detach()
    dist.all_reduce(r, op=dist.ReduceOp.SUM)
    return r


def log(msg, lf=None):
    """Print, and append to ``lf`` if given; rank 0 only."""
    if is_main():
        print(msg, flush=True)
        if lf:
            with open(lf, 'a') as f:
                f.write(msg + '\n')


def unwrap(m):
    """Unwrap DDP/DataParallel/torch.compile wrappers.

    torch.compile's ``OptimizedModule`` prefixes every state_dict key with
    ``_orig_mod.``; loading such a state_dict into an uncompiled model with
    ``strict=False`` silently drops every parameter.
    """
    if isinstance(m, (DDP, nn.DataParallel)):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


class EMAModel:
    """Exponential moving average of model parameters, for smoother validation.

    Update after each optimiser step; ``apply()`` is a context manager that
    swaps the shadow weights in and restores the live ones on exit::

        ema.update(model)
        with ema.apply(model):
            val_loss = evaluate(model)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for k, v in unwrap(model).state_dict().items():
            self.shadow[k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for k, v in unwrap(model).state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v, 1 - self.decay)

    class _ApplyContext:
        def __init__(self, ema, model):
            self.ema = ema
            self.model = model
        def __enter__(self):
            base = unwrap(self.model)
            self.ema.backup = {k: v.clone() for k, v in base.state_dict().items()}
            base.load_state_dict(self.ema.shadow, strict=False)
            return self.model
        def __exit__(self, *args):
            unwrap(self.model).load_state_dict(self.ema.backup, strict=False)
            self.ema.backup = {}

    def apply(self, model):
        return self._ApplyContext(self, model)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ---- Loss and scheduler

def wMSE(pred, target, pos_w=3.0):
    """Mean squared error with present genes (target > 0) weighted by pos_w.

    Presence/absence is strongly imbalanced — roughly 30 % of COGs present
    (+1) in a typical genome, far fewer under heavy false-negative corruption
    — so an unweighted loss is dominated by the absent class.
    """
    w = torch.where(target > 0, pos_w, 1.0)
    return (w * (pred - target).pow(2)).mean()


class WarmupCos(torch.optim.lr_scheduler._LRScheduler):
    r"""Linear warmup → cosine decay learning-rate schedule.

    Two phases over ``total`` optimiser steps, crossing over at ``warmup``::

        t < warmup:          a(t) = t / warmup
        t ∈ [warmup, total]: a(t) = 0.5·(1 + cos(π · (t-warmup)/(total-warmup)))

        lr(t) = eta + (lr_base - eta) · a(t)

    ``lr_base`` is the per-param-group LR registered with the optimiser
    (``base_lrs``); ``eta`` is the floor.  ``warmup=0`` skips the ramp.

    Resume: ``last_epoch`` is the step counter (-1 ⇒ uninitialised).  To resume
    after ``k`` optimiser steps either pass ``last_epoch=k`` or call
    ``sched.step()`` k times after construction; ``train_denoiser_stage`` does
    the latter.
    """
    def __init__(self, opt, warmup, total, eta=1e-7, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.eta = eta
        # _LRScheduler.__init__ steps once internally; suppress its
        # "step() after optimizer.step()" hint, which doesn't apply here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(opt, last_epoch)

    def get_lr(self):
        # last_epoch is incremented before get_lr() runs, so st is the 1-based
        # step count.  The max() guards divide-by-zero at warmup=0 and
        # warmup==total.
        st = self.last_epoch
        if st < self.warmup:
            a = st / max(self.warmup, 1)
        else:
            a = 0.5 * (1 + math.cos(math.pi * (st - self.warmup) /
                                      max(self.total - self.warmup, 1)))
        return [self.eta + (b - self.eta) * a for b in self.base_lrs]


# ---- Checkpoint management

def strip_compile_prefix(sd):
    """Strip torch.compile's '_orig_mod.' prefix from state_dict keys.

    No-op when the prefix is absent.  Applied at every load site: checkpoints
    written from a compiled model carry the prefix and match nothing under
    strict=False.
    """
    if any(k.startswith('_orig_mod.') for k in sd):
        return {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    return sd


def load_model_ckpt(raw, ckpt_path, dev, lf=None, label=''):
    """Load a model state_dict into ``raw``, stripping any compile prefix.

    Warns about missing keys, which strict=False would otherwise hide.
    Returns the (missing, unexpected) lists from load_state_dict.
    """
    sd = torch.load(ckpt_path, map_location=dev, weights_only=True)
    sd = strip_compile_prefix(sd)
    missing, unexpected = raw.load_state_dict(sd, strict=False)
    if missing and len(missing) > 5 and lf is not None:
        log(f"  WARNING [{label}]: {len(missing)} model params missing "
            f"from {ckpt_path}: {missing[:5]}...", lf)
    return missing, unexpected


def _canonical_onsager(model):
    """Canonical onsager string ('none'/'within'/'full'/'tied') for a model.

    NoHiddenDenoiser stores onsager as a bool (True = J² TAP); True maps to
    'full' because 'full' and 'within' coincide when there is no hidden block.
    """
    o = getattr(unwrap(model), 'onsager', None)
    if o is None or o is False:
        return 'none'
    if o is True:
        return 'full'
    return o


def _ckpt_cfg_path(ckpt_path):
    return Path(ckpt_path).with_suffix('.cfg.json')


def _save_ckpt_cfg(model, ckpt_path):
    """Write the model_<name>.cfg.json sidecar.

    Records construction fields that change topology or iteration semantics
    but are not recoverable from a bare state_dict, chiefly ``onsager``.
    """
    base = unwrap(model)
    cfg = {
        'onsager': _canonical_onsager(model),
        'no_hidden': type(base).__name__.startswith('NoHidden'),
        'T': int(getattr(base, 'T', 0)),
    }
    with open(_ckpt_cfg_path(ckpt_path), 'w') as f:
        json.dump(cfg, f, indent=2)


def load_ckpt_cfg(ckpt_path):
    """Read sidecar metadata for model_<name>.pth, or {} if missing.

    Keys when present: 'onsager' (str), 'no_hidden' (bool), 'T' (int).
    Callers must treat {} as "no information" and fall back to the CLI
    defaults.
    """
    p = _ckpt_cfg_path(ckpt_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_ckpt(model, od, name, prog, lf, aux_head=None):
    """Save a completed stage's checkpoint and update progress (rank 0 only).

      model_{name}.pth      model state dict
      model_{name}.cfg.json sidecar with onsager / no_hidden / T
      aux_{name}.pth        aux head state dict, if present
      progress.json         list of completed stage names

    The in-progress marker is cleared, so the next launch skips this stage.
    """
    if not is_main():
        return
    p = od / f'model_{name}.pth'
    torch.save(unwrap(model).state_dict(), p)
    _save_ckpt_cfg(model, p)
    if aux_head is not None:
        torch.save(aux_head.state_dict(), od / f'aux_{name}.pth')
    prog['completed'].append(name)
    prog.pop('in_progress', None)
    with open(od / 'progress.json', 'w') as f:
        json.dump(prog, f, indent=2)
    log(f"  Saved: {p}", lf)


def save_ckpt_mid(model, od, name, ep, prog, lf, aux_head=None):
    """Save a mid-stage checkpoint for resume (rank 0 only).

    Overwrites model_{name}.pth / model_{name}.cfg.json / aux_{name}.pth and
    stamps progress.json with {'in_progress': {'stage': name, 'epoch': ep}}.
    The stage is not added to 'completed': on the next launch, if
    in_progress.stage == name, training resumes from epoch ep + 1.
    """
    if not is_main():
        return
    p = od / f'model_{name}.pth'
    torch.save(unwrap(model).state_dict(), p)
    _save_ckpt_cfg(model, p)
    if aux_head is not None:
        torch.save(aux_head.state_dict(), od / f'aux_{name}.pth')
    prog['in_progress'] = {'stage': name, 'epoch': int(ep)}
    with open(od / 'progress.json', 'w') as f:
        json.dump(prog, f, indent=2)
    log(f"    [ckpt] mid-stage save at ep{ep}: {p}", lf)


def load_progress(od):
    """Read progress.json: 'completed' (finished stage names) and optionally
    'in_progress' ({'stage': name, 'epoch': last_ep})."""
    pf = od / 'progress.json'
    if pf.exists():
        with open(pf) as f:
            return json.load(f)
    return {'completed': []}


# ---- T extension

def load_and_extend(ckpt_path, N, H, old_T, new_T, sK, fK, model_class,
                    dev, **model_kwargs):
    """Load a checkpoint trained at T iterations and extend to new_T >= T.

    T is a relaxation depth, not a temperature, so it can be grown: the
    T8 -> T12 -> T16 -> T20 ladder is built by repeated calls here.
    Per-timestep parameters (skip gates α_t, w_t, V_t; field gates u_t, d_t,
    P_t; temperature a_t, b_t) are copied for t < old_T and interpolated
    toward neutral beyond it, by the rule in the inline comment below.

    ``onsager`` (``'none' | 'within' | 'full' | 'tied'``) is a plain Python
    attribute, not a buffer or parameter, so it is not recovered from the
    state_dict.  Pass it via ``model_kwargs`` or the extended model runs with
    a different TAP correction than the source.

    Args:
        N:            Number of COG families (visible binary variables).
        H:            Hidden state dimension, or ``None`` for NoHiddenDenoiser,
                      whose constructor drops the H positional.
        old_T:        T of the source checkpoint (auto-detected if None).
        new_T:        Target number of iterations; must be >= old_T.
        sK, fK:       Skip and field rank for the low-rank gate components.
        model_class:  ModuleConditionedDenoiser or NoHiddenDenoiser.
        model_kwargs: Extra kwargs for model_class (e.g. ``n_modules=419``,
                      ``adaptive_temp=True``, ``onsager='full'``).
    """
    sd = torch.load(ckpt_path, map_location=dev, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = strip_compile_prefix(sd)
    sd = {k: v for k, v in sd.items()
          if not k.startswith('aux_head.')}

    # Two on-disk layouts: a stacked (T, ...) tensor under 'skip_alpha', and
    # the older ParameterList keys 'skip_alpha.0', 'skip_alpha.1', ...
    if old_T is None:
        if 'skip_alpha' in sd and sd['skip_alpha'].dim() >= 1 and not any(k.startswith('skip_alpha.') for k in sd):
            old_T = sd['skip_alpha'].shape[0]
        else:
            old_T = sum(1 for k in sd if k.startswith('skip_alpha.'))

    if H is None:
        model = model_class(N, new_T, sK, fK, **model_kwargs).to(dev)
    else:
        model = model_class(N, H, new_T, sK, fK, **model_kwargs).to(dev)

    def _get_t(prefix, t):
        """Timestep t of a per-timestep param, from either checkpoint layout."""
        if prefix in sd and sd[prefix].dim() >= 1 and sd[prefix].shape[0] >= t + 1:
            if not any(k.startswith(f'{prefix}.') for k in sd):
                return sd[prefix][t]
        key = f'{prefix}.{t}'
        if key in sd:
            return sd[key]
        return None

    with torch.no_grad():
        for name in ['h', 'J', 'A', 'U', 'W']:
            if name in sd and hasattr(model, name):
                getattr(model, name).copy_(sd[name])

        per_t = ['skip_alpha', 'skip_w', 'skip_V',
                 'field_u', 'field_d', 'field_P',
                 'temp_a', 'temp_b']
        for t in range(min(old_T, new_T)):
            for prefix in per_t:
                src = _get_t(prefix, t)
                if src is not None and hasattr(model, prefix):
                    getattr(model, prefix)[t].data.copy_(src)

        # New timesteps t ∈ [old_T, new_T) interpolate the last trained values
        # toward neutral, which leaves the added iterations near-identity:
        #
        #     param[t] = (1 - frac) · param[old_T - 1] + frac · neutral
        #     frac     = (t - old_T + 1) / (new_T - old_T + 1)   ∈ (0, 1)
        #
        # frac never reaches 1, so every new step keeps a residue of the
        # trained value.  Neutral values:
        #     skip_alpha = 1.0    (skip on, at the trained baseline)
        #     skip_w, V  = 0.0    (no diagonal/low-rank gating)
        #     field_u, d, P = 0.0 (no field gating)
        #     temp_a = 0.5413     (softplus(0.5413) ≈ 1.0 ⇒ τ = 1)
        #     temp_b = 0.0        (no density-dependent correction)
        neutrals = [('skip_alpha', 1.0), ('skip_w', 0.0),
                     ('skip_V', 0.0), ('field_u', 0.0),
                     ('field_d', 0.0), ('field_P', 0.0),
                     ('temp_a', 0.5413), ('temp_b', 0.0)]
        for t in range(old_T, new_T):
            frac = (t - old_T + 1) / (new_T - old_T + 1)
            last = old_T - 1
            for prefix, neutral in neutrals:
                if not hasattr(model, prefix):
                    continue
                src = _get_t(prefix, last)
                if src is not None:
                    tgt = getattr(model, prefix)[t].data
                    tgt.copy_(src * (1 - frac) + neutral * frac)

        # Higher-order attention (HigherOrderDenoiser only).  Every attn.*
        # tensor is T-independent except `t_embedding.weight`, (T, d_model):
        # rows beyond old_T repeat the last source row, so the added
        # iterations see the temporal embedding of the last trained one.
        # A source checkpoint with no attn.* keys leaves attn at its
        # zero-init Δ ≡ 0 state.
        if hasattr(model, 'attn'):
            for sub_name, param in model.attn.named_parameters():
                key = 'attn.' + sub_name
                if key not in sd:
                    continue
                src_v = sd[key]
                if sub_name == 't_embedding.weight':
                    if src_v.dim() != 2 or param.dim() != 2:
                        continue
                    T_src, d_src = src_v.shape
                    T_tgt, d_tgt = param.shape
                    if d_src != d_tgt:
                        continue
                    n_copy = min(T_src, T_tgt)
                    param.data[:n_copy].copy_(src_v[:n_copy])
                    if T_tgt > T_src:
                        param.data[T_src:].copy_(src_v[-1:])
                elif src_v.shape == param.shape:
                    param.data.copy_(src_v)
            for sub_name, buf in model.attn.named_buffers():
                key = 'attn.' + sub_name
                if key in sd and sd[key].shape == buf.shape:
                    buf.data.copy_(sd[key])

        # Module conditioning MLP.  Only the output layer (4) is T-dependent:
        # it maps to 2*T, so its first 2*old_T rows are copied and the rest
        # zero-initialised (γ = 0, neutral, for the new timesteps).
        if hasattr(model, 'module_cond'):
            for name, param in model.module_cond.named_parameters():
                key = 'module_cond.' + name
                if key not in sd:
                    continue
                src_v = sd[key]
                if src_v.shape == param.shape:
                    param.data.copy_(src_v)
                elif 'mlp.4.' in name and src_v.shape[0] == 2 * old_T:
                    n_copy = min(src_v.shape[0], param.shape[0])
                    param.data.zero_()
                    param.data[:n_copy].copy_(src_v[:n_copy])

    return model

