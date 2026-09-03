#!/usr/bin/env python3
"""
denoise_genome.py - run a trained Ising denoiser on a partial
                    gene-presence observation, returning a posterior
                    probability of presence for every COG in the
                    vocabulary.

Inputs:
  --model PATH    Trained checkpoint (.pth).  The model class
                  (HigherOrderDenoiser / ModuleConditioned / NoHidden /
                  SetTransformer) is inferred from the state-dict keys,
                  T from the per-timestep parameter shape, and the
                  onsager mode from the sidecar model_*.cfg.json.

  --input PATH    Two-column file (TSV / CSV / whitespace-separated):
                      COG_ID   probability_of_presence
                      COG0001  0.95
                      COG0002  0.50
                  Lines starting with '#' are ignored.
                  COGs not listed are treated as absent (p = 0).

The input p is mapped to a soft visible state x = 2p - 1, so p = 0.5
carries no information.  --binary thresholds at 0.5 first, recovering
the strict {-1, +1} input the model was trained on.

Output: one row per COG with columns COG_ID, input_prob,
denoised_prob, delta (denoised - input), to stdout unless --output is
given.  Rows are ordered by --sort-by; --top N keeps the first N.

--device auto picks CUDA > MPS > CPU.  At batch = 1 (NoHidden T=8,
N=4789) CPU beats MPS by ~3x, 40 ms against 115 ms per forward:
kernel-dispatch overhead dominates at this size.

Example:
  python3 scripts/denoise_genome.py \\
      --model gsd_results_higher_order_nohidden_T12_split1/model_ho3.pth \\
      --input my_genome.tsv > denoised.tsv
"""
import argparse
import sys
from pathlib import Path

import torch

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from ising_denoiser.data import load_feathers
from ising_denoiser.modules import load_module_matrix
from ising_denoiser.training import load_ckpt_cfg, strip_compile_prefix


# ---- Checkpoint / device helpers

def detect_class(sd):
    """Return one of 'higher_order', 'hidden', 'nohidden', 'set_transformer'."""
    keys = set(sd.keys())
    has_gene_embedding = 'gene_embedding' in keys
    has_attn = any(k.startswith('attn.') for k in keys)
    has_A = 'A' in keys
    if has_gene_embedding:
        return 'set_transformer'
    if has_attn:
        return 'higher_order'
    if has_A:
        return 'hidden'
    return 'nohidden'


def detect_T(sd):
    """Detect T from skip_alpha / skip_gates.

    Accepts both checkpoint layouts: a stacked (T, ...) tensor, and the
    older per-timestep 'skip_alpha.0', 'skip_alpha.1', ... keys.
    """
    for key in ('skip_alpha', 'skip_gates'):
        if key in sd and sd[key].dim() >= 1:
            return int(sd[key].shape[0])
    for prefix in ('skip_alpha.', 'skip_gates.'):
        n = sum(1 for k in sd if k.startswith(prefix))
        if n > 0:
            return n
    raise ValueError("Could not detect T from checkpoint state_dict.")


def pick_device(req):
    """Return (torch.device, kind), kind in {'cuda', 'mps', 'cpu'}."""
    if req == 'cuda' or (req == 'auto' and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device('cuda'), 'cuda'
        print('# WARNING: --device cuda but CUDA unavailable; falling back',
              file=sys.stderr)
    if req == 'mps' or (req == 'auto' and getattr(torch.backends, 'mps', None)
                        and torch.backends.mps.is_available()):
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps'), 'mps'
        print('# WARNING: --device mps but MPS unavailable; falling back',
              file=sys.stderr)
    return torch.device('cpu'), 'cpu'


# ---- Input parsing

def parse_input(path):
    """Parse a 2-column COG / probability file (tab, comma or whitespace
    separated).  Returns {COG: prob}."""
    rows = {}
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = None
            for sep in ('\t', ','):
                if sep in s:
                    parts = [t.strip() for t in s.split(sep)]
                    break
            if parts is None:
                parts = s.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{ln}: expected 2 columns, got {len(parts)}: {s!r}")
            cog = parts[0]
            try:
                p = float(parts[1])
            except ValueError:
                raise ValueError(f"{path}:{ln}: bad probability {parts[1]!r}")
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"{path}:{ln}: probability {p} for {cog} is not in [0, 1]")
            if cog in rows:
                print(f"# WARNING: {cog} appears twice; using last value",
                      file=sys.stderr)
            rows[cog] = p
    return rows


def build_model(cls_kind, sd, T, n_mod, onsager, dev,
                 attn_d_model=128, attn_nhead=4, attn_n_layers=1,
                 attn_dim_ff=512, H=1000):
    """Instantiate the detected model class at the geometry inferred from
    the state dict.  H = 1000 is the trained default; --H overrides it."""
    if cls_kind == 'higher_order':
        from ising_denoiser.models import HigherOrderDenoiser
        # 'tied' is the higher-order training default.
        onsager_val = onsager if onsager and onsager != 'none' else 'tied'
        m = HigherOrderDenoiser(
            N=sd_n(sd), H=H, T=T, n_modules=n_mod,
            adaptive_temp=True, onsager=onsager_val,
            attn_d_model=attn_d_model, attn_nhead=attn_nhead,
            attn_n_layers=attn_n_layers,
            attn_dim_feedforward=attn_dim_ff,
            attn_use_checkpoint=False,   # inference only
        )
    elif cls_kind == 'hidden':
        from ising_denoiser.models import ModuleConditionedDenoiser
        onsager_val = onsager if onsager and onsager != 'none' else 'full'
        m = ModuleConditionedDenoiser(
            N=sd_n(sd), H=H, T=T, n_modules=n_mod,
            adaptive_temp=True, onsager=onsager_val,
        )
    elif cls_kind == 'nohidden':
        from ising_denoiser.models import NoHiddenDenoiser
        # NoHidden takes onsager as a plain on/off bool: it has no
        # within/full/tied modes.
        onsager_bool = bool(onsager and onsager != 'none')
        m = NoHiddenDenoiser(
            N=sd_n(sd), T=T, n_modules=n_mod,
            adaptive_temp=True, onsager=onsager_bool,
        )
    elif cls_kind == 'set_transformer':
        from ising_denoiser.models import SetTransformerDenoiser
        N_ckpt, d_ckpt = sd['gene_embedding'].shape
        m = SetTransformerDenoiser(
            N=N_ckpt, d_model=d_ckpt, nhead=attn_nhead,
            num_layers=attn_n_layers,
            dim_feedforward=attn_dim_ff,
            dropout=0.0, T=T, n_modules=n_mod,
            adaptive_temp=True, use_checkpoint=False,
        )
    else:
        raise ValueError(f"Unknown model class: {cls_kind}")
    return m.to(dev)


def sd_n(sd):
    """Infer N (vocabulary size) from whichever of h, J or gene_embedding
    the model class provides."""
    for k in ('h', 'J', 'gene_embedding'):
        if k in sd:
            return int(sd[k].shape[0])
    raise ValueError("Cannot infer N from state_dict (no 'h', 'J', or 'gene_embedding').")


def main():
    pa = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--model', required=True,
                    help='Path to model checkpoint (.pth).')
    pa.add_argument('--input', required=True,
                    help='Two-column file (COG_ID, probability).')
    pa.add_argument('--vocab-feather', default='data/COG_train1_phylum.feather',
                    help='Feather file used only to recover the COG-name -> '
                         'index ordering the model was trained with.')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt',
                    help='Module-membership matrix.  Pass "" to skip module '
                         'conditioning (only useful if model was trained '
                         'without it).')
    pa.add_argument('--onsager', default='auto',
                    choices=['auto', 'none', 'within', 'full', 'tied'],
                    help='TAP correction mode.  "auto" reads it from the '
                         'sidecar model_*.cfg.json; falls back to "tied" for '
                         'HO and "full" for hidden/nohidden if missing.')
    pa.add_argument('--H', type=int, default=1000,
                    help='Hidden-state dim (must match training; ignored '
                         'for NoHidden / SetTransformer).')
    pa.add_argument('--output', default=None,
                    help='Output TSV (default: stdout).')
    pa.add_argument('--device', default='auto',
                    choices=['auto', 'cpu', 'mps', 'cuda'])
    pa.add_argument('--binary', action='store_true',
                    help='Threshold input probabilities at 0.5 to +/-1 '
                         'instead of using continuous x = 2p - 1.')
    pa.add_argument('--top', type=int, default=None,
                    help='Show only top N COGs by absolute delta.')
    pa.add_argument('--sort-by', default='abs_delta',
                    choices=['abs_delta', 'denoised_prob', 'cog_id'],
                    help='Sort output rows.  Default: largest |delta| first.')
    pa.add_argument('--no-amp', action='store_true',
                    help='Disable autocast.  Default is fp32 on CPU/MPS, '
                         'bf16 on CUDA.')
    A = pa.parse_args()

    dev, dev_kind = pick_device(A.device)
    print(f'# device: {dev_kind}', file=sys.stderr)
    print(f'# torch:  {torch.__version__}', file=sys.stderr)

    # ---- Vocabulary: the COG name -> index ordering
    if not Path(A.vocab_feather).exists():
        sys.exit(f"ERROR: vocab feather not found: {A.vocab_feather}")
    print(f'# vocab:  loading from {A.vocab_feather} ...', file=sys.stderr)
    # frac=0.001: only the COG ordering is needed, the genome rows are dropped.
    _, _, vocab = load_feathers(A.vocab_feather, A.vocab_feather, frac=0.001)
    N_vocab = len(vocab)
    cog_to_idx = {cog: i for i, cog in enumerate(vocab)}
    print(f'# vocab:  N = {N_vocab} COGs', file=sys.stderr)

    input_probs = parse_input(A.input)
    unknown = sorted(c for c in input_probs if c not in cog_to_idx)
    known = {c: p for c, p in input_probs.items() if c in cog_to_idx}
    print(f'# input:  {len(input_probs)} entries '
          f'({len(known)} in-vocab, {len(unknown)} unknown)', file=sys.stderr)
    if unknown:
        sample = ', '.join(unknown[:5]) + ('...' if len(unknown) > 5 else '')
        print(f'# WARNING: {len(unknown)} input COGs not in vocab: {sample}',
              file=sys.stderr)

    # Unlisted COGs stay at -1: absent, with full confidence.
    x_in = torch.full((1, N_vocab), -1.0, device=dev)
    for cog, p in known.items():
        idx = cog_to_idx[cog]
        if A.binary:
            x_in[0, idx] = 1.0 if p > 0.5 else -1.0
        else:
            x_in[0, idx] = 2.0 * p - 1.0

    # ---- Load checkpoint
    print(f'# model:  loading {A.model} ...', file=sys.stderr)
    sd = torch.load(A.model, map_location=dev, weights_only=True)
    # Strip the DDP 'module.' and torch.compile '_orig_mod.' prefixes: left in
    # place they match nothing and load_state_dict(strict=False) would silently
    # return an untrained model.
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = strip_compile_prefix(sd)
    cls_kind = detect_class(sd)
    T = detect_T(sd)
    N_ckpt = sd_n(sd)
    if N_ckpt != N_vocab:
        sys.exit(f"ERROR: checkpoint N={N_ckpt} != vocab N={N_vocab}.  "
                 f"Wrong --vocab-feather?")
    print(f'# model:  class={cls_kind} T={T} N={N_ckpt}', file=sys.stderr)

    # ---- Resolve onsager mode
    onsager = A.onsager
    if onsager == 'auto':
        cfg = load_ckpt_cfg(A.model)
        onsager = cfg.get('onsager', None)
        if onsager:
            print(f'# onsager: {onsager} (from sidecar)', file=sys.stderr)
        else:
            print(f'# onsager: sidecar missing; using default for class',
                  file=sys.stderr)

    # ---- Module matrix
    M_mod = M_sizes = None
    n_mod = 419   # geometry fallback when no module matrix is available
    if A.module_matrix and Path(A.module_matrix).exists():
        _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, A.H)
        n_mod = M_mod.shape[1]
        print(f'# modules: {n_mod} from {A.module_matrix}', file=sys.stderr)
    elif A.module_matrix:
        print(f'# WARNING: --module-matrix {A.module_matrix} not found; '
              f'using n_modules={n_mod}, no conditioning', file=sys.stderr)

    model = build_model(cls_kind, sd, T, n_mod, onsager, dev, H=A.H)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # Benign when --module-matrix is omitted (cond_mod stays freshly
        # initialised) or when an HO architecture is built for a non-HO
        # checkpoint (attn.* absent).
        print(f'# WARNING: {len(missing)} missing keys '
              f'(first 3: {list(missing)[:3]})', file=sys.stderr)
    if unexpected:
        print(f'# WARNING: {len(unexpected)} unexpected keys '
              f'(first 3: {list(unexpected)[:3]})', file=sys.stderr)
    model.eval()

    # ---- Forward pass
    use_amp = (not A.no_amp) and dev_kind == 'cuda'
    print(f'# autocast: {"bf16" if use_amp else "off"}', file=sys.stderr)
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x_in, M_mod=M_mod, M_sizes=M_sizes)
        else:
            out = model(x_in, M_mod=M_mod, M_sizes=M_sizes)
        x_out = out[0] if isinstance(out, tuple) else out
    if dev_kind == 'cuda':
        torch.cuda.synchronize()
    elif dev_kind == 'mps':
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    print(f'# forward: {dt*1000:.0f} ms', file=sys.stderr)

    # ---- Output
    x_in_arr = x_in[0].float().cpu().numpy()
    x_out_arr = x_out[0].float().cpu().numpy()
    in_probs = (x_in_arr + 1.0) / 2.0
    out_probs = (x_out_arr + 1.0) / 2.0
    # The output is a tanh, so out_probs is already in (0, 1); the clip only
    # absorbs floating-point drift at the two ends.
    out_probs = out_probs.clip(0.0, 1.0)
    deltas = out_probs - in_probs

    indices = list(range(N_vocab))
    if A.sort_by == 'abs_delta':
        indices.sort(key=lambda i: -abs(deltas[i]))
    elif A.sort_by == 'denoised_prob':
        indices.sort(key=lambda i: -out_probs[i])
    elif A.sort_by == 'cog_id':
        indices.sort(key=lambda i: vocab[i])
    if A.top is not None:
        indices = indices[:A.top]

    out_fh = open(A.output, 'w') if A.output else sys.stdout
    print('COG_ID\tinput_prob\tdenoised_prob\tdelta', file=out_fh)
    for i in indices:
        print(f'{vocab[i]}\t{in_probs[i]:.4f}\t{out_probs[i]:.4f}\t{deltas[i]:+.4f}',
              file=out_fh)
    if A.output:
        out_fh.close()
        print(f'# wrote {len(indices)} rows to {A.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
