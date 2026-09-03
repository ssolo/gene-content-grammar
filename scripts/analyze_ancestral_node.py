#!/usr/bin/env python3
"""
analyze_ancestral_node.py - run an ensemble of denoiser checkpoints on one
ancestral-node posterior (default node 2012), sweep input thresholds, and
write a summary report.

Reads data/TableAncestralRoot1.tsv and aggregates sub-family copy numbers per
bare COG (sum across COGxxxx_N rows, capped at 1.0 so the value can be used as
a probability), the same aggregation as extract_ancestral_node.py.  Each
split's checkpoint is then run on five input variants -- the aggregated
probabilities ('actual') and binarizations at thresholds 0, 0.01, 0.04 and
0.099 -- and the denoised per-COG probabilities are averaged across splits
(mean +/- sd).  Model class, T and Onsager mode are inferred per checkpoint.

The report gives, per variant, the COGs presented as input, the COGs predicted
present (mean denoised > 0.5), how many of the COGs the threshold filtered out
came back, the rescued and silenced COG listings, and functional-category and
KEGG-module summaries.  --csv-out writes the per-COG mean and sd for every
variant.

Usage:
  python3 scripts/analyze_ancestral_node.py \\
      --node 2012 \\
      --table data/TableAncestralRoot1.tsv \\
      --models 'gsd_results_nohidden_denovo_elbo_T8_split*/model_s3d.pth' \\
      --vocab-feather data/COG_train1_phylum.feather \\
      --module-matrix data/module_matrix_kegg.pt \\
      --output node2012_analysis.txt \\
      --device cpu
"""
import argparse
import glob
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from ising_denoiser.data import load_feathers
from ising_denoiser.modules import load_module_matrix
from ising_denoiser.training import strip_compile_prefix, load_ckpt_cfg


# ----  COG annotations

def load_cog_definitions(path):
    """Load NCBI COG2020 data/cog-20.def.tab into {COG_ID: short-description}.

    Columns: COG_ID, category, name, gene, pathway, pubmed, pdb; the label is
    'name (gene)'.  Returns {} if the file is missing; the report then falls
    back to COG ID + category letter only.
    """
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            t = line.rstrip('\n').split('\t')
            if not t[0].startswith('COG'):
                continue
            cog = t[0]
            name = t[2] if len(t) > 2 else ''
            gene = t[3] if len(t) > 3 else ''
            extras = []
            if gene and gene not in name:
                extras.append(gene)
            label = name
            if extras:
                label += f' ({", ".join(extras)})'
            out[cog] = label
    return out


COG_SUFFIX_RE = re.compile(r'^(COG\d+)(?:_(?:\d+|X))?$')


# ----  Helpers

def strip_cog_suffix(s):
    m = COG_SUFFIX_RE.match(s.strip())
    return m.group(1) if m else None


def detect_T(sd):
    for key in ('skip_alpha', 'skip_gates'):
        if key in sd and sd[key].dim() >= 1:
            return int(sd[key].shape[0])
    for prefix in ('skip_alpha.', 'skip_gates.'):
        n = sum(1 for k in sd if k.startswith(prefix))
        if n > 0:
            return n
    raise ValueError("Could not detect T")


def detect_class(sd):
    """Model class from the state-dict keys: 'higher_order',
    'nohidden_higher_order', 'hidden' or 'nohidden'.

    Attention sits on either a hidden-state base (HigherOrderDenoiser, which
    also carries the A/W hidden matrices) or a no-hidden base
    (NoHiddenHigherOrderDenoiser, no A/W), so attn.* alone does not identify
    the class; the presence of A and W splits the two.
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


def build_model_for_ckpt(sd, dev, N, n_modules, ckpt_path, H=1000):
    """Build the model class matching a checkpoint.

    Class from detect_class, T from the checkpoint, onsager mode from the
    model_*.cfg.json sidecar when present.  H is taken from the checkpoint's A
    so hidden models trained at any width load, overriding the H argument.
    """
    cls_kind = detect_class(sd)
    T = detect_T(sd)
    if 'A' in sd and sd['A'].dim() == 2:
        H = int(sd['A'].shape[1])
    cfg = load_ckpt_cfg(ckpt_path)
    onsager = cfg.get('onsager')

    if cls_kind == 'higher_order':
        from ising_denoiser.models import HigherOrderDenoiser
        onsager_val = onsager if onsager and onsager != 'none' else 'tied'
        model = HigherOrderDenoiser(
            N=N, H=H, T=T, n_modules=n_modules,
            adaptive_temp=True, onsager=onsager_val,
            attn_d_model=128, attn_nhead=4, attn_n_layers=1,
            attn_dim_feedforward=512, attn_use_checkpoint=False,
        )
    elif cls_kind == 'hidden':
        from ising_denoiser.models import ModuleConditionedDenoiser
        onsager_val = onsager if onsager and onsager != 'none' else 'full'
        model = ModuleConditionedDenoiser(
            N=N, H=H, T=T, n_modules=n_modules,
            adaptive_temp=True, onsager=onsager_val,
        )
    elif cls_kind == 'nohidden_higher_order':
        from ising_denoiser.models import NoHiddenHigherOrderDenoiser
        onsager_bool = (onsager is None) or (onsager and onsager != 'none')
        model = NoHiddenHigherOrderDenoiser(
            N=N, T=T, n_modules=n_modules,
            adaptive_temp=True, onsager=onsager_bool,
            attn_d_model=128, attn_nhead=4, attn_n_layers=1,
            attn_dim_feedforward=512, attn_use_checkpoint=False,
        )
    else:  # nohidden
        from ising_denoiser.models import NoHiddenDenoiser
        onsager_bool = (onsager is None) or (onsager and onsager != 'none')
        model = NoHiddenDenoiser(
            N=N, T=T, n_modules=n_modules,
            adaptive_temp=True, onsager=onsager_bool,
        )
    return model.to(dev), cls_kind, T, (onsager or 'auto')


def pick_device(req):
    if req == 'auto':
        # MPS is not auto-selected (some relaxation ops are flaky on Metal);
        # ask for it explicitly with --device mps.
        if torch.cuda.is_available():
            return torch.device('cuda'), 'cuda'
        return torch.device('cpu'), 'cpu'
    if req == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    if req == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps'), 'mps'
    return torch.device('cpu'), 'cpu'


def extract_node_probs(table_path, node_label):
    """Per-COG probability dict: sum sub-family copy numbers, cap at 1.0."""
    accumulator = defaultdict(float)
    seen_zero_only = set()
    headers = None
    col_idx = None
    with open(table_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if headers is None:
                headers = parts
                try:
                    col_idx = headers.index(node_label)
                except ValueError:
                    # Retry ignoring the '(N)' tip suffix.
                    for i, h in enumerate(headers):
                        if h.split('(')[0] == node_label:
                            col_idx = i
                            break
                if col_idx is None:
                    raise SystemExit(
                        f"node {node_label!r} not found in headers.  "
                        f"First few: {headers[:5]}")
                continue
            cog = strip_cog_suffix(parts[0])
            if cog is None:
                continue
            try:
                v = float(parts[col_idx])
            except (ValueError, IndexError):
                continue
            if v > 0.0:
                accumulator[cog] += v
            else:
                seen_zero_only.add(cog)
    out = {c: min(s, 1.0) for c, s in accumulator.items()}
    return out


def build_input_tensor(probs_by_cog, cog_to_idx, N, dev,
                        threshold=None, mode='raw',
                        inject_fp=0.0, inject_seed=42):
    """Build (1, N) x in [-1, +1] from a {COG: prob} dict.

    threshold: if not None, COGs with value <= threshold are treated as
               absent (x = -1).  threshold=0.0 keeps everything > 0.
    mode:
      'raw'      - x = the table copy number, clipped to [-1, 1].
      'binarize' - x = +1 for every COG above threshold, whatever its value.
      'soft'     - x = 2p - 1, reading p as a presence probability.
      'half'     - x = (0.5 + p) / 2 (absent still -1), a false-positive
                   control that under-trusts the input.
    inject_fp: if > 0, flip this fraction of the absent families (x == -1) to
               present after building x, matching the training false-positive
               channel.  Deterministic given inject_seed.
    """
    x = torch.full((1, N), -1.0, device=dev)
    for cog, p in probs_by_cog.items():
        if cog not in cog_to_idx:
            continue
        if threshold is not None and p <= threshold:
            continue
        idx = cog_to_idx[cog]
        if mode == 'binarize':
            x[0, idx] = 1.0
        elif mode == 'soft':
            x[0, idx] = 2.0 * p - 1.0
        elif mode == 'half':
            x[0, idx] = (0.5 + p) / 2.0
        else:  # raw
            x[0, idx] = max(min(p, 1.0), -1.0)
    if inject_fp and inject_fp > 0:
        gen = torch.Generator().manual_seed(int(inject_seed))
        r = torch.rand(N, generator=gen)
        absent = (x[0].detach().cpu() <= -0.999)
        flip = (absent & (r < inject_fp))
        x[0, flip.to(x.device)] = 1.0
    return x


def run_model(model, x_in, M_mod, M_sizes):
    with torch.no_grad():
        out = model(x_in, M_mod=M_mod, M_sizes=M_sizes)
    x_out = out[0] if isinstance(out, tuple) else out
    return x_out[0]


# ----  Main

def main():
    pa = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--node', default='2012', help='Ancestral node label (default 2012).')
    pa.add_argument('--table', default='data/TableAncestralRoot1.tsv')
    pa.add_argument('--models', required=True,
                    help='Glob for split checkpoints, e.g. '
                         '"gsd_results_nohidden_denovo_elbo_T8_split*/model_s3d.pth"')
    pa.add_argument('--vocab-feather', default='data/COG_train1_phylum.feather')
    pa.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    pa.add_argument('--cog-defs', default='data/cog-20.def.tab',
                    help='NCBI COG2020 data/cog-20.def.tab for per-COG names. '
                         'Optional; if missing, per-COG listings show '
                         'only the COG ID + category letter.')
    pa.add_argument('--output', default='node2012_analysis.txt',
                    help='Path for the human-readable report.')
    pa.add_argument('--csv-out', default=None,
                    help='Optional companion TSV with per-COG, '
                         'per-variant predictions (mean + sd).')
    pa.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
                    help="auto (default) = CUDA if available, else CPU.")
    pa.add_argument('--present-cutoff', type=float, default=0.5,
                    help='Denoised prob above this counts as "predicted present" '
                         '(default 0.5).')
    pa.add_argument('--top-rescued', type=int, default=None,
                    help='Per-variant: how many highest-confidence rescued '
                         'COGs to show.  Default: all of them (full output).')
    pa.add_argument('--top-modules', type=int, default=None,
                    help='How many top KEGG modules / categories to list per '
                         'variant.  Default: all (full output).')
    pa.add_argument('--inject-fp', type=float, default=0.0,
                    help='High-FP stress test: flip this fraction of ABSENT '
                         'families to present in the input before relaxation '
                         '(matches the training false-positive channel). '
                         'Default 0.0 (no injection).')
    pa.add_argument('--inject-seed', type=int, default=42,
                    help='Seed for the --inject-fp false-positive injection.')
    pa.add_argument('--drop-low-conf', type=float, default=0.0,
                    help='FP control: among families present in the input '
                         '(prob>0), set the lowest-confidence DROP_LOW_CONF '
                         'fraction (by input probability) to absent before '
                         'relaxation. E.g. 0.10 drops the bottom 10%%. '
                         'Default 0.0 (keep all).')
    pa.add_argument('--actual-mode', default='raw',
                    choices=['raw', 'binarize', 'soft', 'half'],
                    help="Input encoding for the headline 'actual' variant: "
                         "raw (x=p, default), half (x=(0.5+p)/2), soft "
                         "(x=2p-1), binarize (x=+1).")
    pa.add_argument('--bin-thresholds', type=float, nargs='+',
                    default=[0.0, 0.01, 0.04, 0.099],
                    help='Binarize-variant thresholds: each t yields a variant '
                         'where input families with prob>t are set to +1 and the '
                         'rest to absent. t=0 is the 0%% threshold (every family '
                         'with prob>0 -> +1). Default: 0 0.01 0.04 0.099.')
    pa.add_argument('--genome-mass-norm', action='store_true',
                    help='apply the sqrt(kbar/k_n) coupling-drive normalisation '
                         '(analysis/gtdb_gene_content_epistasis.tex Model C) to '
                         'each model -- required for the gtdbctl/gmass fine-tunes, '
                         'whose gmass_kbar is a plain attr not saved in the ckpt.')
    pa.add_argument('--freq-tsv', default='data/cog_marginal_frequency.tsv',
                    help='per-COG presence frequency for --genome-mass-norm.')
    A = pa.parse_args()

    dev, dev_kind = pick_device(A.device)
    print(f'device: {dev_kind}', file=sys.stderr)

    print(f'loading vocab from {A.vocab_feather} ...', file=sys.stderr)
    _, _, vocab = load_feathers(A.vocab_feather, A.vocab_feather, frac=0.001)
    N = len(vocab)
    cog_to_idx = {c: i for i, c in enumerate(vocab)}
    print(f'vocab: N = {N}', file=sys.stderr)

    cog_defs = load_cog_definitions(A.cog_defs)
    if cog_defs:
        print(f'cog defs: {len(cog_defs)} COGs annotated from {A.cog_defs}',
              file=sys.stderr)
    else:
        print(f'cog defs: skipped (no file at {A.cog_defs})', file=sys.stderr)

    def cog_label(c):
        """Return '  <NCBI name>' if known, else ''.  Name trimmed to 60 chars."""
        s = cog_defs.get(c, '')
        if not s:
            return ''
        return f'  {s[:60]}'

    print(f'loading module matrix from {A.module_matrix} ...', file=sys.stderr)
    _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, H=1000)
    n_modules = M_mod.shape[1]
    # load_module_matrix does not return the module names/descriptions, so
    # read the .pt directly for those.
    meta = torch.load(A.module_matrix, weights_only=False)
    cat_names = meta['cat_names']
    cat_descs = meta['cat_descs']
    path_names = meta['path_names']
    kegg_names = meta['kegg_names']
    kegg_descs = meta['kegg_descs']
    # The model's M is the three blocks below concatenated; keep them apart
    # for per-group reporting.
    M_cat = meta['M_cat']
    M_path = meta['M_path']
    M_kegg = meta['M_kegg']
    cat_sizes = meta['cat_sizes']
    path_sizes = meta['path_sizes']
    kegg_sizes = meta['kegg_sizes']
    print(f'modules: {n_modules}  (26 cat + 66 path + 327 KEGG)', file=sys.stderr)

    print(f'extracting node {A.node} from {A.table} ...', file=sys.stderr)
    node_probs = extract_node_probs(A.table, A.node)
    n_input = len(node_probs)
    print(f'node {A.node}: {n_input} COGs with sum > 0', file=sys.stderr)

    # FP control: drop the lowest-confidence input families.
    if A.drop_low_conf and A.drop_low_conf > 0:
        ranked = sorted(node_probs.items(), key=lambda kv: kv[1])
        n_drop = int(round(A.drop_low_conf * len(ranked)))
        for cog, _ in ranked[:n_drop]:
            del node_probs[cog]
        print(f'drop-low-conf {A.drop_low_conf}: removed {n_drop}/{n_input} '
              f'lowest-confidence input COGs ({len(node_probs)} kept)',
              file=sys.stderr)

    # The 'actual' variant plus one binarize variant per --bin-thresholds entry.
    variants = [('actual', dict(threshold=None, mode=A.actual_mode))]
    for t in A.bin_thresholds:
        lab = '>0' if float(t) == 0.0 else ('>%g' % t)
        variants.append((lab, dict(threshold=float(t), mode='binarize')))
    inputs = {}
    for name, kw in variants:
        x = build_input_tensor(node_probs, cog_to_idx, N, dev,
                               inject_fp=A.inject_fp,
                               inject_seed=A.inject_seed, **kw)
        inputs[name] = x
        n_set = int((x > -0.999).sum().item())
        print(f'variant {name:8s}: {n_set} COGs presented as input',
              file=sys.stderr)

    model_paths = sorted(glob.glob(A.models))
    if not model_paths:
        raise SystemExit(f'no models matched: {A.models}')
    print(f'models: {len(model_paths)} found', file=sys.stderr)

    accum = {name: [] for name, _ in variants}
    t0 = time.perf_counter()
    class_seen = []
    for i, mp in enumerate(model_paths):
        print(f'  [{i+1}/{len(model_paths)}] {mp}', file=sys.stderr)
        sd = torch.load(mp, map_location=dev, weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = strip_compile_prefix(sd)
        model, cls_kind, T_ckpt, onsager_used = build_model_for_ckpt(
            sd, dev, N, n_modules, mp, H=1000)
        class_seen.append((cls_kind, T_ckpt, onsager_used))
        # strict=False would load a wrong-class checkpoint with garbage
        # weights and no complaint, so report any key mismatch.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f'    load: {len(missing)} missing, {len(unexpected)} '
                  f'unexpected keys', file=sys.stderr)
            if unexpected:
                print(f'      unexpected[:8]: {unexpected[:8]}', file=sys.stderr)
            if missing:
                print(f'      missing[:8]: {missing[:8]}', file=sys.stderr)
        model.eval()
        if A.genome_mass_norm:
            from ising_denoiser.models import set_coupling_controls
            desc = set_coupling_controls(model, freq_tsv=A.freq_tsv,
                                         genome_mass_norm=True, device=dev)
            if i == 0:
                print(f'    {desc}', file=sys.stderr)

        for name, _ in variants:
            x_in = inputs[name]
            x_out = run_model(model, x_in, M_mod, M_sizes)
            # spin x in (-1, 1) -> probability p in (0, 1)
            accum[name].append(((x_out + 1.0) / 2.0).clamp(0.0, 1.0).cpu())

        del model
        if dev_kind == 'cuda':
            torch.cuda.empty_cache()
    print(f'all {len(model_paths) * len(variants)} forwards in '
          f'{time.perf_counter() - t0:.1f}s', file=sys.stderr)

    results = {}
    for name, _ in variants:
        stack = torch.stack(accum[name], dim=0)
        results[name] = dict(mean=stack.mean(0), sd=stack.std(0))

    cutoff = A.present_cutoff
    lines = []
    def out(s=''):
        lines.append(s)

    out('=' * 72)
    out(f'  ANCESTRAL NODE DENOISING REPORT')
    out('=' * 72)
    out(f'Node:          {A.node}')
    out(f'Table:         {A.table}')
    out(f'Models:        {len(model_paths)} splits matching {A.models}')
    for mp in model_paths:
        out(f'                 {mp}')
    # All splits should share class / T / onsager; warn if any disagree.
    if class_seen:
        cls, T_v, ons = class_seen[0]
        disagreeing = [tup for tup in class_seen[1:] if tup != (cls, T_v, ons)]
        out(f'Model class:   {cls} (T={T_v}, onsager={ons})')
        if disagreeing:
            out(f'WARNING: splits disagree on class/T/onsager: {disagreeing}')
    out(f'Vocab:         {A.vocab_feather}  (N = {N} COGs)')
    out(f'Module matrix: {A.module_matrix}  ({n_modules} modules)')
    out(f'Present cutoff: denoised mean > {cutoff:.2f}')
    out(f'Device:        {dev_kind}')
    out()
    out('Input variants:')
    for name, kw in variants:
        t = kw['threshold']
        m = kw.get('mode', 'raw')
        x = inputs[name]
        n_set = int((x > -0.999).sum().item())
        if m == 'raw':
            desc = 'pass raw copy number as x (0.01 -> x=0.01 etc); absent -> -1'
        elif m == 'binarize':
            desc = f'binarize: input > {t} -> +1, else -1'
        elif m == 'soft':
            desc = 'soft: x = 2p - 1'
        else:
            desc = m
        out(f'  {name:8s} : {n_set:>4d} COGs activated  ({desc})')
    out()

    out('=' * 72)
    out('  PER-VARIANT SUMMARY')
    out('=' * 72)
    for name, kw in variants:
        mean = results[name]['mean']
        sd = results[name]['sd']
        present_mask = mean > cutoff
        n_present = int(present_mask.sum().item())
        x_in = inputs[name]
        # mean/sd were moved to CPU at accumulation; the mask must follow or
        # the combinations below break under --device cuda.
        in_active_mask = (x_in[0] > -0.999).to(present_mask.device)
        out()
        out('-' * 72)
        out(f'Variant: {name}')
        out('-' * 72)
        out(f'  COGs activated as input:                  '
            f'{int(in_active_mask.sum().item())}')
        out(f'  COGs predicted present (mean > {cutoff:.2f}):     '
            f'{n_present}')
        rescued_mask = (~in_active_mask) & present_mask
        n_rescued = int(rescued_mask.sum().item())
        out(f'  COGs rescued  (input absent -> predicted): '
            f'{n_rescued}')
        silenced_mask = in_active_mask & (~present_mask)
        n_silenced = int(silenced_mask.sum().item())
        out(f'  COGs silenced (input present -> absent):   '
            f'{n_silenced}')
        # Of the input COGs this variant's threshold filtered out, how many
        # come back?
        if kw['threshold'] is not None and kw['threshold'] > 0:
            t_val = kw['threshold']
            recoverable = []
            recovered = []
            for cog, p in node_probs.items():
                if cog not in cog_to_idx:
                    continue
                if 0.0 < p <= t_val:
                    recoverable.append(cog)
                    if mean[cog_to_idx[cog]] > cutoff:
                        recovered.append(cog)
            n_rec = len(recovered)
            n_tot = len(recoverable)
            pct = 100.0 * n_rec / n_tot if n_tot else 0.0
            out(f'  Recovery of filtered (0 < p <= {t_val}):  '
                f'{n_rec} / {n_tot}  ({pct:.1f}%)')
        else:
            out(f'  (No filter -- recovery statistic n/a)')

        if n_rescued > 0:
            scores = mean.clone()
            scores[~rescued_mask] = -1.0
            order = torch.argsort(scores, descending=True)
            shown = n_rescued if A.top_rescued is None \
                              else min(A.top_rescued, n_rescued)
            tag = 'All' if A.top_rescued is None else f'Top {shown}'
            out(f'  {tag} rescued COGs (input absent in this variant, '
                f'denoised mean > {cutoff:.2f}):')
            out(f'    {"COG_ID":<10} {"orig_p":>7} {"denoised":>9}'
                f' {"+/-sd":>7}  name')
            for j in range(shown):
                i = int(order[j].item())
                cog = vocab[i]
                orig_p = node_probs.get(cog, 0.0)
                out(f'    {cog:<10} {orig_p:>7.4f} '
                    f'{mean[i].item():>9.4f} {sd[i].item():>7.4f}{cog_label(cog)}')

        if n_silenced > 0:
            silenced_idx = torch.nonzero(silenced_mask, as_tuple=False).squeeze(1).tolist()

            # (a) Silencing rate by original input copy-number bucket.
            bucket_edges = [
                (0.0,   0.01),
                (0.01,  0.04),
                (0.04,  0.099),
                (0.099, 0.25),
                (0.25,  0.5),
                (0.5,   1.0 + 1e-9),  # include the cap
            ]
            bucket_labels = [
                '(0.00, 0.01]',
                '(0.01, 0.04]',
                '(0.04, 0.099]',
                '(0.099, 0.25]',
                '(0.25, 0.50]',
                '(0.50, 1.00]',
            ]
            bucket_total = [0] * len(bucket_edges)
            bucket_silenced = [0] * len(bucket_edges)
            for cog, p in node_probs.items():
                if p <= 0.0 or cog not in cog_to_idx:
                    continue
                idx = cog_to_idx[cog]
                if not in_active_mask[idx].item():
                    continue
                for b, (lo, hi) in enumerate(bucket_edges):
                    if lo < p <= hi:
                        bucket_total[b] += 1
                        if silenced_mask[idx].item():
                            bucket_silenced[b] += 1
                        break
            out()
            out(f'  Silencing rate by ORIGINAL input copy number bucket:')
            out(f'    {"bucket":<14} {"total":>6} {"silenced":>9} {"rate":>6}')
            for b, lab in enumerate(bucket_labels):
                if bucket_total[b] == 0:
                    continue
                rate = 100.0 * bucket_silenced[b] / bucket_total[b]
                out(f'    {lab:<14} {bucket_total[b]:>6d} '
                    f'{bucket_silenced[b]:>9d} {rate:>5.1f}%')

            # (b) Functional-category profile of the silenced COGs.
            sil_per_cat = torch.zeros(len(cat_names))
            sil_per_kegg = torch.zeros(len(kegg_names))
            sil_per_path = torch.zeros(len(path_names))
            for i in silenced_idx:
                sil_per_cat += M_cat[i].float()
                sil_per_kegg += M_kegg[i].float()
                sil_per_path += M_path[i].float()
            out()
            out(f'  Functional categories of silenced COGs:')
            out(f'    {"cat":<3} {"description":<50} {"silenced":>9} '
                f'{"of total":>8} {"share":>6}')
            order = torch.argsort(sil_per_cat, descending=True).tolist()
            for k in order:
                n_sil = int(sil_per_cat[k].item())
                if n_sil == 0:
                    continue
                n_tot = int(M_cat[:, k].sum().item())
                share = 100.0 * n_sil / max(n_tot, 1)
                out(f'    {cat_names[k]:<3} {cat_descs[k][:48]:<50} '
                    f'{n_sil:>9d} {n_tot:>8d} {share:>5.1f}%')

            # (c) The silenced COGs themselves.
            silenced_idx.sort(key=lambda i: -node_probs.get(vocab[i], 0.0))
            shown = n_silenced if A.top_rescued is None \
                                else min(A.top_rescued, n_silenced)
            tag = 'All' if A.top_rescued is None else f'Top {shown}'
            out()
            out(f'  {tag} silenced COGs (input active in this variant, '
                f'denoised mean <= {cutoff:.2f}), sorted by input strength:')
            out(f'    {"COG_ID":<10} {"orig_p":>7} {"denoised":>9}'
                f' {"+/-sd":>7}  name')
            for i in silenced_idx[:shown]:
                cog = vocab[i]
                orig_p = node_probs.get(cog, 0.0)
                out(f'    {cog:<10} {orig_p:>7.4f} '
                    f'{mean[i].item():>9.4f} {sd[i].item():>7.4f}{cog_label(cog)}')

    out()

    out('=' * 72)
    out('  MODULE / CATEGORY SUMMARY')
    out('=' * 72)
    M_kegg_t = M_kegg.float()
    M_cat_t = M_cat.float()
    M_path_t = M_path.float()

    for name, _ in variants:
        mean = results[name]['mean']
        present = (mean > cutoff).float()
        x_in = inputs[name]
        # CPU, to match present; see the device note above.
        in_active = (x_in[0] > -0.999).float().to(present.device)
        silenced = in_active * (1.0 - present)
        out()
        out('-' * 72)
        out(f'Variant: {name}')
        out('-' * 72)

        cat_present = (M_cat_t * present.unsqueeze(1)).sum(0)
        cat_silenced = (M_cat_t * silenced.unsqueeze(1)).sum(0)
        cat_total = M_cat_t.sum(0)
        out(f'  Functional categories  '
            f'(predicted-present / silenced / total in cat):')
        order = torch.argsort(cat_present, descending=True).tolist()
        n_cat_show = len(order) if A.top_modules is None \
                                else min(A.top_modules, len(order))
        for k in order[:n_cat_show]:
            np_ = int(cat_present[k].item())
            ns_ = int(cat_silenced[k].item())
            nt_ = int(cat_total[k].item())
            if nt_ == 0:
                continue
            frac_p = np_ / nt_
            out(f'    {cat_names[k]:<3} {cat_descs[k][:48]:<50} '
                f'{np_:>4} / {ns_:>3} / {nt_:>4}  ({frac_p:.0%} p)')

        kegg_present = (M_kegg_t * present.unsqueeze(1)).sum(0)
        kegg_silenced = (M_kegg_t * silenced.unsqueeze(1)).sum(0)
        kegg_total = M_kegg_t.sum(0)
        kegg_frac = (kegg_present / kegg_total.clamp(min=1)).tolist()
        out()
        order = sorted(range(len(kegg_names)), key=lambda k: -kegg_frac[k])
        eligible = [k for k in order
                    if kegg_total[k].item() >= 3 and kegg_present[k].item() > 0]
        n_kegg_show = len(eligible) if A.top_modules is None \
                                    else min(A.top_modules, len(eligible))
        tag = 'All' if A.top_modules is None else f'Top {n_kegg_show}'
        out(f'  {tag} KEGG modules with >=1 COG predicted (size >= 3), '
            f'sorted by fraction-of-COGs-predicted (cols: present / silenced / total):')
        for k in eligible[:n_kegg_show]:
            np_ = int(kegg_present[k].item())
            ns_ = int(kegg_silenced[k].item())
            nt_ = int(kegg_total[k].item())
            frac = kegg_frac[k]
            desc = kegg_descs[k][:50]
            out(f'    {kegg_names[k]:<8} {desc:<52} '
                f'{np_:>3} / {ns_:>2} / {nt_:>3}  ({frac:.0%})')

        out()
        kegg_sil_frac = (kegg_silenced / kegg_total.clamp(min=1)).tolist()
        order = sorted(range(len(kegg_names)), key=lambda k: -kegg_sil_frac[k])
        eligible_sil = [k for k in order
                        if kegg_total[k].item() >= 3 and kegg_silenced[k].item() > 0]
        n_show = len(eligible_sil) if A.top_modules is None \
                                  else min(A.top_modules, len(eligible_sil))
        tag = 'All' if A.top_modules is None else f'Top {n_show}'
        out(f'  {tag} KEGG modules with >=1 COG SILENCED (size >= 3), '
            f'sorted by fraction-silenced (cols: silenced / present / total):')
        for k in eligible_sil[:n_show]:
            ns_ = int(kegg_silenced[k].item())
            np_ = int(kegg_present[k].item())
            nt_ = int(kegg_total[k].item())
            frac = kegg_sil_frac[k]
            desc = kegg_descs[k][:50]
            out(f'    {kegg_names[k]:<8} {desc:<52} '
                f'{ns_:>2} / {np_:>3} / {nt_:>3}  ({frac:.0%})')

        # COG pathways, handled as the KEGG modules above.
        path_present = (M_path_t * present.unsqueeze(1)).sum(0)
        path_silenced = (M_path_t * silenced.unsqueeze(1)).sum(0)
        path_total = M_path_t.sum(0)
        path_frac = (path_present / path_total.clamp(min=1)).tolist()
        out()
        order = sorted(range(len(path_names)), key=lambda k: -path_frac[k])
        eligible = [k for k in order
                    if path_total[k].item() >= 3 and path_present[k].item() > 0]
        n_path_show = len(eligible) if A.top_modules is None \
                                    else min(A.top_modules, len(eligible))
        tag = 'All' if A.top_modules is None else f'Top {n_path_show}'
        out(f'  {tag} COG pathways with >=1 COG predicted (size >= 3), '
            f'sorted by fraction-of-COGs-predicted (cols: present / silenced / total):')
        for k in eligible[:n_path_show]:
            np_ = int(path_present[k].item())
            ns_ = int(path_silenced[k].item())
            nt_ = int(path_total[k].item())
            frac = path_frac[k]
            out(f'    {path_names[k]:<55} '
                f'{np_:>3} / {ns_:>2} / {nt_:>3}  ({frac:.0%})')

        out()
        path_sil_frac = (path_silenced / path_total.clamp(min=1)).tolist()
        order = sorted(range(len(path_names)), key=lambda k: -path_sil_frac[k])
        eligible_sil = [k for k in order
                        if path_total[k].item() >= 3 and path_silenced[k].item() > 0]
        n_show = len(eligible_sil) if A.top_modules is None \
                                  else min(A.top_modules, len(eligible_sil))
        tag = 'All' if A.top_modules is None else f'Top {n_show}'
        out(f'  {tag} COG pathways with >=1 COG SILENCED (size >= 3), '
            f'sorted by fraction-silenced (cols: silenced / present / total):')
        for k in eligible_sil[:n_show]:
            ns_ = int(path_silenced[k].item())
            np_ = int(path_present[k].item())
            nt_ = int(path_total[k].item())
            frac = path_sil_frac[k]
            out(f'    {path_names[k]:<55} '
                f'{ns_:>2} / {np_:>3} / {nt_:>3}  ({frac:.0%})')

    out()

    out('=' * 72)
    out('  CROSS-VARIANT AGREEMENT (predicted present)')
    out('=' * 72)
    out(f'How many COGs are predicted present (mean > {cutoff:.2f}) under')
    out(f'each variant, vs. under all variants simultaneously:')
    out()
    var_present = {name: (results[name]['mean'] > cutoff)
                   for name, _ in variants}
    for name, _ in variants:
        n = int(var_present[name].sum().item())
        out(f'  {name:8s} : {n} COGs')
    all_var_present = torch.ones_like(var_present[variants[0][0]])
    for name, _ in variants:
        all_var_present &= var_present[name]
    n_intersect = int(all_var_present.sum().item())
    out(f'  intersection of all 5 variants: {n_intersect} COGs')
    n_union = int(sum(var_present[v[0]].long()
                      for v in variants).clamp_(0, 1).sum().item())
    out(f'  union     of all 5 variants:    {n_union} COGs')

    out()
    out(f'  COGs in the all-5-variant intersection '
        f'(highest-confidence consensus, n={n_intersect}):')
    intersect_idx = torch.nonzero(all_var_present, as_tuple=False).squeeze(1).tolist()
    actual_mean = results['actual']['mean']
    intersect_idx.sort(key=lambda i: -actual_mean[i].item())
    out(f'    {"COG_ID":<10} {"orig_p":>7}'
        + ''.join(f' {"mean_" + v[0]:>10}' for v in variants))
    for i in intersect_idx:
        cog = vocab[i]
        orig = node_probs.get(cog, 0.0)
        means = ''.join(f' {results[v[0]]["mean"][i].item():>10.4f}'
                        for v in variants)
        out(f'    {cog:<10} {orig:>7.4f}{means}{cog_label(cog)}')

    out()
    out('=' * 72)
    out('  CROSS-VARIANT SILENCED COGs')
    out('=' * 72)
    out('COGs with copy number > 0 in the original input but predicted')
    out(f'absent (mean <= {cutoff:.2f}) under EVERY variant.  These are the')
    out('cases where the denoiser most strongly disagrees with the')
    out('ancestral-reconstruction posterior.')
    out()
    silenced_global = []
    for cog, p in node_probs.items():
        if p <= 0.0 or cog not in cog_to_idx:
            continue
        i = cog_to_idx[cog]
        all_silent = all(
            results[v[0]]['mean'][i].item() <= cutoff for v in variants
        )
        if all_silent:
            silenced_global.append((cog, p, i))
    silenced_global.sort(key=lambda r: -r[1])
    n_glob_sil = len(silenced_global)
    n_input_pos = sum(1 for p in node_probs.values() if p > 0)
    pct_global = 100.0 * n_glob_sil / max(n_input_pos, 1)
    out(f'  Total: {n_glob_sil} COGs ({pct_global:.1f}% of the {n_input_pos} '
        f'COGs with copy number > 0 in the input)')

    bucket_edges = [
        (0.0,   0.01),
        (0.01,  0.04),
        (0.04,  0.099),
        (0.099, 0.25),
        (0.25,  0.5),
        (0.5,   1.0 + 1e-9),
    ]
    bucket_labels = [
        '(0.00, 0.01]',
        '(0.01, 0.04]',
        '(0.04, 0.099]',
        '(0.099, 0.25]',
        '(0.25, 0.50]',
        '(0.50, 1.00]',
    ]
    bucket_total = [0] * len(bucket_edges)
    bucket_sil = [0] * len(bucket_edges)
    for cog, p in node_probs.items():
        if p <= 0 or cog not in cog_to_idx:
            continue
        for b, (lo, hi) in enumerate(bucket_edges):
            if lo < p <= hi:
                bucket_total[b] += 1
                break
    for cog, p, i in silenced_global:
        for b, (lo, hi) in enumerate(bucket_edges):
            if lo < p <= hi:
                bucket_sil[b] += 1
                break
    out()
    out(f'  Cross-variant silenced count by ORIGINAL input copy number bucket:')
    out(f'    {"bucket":<14} {"input>0":>8} {"silenced":>9} {"rate":>6}')
    for b, lab in enumerate(bucket_labels):
        if bucket_total[b] == 0:
            continue
        rate = 100.0 * bucket_sil[b] / bucket_total[b]
        out(f'    {lab:<14} {bucket_total[b]:>8d} '
            f'{bucket_sil[b]:>9d} {rate:>5.1f}%')

    sil_per_cat = torch.zeros(len(cat_names))
    sil_per_kegg = torch.zeros(len(kegg_names))
    sil_per_path = torch.zeros(len(path_names))
    for _, _, i in silenced_global:
        sil_per_cat += M_cat[i].float()
        sil_per_kegg += M_kegg[i].float()
        sil_per_path += M_path[i].float()
    out()
    out(f'  Functional categories of cross-variant silenced COGs:')
    out(f'    {"cat":<3} {"description":<50} {"silenced":>9} '
        f'{"of total":>8} {"share":>6}')
    order = torch.argsort(sil_per_cat, descending=True).tolist()
    for k in order:
        n_sil = int(sil_per_cat[k].item())
        if n_sil == 0:
            continue
        n_tot = int(M_cat[:, k].sum().item())
        share = 100.0 * n_sil / max(n_tot, 1)
        out(f'    {cat_names[k]:<3} {cat_descs[k][:48]:<50} '
            f'{n_sil:>9d} {n_tot:>8d} {share:>5.1f}%')

    out()
    out(f'  KEGG modules of cross-variant silenced COGs (size >= 3, '
        f'sorted by silenced count):')
    out(f'    {"module":<8} {"description":<52} '
        f'{"silenced":>9} {"of total":>8} {"share":>6}')
    kegg_total_f = M_kegg.float().sum(0)
    order = sorted(range(len(kegg_names)),
                   key=lambda k: (-sil_per_kegg[k].item(),
                                   -kegg_total_f[k].item()))
    for k in order:
        n_sil = int(sil_per_kegg[k].item())
        if n_sil == 0:
            continue
        n_tot = int(kegg_total_f[k].item())
        if n_tot < 3:
            continue
        share = 100.0 * n_sil / max(n_tot, 1)
        out(f'    {kegg_names[k]:<8} {kegg_descs[k][:50]:<52} '
            f'{n_sil:>9d} {n_tot:>8d} {share:>5.1f}%')

    out()
    out(f'  COG pathways of cross-variant silenced COGs (size >= 3, '
        f'sorted by silenced count):')
    out(f'    {"pathway":<55} '
        f'{"silenced":>9} {"of total":>8} {"share":>6}')
    path_total_f = M_path.float().sum(0)
    order = sorted(range(len(path_names)),
                   key=lambda k: (-sil_per_path[k].item(),
                                   -path_total_f[k].item()))
    for k in order:
        n_sil = int(sil_per_path[k].item())
        if n_sil == 0:
            continue
        n_tot = int(path_total_f[k].item())
        if n_tot < 3:
            continue
        share = 100.0 * n_sil / max(n_tot, 1)
        out(f'    {path_names[k]:<55} '
            f'{n_sil:>9d} {n_tot:>8d} {share:>5.1f}%')

    out()
    out(f'  Cross-variant silenced COGs, sorted by input strength:')
    out(f'    {"COG_ID":<10} {"orig_p":>7}'
        + ''.join(f' {"mean_" + v[0]:>10}' for v in variants))
    for cog, p, i in silenced_global:
        means = ''.join(f' {results[v[0]]["mean"][i].item():>10.4f}'
                        for v in variants)
        out(f'    {cog:<10} {p:>7.4f}{means}{cog_label(cog)}')

    Path(A.output).write_text('\n'.join(lines) + '\n')
    print(f'wrote report to {A.output}', file=sys.stderr)

    if A.csv_out:
        with open(A.csv_out, 'w') as f:
            cols = ['COG_ID', 'input_prob']
            for name, _ in variants:
                cols += [f'mean_{name}', f'sd_{name}']
            f.write('\t'.join(cols) + '\n')
            for i, cog in enumerate(vocab):
                row = [cog, f'{node_probs.get(cog, 0.0):.4f}']
                for name, _ in variants:
                    row += [
                        f'{results[name]["mean"][i].item():.4f}',
                        f'{results[name]["sd"][i].item():.4f}',
                    ]
                f.write('\t'.join(row) + '\n')
        print(f'wrote per-COG TSV to {A.csv_out}', file=sys.stderr)


if __name__ == '__main__':
    main()
