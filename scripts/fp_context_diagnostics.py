#!/usr/bin/env python3
"""fp_context_diagnostics.py -- post-hoc false-positive diagnostics, no retraining.

Scores whether a present input gene is still supported once its own input bit is
removed: a gene embedded in a coherent module is rescued by its neighbours, a
spurious one falls back to its marginal base rate.  All scores come from frozen
checkpoints.

The forward pass recomputes module conditioning, skip gates and field gates from
the perturbed input, so masking gene i also drops i from every module and density
statistic; one batched forward over per-gene-masked copies is therefore a
leave-one-out cavity.

Three modes:
  ancestral   dense ancestral input (reconciliation/GLD table or a prediction TSV
              of per-COG input probabilities), ensembled over the pinned splits.
  extant      held-out present-day genome: corrupt its known content (fn / fp /
              fp-mode) and score, with truth labels, on the split whose phylum
              hold-out contains it.
  summarize   ROC-AUC / average precision for known-FP detection from an extant
              output TSV, plus a pruning table vs the q_full>0.5 baseline.

Scores (per candidate input-present gene i):
  q_full              denoised posterior p_i from the original input.
  q_mask_neutral      p_i with x_i = 0   (no evidence either way).
  q_mask_absent       p_i with x_i = -1  (input votes it absent).
  self_anchor_*       q_full - q_mask_*; large => self-anchored => likely FP.
  residual_context    logit(q_mask_absent) - logit(marginal_frequency_i).
  dropout_mean/_cons  drop a fraction of present genes at random, N times; mean
                      p_i and fraction of runs with p_i>0.5.
  anchor_support      p_i under a sparse core built from an anchor prediction
                      (posterior>thr).

Full runs need the checkpoints and, for `extant`, the held-out validation
feathers; without either, the parser, summarize and IO paths still import and
run (scripts/test_fp_context_diagnostics.py).
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np

# Bare-COG normaliser (COGxxxx_N / COGxxxx_X -> COGxxxx).  Local copy so the
# parse/summarize/IO paths stay importable without torch.
_COG_RE = re.compile(r'^(COG\d+)(?:_(?:\d+|X))?$')


def _strip_cog(s):
    m = _COG_RE.match(s.strip())
    return m.group(1) if m else None


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---- CLI helpers

def parse_splits(s):
    """'1-10' -> [1..10]; '1,3,5' -> [1,3,5]; '5' -> [5].  Never '*'."""
    s = s.strip()
    if s == '*':
        raise SystemExit("refuse --splits '*': pin the split set explicitly "
                         "(see memory/feedback_analysis_provenance.md)")
    out = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise SystemExit(f"could not parse --splits {s!r}")
    return out


def resolve_models(template, splits, allow_glob):
    """Expand a {split} template to explicit (split, path) pairs.  Wildcard paths
    are refused unless --allow-glob: the matched split set drifts as checkpoints
    appear, so a glob is not reproducible."""
    if ('*' in template or '?' in template or '[' in template) and not allow_glob:
        raise SystemExit(
            f"refuse wildcard model path {template!r}: pin the split set with "
            f"--models-template '...split{{split}}/model_ho3.pth' --splits 1-10 "
            f"(pass --allow-glob only if you really mean a glob).")
    if '{split}' not in template:
        raise SystemExit("--models-template must contain the {split} placeholder")
    pairs = []
    for s in splits:
        path = template.format(split=s)
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        pairs.append((s, path))
    return pairs


def load_value_tsv(path, value_col):
    """{bare-COG: float} from a TSV with a COG_ID column + value_col.  Sub-family
    rows (COGxxxx_N) are collapsed to the bare COG by max."""
    out = {}
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        try:
            ci = header.index('COG_ID')
            vi = header.index(value_col)
        except ValueError:
            raise SystemExit(f"{path}: need columns COG_ID and {value_col!r}; "
                             f"have {header[:6]}")
        for line in f:
            t = line.rstrip('\n').split('\t')
            if len(t) <= max(ci, vi):
                continue
            cog = _strip_cog(t[ci])
            if cog is None:
                continue
            try:
                v = float(t[vi])
            except ValueError:
                continue
            out[cog] = max(out.get(cog, -1e9), v)
    return out


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ---- model construction and forward passes

def _build_model(path, dev, N, n_modules):
    import torch
    from analyze_ancestral_node import build_model_for_ckpt
    from ising_denoiser.training import strip_compile_prefix
    sd = torch.load(path, map_location=dev, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = strip_compile_prefix(sd)
    model, cls_kind, T, onsager = build_model_for_ckpt(sd, dev, N, n_modules, path, H=1000)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, dict(cls=cls_kind, T=T, onsager=onsager)


def forward_probs(model, X, M_mod, M_sizes):
    """X: (B, N) input in [-1, 1].  The model returns a magnetisation m in
    (-1, 1); this returns the posterior p = (m + 1) / 2, shape (B, N)."""
    import torch
    with torch.no_grad():
        out = model(X, M_mod=M_mod, M_sizes=M_sizes)
    m = out[0] if isinstance(out, tuple) else out
    return ((m + 1.0) / 2.0).clamp(0.0, 1.0)


def cavity(model, x_full, cand_idx, mask_value, M_mod, M_sizes, batch_size):
    """For each candidate i, set x_i = mask_value (others unchanged), run the
    model, and record i's own output probability.  Returns (n_cand,) numpy."""
    import torch
    dev = x_full.device
    cand = torch.as_tensor(cand_idx, device=dev, dtype=torch.long)
    out = np.empty(len(cand_idx), dtype=np.float32)
    for b0 in range(0, len(cand_idx), batch_size):
        ci = cand[b0:b0 + batch_size]
        B = ci.numel()
        X = x_full.expand(B, -1).clone()
        rows = torch.arange(B, device=dev)
        X[rows, ci] = float(mask_value)
        P = forward_probs(model, X, M_mod, M_sizes)
        out[b0:b0 + B] = P[rows, ci].detach().cpu().numpy()
    return out


def dropout_runs(model, x_full, present_idx, drop_frac, runs, gen, M_mod, M_sizes,
                 batch_size):
    """Drop a random drop_frac of the present genes to absent, `runs` times.
    Returns (runs, N) numpy of posteriors."""
    import torch
    dev = x_full.device
    present = torch.as_tensor(present_idx, device=dev, dtype=torch.long)
    n_drop = int(round(drop_frac * len(present_idx)))
    X = x_full.expand(runs, -1).clone()
    for r in range(runs):
        perm = torch.randperm(len(present_idx), generator=gen)[:n_drop]
        X[r, present[perm.to(dev)]] = -1.0
    res = np.empty((runs, x_full.shape[1]), dtype=np.float32)
    for b0 in range(0, runs, batch_size):
        P = forward_probs(model, X[b0:b0 + batch_size], M_mod, M_sizes)
        res[b0:b0 + P.shape[0]] = P.detach().cpu().numpy()
    return res


# ---- diagnostics

def run_diagnostics(model_pairs, x_full, cand_idx, present_idx, anchor_x,
                    marg_vec, args, N, n_modules, dev):
    """Loop the pinned split models; accumulate per-candidate scores; aggregate
    (mean/sd across splits, dropout pooled across split x run).  Returns a dict of
    numpy arrays keyed by output column, plus per-model metadata."""
    import torch
    q_full, q_neu, q_abs, anch = [], [], [], []
    pl = []
    drop_pool = []
    model_meta = []
    cand_arr = np.asarray(cand_idx, dtype=np.int64)
    for k, (split, path) in enumerate(model_pairs):
        print(f'  [{k + 1}/{len(model_pairs)}] split {split}: {os.path.relpath(path, ROOT)}',
              file=sys.stderr)
        model, meta = _build_model(path, dev, N, n_modules)
        meta['split'] = split
        model_meta.append(meta)

        P_full = forward_probs(model, x_full, M_mod=M_MOD, M_sizes=M_SIZES)[0]
        q_full.append(P_full[cand_arr].detach().cpu().numpy())

        # One-step pseudolikelihood local field h + (x @ J~) on the observed
        # configuration: no relaxation, no gates, no Delta.  Monotone in the
        # single-flip energy DeltaE_i, so it ranks candidates identically.  The
        # cavity reads the same neighbour field at the relaxed fixed point, with
        # TAP, tau, the gates and Delta, so Spearman(pl_field, cavity) measures
        # what those corrections add.
        if hasattr(model, '_Js'):
            with torch.no_grad():
                field = x_full[0] @ model._Js() + model.h
            pl.append(field[cand_arr].detach().cpu().numpy())

        q_neu.append(cavity(model, x_full, cand_idx, 0.0, M_MOD, M_SIZES, args.batch_size))
        q_abs.append(cavity(model, x_full, cand_idx, -1.0, M_MOD, M_SIZES, args.batch_size))

        if args.dropout_runs > 0 and len(present_idx) > 0:
            gen = torch.Generator().manual_seed(int(args.seed) + split)
            dr = dropout_runs(model, x_full, present_idx, args.drop_frac,
                              args.dropout_runs, gen, M_MOD, M_SIZES, args.batch_size)
            drop_pool.append(dr[:, cand_arr])

        if anchor_x is not None:
            Pa = forward_probs(model, anchor_x, M_MOD, M_SIZES)[0]
            anch.append(Pa[cand_arr].detach().cpu().numpy())

        del model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    def ms(lst):
        a = np.stack(lst, axis=0)
        return a.mean(0), (a.std(0) if a.shape[0] > 1 else np.zeros(a.shape[1], np.float32))

    qf_m, qf_s = ms(q_full)
    qn_m, qn_s = ms(q_neu)
    qa_m, qa_s = ms(q_abs)
    cols = dict(
        q_full_mean=qf_m, q_full_sd=qf_s,
        q_mask_neutral_mean=qn_m, q_mask_neutral_sd=qn_s,
        q_mask_absent_mean=qa_m, q_mask_absent_sd=qa_s,
        self_anchor_neutral=qf_m - qn_m,
        self_anchor_absent=qf_m - qa_m,
        residual_context=logit(qa_m) - logit(marg_vec[cand_arr]),
    )
    if pl:
        cols['pl_field'] = np.stack(pl, 0).mean(0)
    if drop_pool:
        pool = np.concatenate(drop_pool, axis=0)
        cols['dropout_mean'] = pool.mean(0)
        cols['dropout_consensus'] = (pool > 0.5).mean(0)
    else:
        cols['dropout_mean'] = np.full(len(cand_idx), np.nan, np.float32)
        cols['dropout_consensus'] = np.full(len(cand_idx), np.nan, np.float32)
    if anch:
        cols['anchor_support'] = np.stack(anch, 0).mean(0)
    return cols, model_meta


# Module-conditioning matrices, set by _load_vocab_modules.
M_MOD = None
M_SIZES = None


def _load_vocab_modules(args, dev):
    global M_MOD, M_SIZES
    from ising_denoiser.data import load_feathers
    from ising_denoiser.modules import load_module_matrix
    _, _, vocab = load_feathers(args.vocab_feather, args.vocab_feather, frac=0.001)
    N = len(vocab)
    cog_to_idx = {c: i for i, c in enumerate(vocab)}
    _, M_MOD, M_SIZES = load_module_matrix(args.module_matrix, dev, H=1000)
    n_modules = M_MOD.shape[1]
    return vocab, cog_to_idx, N, n_modules


def _marg_vec(vocab, path):
    freq = load_value_tsv(path, 'p_c') if os.path.exists(path) else {}
    return np.array([freq.get(c, np.nan) for c in vocab], dtype=np.float64)


def _write(out_path, cog_ids, base_cols, score_cols, meta):
    import pandas as pd
    df = pd.DataFrame({'COG_ID': cog_ids, **base_cols, **score_cols})
    df.to_csv(out_path, sep='\t', index=False, float_format='%.6g')
    meta_path = out_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'wrote {out_path}  ({len(df)} candidates) and {meta_path}', file=sys.stderr)


def _base_meta(args, mode, model_pairs, model_meta, n_cand, extra=None):
    import torch
    m = dict(
        mode=mode, command=sys.argv, timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
        torch=torch.__version__, device=str(args.device),
        models=[os.path.relpath(p, ROOT) for _, p in model_pairs],
        splits=[s for s, _ in model_pairs], model_meta=model_meta,
        mask_semantics=dict(neutral='x_i=0 (no evidence)', absent='x_i=-1 (input votes absent)',
                            note='forward recomputes module/field/skip gates from the perturbed '
                                 'input, so masking i removes it from all conditioning (intended cavity)'),
        candidate_threshold=args.candidate_threshold, n_candidates=n_cand,
        dropout_runs=args.dropout_runs, drop_frac=args.drop_frac, seed=args.seed,
    )
    if extra:
        m.update(extra)
    return m


# ---- ancestral mode

def mode_ancestral(args):
    import torch
    from analyze_ancestral_node import pick_device, extract_node_probs, build_input_tensor
    dev, _ = pick_device(args.device)
    args.device = dev
    splits = parse_splits(args.splits)
    model_pairs = [(s, p) for s, p in resolve_models(args.models_template, splits, args.allow_glob)
                   if os.path.exists(p)]
    if not model_pairs:
        raise SystemExit("no checkpoints from --models-template exist (run on the cluster)")
    vocab, cog_to_idx, N, n_modules = _load_vocab_modules(args, dev)
    marg_vec = _marg_vec(vocab, args.marginal_freq)

    if args.input_prob_tsv:
        probs = load_value_tsv(args.input_prob_tsv, 'input_prob')
        src = f'input-prob-tsv {args.input_prob_tsv}'
    elif args.table and args.node:
        probs = extract_node_probs(args.table, args.node)
        src = f'table {args.table} node {args.node}'
    else:
        raise SystemExit("ancestral: give --input-prob-tsv OR (--table and --node)")
    # Prediction TSVs commonly carry explicit zero rows; dropping them makes them
    # encode as absent (x = -1) instead of neutral (x = 0).
    probs = {c: p for c, p in probs.items() if p > 0.0}
    x_full = build_input_tensor(probs, cog_to_idx, N, dev, mode='raw')

    prob_vec = x_full[0].detach().cpu().numpy()
    input_present = prob_vec > 0.0
    cand_mask = input_present & (prob_vec >= args.candidate_threshold)
    cand_idx = np.where(cand_mask)[0].tolist()
    present_idx = np.where(input_present)[0].tolist()
    if not cand_idx:
        raise SystemExit(f"no candidates (input-present and prob>={args.candidate_threshold}); "
                         f"lower --candidate-threshold")
    print(f'{src}: {len(present_idx)} input-present, {len(cand_idx)} candidates '
          f'(>= {args.candidate_threshold})', file=sys.stderr)

    anchor_x = None
    if args.anchor_pred_tsv:
        apost = load_value_tsv(args.anchor_pred_tsv, 'mean_actual')
        core = {c: v for c, v in apost.items() if v > args.anchor_threshold}
        anchor_x = build_input_tensor(core, cog_to_idx, N, dev, mode='raw')
        print(f'anchor: {len(core)} core COGs (posterior>{args.anchor_threshold}) '
              f'from {args.anchor_pred_tsv}', file=sys.stderr)

    cols, model_meta = run_diagnostics(model_pairs, x_full, cand_idx, present_idx,
                                       anchor_x, marg_vec, args, N, n_modules, dev)
    cand_cogs = [vocab[i] for i in cand_idx]
    base = dict(input_prob=prob_vec[cand_idx], input_present=np.ones(len(cand_idx), int),
                candidate=np.ones(len(cand_idx), int))
    meta = _base_meta(args, 'ancestral', model_pairs, model_meta, len(cand_idx),
                      extra=dict(input_source=src, anchor_pred_tsv=args.anchor_pred_tsv,
                                 anchor_threshold=args.anchor_threshold))
    _write(args.out, cand_cogs, base, cols, meta)


# ---- extant mode

def mode_extant(args):
    import torch
    from analyze_ancestral_node import pick_device
    from reconstruct_extant import find_held_out_genome, corrupt
    dev, _ = pick_device(args.device)
    args.device = dev
    splits = parse_splits(args.splits)
    template_pairs = dict(resolve_models(args.models_template, splits, args.allow_glob))
    vocab, cog_to_idx, N, n_modules = _load_vocab_modules(args, dev)
    marg_map = _marg_vec(vocab, args.marginal_freq)

    split, present, acc, spname, ngene = find_held_out_genome(
        args.genome, args.domain, args.val_glob, vocab, profile_glob=args.profile_glob)
    if split not in template_pairs:
        raise SystemExit(f"genome held out in split {split} but that split is not in "
                         f"--splits {args.splits}")
    path = template_pairs[split]
    if not os.path.exists(path):
        raise SystemExit(f"checkpoint missing for held-out split {split}: {path} (run on cluster)")
    print(f'{spname} (acc {acc}, {ngene} genes) -> held-out split {split}', file=sys.stderr)

    truth = (np.asarray(present) > 0).astype(int)
    clean = torch.tensor(2.0 * truth - 1.0, dtype=torch.float32, device=dev)
    marg_t = torch.tensor(np.nan_to_num(marg_map, nan=0.0), dtype=torch.float32, device=dev)
    ny = corrupt(clean, args.fn, args.fp, args.seed, fp_mode=args.fp_mode, marg=marg_t)
    x_full = ny.unsqueeze(0)

    inp = (ny.detach().cpu().numpy() > 0.0).astype(int)
    input_present = inp == 1
    cand_idx = np.where(input_present)[0].tolist()
    present_idx = cand_idx
    if not cand_idx:
        raise SystemExit("no input-present genes after corruption")
    n_fp = int(((truth == 0) & input_present).sum())
    n_ti = int(((truth == 1) & input_present).sum())
    print(f'corrupt fn={args.fn} fp={args.fp} ({args.fp_mode}): {len(cand_idx)} input-present '
          f'= {n_ti} true + {n_fp} injected FP', file=sys.stderr)

    anchor_x = None
    if args.anchor_pred_tsv:
        from analyze_ancestral_node import build_input_tensor
        apost = load_value_tsv(args.anchor_pred_tsv, 'mean_actual')
        core = {c: v for c, v in apost.items() if v > args.anchor_threshold}
        anchor_x = build_input_tensor(core, cog_to_idx, N, dev, mode='raw')

    cols, model_meta = run_diagnostics([(split, path)], x_full, cand_idx, present_idx,
                                       anchor_x, marg_map, args, N, n_modules, dev)
    cand_cogs = [vocab[i] for i in cand_idx]
    ca = np.asarray(cand_idx)
    base = dict(
        input_prob=np.ones(len(cand_idx), np.float32),
        input_present=np.ones(len(cand_idx), int),
        candidate=np.ones(len(cand_idx), int),
        truth=truth[ca],
        known_fp=((truth[ca] == 0)).astype(int),
        true_input=((truth[ca] == 1)).astype(int),
    )
    meta = _base_meta(args, 'extant', [(split, path)], model_meta, len(cand_idx),
                      extra=dict(genome=spname, accession=acc, held_out_split=split,
                                 fn=args.fn, fp=args.fp, fp_mode=args.fp_mode,
                                 n_true_input=n_ti, n_known_fp=n_fp))
    _write(args.out, cand_cogs, base, cols, meta)


# ---- summarize mode

def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks over ties: the tie-corrected Mann-Whitney AUC.
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def average_precision(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if (y == 1).sum() == 0:
        return float('nan')
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int((y == 1).sum()), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def spearman(a, b):
    """Spearman rank correlation, nan-safe."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def mode_summarize(args):
    import pandas as pd
    df = pd.read_csv(args.tsv, sep='\t')
    if 'known_fp' not in df.columns:
        raise SystemExit(f"{args.tsv}: no known_fp column -- summarize is for extant outputs")
    y = df['known_fp'].to_numpy()
    truth = df['truth'].to_numpy() if 'truth' in df else None
    print(f'== known-FP detection ({int((y==1).sum())} FP / {len(y)} candidates) ==')
    scores = {
        '-q_mask_absent_mean':  -df['q_mask_absent_mean'].to_numpy(),
        '-q_mask_neutral_mean': -df['q_mask_neutral_mean'].to_numpy(),
        'self_anchor_absent':    df['self_anchor_absent'].to_numpy(),
        '-dropout_mean':        -df['dropout_mean'].to_numpy(),
    }
    if 'anchor_support' in df.columns:
        scores['-anchor_support'] = -df['anchor_support'].to_numpy()
    if 'pl_field' in df.columns:
        scores['-pl_field'] = -df['pl_field'].to_numpy()
    print(f'  {"score":24s} {"ROC-AUC":>8s} {"AP":>8s}')
    for name, s in scores.items():
        print(f'  {name:24s} {roc_auc(y, s):8.3f} {average_precision(y, s):8.3f}')

    if 'pl_field' in df.columns:
        rho = spearman(df['pl_field'].to_numpy(), df['q_mask_absent_mean'].to_numpy())
        print(f'\n== GATE: one-step PL field vs relaxed cavity ==')
        print(f'  Spearman(pl_field, q_mask_absent_mean) = {rho:+.3f}  '
              f'(near +1 => the bare h+J~ field IS the cavity; relaxation/gates/'
              f'Delta add nothing separable -> no training-free route)')

    print(f'\n== pruning table (support = q_mask_absent_mean; keep if >= thr) ==')
    support = df['q_mask_absent_mean'].to_numpy()
    n_true = int((truth == 1).sum()) if truth is not None else int((df['true_input'] == 1).sum())
    n_fp = int((y == 1).sum())
    tcol = (truth == 1) if truth is not None else (df['true_input'].to_numpy() == 1)
    print(f'  {"thr":>5s} {"true_recall":>11s} {"FP_removed":>10s} '
          f'{"retain_true":>11s} {"retain_FP":>9s}')
    base_keep = df['q_full_mean'].to_numpy() > 0.5
    def row(label, keep):
        tr = (tcol & keep).sum() / max(n_true, 1)
        fr = ((y == 1) & ~keep).sum() / max(n_fp, 1)
        print(f'  {label:>5s} {tr:11.3f} {fr:10.3f} {int((tcol & keep).sum()):11d} '
              f'{int(((y==1) & keep).sum()):9d}')
    row('q>.5', base_keep)
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        row(f'{thr:.2f}', support >= thr)


# ---- main

def add_common(p):
    p.add_argument('--models-template', dest='models_template', required=True,
                   help="checkpoint path with a {split} placeholder, e.g. "
                        "'gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split{split}/model_ho3.pth'")
    p.add_argument('--splits', required=True, help="PINNED, e.g. '1-10' or '1,3,5'. Never '*'.")
    p.add_argument('--allow-glob', action='store_true', help="permit wildcard model paths (discouraged)")
    p.add_argument('--candidate-threshold', type=float, default=0.5,
                   help="a gene is a candidate if input-present and input_prob >= this "
                        "(default 0.5; use 0 to test every input-present gene on a dense input)")
    p.add_argument('--dropout-runs', type=int, default=0, help="context-dropout runs per split (0=off)")
    p.add_argument('--drop-frac', type=float, default=0.5, help="fraction of present genes dropped per run")
    p.add_argument('--anchor-pred-tsv', default=None, help="prediction TSV (COG_ID, mean_actual) for the sparse-core anchor")
    p.add_argument('--anchor-threshold', type=float, default=0.9, help="anchor core: posterior > this")
    p.add_argument('--marginal-freq', default='data/cog_marginal_frequency.tsv')
    p.add_argument('--vocab-feather', default='data/COG_train1_phylum.feather')
    p.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', required=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)

    a = sub.add_parser('ancestral', help='diagnose a dense ancestral input')
    add_common(a)
    a.add_argument('--table', default=None, help='reconciliation/GLD table (with --node)')
    a.add_argument('--node', default=None, help='node-column label in --table')
    a.add_argument('--input-prob-tsv', dest='input_prob_tsv', default=None,
                   help='TSV with COG_ID,input_prob (e.g. marg_soft.tsv)')

    e = sub.add_parser('extant', help='diagnose a corrupted held-out genome')
    add_common(e)
    e.add_argument('--domain', default='', help='d__Bacteria / d__Archaea')
    e.add_argument('--genome', required=True, help="species substring")
    e.add_argument('--fn', type=float, default=0.9)
    e.add_argument('--fp', type=float, default=0.05)
    e.add_argument('--fp-mode', choices=['uniform', 'marginal'], default='marginal')
    e.add_argument('--val-glob', default='data/COG_val{split}_phylum.feather')
    e.add_argument('--profile-glob', default='')

    s = sub.add_parser('summarize', help='ROC/AP + pruning table from an extant output TSV')
    s.add_argument('tsv')

    args = ap.parse_args()
    if args.mode == 'ancestral':
        mode_ancestral(args)
    elif args.mode == 'extant':
        mode_extant(args)
    elif args.mode == 'summarize':
        mode_summarize(args)


if __name__ == '__main__':
    main()
