#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""score_ecoli_replicates.py -- noise-instance replicates for the Fig 3 ladder.

Each Fig 3 point (E. coli recovery MCC/recall/precision at fn=0.4/0.6/0.8 for a
given leave-clade-out rung) comes from a single noisy corruption instance, and
recovery depends on which genes the false-negative channel happens to drop. This
script corrupts E. coli's clean gene-content vector with many independent noise
seeds, denoises each with the rung model and emits confusion counts per
(fn, replicate); plot_ecoli_timeladder.py turns those into an HPD ribbon around
the published markers.

The clean vector is the `truth` column of the committed point-estimate TSV,
identical across rungs. The per-fn corruption seed is the same deterministic base
reconstruct_extant.py uses (42 + 1000*fn + 7919*1000*fp) plus the replicate
index, so replicate 0 reproduces the published point exactly. corrupt() and
run_model() are imported rather than reimplemented, so this shares the point
figure's code path.

Usage:
  python scripts/score_ecoli_replicates.py \
      --model gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_phylum/model_ho3.pth \
      --truth-tsv analysis/ecoli_zoom/recover_ecoli_ecolizoom_phylum_ho3_fn468.tsv \
      --reps 500 --fn 0.4 0.6 0.8 --fp 0.01 \
      --out analysis/ecoli_zoom/reps_ecoli_ecolizoom_phylum_ho3_fn468.tsv
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from reconstruct_extant import load_vocab, corrupt              # noqa: E402
from analyze_ancestral_node import build_model_for_ckpt, run_model  # noqa: E402
from ising_denoiser.modules import load_module_matrix           # noqa: E402
from ising_denoiser.training import strip_compile_prefix        # noqa: E402


def metrics(present, pred):
    """present/pred boolean (N,) -> confusion counts vs the clean truth."""
    t, p = present.astype(bool), pred.astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum())
    fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='rung checkpoint (model_ho3.pth)')
    ap.add_argument('--truth-tsv', required=True,
                    help='committed point TSV; its truth column is the clean vector')
    ap.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    ap.add_argument('--fn', type=float, nargs='+', default=[0.4, 0.6, 0.8])
    ap.add_argument('--fp', type=float, default=0.01)
    ap.add_argument('--fp-mode', choices=['uniform', 'marginal'], default='marginal',
                    help="false-positive injection: 'marginal' (DEFAULT, matches the "
                         "published eval) draws common-gene contamination by cross-genome "
                         "marginal frequency; 'uniform' is random rare junk.")
    ap.add_argument('--marginal-freq', default='data/cog_marginal_frequency.tsv',
                    help='COG_ID<TAB>p_c table for --fp-mode marginal (same default as '
                         'reconstruct_extant.py)')
    ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--allow-tf32', action='store_true',
                    help='allow TF32 matmul on an 80 GB GPU (default: OFF -> full fp32, '
                         'matching the a 16 GB GPU published eval + training precision; '
                         'TF32 systematically lowers the denoiser borderline calls)')
    ap.add_argument('--out', required=True, help='per-(fn,rep) confusion-count TSV')
    ap.add_argument('--dump-typical', action='store_true',
                    help='also dump the per-COG recover TSV for the TYPICAL replicate '
                         '(the one whose fn=0.8 MCC is the median) -- a representative '
                         'noise instance for the Fig 2 mosaic, from this same pipeline')
    ap.add_argument('--dump-tsv', default='',
                    help='path for --dump-typical (COG_ID/fn/truth/input_present/denoised_prob)')
    A = ap.parse_args()

    if A.device == 'auto':
        A.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(A.device)
    if dev.type == 'cuda' and not A.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(f'device={dev}  tf32={A.allow_tf32}', file=sys.stderr)

    cog_names = load_vocab(A.module_matrix)
    N = len(cog_names)
    _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, H=1000)
    n_modules = M_mod.shape[1]

    # Marginal-FP prior: per-COG cross-genome frequency p_c, so injected
    # contamination is common genes rather than random junk.  Without it,
    # corrupt(fp_mode='marginal', marg=None) silently falls back to uniform.
    marg = None
    if A.fp_mode == 'marginal':
        fr = pd.read_csv(A.marginal_freq, sep='\t')
        m = dict(zip(fr['COG_ID'], fr['p_c']))
        marg = torch.tensor([float(m.get(c, 0.0)) for c in cog_names], dtype=torch.float32)
        print(f'[fp-mode marginal] p_c for {int((marg > 0).sum())}/{N} COGs '
              f'<- {A.marginal_freq}', file=sys.stderr)

    # Clean vector from the TSV's truth column, reordered to the model's COG vocab.
    df = pd.read_csv(A.truth_tsv, sep='\t')
    truth_map = dict(zip(df['COG_ID'], df['truth'].astype(int)))
    present = np.array([int(truth_map.get(c, 0)) for c in cog_names], dtype=np.float32)
    ngenes = int(present.sum())
    clean = torch.from_numpy(2.0 * present - 1.0).to(dev)
    print(f'clean E. coli vector: {ngenes}/{N} present (from {A.truth_tsv})',
          file=sys.stderr)

    sd = torch.load(A.model, map_location=dev, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = strip_compile_prefix(sd)
    model, cls_kind, T, onsager = build_model_for_ckpt(sd, dev, N, n_modules, A.model)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'model: {cls_kind} T={T} <- {A.model}', file=sys.stderr)

    rows = []
    for fn in A.fn:
        base = 42 + int(fn * 1000) + 7919 * int(A.fp * 1000)  # reconstruct_extant.py seed base
        for rep in range(A.reps):
            seed = base + rep                                  # rep 0 == published point
            ny = corrupt(clean, fn, A.fp, seed, fp_mode=A.fp_mode, marg=marg)
            x_out = run_model(model, ny.unsqueeze(0), M_mod, M_sizes)
            prob = ((x_out + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy().ravel()
            pred = prob > 0.5
            tp, fp, fn_, tn = metrics(present, pred)
            rows.append((f'{fn:g}', rep, ngenes, tp, fp, fn_, tn))
        print(f'fn={fn:g}: {A.reps} replicates done', file=sys.stderr)

    out = Path(A.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('fn\trep\ttruth\tTP\tFP\tFN\tTN\n')
        for r in rows:
            fh.write('\t'.join(str(x) for x in r) + '\n')
    print(f'wrote {out}  ({len(rows)} rows = {len(A.fn)} fn x {A.reps} reps)',
          file=sys.stderr)

    # Per-COG reconstruction of one representative draw, for the Fig 2 mosaic.
    if A.dump_typical and A.dump_tsv:
        df = pd.DataFrame(rows, columns=['fn', 'rep', 'truth', 'TP', 'FP', 'FN', 'TN'])
        fn_pick = max(A.fn)                                   # deepest operating point
        d = df[df['fn'] == f'{fn_pick:g}'].copy()
        den = np.sqrt((d.TP + d.FP) * (d.TP + d.FN) * (d.TN + d.FP) * (d.TN + d.FN))
        d['mcc'] = np.where(den > 0, (d.TP * d.TN - d.FP * d.FN) / den, np.nan)
        d = d.sort_values('mcc').reset_index(drop=True)
        rstar = int(d.loc[len(d) // 2, 'rep'])                # median-MCC replicate
        out_rows = []
        for fn in A.fn:
            base = 42 + int(fn * 1000) + 7919 * int(A.fp * 1000)
            ny = corrupt(clean, fn, A.fp, base + rstar, fp_mode=A.fp_mode, marg=marg)
            prob = ((run_model(model, ny.unsqueeze(0), M_mod, M_sizes) + 1.0) / 2.0
                    ).clamp(0.0, 1.0).cpu().numpy().ravel()
            inp = (ny.cpu().numpy() > 0).astype(int)
            for i, c in enumerate(cog_names):
                out_rows.append((c, f'{fn:g}', int(present[i]), int(inp[i]), f'{prob[i]:.4f}'))
        dp = Path(A.dump_tsv); dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, 'w') as fh:
            fh.write('COG_ID\tfn\ttruth\tinput_present\tdenoised_prob\n')
            for r in out_rows:
                fh.write('\t'.join(str(x) for x in r) + '\n')
        with open(dp.with_suffix('.meta.tsv'), 'w') as fh:
            fh.write('key\tvalue\nspecies\tEscherichia coli\nmodel_label\tbac-FT-fp-marginal-HQ\n')
            fh.write(f'truth_total\t{ngenes}\nfp\t{A.fp:g}\nfn_levels\t{",".join(f"{f:g}" for f in A.fn)}\n')
            fh.write(f'typical_replicate\t{rstar}\nselected_by\tmedian fn={fn_pick:g} MCC over {A.reps} draws\n')
        print(f'typical replicate = {rstar} (median fn={fn_pick:g} MCC) -> {dp}',
              file=sys.stderr)


if __name__ == '__main__':
    main()
