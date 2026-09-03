#!/usr/bin/env python3
"""Stratified recovery (MCC) and calibration (ECE) over held-out validation genomes.

For each cross-validation split, corrupt every held-out genome with the
reconciliation noise model (false negatives at --fn, false positives at --fp),
denoise it under that split's model and pool by CheckM-completeness bin and by
phylum. MCC uses the pooled TP/FP/TN/FN, ECE the pooled 20-bin reliability
diagram (present vs posterior). Whole-phylum holdout: every genome of a phylum
sits in exactly one split's validation set, so each phylum is scored by a model
that never trained on it.

Output: long TSV with columns (dim, stratum, fn, n_genomes, MCC, ECE).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ancestral_node import build_model_for_ckpt              # noqa: E402
from ising_denoiser.modules import load_module_matrix                # noqa: E402
from ising_denoiser.training import strip_compile_prefix             # noqa: E402


def denoise_batch(model, xb, M_mod, M_sizes):
    """Batched relaxation -> (B, N) posterior spins.

    analyze_ancestral_node.run_model returns only the first genome of a batch.
    """
    with torch.no_grad():
        out = model(xb, M_mod=M_mod, M_sizes=M_sizes)
    return out[0] if isinstance(out, tuple) else out

NB = 20                                  # ECE reliability bins
COMPL_BINS = [(0, 80), (80, 90), (90, 95), (95, 99), (99, 100.01)]


def load_vocab(mm):
    return list(torch.load(mm, weights_only=False)['cog_names'])


def apply_noise(clean, fn, fp, seed):
    """present(+1)->absent(-1) w.p. fn; absent(-1)->present(+1) w.p. fp; seeded."""
    g = torch.Generator().manual_seed(seed)
    ny = clean.clone()
    ny[(clean == 1) & (torch.rand(clean.shape, generator=g) < fn)] = -1
    ny[(clean == -1) & (torch.rand(clean.shape, generator=g) < fp)] = 1
    return ny


def compl_label(c):
    if not np.isfinite(c):
        return 'NA'
    for lo, hi in COMPL_BINS:
        if lo <= c < hi:
            return f'{lo}-{min(hi,100):g}'
    return 'NA'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='gsd_results_higher_order_nohidden_T20_split{split}/model_ho3.pth',
                    help='checkpoint template; {split} -> 1..10 (default: HO-T20 generalist).')
    ap.add_argument('--val-glob', default='data/COG_val{split}_phylum.feather')
    ap.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    ap.add_argument('--fn', type=float, nargs='+', default=[0.5, 0.75, 0.9])
    ap.add_argument('--fp', type=float, default=0.01)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--batch', type=int, default=32,
                    help='genomes/forward. The HO attention materialises a '
                         'B*heads*N^2 score tensor, so keep small (GPU ~32-64; '
                         'CPU ~2-8).')
    ap.add_argument('--splits', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--limit', type=int, default=0,
                    help='random subsample genomes/split (0=all; seed 1000+split).')
    ap.add_argument('--raw-out', default='',
                    help='also pickle the raw per-stratum accumulators here, for '
                         'merging split-sharded array runs (MCC/ECE are not '
                         'additive -- shards must combine at the count level via '
                         'scripts/merge_strat_raw.py).')
    ap.add_argument('--out', default='data/stratified_eval.tsv')
    A = ap.parse_args()

    if A.device in ('cuda', 'cpu', 'mps'):
        dev = torch.device(A.device)
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    cog = load_vocab(A.module_matrix); N = len(cog)
    cogpos = {c: i for i, c in enumerate(cog)}
    _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, 1000)
    n_mod = M_mod.shape[1] if M_mod is not None else 1

    # acc[(fn,dim,label)] = [TP,FP,TN,FN, hist_n(NB), hist_truth(NB), hist_conf(NB), ngen]
    acc = {}

    def slot(fn, dim, label):
        k = (fn, dim, label)
        if k not in acc:
            acc[k] = [0, 0, 0, 0, np.zeros(NB), np.zeros(NB), np.zeros(NB), 0]
        return acc[k]

    for split in A.splits:
        ck = A.models.format(split=split); vp = A.val_glob.format(split=split)
        if not (Path(ck).exists() and Path(vp).exists()):
            print(f'  skip split {split} (missing model/val)', file=sys.stderr); continue
        df = pd.read_feather(vp)
        if A.limit and A.limit < len(df):
            df = df.sample(n=A.limit, random_state=1000 + split).reset_index(drop=True)
        cols = [c for c in df.columns if c in cogpos]
        present = np.zeros((len(df), N), dtype=np.float32)
        present[:, [cogpos[c] for c in cols]] = (df[cols].to_numpy() > 0).astype(np.float32)
        compl = pd.to_numeric(df.get('checkm_completeness'), errors='coerce').to_numpy()
        phy = df['phylum'].astype(str).to_numpy()
        clean = torch.from_numpy(2 * present - 1)

        sd = torch.load(ck, map_location=dev, weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = strip_compile_prefix(sd)
        model, _, _, _ = build_model_for_ckpt(sd, dev, N, n_mod, ck)
        model.load_state_dict(sd, strict=False)
        model.eval()
        print(f'  split {split}: {len(df)} genomes', file=sys.stderr)

        for fn in A.fn:
            ny = apply_noise(clean, fn, A.fp, 42 + int(fn * 1000))
            post = np.zeros((len(df), N), dtype=np.float32)
            with torch.no_grad():
                for b in range(0, len(df), A.batch):
                    xo = denoise_batch(model, ny[b:b + A.batch].to(dev), M_mod, M_sizes)
                    post[b:b + A.batch] = ((xo + 1) / 2).clamp(0, 1).cpu().numpy()
            pred = post > 0.5; truth = present > 0.5
            tp = (pred & truth).sum(1); fp = (pred & ~truth).sum(1)
            tn = (~pred & ~truth).sum(1); fnn = (~pred & truth).sum(1)
            bidx = np.clip((post * NB).astype(int), 0, NB - 1)
            for g in range(len(df)):
                hn = np.bincount(bidx[g], minlength=NB)
                ht = np.bincount(bidx[g], weights=truth[g].astype(float), minlength=NB)
                hc = np.bincount(bidx[g], weights=post[g], minlength=NB)
                for dim, label in (('completeness', compl_label(compl[g])), ('phylum', phy[g])):
                    a = slot(fn, dim, label)
                    a[0] += tp[g]; a[1] += fp[g]; a[2] += tn[g]; a[3] += fnn[g]; a[7] += 1
                    a[4] += hn; a[5] += ht; a[6] += hc

    if A.raw_out:
        import pickle
        with open(A.raw_out, 'wb') as f:
            pickle.dump(acc, f)
        print(f'wrote raw accumulators -> {A.raw_out}', file=sys.stderr)

    rows = []
    for (fn, dim, label), a in acc.items():
        TP, FP, TN, FN = (float(a[i]) for i in range(4))
        den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = (TP * TN - FP * FN) / den if den > 0 else 0.0
        n, st, sc = a[4], a[5], a[6]; tot = n.sum()
        ece = float(sum((n[i] / tot) * abs(st[i] / n[i] - sc[i] / n[i])
                        for i in range(NB) if n[i] > 0)) if tot > 0 else 0.0
        rows.append((dim, label, fn, a[7], round(mcc, 4), round(ece, 4)))
    out = pd.DataFrame(rows, columns=['dim', 'stratum', 'fn', 'n_genomes', 'MCC', 'ECE'])
    out = out.sort_values(['dim', 'stratum', 'fn']).reset_index(drop=True)
    out.to_csv(A.out, sep='\t', index=False)
    print(f'wrote {A.out}: {len(out)} rows '
          f'({out.stratum[out.dim=="phylum"].nunique()} phyla, '
          f'{out.stratum[out.dim=="completeness"].nunique()} completeness bins)')


if __name__ == '__main__':
    main()
