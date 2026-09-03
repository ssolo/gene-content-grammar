#!/usr/bin/env python3
"""Coherent false-positive eval: does the realistic-FP curriculum reject coherent
contamination (a foreign organism's modules grafted in) better than the default
and uniform-FP models?

Random per-gene FP is the easy regime: a lone absent gene rarely fits a genome's
couplings, so every model silences it (Section "Robustness to false positives").
The FPs that survive reconciliation are functionally coherent - whole modules, or
a neighbouring organism's module signature mis-mapped in as a block. That is what
the realistic-FP fine-tune trains on (ReconciliationNoiseDataset._swap_signature
in ising_denoiser/data.py); this script measures the effect at inference.

For each held-out validation genome:
  1. delete present genes at rate --fn (random; the deep-ancestral regime),
  2. graft up to --n-graft modules from another validation genome - modules
     present in the donor (frac > 0.8) and absent in the target (frac < 0.2),
     turning those absent members present,
  3. denoise through the model,
and pool, over a decision-threshold sweep:
  recall              = truly-present genes recovered (post > tau),
  coherent FP-removed = grafted foreign genes switched back off (post <= tau).

Corruption is seeded from --seed and the split index alone, so every model sees
bit-identical contamination and the only difference is the network.

Output: long TSV with columns
  (model, split, fn, tau, n_present, n_grafted, n_recovered, n_fp_removed)
to be pooled across splits (recall = n_recovered / n_present, and so on).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ancestral_node import build_model_for_ckpt              # noqa: E402
from ising_denoiser.models import set_coupling_controls              # noqa: E402
from ising_denoiser.modules import load_module_matrix                # noqa: E402
from ising_denoiser.training import strip_compile_prefix             # noqa: E402

# Per-module presence fraction below which the target counts as lacking a module
# and above which the donor counts as carrying it; same values as the training
# curriculum.
FRAC_ABSENT, FRAC_PRESENT = 0.2, 0.8


def denoise_batch(model, xb, M_mod, M_sizes):
    with torch.no_grad():
        out = model(xb, M_mod=M_mod, M_sizes=M_sizes)
    return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='ckpt template; {split}->1..10')
    ap.add_argument('--label', required=True, help='model label for the output rows')
    ap.add_argument('--val-glob', default='data/COG_val{split}_phylum.feather')
    ap.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    ap.add_argument('--fn', type=float, nargs='+', default=[0.5, 0.9])
    ap.add_argument('--n-graft', type=int, default=3, help='max foreign modules grafted (module_swap_max)')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--splits', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--limit', type=int, default=4000, help='random genomes/split (seed 7+split).')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', required=True)
    # Coupling controls applied post-hoc to the frozen model at inference.
    ap.add_argument('--coupling-shrink-alpha', type=float, default=None,
                    help='support-shrink J~ by s_c=sqrt(p_c/(p_c+alpha)); off if unset')
    ap.add_argument('--genome-mass-norm', action='store_true',
                    help='scale the pairwise coupling drive by sqrt(kbar/k_n)')
    ap.add_argument('--freq-tsv', default='data/cog_marginal_frequency.tsv',
                    help='per-COG presence frequency p_c for the controls')
    A = ap.parse_args()

    if A.device in ('cuda', 'cpu', 'mps'):
        dev = torch.device(A.device)
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    mm = torch.load(A.module_matrix, weights_only=False)
    cog = list(mm['cog_names']); N = len(cog); cogpos = {c: i for i, c in enumerate(cog)}
    _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, 1000)
    n_mod = M_mod.shape[1] if M_mod is not None else 1
    # module -> member gene indices, kept on CPU for the per-genome graft loop
    Mc = M_mod.detach().cpu()
    members = [torch.where(Mc[:, j] > 0)[0] for j in range(n_mod)]
    sizes_c = M_sizes.detach().cpu().clamp(min=1).float()

    rows = []
    for split in A.splits:
        ck = A.models.format(split=split); vp = A.val_glob.format(split=split)
        if not (Path(ck).exists() and Path(vp).exists()):
            print(f'  skip split {split} (missing model/val)', file=sys.stderr); continue
        df = pd.read_feather(vp)
        if A.limit and A.limit < len(df):
            df = df.sample(n=A.limit, random_state=A.seed + split).reset_index(drop=True)
        cols = [c for c in df.columns if c in cogpos]
        present = np.zeros((len(df), N), dtype=np.float32)
        present[:, [cogpos[c] for c in cols]] = (df[cols].to_numpy() > 0).astype(np.float32)
        clean = torch.from_numpy(2.0 * present - 1.0)                    # (B, N) in {-1, +1}
        B = clean.shape[0]
        # per-module presence fraction of every genome, (B, n_mod) in [0, 1]
        fracs = (torch.from_numpy(present) @ Mc) / sizes_c

        sd = torch.load(ck, map_location=dev, weights_only=True)
        sd = strip_compile_prefix({k.replace('module.', ''): v for k, v in sd.items()})
        model, _, _, _ = build_model_for_ckpt(sd, dev, N, n_mod, ck)
        model.load_state_dict(sd, strict=False); model.eval()
        ctl = set_coupling_controls(
            model, freq_tsv=A.freq_tsv, shrink_alpha=A.coupling_shrink_alpha,
            genome_mass_norm=A.genome_mass_norm, device=dev)
        print(f'  [{A.label}] split {split}: {B} genomes | {ctl}', file=sys.stderr)

        for fn in A.fn:
            g = torch.Generator().manual_seed(A.seed * 1000 + split * 10 + int(fn * 10))
            donor = torch.randperm(B, generator=g)
            noisy = clean.clone()
            grafted = torch.zeros(B, N, dtype=torch.bool)
            for i in range(B):
                # (1) random FN: +1 -> -1
                fnmask = (clean[i] == 1) & (torch.rand(N, generator=g) < fn)
                noisy[i][fnmask] = -1.0
                # (2) coherent FP: graft a foreign organism's modules
                j = int(donor[i]);  j = (i + 1) % B if j == i else j
                diff = torch.where((fracs[j] > FRAC_PRESENT) & (fracs[i] < FRAC_ABSENT))[0]
                if diff.numel() == 0:
                    continue
                k = min(A.n_graft, diff.numel())
                pick = diff[torch.randperm(diff.numel(), generator=g)[:k]]
                for m in pick.tolist():
                    idx = members[m]
                    absent = idx[noisy[i][idx] == -1.0]                  # graft only members not already present
                    noisy[i][absent] = 1.0
                    grafted[i][absent] = True

            truth = clean == 1                                           # present in the clean genome
            post = np.zeros((B, N), dtype=np.float32)
            with torch.no_grad():
                for b in range(0, B, A.batch):
                    xo = denoise_batch(model, noisy[b:b + A.batch].to(dev), M_mod, M_sizes)
                    post[b:b + A.batch] = ((xo + 1) / 2).clamp(0, 1).cpu().numpy()
            post_t = torch.from_numpy(post)
            n_present = int(truth.sum()); n_graft = int(grafted.sum())
            for tau in A.thresholds:
                pred = post_t > tau
                n_rec = int((pred & truth).sum())
                n_rem = int((~pred & grafted).sum())
                rows.append((A.label, split, fn, tau, n_present, n_graft, n_rec, n_rem))

    out = pd.DataFrame(rows, columns=['model', 'split', 'fn', 'tau',
                                      'n_present', 'n_grafted', 'n_recovered', 'n_fp_removed'])
    out.to_csv(A.out, sep='\t', index=False)
    print(f'wrote {A.out}: {len(out)} rows', file=sys.stderr)


if __name__ == '__main__':
    main()
