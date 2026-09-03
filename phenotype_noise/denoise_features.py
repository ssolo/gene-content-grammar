#!/usr/bin/env python3
"""Denoise step of the phenotype experiment. Depends on torch alone (no sklearn,
no xgboost) so it can run apart from the prediction step.

For one split and every (fn, fp) noise cell: apply the deterministic noise to the
2677 COG feature counts, embed the noisy presence in the genome's full 4789-COG
profile (real genome context, from aerob_profiles.npz), denoise with the
HQ-marginal cross-input-consistent ensemble, and read the 2677 features back as
binary presence. Only the denoised features are stored; the prediction step
regenerates the same noise from (seed_base, split, fn, fp) for the noisy arms.

Usage (one split per invocation, so splits can be run in parallel):
  python3 phenotype_noise/denoise_features.py --split 0 --ensemble 10 --device cuda \
      --fn-grid 0.0 0.2 0.4 0.6 0.8 1.0 --fp-grid 0.0 0.1 0.2 \
      --out data/pheno_den
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'phenotype_noise'))
from analyze_ancestral_node import build_model_for_ckpt          # noqa: E402
from ising_denoiser.modules import load_module_matrix            # noqa: E402
from ising_denoiser.training import strip_compile_prefix         # noqa: E402
from noise_util import flip_with_fractional_noise, noise_seed    # noqa: E402

PH = os.path.join(REPO, 'phenotype_noise')
ENS = 'gsd_results_consistency_T20_mix_fp_marginal_cons_l1.0_j1.0_hq_split{s}/model_ho3.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--ensemble', type=int, default=10)
    ap.add_argument('--fn-grid', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument('--fp-grid', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    ap.add_argument('--seed-base', type=int, default=7)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(REPO, 'data/pheno_den'))
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    dev = torch.device(A.device)

    Z = np.load(os.path.join(PH, 'aerob_profiles.npz'), allow_pickle=True)
    prof = Z['prof']; covered = Z['covered']; feat_idx = Z['feat_idx']
    acc_to_row = {a: i for i, a in enumerate(Z['acc'])}
    valid = feat_idx >= 0
    fvalid = feat_idx[valid]
    with open(os.path.join(PH, 'denoise_bundle.pkl'), 'rb') as fh:
        bundle = pickle.load(fh)
    S = bundle[A.split]
    X, acc, y = S['X'], S['acc'], S['y']
    n = X.shape[0]

    mm = torch.load(os.path.join(REPO, 'data/module_matrix_kegg.pt'), weights_only=False)
    N = len(mm['cog_names'])
    _, M_mod, M_sizes = load_module_matrix(os.path.join(REPO, 'data/module_matrix_kegg.pt'), dev, 1000)
    n_mod = M_mod.shape[1]
    models = []
    for s in range(1, A.ensemble + 1):
        ck = os.path.join(REPO, ENS.format(s=s))
        sd = strip_compile_prefix({k.replace('module.', ''): v
                                   for k, v in torch.load(ck, map_location=dev, weights_only=True).items()})
        m, _, _, _ = build_model_for_ckpt(sd, dev, N, n_mod, ck)
        m.load_state_dict(sd, strict=False); m.eval()
        models.append(m)
    print(f'split {A.split}: n={n}  ensemble={len(models)}  device={dev}', flush=True)

    # Clean genome presence (n, N) in {0, 1}, in the denoiser's COG column order.
    # A genome with no profile row stays all-absent.
    prof01 = np.zeros((n, N), dtype=np.float32)
    for i, a in enumerate(acc):
        r = acc_to_row.get(a)
        if r is not None and covered[r]:
            prof01[i] = prof[r]

    @torch.no_grad()
    def denoise(x4789):
        out = torch.zeros(x4789.shape[0], N)
        xt = torch.from_numpy(x4789)
        for b in range(0, xt.shape[0], A.batch):
            xb = xt[b:b + A.batch].to(dev)
            acc_p = torch.zeros(xb.shape[0], N, device=dev)
            for m in models:
                acc_p += ((m(xb, M_mod, M_sizes)[0] + 1) * 0.5).clamp(0, 1)
            out[b:b + A.batch] = (acc_p / len(models)).cpu()
        return out.numpy()

    grid = [(fn, fp) for fn in A.fn_grid for fp in A.fp_grid]
    den = {}
    for gi, (fn, fp) in enumerate(grid):
        sd = noise_seed(A.seed_base, A.split, fn, fp)
        Xn = flip_with_fractional_noise(X, fp, fn, sd)            # (n, 2677) noisy feature counts
        # The WHOLE genome is corrupted, context COGs included, so the denoiser
        # never sees a clean-context leak; the draw is independent (seed + 1) of
        # the feature-count noise. The feature slots are then overwritten from
        # the noisy counts so the noisy arm and the denoiser's input agree.
        prof_noisy = flip_with_fractional_noise(prof01, fp, fn, sd + 1)
        x4789 = 2.0 * prof_noisy - 1.0
        x4789[:, fvalid] = 2.0 * (Xn[:, valid] > 0).astype(np.float32) - 1.0
        post = denoise(x4789)
        Xd = (Xn > 0).astype(np.uint8)                      # features with no COG column stay noisy
        Xd[:, valid] = (post[:, fvalid] > 0.5).astype(np.uint8)
        den[f'd{gi}'] = Xd
        print(f'  [{gi+1}/{len(grid)}] fn={fn} fp={fp} done', flush=True)

    np.savez_compressed(os.path.join(A.out, f'den_split_{A.split}.npz'),
                        y=y, acc=acc, grid=np.array(grid),
                        seed_base=A.seed_base, **den)
    print(f'wrote {A.out}/den_split_{A.split}.npz', flush=True)


if __name__ == '__main__':
    main()
