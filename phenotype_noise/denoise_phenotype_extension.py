#!/usr/bin/env python3
"""Aerobicity prediction under gene-content noise, with a DENOISED-input arm.

Aerobicity XGBoost predictors are scored across an fn x fp noise grid in four
arms: the original and the noise-robust predictor, each fed either the noisy COG
profile or the same profile passed through the denoiser as a front-end
(orig_noisy, robust_noisy, orig_denoised, robust_denoised). The denoiser is the
10-split cross-input-consistent marginal-HQ ensemble. Denoised features enter as
BINARY presence (posterior > 0.5), matching the count data (median count 1).

Pipeline per (fn, fp) cell:
  1. corrupt the 2677 COG feature counts once, under the hard-FN convention
     (a false negative sets the count to 0 rather than decrementing it);
  2. embed the noisy feature PRESENCE into the genome's full 4789-COG profile,
     so that the ~2114 non-feature COGs supply real genome context
     (see build_aerob_profiles.py);
  3. denoise the 4789-vector with the ensemble (mean posterior);
  4. read back the 2677 feature COGs as binary presence -> predict.
Row->accession is recovered by content-matching X to all_gene_annotations.tsv.

Writes <out>/pheno_denoise_results.pkl: per arm and per (fn, fp), the
across-split mean and std of MCC, F1, Brier, ECE, accuracy and balanced accuracy.

Usage:
  python3 phenotype_noise/denoise_phenotype_extension.py \
      --n-splits 10 --fp-grid 0.0 0.1 0.2 --ensemble 10 --device mps \
      --out phenotype_noise/results
"""
import argparse
import hashlib
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from analyze_ancestral_node import build_model_for_ckpt            # noqa: E402
from ising_denoiser.modules import load_module_matrix              # noqa: E402
from ising_denoiser.training import strip_compile_prefix           # noqa: E402

PH = os.path.join(REPO, 'phenotype_noise')
ENS_GLOB = 'gsd_results_consistency_T20_mix_fp_marginal_cons_l1.0_j1.0_hq_split{s}/model_ho3.pth'
warnings.filterwarnings('ignore')


# ---- Noise model, on COG COUNTS rather than presence.  Reproduces the
# reference implementation's hard-FN convention (present -> 0, not count - 1):
# false negatives zero a fraction fn_rate of the present entries, false
# positives add one count to a fraction fp_rate of the candidate set, which
# here is every feature, absent or present.
def flip_with_fractional_noise(X, fp_rate, fn_rate, hard_fn_flag=True):
    X_noisy = X.float().clone()
    n_rows = X_noisy.shape[0]
    for i in range(n_rows):
        pos_idx = torch.nonzero(X[i] > 0).flatten()
        n_fn = int(round(fn_rate * len(pos_idx)))
        if n_fn > 0:
            fn_idx = pos_idx[torch.randperm(len(pos_idx))[:n_fn]]
            X_noisy[i, fn_idx] = 0 if hard_fn_flag else X_noisy[i, fn_idx] - 1
        zero_idx = torch.nonzero(X[i] > -1).flatten()
        n_fp = int(round(fp_rate * len(zero_idx)))
        if n_fp > 0:
            fp_idx = zero_idx[torch.randperm(len(zero_idx))[:n_fp]]
            X_noisy[i, fp_idx] += 1
    return torch.clamp(X_noisy, min=0.0)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if np.any(m):
            ece += abs(y_true[m].mean() - y_prob[m].mean()) * m.mean()
    return ece


# ---- Denoiser front-end
class Denoiser:
    def __init__(self, ensemble, device):
        self.dev = torch.device(device)
        mm = torch.load(os.path.join(REPO, 'data/module_matrix_kegg.pt'), weights_only=False)
        self.N = len(mm['cog_names'])
        _, self.M_mod, self.M_sizes = load_module_matrix(
            os.path.join(REPO, 'data/module_matrix_kegg.pt'), self.dev, 1000)
        n_mod = self.M_mod.shape[1]
        self.models = []
        for s in range(1, ensemble + 1):
            ck = os.path.join(REPO, ENS_GLOB.format(s=s))
            sd = torch.load(ck, map_location=self.dev, weights_only=True)
            sd = strip_compile_prefix({k.replace('module.', ''): v for k, v in sd.items()})
            m, _, _, _ = build_model_for_ckpt(sd, self.dev, self.N, n_mod, ck)
            m.load_state_dict(sd, strict=False); m.eval()
            self.models.append(m)
        print(f'  loaded {len(self.models)}-model HQ-marginal consistent ensemble on {self.dev}')

    @torch.no_grad()
    def posterior(self, x_pm1, batch=256):
        """x_pm1: (n, N) +/-1 input -> mean denoised posterior (n, N) in [0,1]."""
        n = x_pm1.shape[0]
        out = torch.zeros(n, self.N)
        for b in range(0, n, batch):
            xb = x_pm1[b:b + batch].to(self.dev)
            acc = torch.zeros(xb.shape[0], self.N, device=self.dev)
            for m in self.models:
                xo = m(xb, self.M_mod, self.M_sizes)[0]
                acc += ((xo + 1) * 0.5).clamp(0, 1)
            out[b:b + batch] = (acc / len(self.models)).cpu()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-splits', type=int, default=10)
    ap.add_argument('--fp-grid', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    ap.add_argument('--fn-grid', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument('--ensemble', type=int, default=10)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--robust-pkl', default=os.path.join(
        PH, 'models/trained_models_fp_0.1_fn_0.5_noise_type_unif_x_100.pkl'))
    ap.add_argument('--out', default=os.path.join(PH, 'results'))
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    torch.manual_seed(A.seed); np.random.seed(A.seed)

    # ---- Full genome profiles and the feature -> denoiser-slot map
    Z = np.load(os.path.join(PH, 'aerob_profiles.npz'), allow_pickle=True)
    prof = Z['prof']                                  # (G, 4789) presence 0/1, denoiser order
    acc_arr = Z['acc']; acc_to_row = {a: i for i, a in enumerate(acc_arr)}
    covered = Z['covered']; feat_idx = Z['feat_idx']  # (2677,) -> denoiser slot, -1 if passthrough
    valid = feat_idx >= 0
    fvalid = feat_idx[valid]                           # denoiser slots for in-universe features
    print(f'profiles: {prof.shape}, covered {covered.sum()}/{len(covered)}, '
          f'{valid.sum()}/{len(valid)} features denoisable')

    # ---- Content matcher: an X count row -> the genome accession it came from
    ann = pd.read_csv(os.path.join(PH, 'all_gene_annotations.tsv'), sep='\t')
    feat = [l.strip() for l in open(os.path.join(PH, 'x_feature_cogs.txt')) if l.strip()]
    Amat = ann[feat].to_numpy().astype(np.float32)
    h2acc = {}
    for r, a in zip(Amat, ann['accession'].to_numpy()):
        h2acc.setdefault(hashlib.md5(r.tobytes()).hexdigest(), a)

    def rows_to_acc(Xnp):
        return [h2acc.get(hashlib.md5(r.astype(np.float32).tobytes()).hexdigest()) for r in Xnp]

    den = Denoiser(A.ensemble, A.device)

    import joblib
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    robust = joblib.load(A.robust_pkl)
    SP = os.path.join(PH, 'splits')
    splits = {}
    for s in range(A.n_splits):
        f = lambda kind: os.path.join(SP, f'{kind}_phylum_split_{s}')
        if not os.path.exists(f('test_data')):
            continue
        Xtr = torch.load(f('train_data'), weights_only=False)
        ytr = torch.load(f('train_annot'), weights_only=False)
        Xte = torch.load(f('test_data'), weights_only=False)
        yte = torch.load(f('test_annot'), weights_only=False)
        accte = rows_to_acc(Xte.numpy())
        orig = make_pipeline(XGBClassifier(tree_method='hist'))
        orig.fit(Xtr.cpu(), ytr.cpu())
        splits[s] = dict(Xte=Xte, yte=yte.numpy(), accte=accte, orig=orig,
                         robust=robust.get(s) or robust.get(str(s)))
        nmatch = sum(a is not None for a in accte)
        print(f'  split {s}: train {tuple(Xtr.shape)} test {tuple(Xte.shape)}  '
              f'acc-matched {nmatch}/{len(accte)}')

    def denoise_X(Xn_np, accte):
        n = Xn_np.shape[0]
        x4789 = np.full((n, den.N), -1.0, dtype=np.float32)        # absent unless matched
        for i, a in enumerate(accte):
            r = acc_to_row.get(a)
            if r is not None and covered[r]:
                x4789[i] = 2.0 * prof[r] - 1.0                     # genome context, +/-1
        # The feature slots carry the NOISY presence, overwriting the clean
        # profile; only the non-feature COGs remain uncorrupted context.
        noisy_pres = 2.0 * (Xn_np[:, valid] > 0).astype(np.float32) - 1.0
        x4789[:, fvalid] = noisy_pres
        post = den.posterior(torch.from_numpy(x4789)).numpy()      # (n, 4789) in [0,1]
        Xd = (Xn_np > 0).astype(np.float32)                        # unmapped features pass through
        Xd[:, valid] = (post[:, fvalid] > 0.5).astype(np.float32)
        return Xd

    from sklearn.metrics import (matthews_corrcoef, accuracy_score, balanced_accuracy_score,
                                  precision_score, recall_score, f1_score, brier_score_loss)
    ARMS = ['orig_noisy', 'robust_noisy', 'orig_denoised', 'robust_denoised']
    results = {arm: {} for arm in ARMS}
    grid = [(fn, fp) for fn in A.fn_grid for fp in A.fp_grid]
    for gi, (fn, fp) in enumerate(grid):
        per = {arm: {k: [] for k in ['mcc', 'f1', 'brier', 'ece', 'accuracy', 'balanced_accuracy']}
               for arm in ARMS}
        for s, S in splits.items():
            # One deterministic noise draw per (split, fn, fp), shared by all
            # four arms, so the arms differ only in predictor and front-end.
            g = torch.Generator().manual_seed(A.seed * 1000 + s * 37 + int(fn * 100) + int(fp * 1000))
            torch.manual_seed(int(g.initial_seed()))
            Xn = flip_with_fractional_noise(S['Xte'], fp, fn, hard_fn_flag=True)
            Xn_np = Xn.numpy()
            y = S['yte']
            Xd = denoise_X(Xn_np, S['accte'])
            feeds = {'orig_noisy': (S['orig'], Xn_np), 'robust_noisy': (S['robust'], Xn_np),
                     'orig_denoised': (S['orig'], Xd), 'robust_denoised': (S['robust'], Xd)}
            for arm, (mdl, Xin) in feeds.items():
                if mdl is None:
                    continue
                yp = mdl.predict(Xin); pr = mdl.predict_proba(Xin)[:, 1]
                per[arm]['mcc'].append(matthews_corrcoef(y, yp))
                per[arm]['f1'].append(f1_score(y, yp, zero_division=0))
                per[arm]['brier'].append(brier_score_loss(y, pr))
                per[arm]['ece'].append(expected_calibration_error(y, pr))
                per[arm]['accuracy'].append(accuracy_score(y, yp))
                per[arm]['balanced_accuracy'].append(balanced_accuracy_score(y, yp))
        for arm in ARMS:
            if per[arm]['mcc']:
                results[arm][(fn, fp)] = {k: (float(np.mean(v)), float(np.std(v)))
                                          for k, v in per[arm].items()}
        print(f'  [{gi+1}/{len(grid)}] fn={fn} fp={fp}  '
              + '  '.join(f"{a}:mcc={results[a].get((fn,fp),{}).get('mcc',(float('nan'),))[0]:.3f}"
                          for a in ARMS if (fn, fp) in results[a]))

    import pickle
    with open(os.path.join(A.out, 'pheno_denoise_results.pkl'), 'wb') as fh:
        pickle.dump({'results': results, 'fp_grid': A.fp_grid, 'fn_grid': A.fn_grid,
                     'n_splits': len(splits), 'ensemble': A.ensemble}, fh)
    print(f'wrote {A.out}/pheno_denoise_results.pkl')


if __name__ == '__main__':
    main()
