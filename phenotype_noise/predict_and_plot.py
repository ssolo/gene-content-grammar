#!/usr/bin/env python3
"""Prediction step of the denoised-COG phenotype experiment.

Consumes the denoise step's output (data/pheno_den/den_split_*.npz from
denoise_features.py). Per split: refit the original aerobicity XGBoost on that
split's training data, take the matching noise-robust predictor from
--robust-pkl, regenerate from (seed_base, split, fn, fp) the same deterministic
noise the denoise step applied, and score four arms over the fn x fp grid:

    orig_noisy      original predictor, noisy counts
    robust_noisy    robust   predictor, noisy counts
    orig_denoised   original predictor, denoised binary presence
    robust_denoised robust   predictor, denoised binary presence

Noisy arms are fed COG counts, denoised arms binary presence, the denoiser being
a presence model. All arms share one noise draw per (split, fn, fp), so the
contrast is over identical feature observations.

Writes <out>/pheno_denoise_results.pkl: per arm and (fn, fp), the across-split
mean and std of MCC, F1, Brier, ECE, accuracy and balanced accuracy.
plot_denoise_comparison.py draws the metric-vs-fn figure from it.
"""
import argparse
import glob
import os
import pickle

import numpy as np
import torch

PH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, PH)
from noise_util import flip_with_fractional_noise, noise_seed   # noqa: E402

ARMS = ['orig_noisy', 'robust_noisy', 'orig_denoised', 'robust_denoised']


def ece(y, p, n_bins=10):
    """Occupancy-weighted calibration error over equal-width probability bins.

    Bins are half-open [e_i, e_i+1), so p = 1.0 falls in no bin and is dropped.
    """
    y = np.asarray(y); p = np.asarray(p); e = np.linspace(0, 1, n_bins + 1); v = 0.0
    for i in range(n_bins):
        m = (p >= e[i]) & (p < e[i + 1])
        if m.any():
            v += abs(y[m].mean() - p[m].mean()) * m.mean()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--den-dir', default=os.path.join(os.path.dirname(PH), 'data/pheno_den'))
    ap.add_argument('--robust-pkl', default=os.path.join(
        PH, 'models/trained_models_fp_0.1_fn_0.5_noise_type_unif_x_100.pkl'))
    ap.add_argument('--out', default=os.path.join(PH, 'results'))
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    import joblib
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    from sklearn.metrics import (matthews_corrcoef, f1_score, brier_score_loss,
                                 accuracy_score, balanced_accuracy_score)
    robust_all = joblib.load(A.robust_pkl)

    res = {arm: {} for arm in ARMS}
    grid = None
    nsplit = 0
    acc_collect = {arm: {} for arm in ARMS}      # (fn,fp) -> list of per-split metric dicts

    for f in sorted(glob.glob(os.path.join(A.den_dir, 'den_split_*.npz'))):
        s = int(f.split('_')[-1].split('.')[0])
        Z = np.load(f, allow_pickle=True)
        y = Z['y']; grid = [(float(a), float(b)) for a, b in Z['grid']]
        seed_base = int(Z['seed_base'])
        # Only the noise-robust predictors ship pre-fitted; the noise-naive one
        # is refit here from the split's own training data.
        Xtr = torch.load(os.path.join(PH, 'splits', f'train_data_phylum_split_{s}'),
                         weights_only=False).numpy()
        ytr = torch.load(os.path.join(PH, 'splits', f'train_annot_phylum_split_{s}'),
                         weights_only=False).numpy()
        Xte = torch.load(os.path.join(PH, 'splits', f'test_data_phylum_split_{s}'),
                         weights_only=False).numpy().astype(np.float32)
        orig = make_pipeline(XGBClassifier(tree_method='hist')); orig.fit(Xtr, ytr)
        rob = robust_all.get(s) if hasattr(robust_all, 'get') else None
        if rob is None and hasattr(robust_all, 'get'):
            rob = robust_all.get(str(s))
        nsplit += 1
        for gi, (fn, fp) in enumerate(grid):
            Xn = flip_with_fractional_noise(Xte, fp, fn, noise_seed(seed_base, s, fn, fp))
            Xd = Z[f'd{gi}'].astype(np.float32)
            feeds = {'orig_noisy': (orig, Xn), 'robust_noisy': (rob, Xn),
                     'orig_denoised': (orig, Xd), 'robust_denoised': (rob, Xd)}
            for arm, (mdl, Xin) in feeds.items():
                if mdl is None:
                    continue
                yp = mdl.predict(Xin); pr = mdl.predict_proba(Xin)[:, 1]
                d = {'mcc': matthews_corrcoef(y, yp), 'f1': f1_score(y, yp, zero_division=0),
                     'brier': brier_score_loss(y, pr), 'ece': ece(y, pr),
                     'accuracy': accuracy_score(y, yp),
                     'balanced_accuracy': balanced_accuracy_score(y, yp)}
                acc_collect[arm].setdefault((fn, fp), []).append(d)
        print(f'split {s}: predicted ({len(grid)} noise cells)')

    for arm in ARMS:
        for key, lst in acc_collect[arm].items():
            res[arm][key] = {k: (float(np.mean([d[k] for d in lst])),
                                 float(np.std([d[k] for d in lst]))) for k in lst[0]}
    fp_grid = sorted(set(fp for _, fp in grid)); fn_grid = sorted(set(fn for fn, _ in grid))
    with open(os.path.join(A.out, 'pheno_denoise_results.pkl'), 'wb') as fh:
        pickle.dump({'results': res, 'fp_grid': fp_grid, 'fn_grid': fn_grid,
                     'n_splits': nsplit, 'ensemble': 10}, fh)
    print(f'wrote {A.out}/pheno_denoise_results.pkl  ({nsplit} splits)')

    print('\n=== mean MCC over fp, per fn  (orig_noisy -> orig_denoised | robust_noisy -> robust_denoised) ===')
    for fn in fn_grid:
        def mean_over_fp(arm, met='mcc'):
            vals = [res[arm][(fn, fp)][met][0] for fp in fp_grid if (fn, fp) in res.get(arm, {})]
            return np.mean(vals) if vals else float('nan')
        print(f'fn={fn:<4}  orig {mean_over_fp("orig_noisy"):.3f}->{mean_over_fp("orig_denoised"):.3f}   '
              f'robust {mean_over_fp("robust_noisy"):.3f}->{mean_over_fp("robust_denoised"):.3f}')


if __name__ == '__main__':
    main()
