#!/usr/bin/env python3
"""Check the denoise->predict pipeline's noisy arm against the reference notebook.

The reference implementation corrupts the 2677-COG test counts with a torch
routine driven by the global RNG; the pipeline instead uses a seeded numpy
routine (noise_util) so that the denoised and noisy arms share an identical
draw. The two draws differ, but the fp/fn fractions, the data and the fitted
models are the SAME, so the per-noise MCC must agree to within split and draw
variance.

The reference noisy evaluation is rerun here and printed next to the cached
base_noisy/robust_noisy values at the shared (fn, fp) cells. It PASSES when the
largest absolute MCC difference is below 0.05.
"""
import os
import pickle
import sys

import numpy as np
import torch

PH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PH)


# ---- the reference torch noise routine, reproduced verbatim.  Its fp draw
# targets X[i] > -1, i.e. every feature of a count table, and it consumes torch's
# global RNG rather than taking an explicit seed.
def flip_with_fractional_noise_nb(X, fp_rate, fn_rate, hard_fn_flag=True):
    X_noisy = X.float().clone()
    n_rows, _ = X_noisy.shape
    for i in range(n_rows):
        pos_idx = torch.nonzero(X[i] > 0).flatten()
        n_fn = int(round(fn_rate * len(pos_idx)))
        if n_fn > 0:
            fn_idx = pos_idx[torch.randperm(len(pos_idx))[:n_fn]]
            if hard_fn_flag:
                X_noisy[i, fn_idx] = 0
            else:
                X_noisy[i, fn_idx] -= 1
        zero_idx = torch.nonzero(X[i] > -1).flatten()
        n_fp = int(round(fp_rate * len(zero_idx)))
        if n_fp > 0:
            fp_idx = zero_idx[torch.randperm(len(zero_idx))[:n_fp]]
            X_noisy[i, fp_idx] += 1
    return torch.clamp(X_noisy, min=0.0)


def main():
    from sklearn.metrics import matthews_corrcoef
    from denoise_then_predict import build_models_from_disk
    torch.manual_seed(0)

    cache = pickle.load(open(os.path.join(PH, "results", "denoise_then_predict_cache.pkl"), "rb"))
    cells = sorted(cache["base_noisy"].keys())
    base, rob = build_models_from_disk()
    splits = sorted(set(base) & set(rob))

    nb = {"base": {}, "robust": {}}
    per = {"base": {c: [] for c in cells}, "robust": {c: [] for c in cells}}
    for s in splits:
        Xte = torch.load(os.path.join(PH, "splits", f"test_data_phylum_split_{s}"),
                         weights_only=False).float()
        yte = torch.load(os.path.join(PH, "splits", f"test_annot_phylum_split_{s}"),
                         weights_only=False).numpy()
        for (fn, fp) in cells:
            Xn = flip_with_fractional_noise_nb(Xte, fp, fn).numpy()
            for who, mdl in (("base", base[s]), ("robust", rob[s])):
                per[who][(fn, fp)].append(matthews_corrcoef(yte, mdl.predict(Xn)))
    for who in nb:
        for c in cells:
            nb[who][c] = float(np.mean(per[who][c]))

    print(f"validation over splits {splits}\n")
    diffs = []
    for who in ("base", "robust"):
        print(f"=== {who}: notebook-noisy vs pipeline base_noisy MCC ===")
        print(f"{'(fn,fp)':>12} {'notebook':>9} {'pipeline':>9} {'|diff|':>7}")
        arm = f"{who}_noisy"
        for c in sorted(cells):
            nbv = nb[who][c]; piv = cache[arm][c]["mcc"][0]; d = abs(nbv - piv)
            diffs.append(d)
            print(f"{str(c):>12} {nbv:>9.3f} {piv:>9.3f} {d:>7.3f}")
        print()
    mad = float(np.mean(diffs)); mx = float(np.max(diffs))
    print(f"mean |MCC diff| = {mad:.4f}   max = {mx:.4f}")
    print("PASS: noisy arm reproduces the notebook" if mx < 0.05
          else "CHECK: larger-than-expected gap")


if __name__ == "__main__":
    main()
