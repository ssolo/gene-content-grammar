#!/usr/bin/env python3
"""Leak-safe denoise-then-predict for the aerobicity phenotype predictor.

Leakage: the predictor uses a reduced vocabulary of 2677 feature COGs out of the
denoiser's 4789, and the denoiser reconstructs a COG from the rest of the genome
through the learned pairwise coupling J. Corrupting only the 2677 in-vocabulary
columns leaves the other 2112 out-of-window COGs pristine, so the feature COGs
are recovered from clean genomic context no genuinely noisy genome would have.
The tell was fn = 1.0 -- every feature deleted -- still scoring MCC > 0.

Fix: corrupt the whole genome presence, out-of-window context included, denoise,
then slice the 2677 feature columns back out. The feature slots carry the same
deterministic noise as the noisy arm, so noisy vs denoised differs only in the
surrounding context, which is corrupted at the same rate.

Arms, each returned as a {(fn, fp): {metric: (mean, std)}} dict
    base_noisy        base predictor   on noisy counts
    robust_noisy      robust predictor on noisy counts
    base_denoised     base predictor   on leak-safe denoised presence
    robust_denoised   robust predictor on leak-safe denoised presence

Evaluation limitation: the predictors were trained on integer counts, so the
noisy arms feed counts while the denoised arms feed binary presence (the denoiser
is a presence model). The denoised-vs-noisy gap therefore answers "did denoising
recover signal", not "counts vs counts".

With predictors already in memory, e.g. from a notebook:
    import sys; sys.path.insert(0, "."); import denoise_then_predict as dtp
    den = dtp.eval_denoised_models(all_splits_dict, base_models=..., robust_models=...)

--smoke runs one split over {fn=0,1} x {fp=0} with a 2-model ensemble and asserts
the leak-safe property: at fn = 1.0 the denoised MCC is ~ 0.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

PH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PH)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, PH)
from analyze_ancestral_node import build_model_for_ckpt          # noqa: E402
from ising_denoiser.modules import load_module_matrix            # noqa: E402
from ising_denoiser.training import strip_compile_prefix         # noqa: E402
from noise_util import flip_with_fractional_noise, noise_seed    # noqa: E402

ENS_TMPL = os.path.join(
    REPO, "gsd_results_consistency_T20_mix_fp_marginal_cons_l1.0_j1.0_hq_split{s}/model_ho3.pth")
ARMS = ["base_noisy", "robust_noisy", "base_denoised", "robust_denoised"]

_PROF = _COVERED = _FEAT_IDX = _ACC2ROW = _BUNDLE = None
_DEN = {"models": None, "ensemble": 0, "device": None, "M_mod": None, "M_sizes": None, "N": None}


def _load_data():
    """Load the full-genome profiles, the reduced <-> full feature map and the
    per-split (X, accession, label) bundle. Cached after the first call."""
    global _PROF, _COVERED, _FEAT_IDX, _ACC2ROW, _BUNDLE
    if _PROF is not None:
        return
    Z = np.load(os.path.join(PH, "aerob_profiles.npz"), allow_pickle=True)
    _PROF = Z["prof"].astype(np.float32)          # (G, 4789) clean presence 0/1, denoiser column order
    _COVERED = Z["covered"]                        # (G,) full profile available
    _FEAT_IDX = Z["feat_idx"]                      # (2677,) reduced col -> full idx, -1 if unmapped
    _ACC2ROW = {a: i for i, a in enumerate(Z["acc"])}
    with open(os.path.join(PH, "denoise_bundle.pkl"), "rb") as fh:
        _BUNDLE = pickle.load(fh)


def _load_denoisers(ensemble, device):
    """Load the cross-input-consistent HQ-marginal denoiser ensemble, cached per
    (ensemble, device). Use cpu or cuda: the higher-order attention hangs on MPS."""
    if _DEN["models"] is not None and _DEN["ensemble"] == ensemble and str(_DEN["device"]) == str(device):
        return
    dev = torch.device(device)
    mmpath = os.path.join(REPO, "data/module_matrix_kegg.pt")
    mm = torch.load(mmpath, weights_only=False)
    N = len(mm["cog_names"])
    _, M_mod, M_sizes = load_module_matrix(mmpath, dev, 1000)
    n_mod = M_mod.shape[1]
    models = []
    for s in range(1, ensemble + 1):
        ck = ENS_TMPL.format(s=s)
        sd = strip_compile_prefix({k.replace("module.", ""): v
                                   for k, v in torch.load(ck, map_location=dev, weights_only=True).items()})
        m, _, _, _ = build_model_for_ckpt(sd, dev, N, n_mod, ck)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models.append(m)
    _DEN.update(models=models, ensemble=ensemble, device=dev, M_mod=M_mod, M_sizes=M_sizes, N=N)


@torch.no_grad()
def _denoise_full(x4789, batch=64):
    """x4789: (n, 4789) in {-1,+1}. Returns the (n, 4789) ensemble-mean presence
    posterior in (0, 1)."""
    models, M_mod, M_sizes, N, dev = (_DEN["models"], _DEN["M_mod"], _DEN["M_sizes"], _DEN["N"], _DEN["device"])
    xt = torch.from_numpy(x4789.astype(np.float32))
    out = torch.zeros(xt.shape[0], N)
    for b in range(0, xt.shape[0], batch):
        xb = xt[b:b + batch].to(dev)
        acc = torch.zeros(xb.shape[0], N, device=dev)
        for m in models:
            acc += ((m(xb, M_mod, M_sizes)[0] + 1) * 0.5).clamp(0, 1)
        out[b:b + batch] = (acc / len(models)).cpu()
    return out.numpy()


def _ece(y, p, n_bins=10):
    y = np.asarray(y); p = np.asarray(p); edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            e += abs(y[m].mean() - p[m].mean()) * m.mean()
    return float(e)


def _metrics(y, yp, pr):
    from sklearn.metrics import (matthews_corrcoef, f1_score, brier_score_loss,
                                 accuracy_score, balanced_accuracy_score,
                                 recall_score, precision_score)
    return {"mcc": matthews_corrcoef(y, yp), "f1": f1_score(y, yp, zero_division=0),
            "brier": brier_score_loss(y, pr), "ece": _ece(y, pr),
            "accuracy": accuracy_score(y, yp), "balanced_accuracy": balanced_accuracy_score(y, yp),
            "recall": recall_score(y, yp, zero_division=0),
            "precision": precision_score(y, yp, zero_division=0)}


def _get(model_dict, s):
    """Per-split model, whether the dict is keyed by int or by str."""
    if model_dict is None:
        return None
    if s in model_dict:
        return model_dict[s]
    return model_dict.get(str(s)) if hasattr(model_dict, "get") else None


def eval_denoised_models(all_splits_dict, base_models, robust_models,
                         fn_grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), fp_grid=(0.0, 0.1, 0.2),
                         ensemble=10, hard_fn=True, seed_base=7, batch=64,
                         device="cpu", verbose=True):
    """Leak-safe denoise-then-predict across the fn x fp grid, over every split
    present both in the bundle and in the two model dicts.

    all_splits_dict is an alignment guard only: it asserts the caller's X_test
    matches the bundle's X, so the in-memory predictors are valid on the denoised
    columns. Pass None to skip the guard.

    Returns {arm: {(fn, fp): {metric: (mean, std)}}} for arm in ARMS.
    """
    _load_data()
    _load_denoisers(ensemble, device)
    N = _DEN["N"]
    valid = _FEAT_IDX >= 0
    fvalid = _FEAT_IDX[valid]

    splits = [s for s in sorted(_BUNDLE) if _get(base_models, s) is not None and _get(robust_models, s) is not None]
    if not splits:
        raise ValueError("no split is present in both the bundle and the base+robust model dicts")
    collect = {a: {} for a in ARMS}

    for s in splits:
        S = _BUNDLE[s]
        X = S["X"].astype(np.float32); acc = S["acc"]; y = np.asarray(S["y"])
        if all_splits_dict is not None and s in all_splits_dict:
            Xnb = all_splits_dict[s]["X_test"]
            Xnb = Xnb.numpy() if hasattr(Xnb, "numpy") else np.asarray(Xnb, dtype=np.float32)
            assert np.array_equal(Xnb.astype(np.float32), X), (
                f"split {s}: notebook X_test != bundle X -- the in-memory predictors "
                f"are not aligned to the denoised columns; rebuild denoise_bundle.pkl")
        base = _get(base_models, s); rob = _get(robust_models, s)
        n = X.shape[0]
        # An uncovered genome gets an all-absent context rather than being dropped.
        prof01 = np.zeros((n, N), dtype=np.float32)
        nun = 0
        for i, a in enumerate(acc):
            r = _ACC2ROW.get(a)
            if r is not None and _COVERED[r]:
                prof01[i] = _PROF[r]
            else:
                nun += 1
        for fn in fn_grid:
            for fp in fp_grid:
                sd = noise_seed(seed_base, s, fn, fp)
                Xn = flip_with_fractional_noise(X, fp, fn, sd, hard_fn=hard_fn)           # noisy feature counts
                # Whole-genome corruption, context included, on an independent draw.
                prof_noisy = flip_with_fractional_noise(prof01, fp, fn, sd + 1, hard_fn=hard_fn)
                x4789 = 2.0 * prof_noisy - 1.0
                # Pin the feature slots to the noise the noisy arm sees, so the two
                # arms differ only in the context available to the denoiser.
                x4789[:, fvalid] = 2.0 * (Xn[:, valid] > 0).astype(np.float32) - 1.0
                post = _denoise_full(x4789, batch=batch)
                Xd = (Xn > 0).astype(np.float32)                                          # passthrough = noisy presence
                Xd[:, valid] = (post[:, fvalid] > 0.5).astype(np.float32)
                feeds = {"base_noisy": (base, Xn), "robust_noisy": (rob, Xn),
                         "base_denoised": (base, Xd), "robust_denoised": (rob, Xd)}
                for arm, (mdl, Xin) in feeds.items():
                    yp = mdl.predict(Xin); pr = mdl.predict_proba(Xin)[:, 1]
                    collect[arm].setdefault((fn, fp), []).append(_metrics(y, yp, pr))
        if verbose:
            print(f"split {s}: denoised+predicted {len(fn_grid)*len(fp_grid)} cells "
                  f"(n={n}, uncovered-context={nun})", flush=True)

    out = {a: {} for a in ARMS}
    for a in ARMS:
        for key, lst in collect[a].items():
            out[a][key] = {k: (float(np.mean([d[k] for d in lst])),
                               float(np.std([d[k] for d in lst]))) for k in lst[0]}
    return out


def eval_from_cached_denoise(all_splits_dict, base_models, robust_models,
                             den_dir=None, seed_base=7, hard_fn=True, verbose=True):
    """The 4-arm evaluation of eval_denoised_models over pre-computed denoised
    features (data/pheno_den/den_split_*.npz) instead of denoising inline.

    The npz are written by denoise_features.py under the same leak-safe recipe
    (whole-genome corruption, out-of-window COGs included); only the matching
    noisy arm is regenerated here, from the identical deterministic seed. Imports
    no torch. Returns {arm: {(fn, fp): {metric: (mean, std)}}}.
    """
    import glob
    _load_data()
    den_dir = den_dir or os.path.join(REPO, "data", "pheno_den")
    collect = {a: {} for a in ARMS}
    used = []
    for f in sorted(glob.glob(os.path.join(den_dir, "den_split_*.npz"))):
        s = int(f.split("_")[-1].split(".")[0])
        base = _get(base_models, s); rob = _get(robust_models, s)
        if base is None or rob is None or s not in _BUNDLE:
            continue
        Z = np.load(f, allow_pickle=True)
        y = np.asarray(Z["y"]); grid = [(float(a), float(b)) for a, b in Z["grid"]]
        sb = int(Z["seed_base"]) if "seed_base" in Z.files else seed_base
        X = _BUNDLE[s]["X"].astype(np.float32)
        if all_splits_dict is not None and s in all_splits_dict:
            Xnb = all_splits_dict[s]["X_test"]
            Xnb = Xnb.numpy() if hasattr(Xnb, "numpy") else np.asarray(Xnb, dtype=np.float32)
            assert np.array_equal(Xnb.astype(np.float32), X), (
                f"split {s}: notebook X_test != bundle X -- predictors not aligned to "
                f"the cached denoised columns; rebuild denoise_bundle.pkl / re-denoise")
        assert len(y) == X.shape[0], f"split {s}: y/X length mismatch"
        for gi, (fn, fp) in enumerate(grid):
            Xn = flip_with_fractional_noise(X, fp, fn, noise_seed(sb, s, fn, fp), hard_fn=hard_fn)
            Xd = Z[f"d{gi}"].astype(np.float32)
            assert Xd.shape == X.shape, f"split {s} cell {gi}: denoised shape {Xd.shape} != {X.shape}"
            feeds = {"base_noisy": (base, Xn), "robust_noisy": (rob, Xn),
                     "base_denoised": (base, Xd), "robust_denoised": (rob, Xd)}
            for arm, (mdl, Xin) in feeds.items():
                yp = mdl.predict(Xin); pr = mdl.predict_proba(Xin)[:, 1]
                collect[arm].setdefault((fn, fp), []).append(_metrics(y, yp, pr))
        used.append(s)
        if verbose:
            print(f"split {s}: predicted {len(grid)} cells from cache (n={X.shape[0]})", flush=True)
    if not used:
        raise ValueError(f"no usable den_split_*.npz in {den_dir} matched to both model dicts")
    out = {a: {} for a in ARMS}
    for a in ARMS:
        for key, lst in collect[a].items():
            out[a][key] = {k: (float(np.mean([d[k] for d in lst])),
                               float(np.std([d[k] for d in lst]))) for k in lst[0]}
    out["_splits"] = used
    return out


def build_models_from_disk(robust_pkl=None, splits=range(10)):
    """Build both predictor sets from disk, for callers with none in memory.

    base retrains the XGBoost pipeline per split from phenotype_noise/splits;
    robust is the saved noise-augmented ensemble, restricted to the splits base
    covers.
    """
    import joblib
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    robust_pkl = robust_pkl or os.path.join(
        PH, "models/trained_models_fp_0.1_fn_0.5_noise_type_unif_x_100.pkl")
    robust = joblib.load(robust_pkl)
    base = {}
    for s in splits:
        f = os.path.join(PH, "splits", f"train_data_phylum_split_{s}")
        if not os.path.exists(f):
            continue
        Xtr = torch.load(f, weights_only=False).numpy()
        ytr = torch.load(os.path.join(PH, "splits", f"train_annot_phylum_split_{s}"),
                         weights_only=False).numpy()
        pipe = make_pipeline(XGBClassifier(tree_method="hist"))
        pipe.fit(Xtr, ytr)
        base[s] = pipe
    return base, {s: _get(robust, s) for s in base if _get(robust, s) is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast 1-split leak-safety check")
    ap.add_argument("--cache", action="store_true",
                    help="predict from the GPU cluster-denoised npz (data/pheno_den); no torch")
    ap.add_argument("--ensemble", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fn-grid", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--fp-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--out", default=os.path.join(PH, "results", "denoise_then_predict.pkl"))
    A = ap.parse_args()

    if A.smoke:
        base, rob = build_models_from_disk(splits=[0])
        den = eval_denoised_models(None, base, rob, fn_grid=(0.0, 1.0), fp_grid=(0.0,),
                                   ensemble=min(A.ensemble, 2), device=A.device)
        mcc0 = den["base_denoised"][(0.0, 0.0)]["mcc"][0]
        mcc1 = den["base_denoised"][(1.0, 0.0)]["mcc"][0]
        print(f"\nSMOKE: base_denoised MCC  fn=0.0 -> {mcc0:.3f}   fn=1.0 -> {mcc1:.3f}")
        print(f"       base_noisy    MCC  fn=1.0 -> {den['base_noisy'][(1.0,0.0)]['mcc'][0]:.3f}")
        assert abs(mcc1) < 0.15, f"LEAK: fn=1.0 denoised MCC={mcc1:.3f} should be ~0 (whole genome deleted)"
        print("       leak-safe: fn=1.0 (whole genome deleted) -> denoised MCC ~ 0  OK")
        return

    base, rob = build_models_from_disk()
    if A.cache:
        den = eval_from_cached_denoise(None, base, rob)
        splits = den.pop("_splits", [])
    else:
        den = eval_denoised_models(None, base, rob, fn_grid=tuple(A.fn_grid), fp_grid=tuple(A.fp_grid),
                                   ensemble=A.ensemble, device=A.device)
        splits = None
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    with open(A.out, "wb") as fh:
        pickle.dump(den, fh)
    print(f"\nwrote {A.out}")
    print("\n=== mean MCC over fp, per fn  (base n->d | robust n->d) ===")
    fn_grid = sorted({fn for fn, _ in den["base_noisy"]})
    fp_grid = sorted({fp for _, fp in den["base_noisy"]})
    for fn in fn_grid:
        def mo(arm):
            v = [den[arm][(fn, fp)]["mcc"][0] for fp in fp_grid if (fn, fp) in den[arm]]
            return np.mean(v) if v else float("nan")
        print(f"fn={fn:<4} base {mo('base_noisy'):.3f}->{mo('base_denoised'):.3f}   "
              f"robust {mo('robust_noisy'):.3f}->{mo('robust_denoised'):.3f}")


if __name__ == "__main__":
    main()
