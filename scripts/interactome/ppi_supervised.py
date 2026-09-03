#!/usr/bin/env python3.11
"""Supervised PPI head over gene-content features, the ProteomeLM-PPI analogue:
does combining the explicit coupling J with the model's per-gene denoising
confidence beat J alone, and does any gain survive whole-phylum holdout?

Per present COG pair (i,j) of a species:
  fJ = APC(J~)_ij                      the explicit coupling (the static read-out)
  fC = c_i c_j,  c_i = 1/(1 - m_i^2)    outer product of per-gene denoising
                                        confidence, m being the one-step denoised
                                        genome. A rank-1 degree/abundance bias,
                                        NOT a coupling.

A logistic regression over [fJ, fC] is fit with pair-level 5-fold CV against
STRING's experimental channel, and reported with the single-feature AUROCs and
the fitted coefficients. Run across the six E. coli zoom checkpoints (species end
= in-distribution, phylum end = whole phylum held out): a combined lead over J
that rises toward the in-distribution end is the confidence/memorisation leak
rather than coupling.

Output: data/interactome/results_ppi/<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from susceptibility_interaction import apc, species_present, cogpair_labels  # noqa: E402
from analyze_ancestral_node import build_model_for_ckpt  # noqa: E402
from ising_denoiser.modules import load_module_matrix  # noqa: E402
from ising_denoiser.training import strip_compile_prefix  # noqa: E402

OUTDIR = os.path.join(REPO, "data", "interactome", "results_ppi")
EPS = 1e-4    # floor on 1 - m^2, so the confidence stays finite as |m| -> 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", default="511145")
    ap.add_argument("--name", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cog-order", default=os.path.join(REPO, "data/interactome/cog_order.txt"))
    ap.add_argument("--module-matrix", default=os.path.join(REPO, "data/module_matrix_kegg.pt"))
    ap.add_argument("--thr", type=int, default=700)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cogs = [l.strip() for l in open(a.cog_order) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    N = len(cogs)
    _, M_mod, M_sizes = load_module_matrix(a.module_matrix, dev, H=1000)
    sd = strip_compile_prefix(torch.load(a.ckpt, map_location=dev, weights_only=True))
    model, cls, T0, _ = build_model_for_ckpt(sd, dev, N, M_mod.shape[1], a.ckpt)
    model.load_state_dict(sd, strict=False); model.eval()
    Jt = 0.5 * (sd["J"].float() + sd["J"].float().t()); Jt.fill_diagonal_(0.0)

    present, p2c = species_present(a.taxon, cog2idx)
    Le, Lc = cogpair_labels(a.taxon, p2c, present)
    iu = np.triu_indices(len(present), k=1)
    pid = np.array(present)
    y = (Le[iu] >= a.thr).astype(np.int8)

    # Gated one-step magnetisation m -> per-gene confidence c_i = 1/(1 - m_i^2).
    g = -torch.ones(N, device=dev); g[torch.tensor(pid.tolist(), device=dev)] = 1.0
    with torch.no_grad():
        out = model(g.unsqueeze(0), M_mod=M_mod, M_sizes=M_sizes, return_trajectory=True)
        m = out[2][0, 0].float()
    c = (1.0 / (1.0 - m * m).clamp_min(EPS)).cpu().numpy()

    fJ = apc(Jt.cpu().numpy())[np.ix_(pid, pid)][iu]
    fC = (c[pid][:, None] * c[pid][None, :])[iu]
    X = np.stack([fJ, fC], 1)

    res = {"taxon": a.taxon, "name": a.name, "ckpt": os.path.basename(os.path.dirname(a.ckpt)), "model": cls,
           "n_pairs": int(y.size), "n_pos": int(y.sum()), "prev": float(y.mean()), "thr": a.thr}
    res["AUROC_J"] = float(roc_auc_score(y, fJ))
    res["AUROC_conf"] = float(roc_auc_score(y, fC))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    res["AUROC_J+conf_linear"] = float(roc_auc_score(y, oof))
    res["AUPRC_J+conf_linear"] = float(average_precision_score(y, oof))
    pipe.fit(X, y)
    res["coef_J"], res["coef_conf"] = map(float, pipe[-1].coef_[0])
    # Rank-based head: the linear one underweights the heavy tail of fC.
    hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=0)
    oofg = cross_val_predict(hgb, X, y, cv=skf, method="predict_proba")[:, 1]
    res["AUROC_J+conf_gbt"] = float(roc_auc_score(y, oofg))
    res["AUPRC_J+conf_gbt"] = float(average_precision_score(y, oofg))
    print("  %-44s | J=%.3f conf=%.3f | linear=%.3f gbt=%.3f | coef[J,conf]=[%.2f,%.2f]"
          % (res["ckpt"], res["AUROC_J"], res["AUROC_conf"], res["AUROC_J+conf_linear"],
             res["AUROC_J+conf_gbt"], res["coef_J"], res["coef_conf"]))
    json.dump(res, open(os.path.join(OUTDIR, "%s.json" % a.tag), "w"), indent=2)
    print("saved", os.path.join(OUTDIR, "%s.json" % a.tag))


if __name__ == "__main__":
    main()
