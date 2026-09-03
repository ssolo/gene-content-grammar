#!/usr/bin/env python3.11
"""Gene essentiality from the mean-field magnetisation TRAJECTORY, the denoiser
analogue of the embedding-based ProteomeLM-Ess head.

essentiality_recon.py scores a gene by the FINAL leave-one-out posterior alone
(demand_i = m_i^T with gene i knocked out; AUROC ~0.64). Here the whole
re-implication trajectory m_i^1..m_i^T, how fast and how strongly the rest of the
genome re-implies the gene at each mean-field sweep, goes to a supervised
classifier together with structural features:

  feature(gene i) = [ m_i^1, ..., m_i^T,  h_i,  sum_j|J~_ij|,  sum_j J~_ij^2 ]
  label           = Goodall et al. 2018 essential (COG-level)

Run the same script under two checkpoints to separate signal from phylogenetic
memorisation: an in-distribution model (E. coli's phylum is in training) and the
E. coli phylum-held-out model. A trajectory gain over demand that survives the
phylum holdout is real; one that collapses was the model leaning on memorised
relatives.

Output: data/interactome/results_ess/<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from essentiality_recon import goodall_cog_labels           # noqa: E402
from dynamic_interaction import species_present, traj_present  # noqa: E402
from analyze_ancestral_node import build_model_for_ckpt      # noqa: E402
from ising_denoiser.modules import load_module_matrix        # noqa: E402
from ising_denoiser.training import strip_compile_prefix     # noqa: E402

STRDIR = os.path.join(REPO, "data", "interactome", "string")
OUTDIR = os.path.join(REPO, "data", "interactome", "results_ess")


def cv_auroc(X, y, clf, seed=0):
    """Out-of-fold (AUROC, average precision); no gene is in both train and test."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), clf)
    oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", default="511145")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cog-order", default=os.path.join(REPO, "data/interactome/cog_order.txt"))
    ap.add_argument("--ess-xlsx", default=os.path.join(REPO, "data/interactome/goodall_st1.xlsx"))
    ap.add_argument("--module-matrix", default=os.path.join(REPO, "data/module_matrix_kegg.pt"))
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cogs = [l.strip() for l in open(a.cog_order) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    N = len(cogs)
    _, M_mod, M_sizes = load_module_matrix(a.module_matrix, dev, H=1000)
    sd = strip_compile_prefix(torch.load(a.ckpt, map_location=dev, weights_only=True))
    model, cls_kind, T0, _ = build_model_for_ckpt(sd, dev, N, M_mod.shape[1], a.ckpt)
    model.load_state_dict(sd, strict=False); model.eval()
    h_field = sd["h"].float().cpu().numpy().reshape(-1)
    Jt = 0.5 * (sd["J"].float() + sd["J"].float().t()); Jt.fill_diagonal_(0.0)
    deg1 = Jt.abs().sum(1).cpu().numpy(); deg2 = (Jt * Jt).sum(1).cpu().numpy()

    present, _ = species_present(a.taxon, cog2idx)
    present = np.array(present); n = len(present)
    present_t = torch.tensor(present.tolist(), device=dev)
    ess_cogs, non_cogs = goodall_cog_labels(
        a.ess_xlsx, os.path.join(STRDIR, "%s.aliases.txt.gz" % a.taxon),
        os.path.join(STRDIR, "%s.protein_to_cog.tsv" % a.taxon), cog2idx)

    # Leave-one-out: one genome per present gene, that gene set absent.
    g = -torch.ones(N, device=dev); g[present_t] = 1.0
    Xko = g.unsqueeze(0).repeat(n, 1)
    Xko[torch.arange(n, device=dev), present_t] = -1.0
    trk = traj_present(model, Xko, M_mod, M_sizes, present_t, chunk=a.chunk)   # (T, n, n)
    T = trk.shape[0]
    traj = trk[:, np.arange(n), np.arange(n)].numpy().T                        # (n, T) own posterior per sweep
    print("loaded %s (class=%s T=%d); present=%d, traj %s" % (os.path.basename(os.path.dirname(a.ckpt)), cls_kind, T0, n, traj.shape))

    # Present COGs that Goodall classifies either way; the rest are unlabelled.
    rows = []
    for k in range(n):
        c = cogs[present[k]]
        if c in ess_cogs: y = 1
        elif c in non_cogs: y = 0
        else: continue
        rows.append((k, y))
    idx = np.array([k for k, _ in rows]); y = np.array([v for _, v in rows])
    demand = traj[idx, -1]
    struct = np.stack([h_field[present[idx]], deg1[present[idx]], deg2[present[idx]]], 1)
    Xtraj = traj[idx]                       # (m, T)
    Xall = np.concatenate([Xtraj, struct], 1)

    res = {"taxon": a.taxon, "ckpt": os.path.basename(os.path.dirname(a.ckpt)), "model": cls_kind,
           "T": int(T), "n_labeled": int(y.size), "n_essential": int(y.sum()), "prevalence": float(y.mean())}
    res["AUROC_demand_final"] = float(roc_auc_score(y, demand))
    res["AUROC_field_h"] = float(roc_auc_score(y, h_field[present[idx]]))
    for name, clf in [("logreg", LogisticRegression(max_iter=2000, C=1.0)),
                      ("gbt", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0))]:
        au_t, ap_t = cv_auroc(Xtraj, y, clf)
        au_a, ap_a = cv_auroc(Xall, y, clf)
        res["AUROC_traj_%s" % name] = au_t
        res["AUROC_traj+struct_%s" % name] = au_a
        res["AUPRC_traj+struct_%s" % name] = ap_a
        print("  %-7s | demand=%.3f | traj=%.3f | traj+struct=%.3f"
              % (name, res["AUROC_demand_final"], au_t, au_a))
    json.dump(res, open(os.path.join(OUTDIR, "%s.json" % a.tag), "w"), indent=2)
    print("saved", os.path.join(OUTDIR, "%s.json" % a.tag))


if __name__ == "__main__":
    main()
