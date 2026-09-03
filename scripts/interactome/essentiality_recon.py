#!/usr/bin/env python3.11
"""Gene essentiality from leave-one-out reconstructability, the denoiser-side
analogue of the ProteomeLM-Ess head.

For E. coli:

  demand_i = x_i^T  evaluated on the E. coli genome with gene i set absent
  h_i      = the learned single-site field, a conservation / base-rate proxy

Benchmark: the experimental essentiality calls of Goodall et al. 2018 (Table
S1). Essential genes are near-universal, so h_i is scored as a conservation
control and demand is residualised on h to isolate the context-dependent term.

Label chain: gene name (Goodall) -> STRING protein (b-number, via the STRING
aliases file) -> COG (via the diamond map) -> index into the model. Labels are
COG-level: a COG is essential if any E. coli gene mapping to it is essential.

  .venv/bin/python scripts/interactome/essentiality_recon.py
Output: data/interactome/essentiality_ecoli.json
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_interaction import species_present, traj_present  # noqa: E402
from analyze_ancestral_node import build_model_for_ckpt  # noqa: E402
from ising_denoiser.modules import load_module_matrix  # noqa: E402
from ising_denoiser.training import strip_compile_prefix  # noqa: E402

STRDIR = "data/interactome/string"
PLAIN_CKPT = "gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth"


def goodall_cog_labels(xlsx, aliases_gz, p2cog_tsv, cog2idx):
    import pandas as pd
    d = pd.read_excel(xlsx, engine="openpyxl", header=1)
    d.columns = [str(c).strip() for c in d.columns]
    gc = d.columns[0]
    ess = set(str(g).lower() for g in d[d["Essential"] == True][gc])      # noqa: E712
    non = set(str(g).lower() for g in d[d["Non-essential"] == True][gc])  # noqa: E712
    name2sp = {}
    with gzip.open(aliases_gz, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                name2sp.setdefault(p[1].lower(), p[0])
    p2c = {}
    with open(p2cog_tsv) as f:
        next(f)
        for line in f:
            a = line.split("\t")
            p2c[a[0]] = a[1]

    def gene_cog(g):
        sp = name2sp.get(g)
        return p2c.get(sp) if sp else None

    ess_cogs = {gene_cog(g) for g in ess}
    non_cogs = {gene_cog(g) for g in non}
    ess_cogs = {c for c in ess_cogs if c in cog2idx}
    non_cogs = {c for c in non_cogs if c in cog2idx} - ess_cogs  # one essential gene makes the COG essential
    return ess_cogs, non_cogs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ess-xlsx", default="data/interactome/goodall_st1.xlsx")
    ap.add_argument("--taxon", default="511145")
    ap.add_argument("--ckpt", default=PLAIN_CKPT)
    ap.add_argument("--cog-order", default="data/interactome/cog_order.txt")
    ap.add_argument("--vocab-feather", default="data/COG_train1_phylum.feather")
    ap.add_argument("--module-matrix", default="data/module_matrix_kegg.pt")
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cogs = [l.strip() for l in open(a.cog_order) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    N = len(cogs)

    _, M_mod, M_sizes = load_module_matrix(a.module_matrix, dev, H=1000)
    n_modules = M_mod.shape[1]
    sd = strip_compile_prefix(torch.load(a.ckpt, map_location=dev, weights_only=True))
    model, cls_kind, T0, _ = build_model_for_ckpt(sd, dev, N, n_modules, a.ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    h_field = sd["h"].float().cpu().numpy().reshape(-1)

    present, p2c = species_present(a.taxon, cog2idx)
    present = np.array(present)
    n = len(present)
    present_t = torch.tensor(present.tolist(), device=dev)
    print("loaded %s (class=%s T=%d); E. coli present COGs=%d" % (a.ckpt, cls_kind, T0, n))

    ess_cogs, non_cogs = goodall_cog_labels(a.ess_xlsx, os.path.join(STRDIR, "%s.aliases.txt.gz" % a.taxon),
                                            os.path.join(STRDIR, "%s.protein_to_cog.tsv" % a.taxon), cog2idx)
    print("Goodall: %d essential COGs, %d non-essential COGs" % (len(ess_cogs), len(non_cogs)))

    # Knock out each present COG, read back its own posterior.
    g = -torch.ones(N, device=dev)
    g[present_t] = 1.0
    Xko = g.unsqueeze(0).repeat(n, 1)
    Xko[torch.arange(n, device=dev), present_t] = -1.0
    trk = traj_present(model, Xko, M_mod, M_sizes, present_t, chunk=a.chunk)   # (T, n, n)
    demand = np.diag(trk[-1].numpy())   # own posterior at the final iteration

    # Present COGs Goodall classifies either way; the rest are unlabelled.
    present_ids = [cogs[i] for i in present]
    y, dem, hh = [], [], []
    for k, c in enumerate(present_ids):
        if c in ess_cogs:
            y.append(1)
        elif c in non_cogs:
            y.append(0)
        else:
            continue
        dem.append(demand[k])
        hh.append(h_field[present[k]])
    y = np.array(y); dem = np.array(dem); hh = np.array(hh)
    res = {"taxon": a.taxon, "model": cls_kind, "n_present": int(n),
           "n_labeled": int(y.size), "n_essential": int(y.sum()), "prevalence": float(y.mean())}
    res["AUROC_demand"] = float(roc_auc_score(y, dem))
    res["AUPRC_demand"] = float(average_precision_score(y, dem))
    res["AUROC_field_h"] = float(roc_auc_score(y, hh))          # conservation control
    res["AUPRC_field_h"] = float(average_precision_score(y, hh))
    # Residualise demand on h.
    A = np.vstack([hh, np.ones_like(hh)]).T
    coef, *_ = np.linalg.lstsq(A, dem, rcond=None)
    resid = dem - A @ coef
    res["AUROC_demand_resid_h"] = float(roc_auc_score(y, resid))

    print("labeled COGs=%d (essential=%d, prev=%.3f)" % (y.size, int(y.sum()), y.mean()))
    print("  AUROC: demand=%.3f | field h (conservation)=%.3f | demand|h (residual)=%.3f"
          % (res["AUROC_demand"], res["AUROC_field_h"], res["AUROC_demand_resid_h"]))
    print("  AUPRC: demand=%.3f | field h=%.3f" % (res["AUPRC_demand"], res["AUPRC_field_h"]))
    os.makedirs("data/interactome", exist_ok=True)
    json.dump(res, open("data/interactome/essentiality_ecoli.json", "w"), indent=2)
    print("saved data/interactome/essentiality_ecoli.json")


if __name__ == "__main__":
    main()
