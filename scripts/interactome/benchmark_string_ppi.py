#!/usr/bin/env python3.11
"""Benchmark the Ising coupling J as a PPI predictor for one bacterial species
against STRING's experimental channel.

Ground truth is STRING's experimental sub-channel (physical/biochemical
evidence), not combined_score: J is a pairwise coupling learned from gene
content, and combined_score folds in STRING's own cooccurrence (phylogenetic
profiling) and textmining channels, so scoring J against it would be circular.

Following the ProteomeLM PPI benchmark, every intra-species protein pair whose
two members both map to a COG family present in J is scored by
J[cog(p1), cog(p2)] and labelled positive when STRING experimental >= threshold.
AUROC and AUPRC are reported for raw and APC-corrected J; the cooccurrence and
combined channels are reported alongside as references, not as ground truth.

Usage:
  .venv/bin/python scripts/interactome/benchmark_string_ppi.py \
      --taxon 511145 --name "Escherichia coli K-12" \
      --J data/interactome/J_plain_pairwise_T20.npy

Output: data/interactome/results/<taxon>.npz and <taxon>.metrics.json
"""
import argparse
import gzip
import json
import os
import subprocess
import urllib.request

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

STRDIR = "data/interactome/string"
RESDIR = "data/interactome/results"
LINKS_URL = "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/{t}.protein.links.detailed.v12.0.txt.gz"


def apc(J):
    """Average product correction (as used for DCA contact prediction)."""
    n = J.shape[0]
    fi = J.mean(axis=1, keepdims=True)
    fj = J.mean(axis=0, keepdims=True)
    f = J.mean()
    # Broadcast rather than matmul: some numpy BLAS builds segfault on the
    # (N, 1) @ (1, N) gemm at this N.
    Jc = J - (fi * fj) / f if f != 0 else J.copy()
    Jc = 0.5 * (Jc + Jc.T)
    np.fill_diagonal(Jc, 0.0)
    return Jc


def fetch(url, dst):
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        urllib.request.urlretrieve(url, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--J", default="data/interactome/J_plain_pairwise_T20.npy")
    ap.add_argument("--cog-order", default="data/interactome/cog_order.txt")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[700, 900])
    a = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)
    t = a.taxon

    cogs = [l.strip() for l in open(a.cog_order) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    J = np.load(a.J).astype(np.float32)
    J_apc = apc(J).astype(np.float32)

    # protein -> COG index (only COGs present in J)
    p2tsv = os.path.join(STRDIR, "%s.protein_to_cog.tsv" % t)
    prot_cogidx = {}
    with open(p2tsv) as f:
        next(f)
        for line in f:
            p, cog = line.split("\t")[:2]
            if cog in cog2idx:
                prot_cogidx[p] = cog2idx[cog]
    proteins = sorted(prot_cogidx)
    pidx = {p: i for i, p in enumerate(proteins)}
    cidx = np.array([prot_cogidx[p] for p in proteins], dtype=np.int64)
    Np = len(proteins)
    print("%s (taxon %s): %d COG-mapped proteins" % (a.name, t, Np))

    gz = os.path.join(STRDIR, "%s.links.detailed.txt.gz" % t)
    fetch(LINKS_URL.format(t=t), gz)
    EXP = np.zeros((Np, Np), dtype=np.int16)
    COO = np.zeros((Np, Np), dtype=np.int16)
    COMB = np.zeros((Np, Np), dtype=np.int16)
    n_links = 0
    with gzip.open(gz, "rt") as f:
        next(f)
        for line in f:
            p = line.split()
            a1, a2 = p[0], p[1]
            i, j = pidx.get(a1), pidx.get(a2)
            if i is None or j is None:
                continue
            # links.detailed cols 5/7/10 (1-based) = cooc/experimental/combined, 0-1000
            cooc, expv, comb = int(p[4]), int(p[6]), int(p[9])
            EXP[i, j] = EXP[j, i] = expv
            COO[i, j] = COO[j, i] = cooc
            COMB[i, j] = COMB[j, i] = comb
            n_links += 1
    print("  %d STRING links among mapped proteins" % n_links)

    iu = np.triu_indices(Np, k=1)
    s_J = J[np.ix_(cidx, cidx)][iu]
    s_Japc = J_apc[np.ix_(cidx, cidx)][iu]
    exp = EXP[iu].astype(np.int32)
    cooc = COO[iu].astype(np.int32)
    comb = COMB[iu].astype(np.int32)
    n_pairs = exp.size

    metrics = {"taxon": t, "name": a.name, "n_proteins_mapped": Np,
               "n_pairs": int(n_pairs), "J_file": os.path.basename(a.J)}
    for thr in a.thresholds:
        y = (exp >= thr).astype(np.int8)
        npos = int(y.sum())
        m = {"n_pos": npos, "prevalence": float(y.mean())}
        if 0 < npos < n_pairs:
            m["AUROC_J"] = float(roc_auc_score(y, s_J))
            m["AUPRC_J"] = float(average_precision_score(y, s_J))
            m["AUROC_Japc"] = float(roc_auc_score(y, s_Japc))
            m["AUPRC_Japc"] = float(average_precision_score(y, s_Japc))
            m["AUROC_cooc_channel"] = float(roc_auc_score(y, cooc))
            m["AUROC_combined_channel"] = float(roc_auc_score(y, comb))
        metrics["thr_%d" % thr] = m
        if "AUROC_J" in m:
            print("  exp>=%d: %d pos / %d  prev=%.2e | J AUROC=%.3f AUPRC=%.3f | "
                  "Japc AUROC=%.3f AUPRC=%.3f | (cooc-channel AUROC=%.3f)"
                  % (thr, npos, n_pairs, m["prevalence"], m["AUROC_J"], m["AUPRC_J"],
                     m["AUROC_Japc"], m["AUPRC_Japc"], m["AUROC_cooc_channel"]))

    np.savez_compressed(os.path.join(RESDIR, "%s.npz" % t),
                        s_J=s_J.astype(np.float32), s_Japc=s_Japc.astype(np.float32),
                        exp=exp.astype(np.int32), cooc=cooc.astype(np.int32),
                        comb=comb.astype(np.int32),
                        taxon=t, name=a.name)
    with open(os.path.join(RESDIR, "%s.metrics.json" % t), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  saved %s.npz + metrics.json" % t)


if __name__ == "__main__":
    main()
