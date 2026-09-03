#!/usr/bin/env python3.11
"""Aggregate the per-species Ising-J interactome benchmark into ProteomeLM-
comparable metrics and a figure.

Reads data/interactome/results/<taxon>.npz (s_J, s_Japc, exp, cooc, comb,
taxon, name) produced by benchmark_string_ppi.py, and produces:

  data/interactome/interactome_summary.json   per-species + pooled metrics
  data/interactome/interactome_per_species.csv
  analysis/figures/fig_interactome.pdf         3-panel figure:
     (A) per-species AUROC of J (exp>=700), the analog of ProteomeLM Fig 3E;
     (B) precision among the top-N J-ranked pairs vs N (pooled), the analog of
         ProteomeLM Fig 3D ("fraction of top predictions in STRING");
     (C) pooled precision-recall curve for J (exp>=700) with the random
         baseline and the AUPRC.

Ground truth is STRING's EXPERIMENTAL channel only, which is non-circular for a
co-occurrence predictor. The pooled AUROC of STRING's own *cooccurrence* channel
against the experimental labels is reported as a reference point: it is high
precisely because it carries the same phylogenetic-profiling signal as J, which
is why J must NOT be scored against combined_score.

The pooled score/label arrays are multi-GB, so this needs a host with tens of GB
of RAM.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

RESDIR = "data/interactome/results"
OUTJSON = "data/interactome/interactome_summary.json"
OUTCSV = "data/interactome/interactome_per_species.csv"
OUTFIG = "analysis/figures/fig_interactome.pdf"
THRS = [700, 900]


def precision_at_topN(score, y, Ns):
    order = np.argsort(score)[::-1]
    ys = y[order]
    csum = np.cumsum(ys)
    out = {}
    for N in Ns:
        N = min(N, len(ys))
        out[N] = float(csum[N - 1] / N)
    return out


def main():
    files = sorted(glob.glob(os.path.join(RESDIR, "*.npz")))
    print("aggregating %d species" % len(files))
    per = []
    pooled_sJ, pooled_sJapc, pooled_exp, pooled_cooc, pooled_comb = [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        name = str(d["name"]); taxon = str(d["taxon"])
        sJ, sJapc, exp, cooc, comb = d["s_J"], d["s_Japc"], d["exp"], d["cooc"], d["comb"]
        rec = {"taxon": taxon, "name": name, "n_pairs": int(exp.size)}
        for thr in THRS:
            y = (exp >= thr).astype(np.int8)
            npos = int(y.sum())
            if 0 < npos < y.size:
                rec["AUROC_J_%d" % thr] = float(roc_auc_score(y, sJ))
                rec["AUPRC_J_%d" % thr] = float(average_precision_score(y, sJ))
                rec["AUROC_Japc_%d" % thr] = float(roc_auc_score(y, sJapc))
                rec["prevalence_%d" % thr] = float(y.mean())
                rec["n_pos_%d" % thr] = npos
        per.append(rec)
        pooled_sJ.append(sJ); pooled_sJapc.append(sJapc)
        pooled_exp.append(exp); pooled_cooc.append(cooc); pooled_comb.append(comb)
        print("  %-34s pairs=%9d  AUROC700=%.3f" % (name, exp.size, rec.get("AUROC_J_700", float("nan"))))

    sJ = np.concatenate(pooled_sJ); sJapc = np.concatenate(pooled_sJapc)
    exp = np.concatenate(pooled_exp); cooc = np.concatenate(pooled_cooc)
    comb = np.concatenate(pooled_comb)
    print("pooled pairs: %d" % exp.size)

    pooled = {"n_species": len(files), "n_pairs": int(exp.size)}
    pr_curves = {}
    for thr in THRS:
        y = (exp >= thr).astype(np.int8)
        pooled["AUROC_J_%d" % thr] = float(roc_auc_score(y, sJ))
        pooled["AUPRC_J_%d" % thr] = float(average_precision_score(y, sJ))
        pooled["AUROC_Japc_%d" % thr] = float(roc_auc_score(y, sJapc))
        pooled["AUPRC_Japc_%d" % thr] = float(average_precision_score(y, sJapc))
        pooled["prevalence_%d" % thr] = float(y.mean())
        pooled["n_pos_%d" % thr] = int(y.sum())
        # Reference only: STRING's own cooccurrence channel against the
        # experimental labels.  Same signal family as J, hence circular.
        pooled["AUROC_cooc_channel_%d" % thr] = float(roc_auc_score(y, cooc))
        prec, rec_, _ = precision_recall_curve(y, sJ)
        pr_curves[thr] = (rec_, prec)

    # J against STRING combined_score, the positive definition ProteomeLM
    # reports AUC against.  combined_score includes the cooccurrence channel,
    # so for a co-occurrence predictor like J it is partly CIRCULAR; it is
    # computed only to compare on the same target.
    for thr in THRS:
        yc = (comb >= thr).astype(np.int8)
        if 0 < yc.sum() < yc.size:
            pooled["AUROC_J_vs_combined_%d" % thr] = float(roc_auc_score(yc, sJ))
            pooled["AUPRC_J_vs_combined_%d" % thr] = float(average_precision_score(yc, sJ))
            pooled["prevalence_combined_%d" % thr] = float(yc.mean())

    Ns = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    patN = precision_at_topN(sJ, (exp >= 700).astype(np.int8), Ns)
    pooled["precision_at_topN_exp700"] = {str(k): v for k, v in patN.items()}

    json.dump({"pooled": pooled, "per_species": per}, open(OUTJSON, "w"), indent=2)
    cols = ["taxon", "name", "n_pairs", "n_pos_700", "prevalence_700",
            "AUROC_J_700", "AUPRC_J_700", "AUROC_Japc_700", "AUROC_J_900", "AUPRC_J_900"]
    with open(OUTCSV, "w") as fo:
        fo.write(",".join(cols) + "\n")
        for r in per:
            fo.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    rows = [r for r in per if "AUROC_J_700" in r]
    rows.sort(key=lambda r: r["AUROC_J_700"])
    def _short(nm):  # "Escherichia coli K-12" / "S.aureus" -> "E. coli" / "S. aureus"
        p = nm.replace(".", " ").split()
        return (p[0][0] + ". " + p[1]) if len(p) >= 2 else nm
    names = [_short(r["name"]) for r in rows]
    vals = [r["AUROC_J_700"] for r in rows]
    ax[0].barh(range(len(rows)), vals, color="#4C72B0")
    ax[0].axvline(0.5, color="0.5", ls="--", lw=1)
    ax[0].axvline(np.mean(vals), color="#C44E52", ls="-", lw=1.5, label="mean %.3f" % np.mean(vals))
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlabel("AUROC (Ising J vs STRING experimental >=700)")
    ax[0].set_xlim(0.45, max(0.8, max(vals) + 0.02))
    ax[0].set_title("(A) Per-species recovery (higher better)\nProteomeLM Fig 3E analog")
    ax[0].legend(loc="lower right", fontsize=9)

    Nsx = sorted(patN)
    ax[1].plot(Nsx, [patN[n] for n in Nsx], "o-", color="#4C72B0", label="Ising J")
    ax[1].axhline(pooled["prevalence_700"], color="0.5", ls="--", lw=1,
                  label="random %.1e" % pooled["prevalence_700"])
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("number of top J-ranked pairs (N)")
    ax[1].set_ylabel("fraction experimentally validated (exp>=700)")
    ax[1].set_title("(B) Precision of top predictions (higher better)\nProteomeLM Fig 3D analog")
    ax[1].legend(fontsize=9)

    rec_, prec = pr_curves[700]
    ax[2].plot(rec_, prec, color="#4C72B0", lw=2,
               label="J  AUPRC=%.3f" % pooled["AUPRC_J_700"])
    ax[2].axhline(pooled["prevalence_700"], color="0.5", ls="--", lw=1,
                  label="random %.1e" % pooled["prevalence_700"])
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
    ax[2].set_xlim(0, 1); ax[2].set_ylim(bottom=0)
    ax[2].set_title("(C) Pooled precision-recall (exp>=700)\nAUROC=%.3f" % pooled["AUROC_J_700"])
    ax[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTFIG, bbox_inches="tight")
    print("wrote", OUTFIG)
    print("POOLED: AUROC700=%.3f AUPRC700=%.3f | AUROC900=%.3f | cooc-channel AUROC700=%.3f (circular ref)"
          % (pooled["AUROC_J_700"], pooled["AUPRC_J_700"], pooled["AUROC_J_900"],
             pooled["AUROC_cooc_channel_700"]))


if __name__ == "__main__":
    main()
