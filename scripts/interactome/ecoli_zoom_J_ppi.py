#!/usr/bin/env python3.11
"""E. coli interactome AUROC of the coupling J as a function of training
distance to E. coli.

The interactome benchmark in the paper scores the 10-split ensemble mean J
(extract_J.py). E. coli's phylum (Pseudomonadota) is in the training set in ~8
of the 10 whole-phylum splits, so the reported J is not a held-out coupling.
This script measures how that E. coli number depends on training proximity,
using the E. coli-zoom models (each holds out E. coli's clade at a
progressively finer rank) and the generalist splits partitioned by whether
Pseudomonadota was held out.

For each condition it extracts a single coupling matrix J (N x N, checkpoint key
'J'), symmetrises and hollows it, saves it as data/interactome/J_<cond>.npy, then
runs the E. coli PPI benchmark (benchmark_string_ppi.py, taxon 511145, STRING
v12.0 experimental channel) and records AUROC at exp >= 700 and >= 900, for raw
J and APC-corrected J.

Conditions
----------
  zoom_<rank>   rank in {phylum,class,order,intermediate,family,species}
                gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_<rank>/model_ho3.pth
                (Zenodo bundle: ecolizoom/<rank>/model_ho3.pth)
  gen_all       generalist HO-T20 ensemble over all 10 splits (== the paper's J)
  gen_heldout   generalist ensemble over splits where Pseudomonadota is in VAL
  gen_intrain   generalist ensemble over splits where Pseudomonadota is in TRAIN
                gsd_results_higher_order_nohidden_T20_split<S>/model_ho3.pth

Needs the checkpoints, the vocabulary feather and the STRING cache locally, and
a working pyarrow for the feather (see extract_J.py):
  .venv/bin/python scripts/interactome/ecoli_zoom_J_ppi.py               # zoom only
  .venv/bin/python scripts/interactome/ecoli_zoom_J_ppi.py --generalist
Output: data/interactome/ecoli_zoom_J_ppi.csv, one row per condition.

Interpretation limits: the zoom models are bac_fp_marginal_hq fine-tunes sharing
a backbone J, and the HO3 stage trains J at j_lr_frac x base_lr, so the per-rank
Js differ little; the fine-tune holds E. coli's clade out only of the fine-tuning
set, while the backbone is not phylum-clean. The gen_heldout vs gen_intrain
contrast is the controlled test of whether holding E. coli's phylum out of
training changes the interactome signal.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO, "data", "interactome")
RESDIR = os.path.join(OUT_DIR, "results")
VOCAB_FEATHER = os.path.join(REPO, "data", "COG_train1_phylum.feather")
BENCH = os.path.join(HERE, "benchmark_string_ppi.py")

TAXON = "511145"          # Escherichia coli K-12 MG1655 (STRING)
TAXON_NAME = "Escherichia coli K-12 MG1655"
PHYLUM = "p__Pseudomonadota"  # GTDB r220 name for Proteobacteria
RANKS = ["phylum", "class", "order", "intermediate", "family", "species"]
ZOOM_TMPL = ("gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_"
             "ecolizoom_{rank}/model_ho3.pth")
GEN_TMPL = "gsd_results_higher_order_nohidden_T20_split{s}/model_ho3.pth"


def load_sd(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model", sd.get("state_dict", sd))


def extract_J(paths):
    """Ensemble-mean J over one or more checkpoints, symmetrised and hollowed.

    Returns (J, used) with J of shape (N, N) float32, or (None, []) if no
    checkpoint in ``paths`` carries a 'J' entry.
    """
    acc, used = None, []
    for p in paths:
        sd = load_sd(p)
        if "J" not in sd:
            print("  [skip] %s has no 'J'" % p)
            continue
        J = sd["J"].float().numpy()
        acc = J.copy() if acc is None else acc + J
        used.append(p)
    if not used:
        return None, []
    J = acc / len(used)
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return J.astype(np.float32), used


def save_cog_order():
    """Ensure data/interactome/cog_order.txt exists; return the number of COGs.

    Reuses an existing file (written by extract_J.py) so the vocabulary feather,
    which needs a working pyarrow, need not be read. This order is the column
    order of the model's J and must be the order benchmark_string_ppi.py indexes
    with.
    """
    cop = os.path.join(OUT_DIR, "cog_order.txt")
    if os.path.exists(cop):
        n = sum(1 for l in open(cop) if l.strip())
        if n > 0:
            print("using existing %s (%d COGs) -- no feather read needed" % (cop, n))
            return n
    import pandas as pd  # needs pyarrow
    cols = list(pd.read_feather(VOCAB_FEATHER).columns)
    cogs = [c for c in cols if c.startswith("COG")]
    with open(cop, "w") as f:
        f.write("\n".join(cogs) + "\n")
    return len(cogs)


def split_partition_by_phylum():
    """Return (heldout_splits, intrain_splits): splits where PHYLUM is in the
    validation vs the training feather. Reads COG_val<S>_phylum.feather, which
    needs pyarrow; returns ([], []) rather than raising if the feathers cannot
    be read, so the zoom sweep still completes."""
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        print("  [warn] pandas/pyarrow unavailable (%s) -- skipping generalist" % e)
        return [], []
    heldout, intrain = [], []
    for s in range(1, 11):
        valf = os.path.join(REPO, "data", "COG_val%d_phylum.feather" % s)
        if not os.path.exists(valf):
            print("  [warn] missing %s -- skipping split %d" % (valf, s))
            continue
        try:
            val_phyla = set(pd.read_feather(valf, columns=["phylum"])["phylum"].unique())
        except Exception as e:
            print("  [warn] cannot read %s (%s) -- run --generalist where pyarrow can read the feathers" % (valf, e))
            return [], []
        (heldout if PHYLUM in val_phyla else intrain).append(s)
    return heldout, intrain


def score(jpath):
    """Run the E. coli PPI benchmark on one saved J; return its metrics dict."""
    cmd = [sys.executable, BENCH, "--taxon", TAXON, "--name", TAXON_NAME,
           "--J", jpath, "--cog-order", os.path.join(OUT_DIR, "cog_order.txt")]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)
    with open(os.path.join(RESDIR, "%s.metrics.json" % TAXON)) as f:
        return json.load(f)


def row(cond, ckpts, metrics):
    r = {"condition": cond, "n_ckpt": len(ckpts)}
    for thr in (700, 900):
        m = metrics.get("thr_%d" % thr, {})
        r["AUROC_J_%d" % thr] = m.get("AUROC_J")
        r["AUROC_Japc_%d" % thr] = m.get("AUROC_Japc")
        r["n_pos_%d" % thr] = m.get("n_pos")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generalist", action="store_true",
                    help="also build gen_all / gen_heldout / gen_intrain ensembles")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)
    n_cog = save_cog_order()
    print("vocab: %d COG columns" % n_cog)

    conditions = []  # (cond_name, [checkpoint paths])
    for rank in RANKS:
        p = os.path.join(REPO, ZOOM_TMPL.format(rank=rank))
        conditions.append(("zoom_%s" % rank, [p] if os.path.exists(p) else []))

    if a.generalist:
        gen = sorted(glob.glob(os.path.join(REPO, GEN_TMPL.format(s="*"))))
        held, intrn = split_partition_by_phylum()
        print("Pseudomonadota held-out in splits %s; in-training in %s" % (held, intrn))
        conditions.append(("gen_all", gen))
        conditions.append(("gen_heldout",
                            [os.path.join(REPO, GEN_TMPL.format(s=s)) for s in held]))
        conditions.append(("gen_intrain",
                            [os.path.join(REPO, GEN_TMPL.format(s=s)) for s in intrn]))

    rows = []
    for cond, ckpts in conditions:
        print("\n== %s (%d ckpt) ==" % (cond, len(ckpts)))
        ckpts = [c for c in ckpts if os.path.exists(c)]
        if not ckpts:
            print("  [skip] no checkpoints found for %s" % cond)
            continue
        J, used = extract_J(ckpts)
        if J is None:
            print("  [skip] no 'J' in checkpoints for %s" % cond)
            continue
        jpath = os.path.join(OUT_DIR, "J_%s.npy" % cond)
        np.save(jpath, J)
        rows.append(row(cond, used, score(jpath)))

    import csv
    csvp = os.path.join(OUT_DIR, "ecoli_zoom_J_ppi.csv")
    cols = ["condition", "n_ckpt",
            "AUROC_J_700", "AUROC_Japc_700", "n_pos_700",
            "AUROC_J_900", "AUROC_Japc_900", "n_pos_900"]
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s" % csvp)
    print("\n%-14s %8s %8s %8s %8s" % ("condition", "Japc700", "Japc900", "J700", "J900"))
    for r in rows:
        print("%-14s %8.3f %8.3f %8.3f %8.3f" % (
            r["condition"], r["AUROC_Japc_700"] or 0, r["AUROC_Japc_900"] or 0,
            r["AUROC_J_700"] or 0, r["AUROC_J_900"] or 0))


if __name__ == "__main__":
    main()
