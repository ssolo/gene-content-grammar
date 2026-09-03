#!/usr/bin/env python3.11
"""Phylum-holdout control for the interactome result over 19 human bacterial
pathogens: whether a species' interactome AUROC depends on its own GTDB phylum
having been in training. Generalises the single-species check in
ecoli_zoom_J_ppi.py.

Three couplings are scored per species against STRING's EXPERIMENTAL channel
(physical/biochemical evidence; combined_score is excluded because it embeds a
co-occurrence channel):
  held    : ensemble-mean J over the generalist splits whose VALIDATION set
            contains the species' phylum, so that phylum is never trained on
  intrain : ensemble-mean J over the remaining splits
  all     : ensemble-mean over all 10 splits (the coupling used in the paper)
held ~= intrain ~= all makes the interactome signal a property of the learned
couplings rather than of whether the species' clade was in training.

Inputs:
  gsd_results_higher_order_nohidden_T20_split{1..10}/model_ho3.pth   per-split J
  data/COG_val{1..10}_phylum.feather                    split -> validation phyla
  data/interactome/cog_order.txt
  data/interactome/string/<taxon>.protein_to_cog.tsv + .links.detailed.txt.gz
Output, appended row by row so a re-run resumes past the taxa already present:
  data/interactome/all19_holdout_J_ppi.csv
"""
import csv
import gzip
import os

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO, "data", "interactome")
STRDIR = os.path.join(OUT_DIR, "string")
COG_ORDER = os.path.join(OUT_DIR, "cog_order.txt")
CSV = os.path.join(OUT_DIR, "all19_holdout_J_ppi.csv")
GEN_TMPL = os.path.join(REPO, "gsd_results_higher_order_nohidden_T20_split%d", "model_ho3.pth")
THRESHOLDS = (700, 900)

# (STRING taxon, name, GTDB r220 phylum). The phylum selects the held-out
# splits, matched at runtime against the split feathers; a phylum in no
# validation set leaves the held set empty.
PATHOGENS = [
    ("511145", "Escherichia coli K-12 MG1655",      "p__Pseudomonadota"),
    ("99287",  "Salmonella enterica Typhimurium",   "p__Pseudomonadota"),
    ("208964", "Pseudomonas aeruginosa PAO1",       "p__Pseudomonadota"),
    ("93061",  "Staphylococcus aureus NCTC 8325",   "p__Bacillota"),
    ("272620", "Klebsiella pneumoniae MGH 78578",   "p__Pseudomonadota"),
    ("400667", "Acinetobacter baumannii ATCC 17978","p__Pseudomonadota"),
    ("83332",  "Mycobacterium tuberculosis H37Rv",  "p__Actinomycetota"),
    ("85962",  "Helicobacter pylori 26695",         "p__Campylobacterota"),
    ("171101", "Streptococcus pneumoniae R6",       "p__Bacillota"),
    ("1314",   "Streptococcus pyogenes",            "p__Bacillota"),
    ("242231", "Neisseria gonorrhoeae FA 1090",     "p__Pseudomonadota"),
    ("122586", "Neisseria meningitidis MC58",       "p__Pseudomonadota"),
    ("71421",  "Haemophilus influenzae Rd KW20",    "p__Pseudomonadota"),
    ("243277", "Vibrio cholerae N16961",            "p__Pseudomonadota"),
    ("169963", "Listeria monocytogenes EGD-e",      "p__Bacillota"),
    ("272563", "Clostridioides difficile 630",      "p__Bacillota"),
    ("226185", "Enterococcus faecalis V583",        "p__Bacillota"),
    ("192222", "Campylobacter jejuni NCTC 11168",   "p__Campylobacterota"),
    ("257313", "Bordetella pertussis Tohama I",     "p__Pseudomonadota"),
]


def load_sd(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model", sd.get("state_dict", sd))


def load_split_Js():
    """{split: J} for the generalist higher-order splits, symmetrised and hollow."""
    out = {}
    for s in range(1, 11):
        p = GEN_TMPL % s
        if not os.path.exists(p):
            print("  [warn] missing %s" % p); continue
        sd = load_sd(p)
        if "J" not in sd:
            print("  [warn] no 'J' in %s" % p); continue
        J = sd["J"].float().numpy()
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        out[s] = J.astype(np.float32)
    print("loaded per-split J for splits %s" % sorted(out))
    return out


def split_phyla():
    """{split: set of phyla in that split's VALIDATION feather}."""
    import pandas as pd
    out = {}
    for s in range(1, 11):
        valf = os.path.join(REPO, "data", "COG_val%d_phylum.feather" % s)
        if not os.path.exists(valf):
            print("  [warn] missing %s" % valf); continue
        out[s] = set(pd.read_feather(valf, columns=["phylum"])["phylum"].unique())
    return out


def apc(J):
    """Average product correction (as for DCA contact prediction)."""
    fi = J.mean(axis=1, keepdims=True)
    fj = J.mean(axis=0, keepdims=True)
    f = J.mean()
    Jc = J - (fi * fj) / f if f != 0 else J.copy()
    Jc = 0.5 * (Jc + Jc.T)
    np.fill_diagonal(Jc, 0.0)
    return Jc.astype(np.float32)


def read_species(taxon, cog2idx):
    """One species' STRING experimental labels and its COG index vector.

    Returns (cidx, iu, {threshold: labels}, n_pairs, n_proteins), or None when the
    cached STRING files are missing; cidx maps each protein to its column in J."""
    p2tsv = os.path.join(STRDIR, "%s.protein_to_cog.tsv" % taxon)
    gz = os.path.join(STRDIR, "%s.links.detailed.txt.gz" % taxon)
    if not (os.path.exists(p2tsv) and os.path.exists(gz)):
        print("  [skip] missing STRING cache for %s" % taxon); return None
    prot_cogidx = {}
    with open(p2tsv) as f:
        next(f)
        for line in f:
            pr, cog = line.split("\t")[:2]
            if cog in cog2idx:
                prot_cogidx[pr] = cog2idx[cog]
    proteins = sorted(prot_cogidx)
    pidx = {p: i for i, p in enumerate(proteins)}
    cidx = np.array([prot_cogidx[p] for p in proteins], dtype=np.int64)
    Np = len(proteins)
    EXP = np.zeros((Np, Np), dtype=np.int16)
    with gzip.open(gz, "rt") as f:
        next(f)
        for line in f:
            c = line.split()
            i, j = pidx.get(c[0]), pidx.get(c[1])
            if i is None or j is None:
                continue
            EXP[i, j] = EXP[j, i] = int(c[6])  # links.detailed col 7 = experimental, 0-1000
    iu = np.triu_indices(Np, k=1)
    exp = EXP[iu].astype(np.int32)
    labels = {thr: (exp >= thr).astype(np.int8) for thr in THRESHOLDS}
    return cidx, iu, labels, exp.size, Np


def score(J, cidx, iu, labels):
    """AUROC of J (APC-corrected, as in the paper) per threshold."""
    s = apc(J)[np.ix_(cidx, cidx)][iu]
    out = {}
    for thr, y in labels.items():
        npos = int(y.sum())
        out[thr] = roc_auc_score(y, s) if 0 < npos < y.size else None
        out["npos_%d" % thr] = npos
    return out


def ens(Js, splits):
    return np.mean([Js[s] for s in splits], axis=0).astype(np.float32) if splits else None


def main():
    cogs = [l.strip() for l in open(COG_ORDER) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    print("vocab: %d COGs" % len(cogs))

    Js = load_split_Js()
    all_splits = sorted(Js)
    phyla = split_phyla()
    for s in all_splits:
        print("  split %d val phyla: %d" % (s, len(phyla.get(s, set()))))

    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            done = {r["taxon"] for r in csv.DictReader(f)}
        print("resume: %d taxa already done" % len(done))

    cols = ["taxon", "name", "phylum", "n_held", "n_intrain", "n_proteins", "n_pairs"]
    for thr in THRESHOLDS:
        cols += ["held_%d" % thr, "intrain_%d" % thr, "all_%d" % thr, "delta_%d" % thr]
    new_file = not os.path.exists(CSV)
    fcsv = open(CSV, "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=cols)
    if new_file:
        w.writeheader(); fcsv.flush()

    for taxon, name, phylum in PATHOGENS:
        if taxon in done:
            print("== %s skip (done) ==" % name); continue
        held = [s for s in all_splits if phylum in phyla.get(s, set())]
        intrain = [s for s in all_splits if s not in held]
        print("\n== %s [%s] held-out splits %s | in-train %s ==" % (name, phylum, held, intrain))
        sp = read_species(taxon, cog2idx)
        if sp is None:
            continue
        cidx, iu, labels, n_pairs, Np = sp
        J_held, J_in, J_all = ens(Js, held), ens(Js, intrain), ens(Js, all_splits)
        m_held = score(J_held, cidx, iu, labels) if J_held is not None else {}
        m_in = score(J_in, cidx, iu, labels) if J_in is not None else {}
        m_all = score(J_all, cidx, iu, labels)
        row = {"taxon": taxon, "name": name, "phylum": phylum,
               "n_held": len(held), "n_intrain": len(intrain),
               "n_proteins": Np, "n_pairs": n_pairs}
        for thr in THRESHOLDS:
            h, i, a = m_held.get(thr), m_in.get(thr), m_all.get(thr)
            row["held_%d" % thr] = None if h is None else round(h, 4)
            row["intrain_%d" % thr] = None if i is None else round(i, 4)
            row["all_%d" % thr] = None if a is None else round(a, 4)
            row["delta_%d" % thr] = (None if (h is None or i is None) else round(h - i, 4))
            print("  exp>=%d  held=%s  intrain=%s  all=%s  (held-intrain delta=%s)"
                  % (thr, row["held_%d" % thr], row["intrain_%d" % thr],
                     row["all_%d" % thr], row["delta_%d" % thr]))
        w.writerow(row); fcsv.flush()

    fcsv.close()
    print("\nwrote %s" % CSV)


if __name__ == "__main__":
    main()
