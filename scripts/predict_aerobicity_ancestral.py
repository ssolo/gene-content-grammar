#!/usr/bin/env python3
"""Aerobe/anaerobe prediction on every LBCA and LACA reconstruction, before and
after denoising.

Protocol follows phenotype_noise/ancestral_phenotype.py: ancestral
reconstructions are presence/absence, so the extant training genomes are
BINARISED (copy number > 0) and an XGBoost classifier is refit on binary
features before being applied to the ancestral binary feature vectors.  One
classifier per phylum-holdout split (0-9); the reported P(aerobe) is the mean
and s.d. over the ten splits.

Inputs:
  phenotype_noise/aerob_splits/train_{data,annot}_phylum_tax_level_split_{0..9}
  phenotype_noise/aerob_all_gene_annotations.tsv   (defines the 2,677 feature COGs)
  data/lbca_cog_lists.tsv, data/laca_cog_lists.tsv

Output: data/ancestral_aerobicity.tsv
"""
import numpy as np, pandas as pd, csv, os
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PH = os.path.join(REPO, "phenotype_noise")
SP = os.path.join(PH, "aerob_splits")
THR = 0.5

feat = [c for c in pd.read_csv(os.path.join(PH, "aerob_all_gene_annotations.tsv"),
                               sep="\t", nrows=0).columns if c != "accession"]
fidx = {c: i for i, c in enumerate(feat)}
print(f"{len(feat)} feature COGs")

# Reconstruction COG sets: (node, label, input set, denoised set)
def sets_from(path, conds):
    """Input sets at BOTH conventions: p > 0, the count the paper quotes (e.g. the
    493-family sparse LBCA), and p > 0.5, the rule used for the denoised call."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    out = []
    for key, lab in conds:
        in0 = {r["COG_ID"] for r in rows if float(r[f"{key}_input"]) > 0.0}
        in5 = {r["COG_ID"] for r in rows if float(r[f"{key}_input"]) > THR}
        den = {r["COG_ID"] for r in rows if r[f"{key}_present"] == "1"}
        out.append((lab, in0, in5, den))
    return out

recs = []
for r in sets_from(os.path.join(REPO, "data/lbca_cog_lists.tsv"),
        [("sparse", "sparse (reconciliation)"), ("GLD_min1", "copy-number min-1"),
         ("GLD_min4", "copy-number min-4")]):
    recs.append(("LBCA",) + r)
for r in sets_from(os.path.join(REPO, "data/laca_cog_lists.tsv"),
        [("combined", "combined root"), ("euryroot_ml", "Euryarchaeota (ML)"),
         ("euryroot_unif", "Euryarchaeota (uniform)"),
         ("gld_min1", "copy-number min-1"), ("gld_min4", "copy-number min-4")]):
    recs.append(("LACA",) + r)

def vec(cogset):
    v = np.zeros(len(feat), dtype=np.float32)
    for c in cogset:
        j = fidx.get(c)
        if j is not None:
            v[j] = 1.0
    return v

X_anc, meta = [], []
fset = set(feat)
for node, lab, in0, in5, den in recs:
    for arm, s in (("input (p>0)", in0), ("input (p>0.5)", in5), ("denoised", den)):
        X_anc.append(vec(s)); meta.append((node, lab, arm, len(s), len(s & fset)))
X_anc = np.vstack(X_anc)

# One classifier per split, trained on binarised extant genomes
P = []
for s in range(10):
    d = pd.read_csv(f"{SP}/train_data_phylum_tax_level_split_{s}", sep="\t")
    a = pd.read_csv(f"{SP}/train_annot_phylum_tax_level_split_{s}", sep="\t")
    lab = dict(zip(a["accession"], (a["annotation"] == "aerobe").astype(int)))
    d = d[d["accession"].isin(lab)]
    Xtr = (d[feat].to_numpy() > 0).astype(np.float32)
    ytr = d["accession"].map(lab).to_numpy()
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.7, colsample_bytree=0.7, tree_method="hist",
                        eval_metric="logloss")
    clf.fit(Xtr, ytr)
    P.append(clf.predict_proba(X_anc)[:, 1])
    print(f"  split {s}: trained on {len(ytr)} genomes ({ytr.mean()*100:.0f}% aerobe)")
P = np.vstack(P)
mean, sd = P.mean(0), P.std(0)

rows = []
for i, (node, lab, arm, n, nfeat) in enumerate(meta):
    rows.append(dict(node=node, reconstruction=lab, arm=arm, n_cogs=n,
                     n_feature_cogs=nfeat, p_aerobe=round(float(mean[i]), 4),
                     sd=round(float(sd[i]), 4),
                     call="aerobe" if mean[i] > 0.5 else "anaerobe"))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(REPO, "data/ancestral_aerobicity.tsv"), sep="\t", index=False)
print("\n" + df.to_string(index=False))
print(f"\nwrote data/ancestral_aerobicity.tsv")
