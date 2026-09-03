#!/usr/bin/env python3
"""Run all three trait predictors on every LBCA and LACA reconstruction, before and
after denoising.

  aerobicity      binary  (aerobe / anaerobe)     2,677 COG features
  cell envelope   binary  (diderm / monoderm)     2,665 COG features
  OGT             regression (degrees C)          4,789 COG features

Protocol follows phenotype_noise/ancestral_phenotype.py: ancestral reconstructions
are presence/absence, so the extant training genomes are BINARISED (copy number > 0)
and each model is refit on binary features, one fit per phylum-holdout split (0-9);
the reported value is the mean and s.d. over the ten splits.

The OGT training table also carries 20 amino-acid composition columns. Amino-acid
composition cannot be derived from a gene-content reconstruction, so the ancestral
OGT model is refit on the COG columns ALONE and is weaker than the full OGT
predictor; the held-out RMSE printed below quantifies that.

Out: data/ancestral_phenotypes.tsv
"""
import numpy as np, pandas as pd, csv, os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import matthews_corrcoef, r2_score, mean_squared_error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PH = os.path.join(REPO, "phenotype_noise")
THR = 0.5
PHENO = {
    "aerobicity":    dict(dir="aerob_splits",  kind="bin", pos="aerobe",   neg="anaerobe"),
    "cell_envelope": dict(dir="diderm_splits", kind="bin", pos="Diderm",   neg="Monoderm"),
    "OGT":           dict(dir="ogt_splits",    kind="reg"),
}

# ---- ancestral COG sets
def sets_from(path, conds):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    out = []
    for key, lab in conds:
        out.append((lab,
                    {r["COG_ID"] for r in rows if float(r[f"{key}_input"]) > 0.0},
                    {r["COG_ID"] for r in rows if r[f"{key}_present"] == "1"}))
    return out

recs = [("LBCA",) + r for r in sets_from(os.path.join(REPO, "data/lbca_cog_lists.tsv"),
        [("sparse", "sparse (reconciliation)"), ("GLD_min1", "copy-number min-1"),
         ("GLD_min4", "copy-number min-4")])]
recs += [("LACA",) + r for r in sets_from(os.path.join(REPO, "data/laca_cog_lists.tsv"),
        [("combined", "combined root"), ("euryroot_ml", "Euryarchaeota (ML)"),
         ("euryroot_unif", "Euryarchaeota (uniform)"),
         ("gld_min1", "copy-number min-1"), ("gld_min4", "copy-number min-4")])]

rows_out = []
for pname, cfg in PHENO.items():
    SP = os.path.join(PH, cfg["dir"])
    d0 = pd.read_csv(f"{SP}/train_data_phylum_tax_level_split_0", sep="\t", nrows=0)
    feat = [c for c in d0.columns if c.startswith("COG")]      # drops the 20 AA-composition cols
    fset, fidx = set(feat), {c: i for i, c in enumerate(feat)}

    def vec(s):
        v = np.zeros(len(feat), dtype=np.float32)
        for c in s:
            j = fidx.get(c)
            if j is not None: v[j] = 1.0
        return v

    X_anc, meta = [], []
    for node, lab, in0, den in recs:
        for arm, s in (("input", in0), ("denoised", den)):
            X_anc.append(vec(s)); meta.append((node, lab, arm, len(s), len(s & fset)))
    X_anc = np.vstack(X_anc)

    P, scores = [], []
    for s in range(10):
        d = pd.read_csv(f"{SP}/train_data_phylum_tax_level_split_{s}", sep="\t")
        a = pd.read_csv(f"{SP}/train_annot_phylum_tax_level_split_{s}", sep="\t")
        if cfg["kind"] == "bin":
            lab = dict(zip(a["accession"], (a["annotation"] == cfg["pos"]).astype(int)))
        else:
            lab = dict(zip(a["accession"], pd.to_numeric(a["annotation"], errors="coerce")))
        d = d[d["accession"].isin(lab)]
        X = (d[feat].to_numpy() > 0).astype(np.float32)
        y = d["accession"].map(lab).to_numpy()
        ok = ~pd.isna(y); X, y = X[ok], y[ok].astype(float)
        n = len(y); cut = int(n * 0.85); rng = np.random.RandomState(0)
        perm = rng.permutation(n); tr, te = perm[:cut], perm[cut:]
        kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.7,
                  colsample_bytree=0.7, tree_method="hist")
        M = XGBClassifier(eval_metric="logloss", **kw) if cfg["kind"] == "bin" else XGBRegressor(**kw)
        M.fit(X[tr], y[tr])
        if cfg["kind"] == "bin":
            scores.append(matthews_corrcoef(y[te], (M.predict_proba(X[te])[:, 1] > .5).astype(int)))
            P.append(M.predict_proba(X_anc)[:, 1])
        else:
            scores.append(np.sqrt(mean_squared_error(y[te], M.predict(X[te]))))
            P.append(M.predict(X_anc))
    P = np.vstack(P); mean, sd = P.mean(0), P.std(0)
    sname = "MCC" if cfg["kind"] == "bin" else "RMSE(C)"
    print(f"{pname}: {len(feat)} COG features | internal held-out {sname} = "
          f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    for i, (node, lab, arm, n, nf) in enumerate(meta):
        r = dict(phenotype=pname, node=node, reconstruction=lab, arm=arm,
                 n_cogs=n, n_feature_cogs=nf, value=round(float(mean[i]), 4),
                 sd=round(float(sd[i]), 4))
        r["call"] = (cfg["pos"] if mean[i] > .5 else cfg["neg"]) if cfg["kind"] == "bin" \
                    else f"{mean[i]:.1f} C"
        rows_out.append(r)

df = pd.DataFrame(rows_out)
df.to_csv(os.path.join(REPO, "data/ancestral_phenotypes.tsv"), sep="\t", index=False)
for p in PHENO:
    print("\n" + df[df.phenotype == p].drop(columns="phenotype").to_string(index=False))
print("\nwrote data/ancestral_phenotypes.tsv")
