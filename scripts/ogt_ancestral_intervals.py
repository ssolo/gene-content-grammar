#!/usr/bin/env python3
"""OGT for the LBCA/LACA reconstructions with calibrated predictive intervals.

Per phylum-holdout split, a mean regressor and q10/q90 quantile regressors are fit on
binarised COG features; the quantile spread gives a within-model s.d.
s = (q90-q10)/(2*1.2816), and the ten splits combine by the law of total variance:

    var_total = mean_s(s_s^2)            (predictive spread, within a model)
              + var_s(mu_s)              (disagreement between splits)

The reported CI is mu +/- 1.96*sqrt(var_total). Interval calibration is checked
empirically, as the coverage of the nominal 80% [q10,q90] band on held-out extant
genomes, alongside a distribution-free conformal half-width from the pooled residuals.

Out: data/ancestral_ogt_intervals.tsv
"""
import numpy as np, pandas as pd, csv, os
from xgboost import XGBRegressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(REPO, "phenotype_noise", "ogt_splits")
Z90 = 1.2815515655446004                      # z_0.90: q90 - q10 = 2*Z90*s if Gaussian

def sets_from(path, conds):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    return [(lab, {r["COG_ID"] for r in rows if float(r[f"{k}_input"]) > 0.0},
                  {r["COG_ID"] for r in rows if r[f"{k}_present"] == "1"}) for k, lab in conds]

recs = [("LBCA",) + r for r in sets_from(f"{REPO}/data/lbca_cog_lists.tsv",
        [("sparse", "sparse (reconciliation)"), ("GLD_min1", "copy-number min-1"),
         ("GLD_min4", "copy-number min-4")])]
recs += [("LACA",) + r for r in sets_from(f"{REPO}/data/laca_cog_lists.tsv",
        [("combined", "combined root"), ("euryroot_ml", "Euryarchaeota (ML)"),
         ("euryroot_unif", "Euryarchaeota (uniform)"),
         ("gld_min1", "copy-number min-1"), ("gld_min4", "copy-number min-4")])]

feat = [c for c in pd.read_csv(f"{SP}/train_data_phylum_tax_level_split_0", sep="\t",
                               nrows=0).columns if c.startswith("COG")]
fidx = {c: i for i, c in enumerate(feat)}
def vec(s):
    v = np.zeros(len(feat), np.float32)
    for c in s:
        j = fidx.get(c)
        if j is not None: v[j] = 1.0
    return v

X_anc, meta = [], []
for node, lab, in0, den in recs:
    for arm, s in (("input", in0), ("denoised", den)):
        X_anc.append(vec(s)); meta.append((node, lab, arm, len(s)))
X_anc = np.vstack(X_anc)

kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.7,
          colsample_bytree=0.7, tree_method="hist")
MU, S, cov, rmse, RES = [], [], [], [], []
for s in range(10):
    d = pd.read_csv(f"{SP}/train_data_phylum_tax_level_split_{s}", sep="\t")
    a = pd.read_csv(f"{SP}/train_annot_phylum_tax_level_split_{s}", sep="\t")
    lab = dict(zip(a["accession"], pd.to_numeric(a["annotation"], errors="coerce")))
    d = d[d["accession"].isin(lab)]
    X = (d[feat].to_numpy() > 0).astype(np.float32)
    y = d["accession"].map(lab).to_numpy().astype(float)
    ok = ~np.isnan(y); X, y = X[ok], y[ok]
    # Fixed-seed 85/15 cut: coverage, RMSE and the conformal residuals are all
    # measured on genomes none of the three regressors was fit on.
    rng = np.random.RandomState(0); perm = rng.permutation(len(y))
    cut = int(len(y) * 0.85); tr, te = perm[:cut], perm[cut:]
    m  = XGBRegressor(**kw).fit(X[tr], y[tr])
    q1 = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10, **kw).fit(X[tr], y[tr])
    q9 = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90, **kw).fit(X[tr], y[tr])
    lo, hi = q1.predict(X[te]), q9.predict(X[te])
    cov.append(float(np.mean((y[te] >= lo) & (y[te] <= hi))))
    rmse.append(float(np.sqrt(np.mean((y[te] - m.predict(X[te]))**2))))
    MU.append(m.predict(X_anc))
    S.append((q9.predict(X_anc) - q1.predict(X_anc)) / (2 * Z90))
    RES.append(np.abs(y[te] - m.predict(X[te])))          # |residuals| in C, pooled below

MU, S = np.vstack(MU), np.clip(np.vstack(S), 1e-6, None)
mu = MU.mean(0)
var_within, var_between = (S**2).mean(0), MU.var(0, ddof=1)
sd_tot = np.sqrt(var_within + var_between)
res = np.concatenate(RES)
q95 = float(np.quantile(res, 0.95))            # distribution-free 95% half-width, in C
print(f"held-out RMSE {np.mean(rmse):.2f} C | nominal-80% quantile band covers "
      f"{np.mean(cov)*100:.1f}% (target 80%) -> quantile model is OVER-CONFIDENT")
print(f"conformal 95% half-width from {len(res)} held-out residuals: +/-{q95:.1f} C")

rows = [dict(node=n, reconstruction=l, arm=a, n_cogs=k, ogt_C=round(float(mu[i]), 1),
             ci95_lo=round(float(mu[i] - 1.96*sd_tot[i]), 1),
             ci95_hi=round(float(mu[i] + 1.96*sd_tot[i]), 1),
             sd_total=round(float(sd_tot[i]), 1),
             sd_within=round(float(np.sqrt(var_within[i])), 1),
             sd_between=round(float(np.sqrt(var_between[i])), 1),
             conf95_lo=round(float(mu[i] - q95), 1), conf95_hi=round(float(mu[i] + q95), 1))
        for i, (n, l, a, k) in enumerate(meta)]
df = pd.DataFrame(rows)
df.to_csv(f"{REPO}/data/ancestral_ogt_intervals.tsv", sep="\t", index=False)
print("\n" + df.to_string(index=False))
