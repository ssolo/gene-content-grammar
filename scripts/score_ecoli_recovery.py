#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""score_ecoli_recovery.py -- MCC / recall / precision / #FP from recovery TSVs.

Scores the per-COG E. coli recovery TSVs written by reconstruct_extant.py
(columns: COG_ID, fn, truth, input_present, denoised_prob), calling a family
present at denoised_prob > 0.5. One row per (label, fn), with the 2x2 counts.

Arguments are LABEL=path.tsv and print in the order given, so the nested-holdout
ladder reads phylum -> class -> order -> family:

  python scripts/score_ecoli_recovery.py \
      phylum=recover_ecoli_ecolizoom_phylum_fn468.tsv \
      class=recover_ecoli_ecolizoom_class_fn468.tsv ...
"""
import math
import sys

import pandas as pd

THR = 0.5


def score(df, fn):
    d = df[df["fn"].astype(str) == f"{fn:g}"]
    truth = d["truth"].to_numpy().astype(bool)
    pred = (d["denoised_prob"].to_numpy().astype(float) > THR)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn_ = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    recall = tp / (tp + fn_) if (tp + fn_) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
    mcc = (tp * tn - fp * fn_) / denom if denom else float("nan")
    return dict(truth=tp + fn_, MCC=mcc, recall=recall, prec=prec,
                nFP=fp, TP=tp, FN=fn_, TN=tn)


def main():
    items = []
    for a in sys.argv[1:]:
        if "=" not in a:
            sys.exit(f"args must be LABEL=path.tsv (got {a!r})")
        lab, path = a.split("=", 1)
        items.append((lab, pd.read_csv(path, sep="\t")))
    if not items:
        sys.exit("no TSVs given")
    fns = sorted({float(x) for _, df in items for x in df["fn"].unique()})

    print(f"{'rung':10s} {'fn':>4s} {'truth':>5s} {'MCC':>7s} "
          f"{'recall':>7s} {'prec':>7s} {'#FP':>5s}  (TP/FN/TN)")
    print("-" * 64)
    for lab, df in items:
        for fn in fns:
            r = score(df, fn)
            print(f"{lab:10s} {fn:4.1f} {r['truth']:5d} {r['MCC']:7.4f} "
                  f"{r['recall']:7.3f} {r['prec']:7.3f} {r['nFP']:5d}  "
                  f"({r['TP']}/{r['FN']}/{r['TN']})")
        print()


if __name__ == "__main__":
    main()
