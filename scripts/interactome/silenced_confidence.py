#!/usr/bin/env python3.11
"""Input confidence of the families the denoiser SILENCES at LBCA/LACA.

Reads the per-COG reconstruction tables (input_prob = reconciliation copy-number
posterior; mean_actual = denoised posterior) and compares the input-confidence
distribution of silenced (input present, denoised absent) against kept (present
in both) families.

Confidence bands follow the reconstruction's own input thresholds: low < 0.04,
mid [0.04, 0.5), high >= 0.5.

Usage:
  python scripts/interactome/silenced_confidence.py \
      --lbca node2012_T20bachard_pred.tsv --laca LACA_combined_T20mix_pred.tsv
"""
import argparse
import os

import numpy as np
import pandas as pd


def analyze(tag, path):
    if not os.path.exists(path):
        print("%s: MISSING %s" % (tag, path)); return
    d = pd.read_csv(path, sep="\t")
    present = d["input_prob"] > 0
    sil = d[present & (d["mean_actual"] <= 0.5)]["input_prob"].values
    kep = d[present & (d["mean_actual"] > 0.5)]["input_prob"].values

    def bands(v):
        return (100 * float((v < 0.04).mean()),
                100 * float(((v >= 0.04) & (v < 0.5)).mean()),
                100 * float((v >= 0.5).mean()))

    for name, v in (("silenced", sil), ("kept", kep)):
        lo, mid, hi = bands(v)
        print("%s %-9s n=%4d  median=%.3f mean=%.3f | low<0.04 %2.0f%%  mid %2.0f%%  high>=0.5 %2.0f%%"
              % (tag, name, len(v), np.median(v), v.mean(), lo, mid, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbca", default="node2012_T20bachard_pred.tsv")
    ap.add_argument("--laca", default="LACA_combined_T20mix_pred.tsv")
    a = ap.parse_args()
    analyze("LBCA(bac-FT)", a.lbca)
    analyze("LACA(mix-FT)", a.laca)


if __name__ == "__main__":
    main()
