#!/usr/bin/env python3.11
"""Tabulate the lowest-presence input-removal sweep (0%, 10%, 25%) for the
ancestral reconstructions. Each arm drops the lowest 0/10/25% of families by
reconciliation presence probability, then binarizes the rest to +1 (>0 -> +1,
the most-liberal input) before relaxation. Per node:
  kept     = input families presented as +1
  removed  = input families dropped (the lowest-presence fraction)
  denoised = families called present (posterior > 0.5; mean_>0 column).

Reads the default-model fpcontrol reconstructions:
  data/interactome/fpcontrol/<node>_default_{baseline,drop10,drop25}_pred.tsv
"""
import os
import pandas as pd

FP = "data/interactome/fpcontrol"
DROPS = [("baseline", "0%"), ("drop10", "10%"), ("drop25", "25%")]
NODES = [("LBCA", "LBCA"), ("LACA", "LACA combined"),
         ("LACA-MHH", "LACA MHH-root"), ("LACA-Eury", "LACA Eury-root")]


def main():
    print(f"{'node':16} {'drop':>6} {'kept':>6} {'removed':>8} {'denoised(+1)':>13}")
    latex = []
    for stem, lab in NODES:
        b = pd.read_csv(f"{FP}/{stem}_default_baseline_pred.tsv", sep="\t")
        n_in = int((b.input_prob > 0).sum())
        dens = []
        for cond, dl in DROPS:
            p = f"{FP}/{stem}_default_{cond}_pred.tsv"
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p, sep="\t")
            kept = int((d.input_prob > 0).sum())
            den = int((d["mean_>0"] > 0.5).sum())
            dens.append(den)
            print(f"{lab:16} {dl:>6} {kept:>6} {n_in - kept:>8} {den:>13}")
        latex.append(f"    {lab:18}& {n_in} & " + " & ".join(str(x) for x in dens) + r" \\")
    print("\n=== LaTeX (node & input & denoised@[0% 10% 25% removed]) ===")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
