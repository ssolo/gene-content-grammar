#!/usr/bin/env python3.11
"""Figure: how the denoised present-count responds when the least-confident
input families are removed before relaxation, the kept families binarised to +1.

The cut differs by node because the two reconciliations have different
resolution:
  * LBCA uses VALUE gates (>0, >0.01, >0.02).  Half of its 493 input families
    sit at exactly p = 0.01 (the two-decimal reconciliation floor), so a rank
    quantile is degenerate: the lowest 10% and 25% both fall inside the 0.01
    tie.  A value gate is well defined, but clearing the 0.01 / 0.02 levels
    removes 0 / ~50 / ~64% of the input.
  * LACA (combined) and the two candidate roots use the lowest-presence
    QUANTILE (0 / 10 / 25% removed); their archaeal reconciliation is graded
    below 0.01, so the quantile is well defined.

Both are plotted against the common x-axis "input families removed (%)".
Reads the default-model fpcontrol reconstructions in --input-dir:
  LBCA_default_baseline_pred.tsv         (cols mean_>0, mean_>0.01, mean_>0.02)
  {LACA,LACA-MHH,LACA-Eury}_default_{baseline,drop10,drop25}_pred.tsv (col mean_>0)
"""
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FP = "data/interactome/fpcontrol"
# LBCA value gates: (threshold, denoised-column, label)
LBCA_GATES = [(0.0, "mean_>0", ">0"), (0.01, "mean_>0.01", ">0.01"), (0.02, "mean_>0.02", ">0.02")]
# LACA quantile conditions: (file-stem suffix, % removed)
LACA_CONDS = [("baseline", 0), ("drop10", 10), ("drop25", 25)]
LACA_NODES = [("LACA", "LACA combined", "#1f77b4"),
              ("LACA-MHH", "LACA MHH-root", "#2ca02c"),
              ("LACA-Eury", "LACA Eury-root", "#ff7f0e")]
LBCA_COL = "#d62728"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot threshold-removal controls from fpcontrol reconstructions."
    )
    p.add_argument("--input-dir", default=FP,
                   help="Directory containing the *_default_*_pred.tsv files.")
    p.add_argument("--out", default="analysis/figures/fig_threshold_removal.pdf",
                   help="Output PDF path.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.2))

    # LBCA: value gates
    dfL = pd.read_csv(os.path.join(args.input_dir, "LBCA_default_baseline_pred.tsv"), sep="\t")
    n_in = int((dfL.input_prob > 0).sum())
    xs, den, rem = [], [], []
    for g, col, _lab in LBCA_GATES:
        kept = int((dfL.input_prob > g).sum())
        xs.append(100.0 * (n_in - kept) / n_in)
        den.append(int((dfL[col] > 0.5).sum()))
        rem.append(n_in - kept)
    axA.plot(xs, den, "o-", color=LBCA_COL, label="LBCA (bac-FT), value gate", lw=1.8, ms=5)
    axB.plot(xs, rem, "o-", color=LBCA_COL, label="LBCA (bac-FT), value gate", lw=1.8, ms=5)
    for x, y, (_g, _c, lab) in zip(xs, den, LBCA_GATES):
        axA.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 5),
                     fontsize=7, color=LBCA_COL)

    # LACA and the two candidate roots: lowest-presence quantile
    for stem, lab, col in LACA_NODES:
        xs, den, rem = [], [], []
        n_in = None
        for cond, pct in LACA_CONDS:
            p = os.path.join(args.input_dir, f"{stem}_default_{cond}_pred.tsv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p, sep="\t")
            kept = int((d.input_prob > 0).sum())
            if n_in is None:
                n_in = kept
            xs.append(pct)
            den.append(int((d["mean_>0"] > 0.5).sum()))
            rem.append(n_in - kept)
        axA.plot(xs, den, "o-", color=col, label=lab, lw=1.8, ms=5)
        axB.plot(xs, rem, "o-", color=col, label=lab, lw=1.8, ms=5)

    for ax in (axA, axB):
        ax.set_xlabel("input families removed (%)")
    axA.set_ylabel("families called present (posterior > 0.5)")
    axA.set_title("A  denoised present vs input removed", fontsize=9, loc="left")
    axB.set_ylabel("input families removed (count)")
    axB.set_title("B  input removed", fontsize=9, loc="left")
    axA.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
