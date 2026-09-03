#!/usr/bin/env python3
"""Two-panel Jaccard heatmap: the five LACA reconstructions converge on denoising.

Left  = pairwise overlap of the five INPUT present sets (reconciliation / recount
        calls at input_prob > 0.5), mean off-diagonal Jaccard ~0.23.
Right = pairwise overlap of the five DENOISED reconstructions (posterior mean
        > 0.5), mean off-diagonal Jaccard ~0.71.

Each cell is the Jaccard index of two present-gene sets.  Both panels share one
0-1 colour scale, so the pale left panel against the warm right panel is itself
the convergence result.

Inputs: the five LACA pred TSVs (columns COG_ID, input_prob, mean_actual).
Output: analysis/figures/fig_laca_jaccard.{pdf,png}.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "analysis", "figures")
# (label, pred file), in figure row and column order.
RECON = [
    ("combined root",           "LACA_merged_pred.tsv"),
    ("Euryarchaeota (ML)",      "LACA_euryroot_pred.tsv"),
    ("Euryarchaeota (uniform)", "LACA_euryroot_uniform_pred.tsv"),
    ("copy-number (min-1)",     "LACA_gld_min1_pred.tsv"),
    ("copy-number (min-4)",     "LACA_gld_min4_pred.tsv"),
]


def present_sets(path):
    """Return (input>0.5, output>0.5) present-COG sets for one reconstruction."""
    inp, out = set(), set()
    with open(path) as fh:
        ci = {k: i for i, k in enumerate(fh.readline().rstrip("\n").split("\t"))}
        for line in fh:
            x = line.rstrip("\n").split("\t")
            if float(x[ci["input_prob"]]) > 0.5:
                inp.add(x[ci["COG_ID"]])
            if float(x[ci["mean_actual"]]) > 0.5:
                out.add(x[ci["COG_ID"]])
    return inp, out


def jaccard_matrix(sets):
    n = len(sets)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                u = len(sets[i] | sets[j])
                M[i, j] = len(sets[i] & sets[j]) / u if u else 0.0
    return M


def offdiag_mean(M):
    n = len(M)
    return float(np.mean([M[i, j] for i in range(n) for j in range(n) if i < j]))


def main():
    labs = [k for k, _ in RECON]
    inp_sets, out_sets = [], []
    for _, f in RECON:
        a, b = present_sets(os.path.join(REPO, f))
        inp_sets.append(a)
        out_sets.append(b)
    Ji, Jo = jaccard_matrix(inp_sets), jaccard_matrix(out_sets)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for k, (ax, M, ttl) in enumerate([(axes[0], Ji, "Before denoising (inputs)"),
                                      (axes[1], Jo, "After denoising (reconstructions)")]):
        im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=10)
        # Both panels keep the RECON row order; y labels on the left panel only.
        ax.set_yticklabels(labs if k == 0 else [""] * 5, fontsize=10)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] > 0.7 else "0.15", fontsize=10)
        ax.set_title(f"{ttl}\nmean off-diagonal Jaccard = {offdiag_mean(M):.2f}",
                     fontsize=11.5)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="Jaccard overlap of present sets (> 0.5)")
    fig.suptitle("Five LACA reconstructions converge on denoising",
                 fontsize=13, y=0.99)

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_laca_jaccard.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"input mean Jaccard {offdiag_mean(Ji):.3f} -> output {offdiag_mean(Jo):.3f}")
    print(f"wrote {FIGDIR}/fig_laca_jaccard.(pdf,png)")


if __name__ == "__main__":
    main()
