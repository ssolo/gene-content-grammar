#!/usr/bin/env python3
"""Genome-as-image denoising mosaic for the E. coli recovery benchmark.

Each gene family of the true E. coli genome is one tile, coloured by its COG
functional-category group and laid out in a fixed grid ordered by category, so
the four groups form contiguous bands. Three panels share that canvas:

    input   the corrupted genome: surviving genes coloured, the rest blanked
    recon   the bacterial specialist's present calls, with a red strip below the
            canvas for false positives (genes the truth does not contain)
    truth   every gene present: the clean target

Reads per-COG truth/input/denoised calls from recover_ecoli_bachard.tsv and
category letters from data/cog-20.def.tab.
Output: analysis/figures/fig_ecoli_mosaic.{pdf,png}.

Usage: python scripts/plot_genome_denoise_mosaic.py [--fn 0.9]
"""
import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "analysis", "figures")

# COG functional-category groups. The letter order also fixes the tile order, so
# the three panels share one canvas and are comparable tile for tile.
GROUPS = [
    ("Information storage & processing", set("JAKLB"),   "#0072B2"),
    ("Cellular processes & signalling",  set("DYVTMNZWUOX"), "#E69F00"),
    ("Metabolism",                       set("CGEFHIPQ"), "#009E73"),
    ("Poorly characterised",             set("RS"),       "#9e9e9e"),
]
CAT_ORDER = "JAKLB" + "DYVTMNZWUOX" + "CGEFHIPQ" + "RS"
CAT_RANK = {c: i for i, c in enumerate(CAT_ORDER)}
MISS = "#e9e9e9"   # blanked / missing tile
WHITE = "#ffffff"


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def load_catmap(path):
    cat = {}
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0].startswith("COG"):
                cat[f[0]] = f[1][0] if f[1] else "S"
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", type=float, default=0.9)
    A = ap.parse_args()
    cat = load_catmap(os.path.join(REPO, "data", "cog-20.def.tab"))

    genes = []
    with open(os.path.join(REPO, "recover_ecoli_bachard.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if abs(float(r["fn"]) - A.fn) > 1e-9 or int(r["truth"]) != 1:
                continue
            c = cat.get(r["COG_ID"], "S")
            genes.append((CAT_RANK.get(c, len(CAT_ORDER)), r["COG_ID"], c,
                          int(r["input_present"]) == 1,
                          float(r["denoised_prob"]) > 0.5))
    genes.sort(key=lambda g: (g[0], g[1]))
    N = len(genes)

    grpcol = {}
    for _, letters, col in GROUPS:
        for L in letters:
            grpcol[L] = col

    # False positives are genes the reconstruction adds that the truth lacks, so
    # they have no tile on the true-genome canvas and get their own strip below it.
    FP = "#b2182b"
    fps = []
    with open(os.path.join(REPO, "recover_ecoli_bachard.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if abs(float(r["fn"]) - A.fn) > 1e-9:
                continue
            if int(r["truth"]) == 0 and float(r["denoised_prob"]) > 0.5:
                fps.append(r["COG_ID"])

    # ncol = sqrt(2N) gives a roughly 2:1 landscape grid.
    ncol = int(round(math.sqrt(N * 2)))
    nrow = math.ceil(N / ncol)
    fp_rows = max(1, math.ceil(len(fps) / ncol))

    def main_canvas(state):
        img = np.ones((nrow, ncol, 3))            # (nrow, ncol, 3) RGB, white ground
        for k, (_, _, c, inp, post) in enumerate(genes):
            r, cc = divmod(k, ncol)
            base = hex2rgb(grpcol.get(c, "#9e9e9e"))
            on = {"truth": True, "input": inp, "recon": post}[state]
            img[r, cc] = base if on else hex2rgb(MISS)
        return img

    def recon_with_fp():
        top = main_canvas("recon")
        sep = np.ones((1, ncol, 3))
        strip = np.ones((fp_rows, ncol, 3))
        for k in range(len(fps)):
            r, cc = divmod(k, ncol)
            strip[r, cc] = hex2rgb(FP)
        return np.vstack([top, sep, strip])

    n_in = sum(g[3] for g in genes)
    n_rec = sum(g[4] for g in genes)
    panels = [
        ("input", main_canvas("input"),
         "Corrupted input  (false-negative rate %g)" % A.fn,
         "%d of %d genes survive  (%.0f%%)" % (n_in, N, 100 * n_in / N)),
        ("recon", recon_with_fp(),
         "Denoised reconstruction  (bacterial specialist)",
         "%d of %d genes recovered  (%.0f%%);  red strip = %d false positives"
         % (n_rec, N, 100 * n_rec / N, len(fps))),
        ("truth", main_canvas("truth"),
         "True $E.\\ coli$ genome  (target)", "all %d genes" % N),
    ]

    heights = [nrow, nrow + 1 + fp_rows, nrow]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, (sum(heights) / ncol) * 8.6 + 1.6),
                             gridspec_kw={"height_ratios": heights})
    for ax, (_, img, title, sub) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb")
        ax.set_title("%s\n%s" % (title, sub), fontsize=12.5, linespacing=1.3, pad=5)

    handles = [Patch(color=col, label=name) for name, _, col in GROUPS]
    handles += [Patch(facecolor=MISS, edgecolor="#cccccc", label="absent / not recovered"),
                Patch(color=FP, label="false positive (wrongly added)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("Denoising the $E.\\ coli$ genome: each tile is a gene family, "
                 "coloured by function", fontsize=13.5, y=0.999)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06, hspace=0.16)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "fig_ecoli_mosaic.%s" % ext),
                    dpi=200, bbox_inches="tight")
    print("wrote %s/fig_ecoli_mosaic.(pdf,png)  grid %dx%d, N=%d, input %d, recon %d, FP %d"
          % (FIGDIR, nrow, ncol, N, n_in, n_rec, len(fps)))


if __name__ == "__main__":
    main()
