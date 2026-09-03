#!/usr/bin/env python3
"""One composite E. coli recovery figure per false-negative level: the genome as
an image beside the before/after metabolic map.

Left column -- one tile per COG family, coloured by functional-category group.
Within a group the genome's true genes lead and the families it lacks follow, so
a false positive reads as a red-tinted tile in that tail and its functional type
stays legible.  The three panels are the corrupted input, the denoised
reconstruction, and the clean target.

Right column -- the same denoising on the iPath3 global metabolic map, from the
PNGs written by plot_metabolic_ipath.py (--node ecoli --tsv <tsv> --fn <fn>
--tag <tag>).

Output: analysis/figures/fig_ecoli_recovery_<tag>.{pdf,png}.

--tsv fixes the noise instance, so a hand-picked TSV gives a different draw from
the published figure.  The driver pins it to the typical draw:
    python scripts/build_mbe_figures.py --network --only fig_ecoli_recovery_fn08.pdf
"""
import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "analysis", "figures")

GROUPS = [
    ("Information storage & processing", set("JAKLB"),       "#0072B2"),
    ("Cellular processes & signalling",  set("DYVTMNZWUOX"), "#E69F00"),
    ("Metabolism",                       set("CGEFHIPQ"),    "#009E73"),
    ("Poorly characterised",             set("RS"),          "#9e9e9e"),
]
MISS = "#e9e9e9"
FP = "#b2182b"
FP_ALPHA = 0.5   # fraction of red mixed into an FP tile's category colour


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def fp_tint(rgb):
    """Category colour mixed with red, reading as "a false positive of this type"."""
    return (1 - FP_ALPHA) * rgb + FP_ALPHA * hex2rgb(FP)


def group_of(cat_letter):
    for gi, (_, letters, col) in enumerate(GROUPS):
        if cat_letter in letters:
            return gi, hex2rgb(col)
    return len(GROUPS) - 1, hex2rgb(GROUPS[-1][2])


def load_catmap(path):
    cat = {}
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0].startswith("COG"):
                cat[f[0]] = f[1][0] if f[1] else "S"
    return cat


def autocrop_white(img, thresh=0.985):
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = img[:, :, :3].mean(axis=2)
    else:
        lum = img
    mask = lum < thresh
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = max(1, int(0.004 * max(img.shape[0], img.shape[1])))
    r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad + 1)
    c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad + 1)
    return img[r0:r1, c0:c1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--fn", type=float, required=True)
    ap.add_argument("--tag", required=True, help="metabolic PNG suffix, e.g. _fn08")
    ap.add_argument("--model", default="the production denoiser (marginal-HQ, split 5)")
    A = ap.parse_args()
    cat = load_catmap(os.path.join(REPO, "data", "cog-20.def.tab"))

    # Every one of the 4,789 COG families gets a tile, so a grey "absent" tile
    # is meaningful: a family the genome does not carry.
    # item: (group_idx, is_nongene, cog, group_rgb, truth, input_present, recon_present)
    items = []
    N_true = n_in_true = n_rec = n_inj = n_fp_out = 0
    with open(os.path.join(REPO, A.tsv)) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if abs(float(r["fn"]) - A.fn) > 1e-9:
                continue
            gi, rgb = group_of(cat.get(r["COG_ID"], "S"))
            tru = int(r["truth"]) == 1
            inp = int(r["input_present"]) == 1
            rec = float(r["denoised_prob"]) > 0.5
            items.append((gi, 0 if tru else 1, r["COG_ID"], rgb, tru, inp, rec))
            if tru:
                N_true += 1; n_in_true += inp; n_rec += rec
            else:
                n_inj += inp; n_fp_out += rec
    # Group-major, true genes before absent ones, then COG id: the tile layout is
    # identical across the three panels, so a change of colour is a change of call.
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    N_univ = len(items)
    deleted = N_true - n_in_true
    recall = n_rec / N_true if N_true else 0.0
    prec = n_rec / (n_rec + n_fp_out) if (n_rec + n_fp_out) else 0.0

    # 3.4 gives a ~3:1 mosaic, wide enough that the three stacked panels fill the
    # column and sit beside the maps instead of floating with side margins.
    ncol = int(round(math.sqrt(N_univ * 3.4)))
    nrow = math.ceil(N_univ / ncol)

    def canvas(state):
        img = np.ones((nrow, ncol, 3))
        for k, (_, is_ng, _, rgb, tru, inp, rec) in enumerate(items):
            r, cc = divmod(k, ncol)
            present = {"truth": tru, "input": inp, "recon": rec}[state]
            if present:
                img[r, cc] = fp_tint(rgb) if is_ng else rgb
            else:
                img[r, cc] = hex2rgb(MISS)
        return img

    mos = [
        (canvas("input"), "Corrupted input",
         "%d of %d genes survive (%.0f%%); %d spurious present"
         % (n_in_true, N_true, 100 * n_in_true / N_true, n_inj)),
        (canvas("recon"), "Denoised reconstruction",
         "%d recovered (recall %.0f%%, precision %.0f%%); %d false positives (red-tinted)"
         % (n_rec, 100 * recall, 100 * prec, n_fp_out)),
        (canvas("truth"), "True genome (target)",
         "%d of %d COGs present" % (N_true, N_univ)),
    ]

    fig = plt.figure(figsize=(14.0, 10.0))
    subL, subR = fig.subfigures(1, 2, width_ratios=[1.0, 1.0], wspace=0.0)

    axL = subL.subplots(3, 1)
    for lab, ax, (img, title, sub) in zip("ABC", axL, mos):
        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb")
        ax.set_title("(%s)  %s\n%s" % (lab, title, sub), fontsize=11.5, linespacing=1.25, pad=4)
    handles = [Patch(color=col, label=name) for name, _, col in GROUPS]
    handles += [Patch(facecolor=MISS, edgecolor="#cccccc", label="absent (COG not present)"),
                Patch(facecolor=fp_tint(hex2rgb("#777777")), edgecolor="#cccccc",
                      label="false positive (category colour, red-tinted)")]
    subL.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                fontsize=9, bbox_to_anchor=(0.5, -0.012))
    subL.subplots_adjust(left=0.04, right=1.0, top=0.86, bottom=0.08, hspace=0.30)

    # Both maps use the same correctness colours (true positive blue, false
    # positive red) and mark input genes with a black centre-line, so the input
    # scaffold reads identically on the two; RECOVERED genes carry no centre-line
    # and stand out as the fill-in.  Opacity on the reconstruction map is the
    # denoised confidence; the input is solid, being a binary observation.
    axR = subR.subplots(2, 1)
    maps = [
        ("metabolic_ecoli_input%s.png" % A.tag, "Corrupted input on the metabolic map"),
        ("metabolic_ecoli_recon%s.png" % A.tag, "Denoised reconstruction on the metabolic map"),
    ]
    for lab, ax, (png, title) in zip("DE", axR, maps):
        p = os.path.join(FIGDIR, png)
        if os.path.exists(p):
            ax.imshow(autocrop_white(mpimg.imread(p)), interpolation="bilinear")
        else:
            ax.text(0.5, 0.5, "missing %s" % png, ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("(%s)  %s" % (lab, title), fontsize=11.5, pad=4)
    handR = [Patch(color="#2166ac", label="recovered / kept true gene (correct)"),
             Patch(color="#9aa0a6", label="true gene not recovered (missed)"),
             Patch(color="#b2182b", label="false positive (spurious gene)"),
             Line2D([0], [0], color="black", lw=2.4,
                    label="survived the input (thin black centre-line)")]
    subR.legend(handles=handR, loc="lower center", ncol=2, frameon=False,
                fontsize=9, bbox_to_anchor=(0.5, -0.012))
    subR.subplots_adjust(left=0.0, right=0.99, top=0.88, bottom=0.06, hspace=0.16)

    fig.text(0.5, 0.975, r"$E.\ coli$ reconstruction after %d of %d genes deleted, %d spurious genes added"
             % (deleted, N_true, n_inj), ha="center", fontsize=15.5, weight="bold")
    fig.text(0.25, 0.918, "the genome as an image (all %d COGs)" % N_univ, ha="center",
             fontsize=12.5, style="italic", color="#444444")
    fig.text(0.75, 0.918, "the same genome on the metabolic map", ha="center",
             fontsize=12.5, style="italic", color="#444444")

    out = os.path.join(FIGDIR, "fig_ecoli_recovery%s" % A.tag)
    # The manuscript embeds the PDF, and matplotlib rasterises the embedded
    # metabolic-map images at the figure dpi, so the maps are only as crisp as the
    # PDF dpi.  The PNG is a preview and stays at a lower dpi to keep it small.
    for ext, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(out + "." + ext, dpi=dpi, bbox_inches="tight")
    print("wrote %s.(pdf,png)  fn=%g Ntrue=%d/%d deleted=%d added=%d recall=%.1f%% prec=%.1f%% FPout=%d"
          % (out, A.fn, N_true, N_univ, deleted, n_inj, 100 * recall, 100 * prec, n_fp_out))


if __name__ == "__main__":
    main()
