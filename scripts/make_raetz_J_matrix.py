#!/usr/bin/env python3
"""The learned coupling sub-block J_ij for the KEGG M00060 (lipid A / Kdo2-lipid
A) genes on its own, gene-labelled, in the green(+)/red(-)-on-black style of
Figure 1.

The J values, labels and colormap are those of scripts/make_raetz_lightup.py,
embedded here so the panel needs no checkpoint to redraw.  Rows and columns run
in pathway order, followed by the false positive FmhB (COG2348), which the model
anti-couples to all nine enzymes because it is mutually exclusive with the
diderm lipid-A envelope:
  LpxA=COG1043 LpxC=COG0774 LpxD=COG1044 LpxH=COG2908 LpxB=COG0763 LpxK=COG1663
  WaaA=COG1519 LpxL=COG1560   LpxI=COG3494 (non-homologous ALTERNATIVE to LpxH;
  the two anti-correlate, a genome carries LpxH OR LpxI, not both).

Output: analysis/figures/fig_raetz_J_matrix.{pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent.parent / "analysis" / "figures"

labels = ['LpxA', 'LpxC', 'LpxD', 'LpxH', 'LpxB', 'LpxK', 'WaaA', 'LpxL', 'LpxI', 'FmhB']
J = np.array([
    [0.0000, 0.1512, 0.2160, 0.0375, 0.2064, 0.0759, 0.0726, 0.0716, 0.2015, -0.0243],
    [0.1512, 0.0000, 0.1058, -0.0017, 0.1316, 0.0870, 0.0719, 0.0532, 0.0878, -0.0285],
    [0.2160, 0.1058, 0.0000, 0.0201, 0.1578, 0.0494, 0.0605, 0.0351, 0.1449, -0.0114],
    [0.0375, -0.0017, 0.0201, 0.0000, 0.0294, 0.0381, 0.0234, 0.0309, -0.1635, -0.0073],
    [0.2064, 0.1316, 0.1578, 0.0294, 0.0000, 0.1018, 0.0855, 0.0722, 0.1760, -0.0071],
    [0.0759, 0.0870, 0.0494, 0.0381, 0.1018, 0.0000, 0.1650, 0.1368, 0.0769, -0.0187],
    [0.0726, 0.0719, 0.0605, 0.0234, 0.0855, 0.1650, 0.0000, 0.1353, 0.0810, -0.0177],
    [0.0716, 0.0532, 0.0351, 0.0309, 0.0722, 0.1368, 0.1353, 0.0000, 0.0705, -0.0324],
    [0.2015, 0.0878, 0.1449, -0.1635, 0.1760, 0.0769, 0.0810, 0.0705, 0.0000, -0.0230],
    [-0.0243, -0.0285, -0.0114, -0.0073, -0.0071, -0.0187, -0.0177, -0.0324, -0.0230, 0.0000],
])
N = len(labels)
FP = 9  # index of the false positive FmhB

# Colormap and scale as in Figure 1: linear and symmetric about zero, clipped
# at max|J| so the sign of a coupling reads off the hue.
DIV = LinearSegmentedColormap.from_list(
    "divglow", [(0.0, "#ff9166"), (0.28, "#a83226"), (0.5, "#0d0d0d"),
                (0.72, "#2f8f4e"), (1.0, "#7fe0a0")])
CLIP = float(np.abs(J).max())
RED = "#cc2b1d"

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm"})

fig, ax = plt.subplots(figsize=(4.1, 3.7))
im = ax.imshow(J, cmap=DIV, vmin=-CLIP, vmax=CLIP, aspect="equal", interpolation="nearest")

ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for tick in (ax.get_xticklabels()[FP], ax.get_yticklabels()[FP]):
    tick.set_color(RED); tick.set_fontstyle("italic")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

# Separator between the nine pathway enzymes and the false-positive row/column.
ax.axhline(FP - 0.5, color="0.5", lw=0.7, ls=(0, (3, 2)))
ax.axvline(FP - 0.5, color="0.5", lw=0.7, ls=(0, (3, 2)))

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-CLIP, 0, CLIP])
cb.ax.set_yticklabels([f"$-{CLIP:.2f}$", "0", f"$+{CLIP:.2f}$"], fontsize=7)
cb.set_label("$J_{ij}$  (co-occur $>0$ / exclude $<0$)", fontsize=8)
cb.outline.set_visible(False)

ax.set_title("Raetz module (KEGG M00060)\nlearned couplings $J_{ij}$",
             fontsize=9, pad=6)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_raetz_J_matrix.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_raetz_J_matrix.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT/'fig_raetz_J_matrix.pdf'}  (CLIP={CLIP:.3f}, {N} genes, "
      f"LpxH-LpxI J={J[3,8]:+.3f})")
