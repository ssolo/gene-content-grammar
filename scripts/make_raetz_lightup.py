#!/usr/bin/env python3
"""Real-data counterpart of the toy schematic in scripts/make_toy_J_lightup.py.

Same three-row layout as the toy (barcode, presence-gated J, pathway map), but the
couplings are the learned J of the trained model sliced for KDO2-lipid A biosynthesis
(the Raetz pathway, KEGG module M00060).

Genes are drawn as EDGES of the linear pathway in biochemical order
(UDP-GlcNAc -> Kdo2-lipid A): LpxA -> LpxC -> LpxD -> LpxH -> LpxB -> LpxK ->
WaaA -> LpxL, with LpxI, the non-homologous alternative to LpxH for the same step,
as a parallel edge.  FmhB, a monoderm peptidoglycan interpeptide-bridge
glycyltransferase, is the named false positive: J < 0 to all nine enzymes.
Mean-field reconstruction from two observed enzymes recovers LpxI and leaves LpxH
as the gap, picking one of two mutually exclusive alternatives (J = -0.16).

The coupling sub-block is embedded below, so this script needs no external data.
Writes analysis/figures/fig_raetz_J_lightup[_altJ].{png,pdf}.

  python3 scripts/make_raetz_lightup.py            # presence-gated J on black
  python3 scripts/make_raetz_lightup.py --alt-j    # raw J faint on off-white
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def lerp(c0, c1, f):
    c0 = np.array(matplotlib.colors.to_rgb(c0)); c1 = np.array(matplotlib.colors.to_rgb(c1))
    return tuple((1 - f) * c0 + f * c1)


# ---- coupling sub-block: 9 Raetz enzymes in pathway order + 1 named false positive
# Learned couplings J_ij for the KEGG M00060 COGs, sliced in pathway order from
# data/interactome/J_plain_pairwise_T20.npy (written by scripts/interactome/extract_J.py).
#   LpxA=COG1043 LpxC=COG0774 LpxD=COG1044 LpxH=COG2908 LpxB=COG0763 LpxK=COG1663
#   WaaA=COG1519 LpxL=COG1560 LpxI=COG3494 (DUF1009)
#   FmhB=COG2348 (Lipid II:glycine glycyltransferase)
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
N, FP, NT = 10, 9, 9                            # index 9 = spurious gene; 9 true enzymes

# ---- reconstruction dynamics on the real sub-block
hT, beta, seeds = -0.46, 2.90, [0, 7]           # LpxA and LpxL clamped present
h = np.full(N, hT)
p = np.zeros(N); p[seeds] = 1.0
P = [p.copy()]
for _ in range(8):
    p = sig(beta * (h + J @ p)); p[seeds] = 1.0; P.append(p.copy())
P = np.array(P); P[:, FP] = 0.0
pfp = np.array([1.0, 0.5, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
KCOLS = 7

# ---- styling, matching the toy schematic
DIV = LinearSegmentedColormap.from_list(
    "divglow", [(0.0, "#ff9166"), (0.28, "#a83226"), (0.5, "#0d0d0d"),
                (0.72, "#2f8f4e"), (1.0, "#7fe0a0")])
CLIP = float(np.abs(J).max())                   # linear scale, so the LpxH/LpxI negative shows
GREEN = "#1a9850"; REC_LO = "#c6dbef"; REC_HI = "#08306b"; GREY = "0.72"; RED = "#cc2b1d"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm"})

# --alt-j: raw J faint on off-white (red = J<0, green = J>0), each cell brightening
# to full saturation as both its genes come present.
ALT_J = "--alt-j" in sys.argv
OFFW = np.array([0.965, 0.955, 0.935])
POS_LIT = np.array(matplotlib.colors.to_rgb("#12d95a"))   # fully lit cell, J > 0
NEG_LIT = np.array(matplotlib.colors.to_rgb("#ff2e21"))   # fully lit cell, J < 0
POS_HOT = np.array(matplotlib.colors.to_rgb("#e8ffef"))   # pale core of a lit J > 0 cell
NEG_HOT = np.array(matplotlib.colors.to_rgb("#ffe6df"))   # pale core of a lit J < 0 cell
DIV_OW = LinearSegmentedColormap.from_list("ow_div", [tuple(NEG_LIT), tuple(OFFW), tuple(POS_LIT)])


def recon_shade(pv):                             # sqrt: moderate confidence still reads blue
    return lerp(REC_LO, REC_HI, np.sqrt(np.clip((pv - 0.5) / 0.5, 0, 1)))


_K = 26                                          # sub-pixels per matrix cell (radial glow)
_u = np.linspace(-1.0, 1.0, _K)
_R = np.sqrt(_u[:, None] ** 2 + _u[None, :] ** 2)
_FALL = np.exp(-(_R / 0.62) ** 2)


def altJ_rgb(pvec, faint=0.05):
    # Steep both-present gate rather than the p_i p_j of the gated branch, which
    # leaves the matrix dim while the reconstruction is still filling.
    w = 1.0 / (1.0 + np.exp(-13.0 * (pvec - 0.45)))       # per-gene presence weight ~0/1
    gate = np.outer(w, w)
    s = np.clip(np.abs(J) / CLIP, 0.0, 1.0)
    amt = np.clip(s * gate * 1.35, 0.0, 1.0)             # lit-ness, overdriven then clipped
    pos = J >= 0
    lit = np.where(pos[..., None], POS_LIT, NEG_LIT)
    hot = np.where(pos[..., None], POS_HOT, NEG_HOT)
    bgw = (s * faint)[..., None]                          # faint raw-J tint between blobs
    bg = (1.0 - bgw) * OFFW + bgw * lit
    img = np.empty((N * _K, N * _K, 3))
    for i in range(N):
        for j in range(N):
            g = (amt[i, j] * _FALL)[..., None]
            cell = (1.0 - g) * bg[i, j] + g * lit[i, j]
            b = (np.clip((amt[i, j] - 0.3) / 0.7, 0, 1) * _FALL ** 1.5)[..., None]  # hot core
            cell = (1.0 - 0.7 * b) * cell + 0.7 * b * hot[i, j]
            img[i * _K:(i + 1) * _K, j * _K:(j + 1) * _K] = cell
    return np.clip(img, 0, 1)


def edge(ax, pa, pb, color, lw, alpha, zorder, rad=0.0):   # straight, or bowed for alternatives
    if rad == 0.0:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round")
    else:
        ax.add_patch(FancyArrowPatch(pa, pb, connectionstyle=f"arc3,rad={rad}",
                     arrowstyle="-", color=color, lw=lw, alpha=alpha, zorder=zorder,
                     capstyle="round"))


# metabolite nodes, laid out serpentine
m = {0: (0.0, 1.15), 1: (1.0, 1.23), 2: (2.0, 1.15), 3: (3.0, 1.23), 4: (4.0, 1.15),
     5: (4.0, -0.15), 6: (3.0, -0.23), 7: (2.0, -0.15), 8: (1.0, -0.23)}
mFP = (4.75, -1.05)                             # lone metabolite of the spurious reaction
# gene index -> (metabolite a, metabolite b).  0..7 are the serial Raetz steps; gene 8
# (LpxI) shares step 3->4 with gene 3 (LpxH); gene 9 is the spurious dangling edge.
E = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (3, 4), (5, "FP")]
RAD = {3: -0.72, 8: 0.72}                        # LpxH / LpxI bowed opposite ways
xlim = (-0.85, 5.15); ylim = (-1.6, 1.95)

fig = plt.figure(figsize=(18.2, 6.1))
gs = fig.add_gridspec(3, KCOLS + 2, width_ratios=[1.6, 0.5] + [1.0] * KCOLS,
                      height_ratios=[0.7, 5.0, 7.3], hspace=0.12, wspace=0.10)

# ---- reference panel (col 0): raw learned J, gene-labelled.  The T-sequence to its
#      right is the same J gated by presence, J_ij x_i x_j.
axR = fig.add_subplot(gs[1, 0])
axR.imshow(J, cmap=DIV, vmin=-CLIP, vmax=CLIP, aspect="equal", interpolation="nearest")
axR.set_xticks(range(N)); axR.set_yticks(range(N))
axR.set_xticklabels(labels, rotation=90, fontsize=5.6)
axR.set_yticklabels(labels, fontsize=5.6)
for _tk in (axR.get_xticklabels()[FP], axR.get_yticklabels()[FP]):
    _tk.set_color(RED); _tk.set_fontstyle("italic")
axR.tick_params(length=0)
for _s in axR.spines.values():
    _s.set_linewidth(0.5); _s.set_color("0.5")
axR.axhline(FP - 0.5, color="0.55", lw=0.6, ls=(0, (2, 2)))
axR.axvline(FP - 0.5, color="0.55", lw=0.6, ls=(0, (2, 2)))
axR.set_title("learned couplings $J_{ij}$", fontsize=10, pad=5)
# gate cue in the spacer column
axS = fig.add_subplot(gs[1, 1]); axS.axis("off")
axS.text(0.5, 0.60, r"$\times\,x_ix_j$", ha="center", va="center", fontsize=9, color="#444")
axS.annotate("", xy=(0.95, 0.36), xytext=(0.05, 0.36), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))

lastJ = None
for t in range(KCOLS):
    # row 0: barcode
    ax0 = fig.add_subplot(gs[0, t + 2]); strip = np.ones((1, N, 4))
    for g in range(N):
        if g in seeds:
            strip[0, g, :3] = matplotlib.colors.to_rgb(GREEN)
        elif g == FP:
            strip[0, g, :3] = lerp("white", RED, pfp[t])
        else:
            strip[0, g, :3] = lerp("white", REC_HI, min(1.0, P[t, g] * 1.3))
    ax0.imshow(strip, aspect="auto"); ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(f"$T={t}$", fontsize=12, fontweight="bold", pad=3)
    if t == 0:
        ax0.set_ylabel("genes", rotation=0, ha="right", va="center", fontsize=9)
    # row 1: coupling sub-block, presence-gated on black (or --alt-j on off-white)
    ax1 = fig.add_subplot(gs[1, t + 2])
    if ALT_J:
        ax1.imshow(altJ_rgb(P[t]), aspect="equal", interpolation="bilinear")
    else:
        A = J * np.outer(P[t], P[t])
        ax1.imshow(A, cmap=DIV, vmin=-CLIP, vmax=CLIP, aspect="equal", interpolation="nearest")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel(f"{int((P[t, :NT] > 0.5).sum())}/9", fontsize=9.5)
    lastJ = ax1
    # row 2: metabolic map; enzymes are edges that light up, the spurious edge fades out
    ax2 = fig.add_subplot(gs[2, t + 2]); ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_xlim(*xlim); ax2.set_ylim(*ylim)
    for gi, (a, b) in enumerate(E):
        if gi == FP:
            continue
        pa = m[a]; pb = m[b]; v = P[t, gi]; rad = RAD.get(gi, 0.0)
        if gi in seeds:
            edge(ax2, pa, pb, GREEN, 5.0, 1.0, 4, rad)
        elif v < 0.5:
            edge(ax2, pa, pb, GREY, 1.3, 0.9, 2, rad)
        else:
            edge(ax2, pa, pb, recon_shade(v), 5.0, 1.0, 3, rad)
        if t == 0:                                # enzyme labels once, clear of the edges
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            if gi == 8:                            # above the LpxI up-arc
                ax2.text(3.5, 1.72, "LpxI", ha="center", va="bottom", fontsize=7,
                         color="#444", zorder=6, rotation=20)
            elif gi == 3:                          # below the LpxH down-arc
                ax2.text(3.4, 0.58, "LpxH", ha="center", va="top", fontsize=7,
                         color="#444", zorder=6, rotation=20)
            elif abs(pa[1] - pb[1]) > 0.5:         # vertical edge -> to the right
                ax2.text(mx + 0.24, my, labels[gi], ha="left", va="center", fontsize=7,
                         color="#444", zorder=6)
            elif my > 0.5:                          # top row -> above
                ax2.text(mx - 0.05, my + 0.32, labels[gi], ha="center", va="bottom",
                         fontsize=7, color="#444", zorder=6, rotation=20)
            else:                                   # bottom row -> below
                ax2.text(mx - 0.05, my - 0.34, labels[gi], ha="center", va="top",
                         fontsize=7, color="#444", zorder=6, rotation=20)
    if pfp[t] > 0.01:                              # spurious enzyme, fades out
        xa, ya = m[5]; a = pfp[t]
        ax2.plot([xa, mFP[0]], [ya, mFP[1]], color=RED, lw=1.4 + 3.4 * a,
                 alpha=min(1, 0.25 + 0.9 * a), zorder=5, solid_capstyle="round")
        ax2.scatter(*mFP, s=40, facecolors="white", edgecolors=RED, lw=1.4, zorder=6,
                    alpha=min(1, 0.3 + a))
        if t == 0:
            ax2.text(mFP[0] + 0.02, mFP[1] - 0.28, "FmhB", ha="center", va="top",
                     fontsize=7, color=RED, rotation=20, zorder=6)
    for k in range(9):
        ax2.scatter(*m[k], s=40, c="white", edgecolors="0.35", lw=1.0, zorder=7)
    if t == 0:
        ax2.text(xlim[0] - 0.12, 0.5, "Raetz\npathway", ha="right", va="center",
                 fontsize=9, color="#555")

cax = lastJ.inset_axes([1.13, 0.05, 0.09, 0.9])
cb = fig.colorbar(ScalarMappable(norm=Normalize(-CLIP, CLIP),
                                 cmap=DIV_OW if ALT_J else DIV), cax=cax)
cb.set_ticks([-CLIP, 0, CLIP])
if ALT_J:
    cb.set_ticklabels(["$J_{ij}{<}0$", "$0$", "$J_{ij}{>}0$"])
    cb.set_label("raw coupling $J_{ij}$  (faint until both genes present)",
                 fontsize=8.5, labelpad=6)
else:
    cb.set_ticklabels(["$J_{ij}x_ix_j{<}0$", "$0$", "$J_{ij}x_ix_j{>}0$"])
    cb.set_label("presence-gated activation", fontsize=8.5, labelpad=6)
cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.5)

leg = [Line2D([0], [0], color=GREEN, lw=3.5, label="present in the input (kept)"),
       Line2D([0], [0], color=REC_HI, lw=3.5, label="recovered (shade = posterior confidence)"),
       Line2D([0], [0], color=GREY, lw=2, label="not recovered (the gap)"),
       Line2D([0], [0], color=RED, lw=3.5, label="false positive, silenced")]
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
SUF = "_altJ" if ALT_J else ""
fig.savefig(f"analysis/figures/fig_raetz_J_lightup{SUF}.png", dpi=175, bbox_inches="tight")
fig.savefig(f"analysis/figures/fig_raetz_J_lightup{SUF}.pdf", bbox_inches="tight")
print("Raetz present/step:", [int((P[t, :NT] > 0.5).sum()) for t in range(KCOLS)])
print("final per-enzyme p:", dict(zip(labels[:9], P[-1, :9].round(2))))
print(f"wrote analysis/figures/fig_raetz_J_lightup{SUF}.png/.pdf")
