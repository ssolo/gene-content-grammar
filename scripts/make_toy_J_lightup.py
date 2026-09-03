#!/usr/bin/env python3
"""Toy schematic of the mean-field denoiser reconstructing a metabolic module.

Three aligned rows, read left to right over denoiser steps T=0..6:
  (top)    gene-presence barcode: observed genes green, reconstructed genes in
           shades of blue (deeper = more confident), false positive red.
  (middle) the presence-gated activation J_ij x_i x_j, which lights up from
           black as both partner genes appear. J itself is FIXED across T; only
           the state x^t changes. Positive couplings go toward GREEN, negative
           (anti-correlation) couplings toward RED, on a diverging colour map.
  (bottom) metabolic map (iPath-style): metabolites are NODES, genes/reactions
           are EDGES; observed genes solid GREEN, reconstructed genes BLUE
           (shade = confidence), the false-positive reaction red then removed.

The module is a glycolysis-like feeder chain into a TCA-like ring with two
branches (11 genes). Couplings are the line graph of the pathway (adjacent
reactions co-occur, positive), plus four anti-correlation couplings between
well-connected non-adjacent genes. Two genes are observed (clamped seeds); the
rest are missing (false negatives) and get filled in. One spurious gene (false
positive) is uncoupled, so the denoiser removes it first.

Writes analysis/figures/fig_toy_J_lightup.{png,pdf}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def lerp(c0, c1, f):
    c0 = np.array(matplotlib.colors.to_rgb(c0)); c1 = np.array(matplotlib.colors.to_rgb(c1))
    return tuple((1 - f) * c0 + f * c1)


# ---- metabolic map: NODES = metabolites, EDGES = genes (reactions)
C = np.array([3.65, -0.55]); R = 1.05
ring = [C + R * np.array([np.cos(a), np.sin(a)])
        for a in np.deg2rad([150, 90, 30, -30, -90, -150])]
m = {0: (0.15, 1.55), 1: (1.15, 1.05), 2: (1.8, 0.25), 3: tuple(ring[0]), 4: tuple(ring[1]),
     5: tuple(ring[2]), 6: tuple(ring[3]), 7: tuple(ring[4]), 8: tuple(ring[5])}
m[9] = (ring[2][0] + 1.0, ring[2][1] + 0.55)      # branch off m5
m[10] = (ring[4][0] + 0.2, ring[4][1] - 1.05)     # branch off m7
mFP = (1.05, -1.25)                               # lone metabolite of the spurious reaction

E = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 3), (5, 9), (7, 10)]
gw = np.array([0.95, 0.55, 0.90, 0.70, 0.55, 0.85, 0.55, 0.80, 0.65, 0.60, 0.55])
NT, N, FP = 11, 12, 11

J = np.zeros((N, N))                               # positive: line graph (adjacent reactions)
for i in range(NT):
    for j in range(i + 1, NT):
        if set(E[i]) & set(E[j]):
            J[i, j] = J[j, i] = np.sqrt(gw[i] * gw[j]) * 1.2
for i, j, w in [(3, 7, -0.50), (2, 6, -0.50), (4, 8, -0.45), (5, 8, -0.40)]:  # anti-correlation
    J[i, j] = J[j, i] = w
# FP row/col stay 0: a spurious gene is uncorrelated with the module, so the
# local field never supports it and it is removed.

# ---- dynamics: true genes fill in via the couplings from two clamped seeds
hT, beta, seeds = -0.76, 2.6, [0, 10]
h = np.full(N, hT)
p = np.zeros(N); p[seeds] = 1.0
P = [p.copy()]
for _ in range(8):
    p = sig(beta * (h + J @ p)); p[seeds] = 1.0; P.append(p.copy())
P = np.array(P); P[:, FP] = 0.0
pfp = np.array([1.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # FP fade-out, schematic
KCOLS = 7

# ---- styling
DIV = LinearSegmentedColormap.from_list(              # negative->red, 0->black, positive->green
    "divglow", [(0.0, "#ff9166"), (0.28, "#a83226"), (0.5, "#0d0d0d"),
                (0.72, "#2f8f4e"), (1.0, "#7fe0a0")])
CLIP = 0.70
GREEN = "#1a9850"; REC_LO = "#c6dbef"; REC_HI = "#08306b"; GREY = "0.72"; RED = "#cc2b1d"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm"})
allnodes = np.array([m[k] for k in range(11)] + [mFP])
xlim = (allnodes[:, 0].min() - 0.5, allnodes[:, 0].max() + 0.6)
ylim = (allnodes[:, 1].min() - 0.4, allnodes[:, 1].max() + 0.4)


def recon_shade(pv):                                  # reconstruction confidence -> blue shade
    return lerp(REC_LO, REC_HI, np.clip((pv - 0.5) / 0.5, 0, 1))


fig = plt.figure(figsize=(15.5, 5.6))
gs = fig.add_gridspec(3, KCOLS, height_ratios=[0.7, 5.6, 6.2], hspace=0.12, wspace=0.10)
lastJ = None
for t in range(KCOLS):
    # -- row 0: gene-presence barcode
    ax0 = fig.add_subplot(gs[0, t]); strip = np.ones((1, N, 4))
    for g in range(N):
        if g in seeds:
            strip[0, g, :3] = matplotlib.colors.to_rgb(GREEN)
        elif g == FP:
            strip[0, g, :3] = lerp("white", RED, pfp[t])
        else:
            strip[0, g, :3] = lerp("white", REC_HI, P[t, g])
    ax0.imshow(strip, aspect="auto"); ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(f"$T={t}$", fontsize=12, fontweight="bold", pad=3)
    if t == 0:
        ax0.set_ylabel("genes", rotation=0, ha="right", va="center", fontsize=9)
    # -- row 1: presence-gated activation J_ij x_i x_j (green = +, red = -)
    ax1 = fig.add_subplot(gs[1, t]); S = J * np.outer(P[t], P[t])
    ax1.imshow(S, cmap=DIV, vmin=-CLIP, vmax=CLIP, aspect="equal", interpolation="nearest")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel(f"{int((P[t, :NT] > 0.5).sum())}/11", fontsize=9.5)
    if t == 0:
        ax1.set_ylabel("$J_{ij}\\,x_ix_j$", rotation=0, ha="right", va="center", fontsize=11)
    lastJ = ax1
    # -- row 2: metabolic map
    ax2 = fig.add_subplot(gs[2, t]); ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_xlim(*xlim); ax2.set_ylim(*ylim)
    for gi, (a, b) in enumerate(E):
        xa, ya = m[a]; xb, yb = m[b]; v = P[t, gi]; lw = 1.6 + 3.4 * gw[gi]
        if gi in seeds:                                # observed
            ax2.plot([xa, xb], [ya, yb], color=GREEN, lw=lw, zorder=4, solid_capstyle="round")
        elif v < 0.5:                                  # still missing
            ax2.plot([xa, xb], [ya, yb], color=GREY, lw=1.3, alpha=0.9, zorder=2,
                     solid_capstyle="round")
        else:                                          # reconstructed, shade = confidence
            ax2.plot([xa, xb], [ya, yb], color=recon_shade(v), lw=lw, zorder=3,
                     solid_capstyle="round")
    if pfp[t] > 0.01:                                  # false-positive reaction, fades out
        xa, ya = m[2]; xb, yb = mFP; a = pfp[t]
        ax2.plot([xa, xb], [ya, yb], color=RED, lw=1.4 + 3.4 * a, alpha=min(1, 0.25 + 0.9 * a),
                 zorder=5, solid_capstyle="round")
        ax2.scatter(*mFP, s=42, facecolors="white", edgecolors=RED, lw=1.4, zorder=6,
                    alpha=min(1, 0.3 + a))
    for k in range(11):
        ax2.scatter(*m[k], s=42, c="white", edgecolors="0.35", lw=1.0, zorder=7)
    if t == 0:
        ax2.text(xlim[0] - 0.15, (ylim[0] + ylim[1]) / 2, "metabolic\nmap",
                 rotation=0, ha="right", va="center", fontsize=9)

cax = lastJ.inset_axes([1.12, 0.06, 0.09, 0.88])
cb = fig.colorbar(ScalarMappable(norm=Normalize(-CLIP, CLIP), cmap=DIV), cax=cax)
cb.set_ticks([-CLIP, 0, CLIP])
cb.set_ticklabels(["$J_{ij}x_ix_j{<}0$", "$0$", "$J_{ij}x_ix_j{>}0$"])
cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.5)
cb.set_label("presence-gated activation", fontsize=8.5, labelpad=6)

leg = [Line2D([0], [0], color=GREEN, lw=3.5, label="present in the input (kept)"),
       Line2D([0], [0], color=REC_HI, lw=3.5, label="recovered (shade = posterior confidence)"),
       Line2D([0], [0], color=GREY, lw=2, label="not recovered (the gap)"),
       Line2D([0], [0], color=RED, lw=3.5, label="false positive, silenced")]
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.savefig("analysis/figures/fig_toy_J_lightup.png", dpi=175, bbox_inches="tight")
fig.savefig("analysis/figures/fig_toy_J_lightup.pdf", bbox_inches="tight")
print("true present/step:", [int((P[t, :NT] > 0.5).sum()) for t in range(KCOLS)])
print("J range:", round(J[:NT, :NT].min(), 2), round(J[:NT, :NT].max(), 2),
      "| n_neg:", int((J[:NT, :NT] < 0).sum()) // 2, "| n_pos:", int((J[:NT, :NT] > 0).sum()) // 2)
print("wrote analysis/figures/fig_toy_J_lightup.{png,pdf}")
