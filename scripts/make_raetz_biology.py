#!/usr/bin/env python3
"""Biology companion to Figure 1: the Raetz pathway (KEGG M00060, Kdo2-lipid A
biosynthesis) drawn as molecular structures rather than as the abstract node/edge
cartoon of Fig. 1. Drawing conventions: glucosamine as a hexagon, 3-hydroxyacyl
primary and acyloxyacyl secondary chains hanging into the membrane, phosphate as
a circled P, Kdo as a sugar cap. The molecule grows one enzyme at a time along
the same left-to-right serpentine as the Fig. 1 pathway row.

  python3 scripts/make_raetz_biology.py  ->  analysis/figures/fig_raetz_biology.{pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "analysis" / "figures"

SUGAR = "#cfe3f5"; SUGAR_E = "#3b6ea5"      # glucosamine
ACYL = "#33414d"                             # primary 3-OH-acyl chain
SEC = "#e08a2c"                              # secondary (acyloxyacyl) chain
PCOL = "#d24d3e"                             # phosphate
KDO = "#7fbf7f"; KDO_E = "#2f8f4e"           # Kdo sugar
UDP = "#b39ddb"                              # UDP leaving group

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm"})


def hexagon(ax, cx, cy, r=0.42, label="GlcN"):
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=r, orientation=np.pi / 6,
                                facecolor=SUGAR, edgecolor=SUGAR_E, lw=1.3, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6.4, color=SUGAR_E, zorder=5)


def acyl(ax, x, ytop, length=1.0, sec=False, wig=0.09):
    """3-OH primary chain (with an OH knuckle) or, if sec, a secondary acyloxyacyl chain."""
    col = SEC if sec else ACYL
    n = 7
    ys = np.linspace(ytop, ytop - length, n)
    xs = x + wig * np.array([0, 1, 0, 1, 0, 1, 0])
    ax.plot(xs, ys, color=col, lw=1.5, zorder=3, solid_capstyle="round")
    ax.add_patch(Circle((x, ytop), 0.05, facecolor="white", edgecolor=col, lw=1.1, zorder=4))
    if not sec:  # 3-OH knuckle where secondary chains esterify
        ax.text(x - 0.14, ytop - length * 0.34, "OH", ha="right", va="center",
                fontsize=4.6, color=col, zorder=4)


def phosphate(ax, x, y):
    ax.add_patch(Circle((x, y), 0.16, facecolor=PCOL, edgecolor="none", zorder=6))
    ax.text(x, y, "P", ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold", zorder=7)


def kdo(ax, cx, cy):
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.30, orientation=0,
                                facecolor=KDO, edgecolor=KDO_E, lw=1.1, zorder=4))
    ax.text(cx, cy, "Kdo", ha="center", va="center", fontsize=5.0, color=KDO_E, zorder=5)


def udp(ax, x, y):
    ax.add_patch(FancyBboxPatch((x - 0.34, y - 0.16), 0.68, 0.32,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=UDP, edgecolor="#6a4fa3", lw=1.0, zorder=5))
    ax.text(x, y, "UDP", ha="center", va="center", fontsize=6.0, color="#3d2c66", zorder=6)


def structure(ax, cx, cy, sugars=1, acyl_per=(0,), sec=0, p1=False, p4=False,
              use_udp=False, nac=False, kdo_cap=False, name="", sub=""):
    """Draw one lipid-A biosynthetic intermediate centred at (cx, cy)."""
    if sugars == 1:
        xs = [cx]
    else:                                   # distal (left, 4'/Kdo) + proximal (right, 1-P)
        xs = [cx - 0.92, cx + 0.92]
    for si, sx in enumerate(xs):
        hexagon(ax, sx, cy)
        # N-linked (left knuckle) + O3 (right) primary chains hang down into the membrane
        na = acyl_per[si] if si < len(acyl_per) else 0
        offs = [-0.22, 0.22][:na] if na <= 2 else [-0.22, 0.0, 0.22]
        for k, dx in enumerate(offs):
            acyl(ax, sx + dx, cy - 0.42)
        # secondary chains only on the distal (non-reducing) sugar
        if sec and si == 0:
            for dx in [-0.22, 0.22][:sec]:
                acyl(ax, sx + dx + 0.06, cy - 0.42 - 0.55, length=0.72, sec=True)
        if nac and si == len(xs) - 1:       # N-acetyl on the proximal amine (pre-LpxC)
            ax.text(sx - 0.30, cy + 0.30, "NAc", ha="right", va="center",
                    fontsize=5.2, color="#555", zorder=6)
    # phosphates / UDP at the reducing (proximal, right) and 4' (distal, left) positions
    prox = xs[-1]; dist = xs[0]
    if use_udp:
        udp(ax, prox + 0.62, cy - 0.30)
    elif p1:
        phosphate(ax, prox + 0.40, cy - 0.34)
    if p4:
        phosphate(ax, dist - 0.40, cy - 0.34)
    if kdo_cap:                              # two Kdo sugars cap the distal 6'-OH, pointing out
        kdo(ax, dist - 0.10, cy + 0.70)
        kdo(ax, dist + 0.34, cy + 1.02)
        ax.plot([dist, dist - 0.10], [cy + 0.42, cy + 0.48], color=KDO_E, lw=1.0, zorder=3)
        ax.plot([dist - 0.10, dist + 0.34], [cy + 0.86, cy + 0.86], color=KDO_E, lw=1.0, zorder=3)
    ax.text(cx, cy - 1.95, name, ha="center", va="top", fontsize=8.2, zorder=6)
    if sub:
        ax.text(cx, cy - 2.2, sub, ha="center", va="top", fontsize=6.6,
                color="#666", style="italic", zorder=6)


def enzyme_arrow(ax, x0, y0, x1, y1, enz, note="", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                 lw=1.6, color="#333", connectionstyle=f"arc3,rad={rad}", zorder=2))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dyy = 0.30 if abs(y1 - y0) < 0.5 else 0.0
    ax.text(mx, my + 0.34 + dyy, enz, ha="center", va="bottom", fontsize=8.6,
            fontweight="bold", color="#1a4a7a", zorder=6)
    if note:
        ax.text(mx, my - 0.30, note, ha="center", va="top", fontsize=6.2, color="#777", zorder=6)


def alt_arrow(ax, x0, x1, y, top_lbl, bot_lbl, note=""):
    """Two parallel routes for one step: the non-orthologous alternative (Fig. 1 loop)."""
    for rad, lbl, dy in [(-0.55, top_lbl, 0.66), (0.55, bot_lbl, -0.66)]:
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), connectionstyle=f"arc3,rad={rad}",
                     arrowstyle="-|>", mutation_scale=13, lw=1.6, color="#8a2f2f", zorder=2))
        ax.text((x0 + x1) / 2, y + dy, lbl, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color="#8a2f2f", zorder=6)
    if note:
        ax.text((x0 + x1) / 2, y - 1.02, note, ha="center", va="top", fontsize=6.0,
                color="#8a2f2f", zorder=6)


fig, ax = plt.subplots(figsize=(15.2, 9.7))
ax.set_xlim(0, 15.6); ax.set_ylim(-1.9, 10.4); ax.axis("off")
R1, R2 = 7.5, 1.7                             # y of the two serpentine rows

# ---- row 1, left to right: monosaccharide assembly, ending at LpxH/LpxI
structure(ax, 1.9, R1, sugars=1, acyl_per=(0,), use_udp=True, nac=True,
          name="UDP-GlcNAc", sub="activated sugar")
enzyme_arrow(ax, 3.1, R1, 4.1, R1, "LpxA", "O-3 acylation")
structure(ax, 5.4, R1, sugars=1, acyl_per=(1,), use_udp=True,
          name="UDP-3-O-acyl-GlcNAc", sub="mono-acyl")
enzyme_arrow(ax, 6.7, R1, 7.7, R1, "LpxC $\\cdot$ LpxD", "deacetylate + N-acylate")
structure(ax, 9.2, R1, sugars=1, acyl_per=(2,), use_udp=True,
          name="UDP-2,3-diacyl-GlcN", sub="di-acyl")
alt_arrow(ax, 10.55, 11.95, R1, "LpxH", "LpxI", note="alternative enzymes\n(one per genome)")
structure(ax, 13.3, R1, sugars=1, acyl_per=(2,), p1=True,
          name="Lipid X", sub="2,3-diacyl-GlcN-1-P")

# ---- serpentine wrap, row 1 Lipid X down to the row 2 disaccharide
enzyme_arrow(ax, 13.3, 5.2, 13.3, 3.2, "LpxB", "")
ax.text(13.72, 4.2, "condense\n(beta-1',6)", ha="left", va="center", fontsize=6.4, color="#555")

# ---- row 2, right to left: disaccharide -> 4'-P -> 2 Kdo -> hexa-acylation
structure(ax, 13.3, R2, sugars=2, acyl_per=(2, 2), p1=True,
          name="Lipid A disaccharide", sub="tetra-acyl, 1-P")
enzyme_arrow(ax, 11.4, R2, 10.3, R2, "LpxK", "+ 4'-P")
structure(ax, 8.7, R2, sugars=2, acyl_per=(2, 2), p1=True, p4=True,
          name="Lipid IVA", sub="1,4'-bis-P")
enzyme_arrow(ax, 6.9, R2, 5.8, R2, "WaaA", "+ 2 Kdo")
structure(ax, 4.3, R2, sugars=2, acyl_per=(2, 2), p1=True, p4=True, kdo_cap=True,
          name="Kdo2-lipid IVA", sub="tetra-acyl")
enzyme_arrow(ax, 2.6, R2, 1.6, R2, "LpxL $\\cdot$ LpxM", "+ 2 sec. acyl")
structure(ax, 1.4, R2 + 0.1, sugars=2, acyl_per=(2, 2), sec=2, p1=True, p4=True, kdo_cap=True,
          name="Kdo2-lipid A", sub="hexa-acyl")

ax.text(0.2, 10.25, "The Raetz pathway: Kdo$_2$-lipid A biosynthesis (KEGG M00060)",
        ha="left", va="top", fontsize=13, fontweight="bold")
ax.text(0.2, 9.78, "the outer-membrane (diderm) endotoxin anchor, built one enzyme at a time "
        "-- each enzyme is one edge of the Figure 1 module; LpxH / LpxI are the "
        "alternative route", ha="left", va="top", fontsize=8.4, color="#555", style="italic")

# ---- building-blocks key, on a bottom strip clear of the structures
ky = -1.4
ax.text(0.5, ky, "building blocks:", fontsize=7.8, fontweight="bold", color="#444",
        ha="left", va="center")
hexagon(ax, 3.0, ky, 0.22, ""); ax.text(3.3, ky, "glucosamine", fontsize=7.2, va="center")
acyl(ax, 5.5, ky + 0.34, 0.5); ax.text(5.72, ky, "3-OH acyl", fontsize=7.2, va="center")
acyl(ax, 7.35, ky + 0.34, 0.4, sec=True); ax.text(7.57, ky, "sec. acyl", fontsize=7.2, va="center")
phosphate(ax, 9.3, ky); ax.text(9.5, ky, "phosphate", fontsize=7.2, va="center")
kdo(ax, 11.3, ky); ax.text(11.62, ky, "Kdo", fontsize=7.2, va="center")
udp(ax, 12.9, ky); ax.text(13.3, ky, "UDP", fontsize=7.2, va="center")

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_raetz_biology.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_raetz_biology.png", dpi=180, bbox_inches="tight")
print(f"wrote {OUT/'fig_raetz_biology.pdf'}")
