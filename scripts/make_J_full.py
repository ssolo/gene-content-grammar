#!/usr/bin/env python3
"""Plot the ENTIRE learned coupling matrix J (all N=4,789 COGs), the global object the
Fig. 1 Raetz sub-block is sliced from. J is symmetric, hollow, and very sparse (~0.1%
of pairs carry |J|>0.05), so in raw COG order it reads as noise; under a spectral
reordering of the COG index the co-occurring gene modules emerge as bright diagonal
blocks. Same green(+)/red(-)-on-black style as Fig. 1.

Reads data/interactome/J_plain_pairwise_T20.npy, the checkpoint-ensembled coupling
written by scripts/interactome/extract_J.py, and J_spectral_order.npy, a permutation
of the COG index that groups spectrally similar rows. Neither is shipped.

  python3 scripts/make_J_full.py  ->  analysis/figures/fig_J_full.{pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "figures"
DAT = ROOT / "data" / "interactome"

J = np.load(DAT / "J_plain_pairwise_T20.npy")
order = np.load(DAT / "J_spectral_order.npy")
N = J.shape[0]
Js = J[np.ix_(order, order)]

off = J[~np.eye(N, dtype=bool)]
CLIP = float(np.percentile(np.abs(off), 99.5))          # a wider scale hides the sparse structure
frac = float(np.mean(np.abs(off) > 0.05))

DIV = LinearSegmentedColormap.from_list(
    "divglow", [(0.0, "#ff9166"), (0.28, "#a83226"), (0.5, "#0d0d0d"),
                (0.72, "#2f8f4e"), (1.0, "#7fe0a0")])
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 6.9))
for ax, M, ttl in [(axA, J, "raw COG order"),
                   (axB, Js, "spectral order (modules $=$ diagonal blocks)")]:
    ax.imshow(M, cmap=DIV, vmin=-CLIP, vmax=CLIP, interpolation="nearest",
              rasterized=True, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=10.5, pad=6)
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("0.55")

cb = fig.colorbar(ScalarMappable(norm=Normalize(-CLIP, CLIP), cmap=DIV),
                  ax=(axA, axB), fraction=0.026, pad=0.02, ticks=[-CLIP, 0, CLIP])
cb.set_ticklabels([f"$J_{{ij}}{{<}}0$", "$0$", f"$J_{{ij}}{{>}}0$"])
cb.set_label(f"coupling $J_{{ij}}$  (co-occur $>0$ / exclude $<0$; scale clipped at "
             f"$\\pm{CLIP:.3f}$)", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5); cb.outline.set_visible(False)

fig.suptitle(f"The entire learned coupling matrix $J$  ($N={N:,}$ COGs, symmetric, hollow; "
             f"only {frac*100:.2f}% of pairs have $|J|>0.05$)",
             fontsize=12.5, fontweight="bold", y=0.98)
fig.savefig(OUT / "fig_J_full.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "fig_J_full.png", bbox_inches="tight", dpi=200)
print(f"wrote {OUT/'fig_J_full.pdf'}  (N={N}, CLIP={CLIP:.4f}, |J|>0.05 frac={frac:.4f})")
