#!/usr/bin/env python3
"""Source data behind Figure 8 (the five LACA reconstructions converging on
denoising): one per-COG table plus the two 5x5 pairwise Jaccard matrices plotted.

Reads the same five pred files as scripts/plot_laca_jaccard_heatmap.py (columns
COG_ID, input_prob, mean_actual, sd_actual).  Present sets are input_prob > 0.5
for the left panel and mean_actual > 0.5 for the right.  COG category and name
are joined from data/laca_cog_lists.tsv.

Outputs:
  data/laca_figure8_cog_lists.tsv   per-COG: COG_ID, category, name, then per
                                    reconstruction {key}_input (input_prob),
                                    {key}_post (denoised mean), {key}_sd
                                    (ensemble sd), {key}_present (post > 0.5)
  data/laca_figure8_jaccard.tsv     tidy 5x5 pairwise Jaccard for both panels
  data/laca_figure8_cog_lists.meta.tsv   key/figure-label map and panel means
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNO = ROOT / "data/laca_cog_lists.tsv"
OUTC = ROOT / "data/laca_figure8_cog_lists.tsv"
OUTJ = ROOT / "data/laca_figure8_jaccard.tsv"
THR = 0.5

# (column key, figure display label, pred file); the order is the Figure 8 axis order.
RECON = [
    ("combined_root",  "combined root",           "LACA_merged_pred.tsv"),
    ("euryarch_ml",    "Euryarchaeota (ML)",       "LACA_euryroot_pred.tsv"),
    ("euryarch_unif",  "Euryarchaeota (uniform)",  "LACA_euryroot_uniform_pred.tsv"),
    ("copynum_min1",   "copy-number (min-1)",      "LACA_gld_min1_pred.tsv"),
    ("copynum_min4",   "copy-number (min-4)",      "LACA_gld_min4_pred.tsv"),
]

anno = {r["COG_ID"]: (r["category"], r["name"])
        for r in csv.DictReader(open(ANNO), delimiter="\t")}

# Per reconstruction: COG -> (input_prob, mean_actual, sd_actual).  The COG row
# order comes from the first reconstruction, which every other one must cover.
data, cogs = {}, []
for key, _, f in RECON:
    d = {}
    for r in csv.DictReader(open(ROOT / f), delimiter="\t"):
        c = r["COG_ID"]
        d[c] = (float(r["input_prob"]), float(r["mean_actual"]), float(r["sd_actual"]))
        if key == RECON[0][0]:
            cogs.append(c)
    data[key] = d

# Per-COG table
header = ["COG_ID", "category", "name"]
for key, _, _ in RECON:
    header += [f"{key}_input", f"{key}_post", f"{key}_sd", f"{key}_present"]
with open(OUTC, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(header)
    for c in cogs:
        cat, name = anno.get(c, ("?", "?"))
        row = [c, cat, name]
        for key, _, _ in RECON:
            inp, post, sd = data[key][c]
            row += [f"{inp:.4f}", f"{post:.4f}", f"{sd:.4f}", int(post > THR)]
        w.writerow(row)

# Pairwise Jaccard for both panels: the numbers the figure plots.
def present(key, which):  # which: 0=input, 1=post
    return {c for c in cogs if data[key][c][which] > THR}

panels = {"before_input": {k: present(k, 0) for k, _, _ in RECON},
          "after_denoised": {k: present(k, 1) for k, _, _ in RECON}}
keys = [k for k, _, _ in RECON]
lbl = {k: L for k, L, _ in RECON}

with open(OUTJ, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["panel", "recon_i", "recon_j", "jaccard"])
    means = {}
    for pan, sets in panels.items():
        off = []
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                u = len(sets[ki] | sets[kj])
                jac = len(sets[ki] & sets[kj]) / u if u else 0.0
                w.writerow([pan, lbl[ki], lbl[kj], f"{jac:.4f}"])
                if i < j:
                    off.append(jac)
        means[pan] = sum(off) / len(off)
    w.writerow(["before_input", "MEAN_OFFDIAG", "", f"{means['before_input']:.4f}"])
    w.writerow(["after_denoised", "MEAN_OFFDIAG", "", f"{means['after_denoised']:.4f}"])

meta = [("figure", "Figure 8 (fig:laca_jacc) -- five LACA reconstructions converge"),
        ("present_threshold", THR),
        ("mean_offdiag_before", f"{means['before_input']:.4f}"),
        ("mean_offdiag_after", f"{means['after_denoised']:.4f}")]
meta += [(f"label::{k}", L) for k, L, _ in RECON]
with open(str(OUTC).replace(".tsv", ".meta.tsv"), "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["key", "value"]); w.writerows(meta)

print(f"wrote {OUTC.name} ({len(cogs)} COGs x {len(header)} cols) + {OUTJ.name} + meta")
print(f"mean off-diagonal Jaccard: before {means['before_input']:.3f} -> "
      f"after {means['after_denoised']:.3f}   (caption: 0.23 -> 0.71)")
