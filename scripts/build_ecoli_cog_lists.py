#!/usr/bin/env python3
"""Per-COG reconstruction table for the E. coli example in Figure 3, in the same
shape as data/lbca_cog_lists.tsv but for an extant genome, so it also carries the
ground truth and a per-COG correctness state.

Source is the recovery TSV at SRC below: the production denoiser
(bac-FT-fp-marginal-HQ, split 5, typical replicate), E. coli held out at both
training stages, corrupted at fn in {0.4, 0.6, 0.8} with fp = 0.01. COG category
and name are joined from data/lbca_cog_lists.tsv.

Columns: COG_ID, category, name, truth, then per fn in {04,06,08}:
  fnXX_input   (0/1)  the corrupted observation the model was given
  fnXX_post    (0-1)  denoised posterior P(present)
  fnXX_present (0/1)  denoised call, posterior > 0.5, the LBCA/LACA threshold
  fnXX_state   kept (true, in input, called present) / recovered (true, deleted
               by noise, restored) / gap (true, not recovered) / FP (absent in
               truth, called present) / absent (true negative)

  python3 scripts/build_ecoli_cog_lists.py  ->  data/ecoli_cog_lists.tsv (+ .meta.tsv)
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "analysis/leakage_audit/scratch_ecoli_split5/recover_ecoli_marginal_hq_split5_typical_fn468.tsv"
ANNO = ROOT / "data/lbca_cog_lists.tsv"
OUT = ROOT / "data/ecoli_cog_lists.tsv"
FNS = [("0.4", "fn04"), ("0.6", "fn06"), ("0.8", "fn08")]
THR = 0.5

# COG_ID -> (category, name)
anno = {}
for r in csv.DictReader(open(ANNO), delimiter="\t"):
    anno[r["COG_ID"]] = (r["category"], r["name"])

# (COG_ID, fn) -> (truth, input_present, denoised_prob)
rec, truth_of, cogs = {}, {}, []
for r in csv.DictReader(open(SRC), delimiter="\t"):
    c, fn = r["COG_ID"], r["fn"]
    if c not in truth_of:
        truth_of[c] = int(r["truth"]); cogs.append(c)
    rec[(c, fn)] = (int(r["input_present"]), float(r["denoised_prob"]))


def state(truth, inp, present):
    if truth and present:      return "kept" if inp else "recovered"
    if truth and not present:  return "gap"
    if not truth and present:  return "FP"
    return "absent"


header = ["COG_ID", "category", "name", "truth"]
for _, tag in FNS:
    header += [f"{tag}_input", f"{tag}_post", f"{tag}_present", f"{tag}_state"]

rows = []
for c in cogs:
    cat, name = anno.get(c, ("?", "?"))
    t = truth_of[c]
    row = [c, cat, name, t]
    for fn, tag in FNS:
        inp, post = rec[(c, fn)]
        pres = int(post > THR)
        row += [inp, f"{post:.4f}", pres, state(t, inp, pres)]
    rows.append(row)

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(header)
    w.writerows(rows)

# The printed summary checks the Figure 3 caption: 59% recall, 89% precision at fn=0.8.
def stats(fn):
    tp = fp = fn_ = tn = 0
    for c in cogs:
        t = truth_of[c]; post = rec[(c, fn)][1]; pres = post > THR
        if t and pres: tp += 1
        elif t and not pres: fn_ += 1
        elif not t and pres: fp += 1
        else: tn += 1
    rec_ = tp / (tp + fn_) if tp + fn_ else 0
    prec = tp / (tp + fp) if tp + fp else 0
    return tp, fp, fn_, tn, rec_, prec


truth_total = sum(truth_of.values())
meta = [("species", "Escherichia coli"), ("model_label", "bac-FT-fp-marginal-HQ (split 5)"),
        ("truth_total_genes", truth_total), ("fp", "0.01"), ("fn_levels", "0.4,0.6,0.8"),
        ("call_threshold", THR), ("n_cogs", len(cogs)),
        ("source_tsv", str(SRC.relative_to(ROOT)))]
print(f"wrote {OUT}  ({len(rows)} COGs x {len(header)} cols)")
print(f"E. coli truth = {truth_total} genes")
for fn, tag in FNS:
    tp, fp, fn_, tn, rec_, prec = stats(fn)
    print(f"  fn={fn}: present_in_input={sum(rec[(c,fn)][0] for c in cogs):4d}  "
          f"TP={tp} FP={fp} gap={fn_}  recall={rec_*100:.1f}%  precision={prec*100:.1f}%")
    meta += [(f"recall_fn{tag[2:]}", f"{rec_:.4f}"), (f"precision_fn{tag[2:]}", f"{prec:.4f}")]

with open(str(OUT).replace(".tsv", ".meta.tsv"), "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["key", "value"]); w.writerows(meta)
