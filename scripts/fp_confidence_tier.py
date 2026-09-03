#!/usr/bin/env python3
"""Multi-axis per-gene confidence tier on a denoised reconstruction.

Reads the marginal-HQ post-hoc FP-context diagnostic
(data/fp_context_{lbca,laca}_dense.tsv, ten-split ensemble means and SDs) and
partitions the present calls (q_full_mean > 0.5) into a confident core and a
borderline shell along three axes:

  weakly proposed  : input_prob         below the 33rd pct of present calls
  self-anchored    : self_anchor_absent above the 67th pct (cavity self-support)
  split-disagreed  : q_full_sd          above the 67th pct

borderline shell = concerning on >= 2 axes; confident core = 0 concerns.

The input axis is near-binary: over 77% of present calls have input_prob exactly
1.0, so its 33rd-pct threshold collapses to "input_prob < 1.0" and flags the
~22% of softly proposed calls rather than a true bottom tercile.

The three axes are not independent. self_anchor_absent and q_full_sd are
strongly collinear (Spearman +0.86 to +0.89), so the tier separates input
weakness from a single context/disagreement cluster; the pairwise Spearman
correlations are printed alongside the counts.

Usage:  python3 scripts/fp_confidence_tier.py
"""
import csv
from collections import Counter

CATFILE = "data/cog-20.def.tab"
LETTER_NAME = {
    "J": "translation", "A": "RNA-processing", "K": "transcription",
    "L": "replication/repair", "B": "chromatin", "D": "cell-cycle",
    "Y": "nuclear", "V": "defense", "T": "signal-transduction",
    "M": "cell-wall/membrane", "N": "cell-motility", "Z": "cytoskeleton",
    "W": "extracellular", "U": "trafficking/secretion",
    "O": "ptm/chaperones", "C": "energy", "G": "carbohydrate",
    "E": "amino-acid", "F": "nucleotide", "H": "coenzyme", "I": "lipid",
    "P": "inorganic-ion", "Q": "secondary-metab", "R": "general-func",
    "S": "unknown", "X": "mobilome",
}
SUPER = {
    "informational": set("JAKLB"),
    "metabolic": set("CGEFHIPQ"),
    "general/unknown": set("RS"),
    # everything else (cellular processes D M N O T U V W Y Z X) -> "cellular"
}


def load_cat():
    cat = {}
    with open(CATFILE, encoding="latin-1") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 2 and row[0].startswith("COG"):
                cat[row[0]] = row[1][:1]  # a COG may list several letters; the first is primary
    return cat


def pct(sorted_vals, p):
    """p-th percentile (0..100) by nearest-rank on a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def spearman(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def analyze(tag, path, cat):
    rows = []
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            try:
                qf = float(r["q_full_mean"])
            except (KeyError, ValueError):
                continue
            if qf <= 0.5:
                continue
            rows.append({
                "cog": r["COG_ID"],
                "input": float(r["input_prob"]),
                "anchor": float(r["self_anchor_absent"]),
                "sd": float(r["q_full_sd"]),
            })
    n = len(rows)
    inp = sorted(x["input"] for x in rows)
    anc = sorted(x["anchor"] for x in rows)
    sd = sorted(x["sd"] for x in rows)
    t_inp = pct(inp, 33)      # concerning below
    t_anc = pct(anc, 67)      # concerning above
    t_sd = pct(sd, 67)        # concerning above

    core, shell = [], []
    for x in rows:
        c = 0
        c += x["input"] < t_inp
        c += x["anchor"] > t_anc
        c += x["sd"] > t_sd
        x["concerns"] = c
        if c == 0:
            core.append(x)
        elif c >= 2:
            shell.append(x)

    print(f"\n===== {tag}  present={n} =====")
    print(f"thresholds: input<{t_inp:.3f}  anchor>{t_anc:.3f}  sd>{t_sd:.3f}")
    print(f"confident core (0 concerns): {len(core)} ({100*len(core)//n}%)")
    print(f"borderline shell (>=2):      {len(shell)} ({100*len(shell)//n}%)")
    print(f"one concern:                 {n-len(core)-len(shell)}")

    print("axis collinearity (Spearman over present calls):")
    print(f"  input vs anchor : {spearman([x['input'] for x in rows],[x['anchor'] for x in rows]):+.2f}")
    print(f"  input vs sd     : {spearman([x['input'] for x in rows],[x['sd'] for x in rows]):+.2f}")
    print(f"  anchor vs sd    : {spearman([x['anchor'] for x in rows],[x['sd'] for x in rows]):+.2f}")

    for name, members in (("CORE", core), ("SHELL", shell)):
        let = Counter(cat.get(x["cog"], "?") for x in members)
        sup = Counter()
        for x in members:
            L = cat.get(x["cog"], "?")
            placed = False
            for s, ls in SUPER.items():
                if L in ls:
                    sup[s] += 1
                    placed = True
                    break
            if not placed:
                sup["cellular"] += 1
        top = ", ".join(f"{LETTER_NAME.get(k,k)} {v}" for k, v in let.most_common(6))
        m = len(members) or 1
        supstr = ", ".join(f"{k} {100*v//m}%" for k, v in
                            sorted(sup.items(), key=lambda kv: -kv[1]))
        print(f"  {name} top cats: {top}")
        print(f"  {name} super:    {supstr}")
    return n, len(core), len(shell)


def main():
    cat = load_cat()
    analyze("LBCA  (bac marginal-HQ, GLD min1 dense)", "data/fp_context_lbca_dense.tsv", cat)
    analyze("LACA  (mix marginal-HQ, recount min1 dense)", "data/fp_context_laca_dense.tsv", cat)


if __name__ == "__main__":
    main()
