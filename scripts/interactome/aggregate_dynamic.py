#!/usr/bin/env python3.11
"""Aggregate dynamic_interaction.py outputs (data/interactome/results_dyn/*.json)
into two CSVs comparing the static coupling J against the generative knockout
response R(T) as an interactome predictor:
  data/interactome/ecoli_zoom_dyn.csv  E. coli on the 6 zoom-distance models
  data/interactome/all19_dyn.csv       all 19 pathogens (plain pairwise model)
Each row carries AUROC on the STRING experimental (non-circular) target at >=700
and on the combined target at >=900, for static J and for R(T) at depths 1, 2, 3
and 5, plus the best depth and its gain over J.
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "data", "interactome", "results_dyn")
OUT = os.path.join(REPO, "data", "interactome")
DEPTHS = ["1", "2", "3", "5"]
ZOOM_ORDER = ["phylum", "class", "order", "intermediate", "family", "species"]


def rnd(x):
    return None if x is None else round(x, 4)


def row(d, label=None):
    J, dyn = d.get("J", {}), d.get("dyn", {})
    r = {}
    if label is not None:
        r["rank"] = label
    r.update({"name": d.get("name"), "taxon": d.get("taxon"), "model": d.get("model"),
              "n_present": d.get("n_present"), "n_pairs": d.get("n_pairs"),
              "J_exp700": rnd(J.get("AUROC_exp_700")),
              "J_exp900": rnd(J.get("AUROC_exp_900")),
              "J_comb900": rnd(J.get("AUROC_comb_900"))})
    for T in DEPTHS:
        m = dyn.get(T, {})
        r["dynT%s_exp700" % T] = rnd(m.get("AUROC_exp_700"))
        r["dynT%s_comb900" % T] = rnd(m.get("AUROC_comb_900"))
    ex = [(T, dyn.get(T, {}).get("AUROC_exp_700")) for T in DEPTHS
          if dyn.get(T, {}).get("AUROC_exp_700") is not None]
    if ex:
        bT, bv = max(ex, key=lambda kv: kv[1])
        r["best_dyn_exp700"], r["best_T"] = rnd(bv), bT
        r["gain_over_J_exp700"] = rnd(bv - (J.get("AUROC_exp_700") or 0))
    return r


def write_csv(rows, path):
    if not rows:
        print("no rows for", path); return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (path, len(rows)))


def main():
    zrows = []
    for rank in ZOOM_ORDER:
        f = os.path.join(RES, "511145_zoom_%s.json" % rank)
        if os.path.exists(f):
            zrows.append(row(json.load(open(f)), label=rank))
    write_csv(zrows, os.path.join(OUT, "ecoli_zoom_dyn.csv"))

    drows = [row(json.load(open(f)))
             for f in sorted(glob.glob(os.path.join(RES, "*_dyn19.json")))]
    write_csv(drows, os.path.join(OUT, "all19_dyn.csv"))


if __name__ == "__main__":
    main()
