#!/usr/bin/env python3
"""Rank the eight ancestral reconstructions by per-call confidence and noise.

Three reconstructions of the last bacterial common ancestor and five of the last
archaeal common ancestor, put on a common footing:

  LBCA (3): GLD min1 (dense), GLD min4, sparse reconciliation     uncert_lbca_*.tsv
  LACA (5): COG-based uniform (merged), Euryroot ML-orig,
            Euryroot uniform-orig, recount min1 (dense), recount min4 (dense)
                                                                  LACA_*_pred.tsv

The LACA files are the same canonical *_pred.tsv that
scripts/build_paper_artifacts.py reads for the paper's confidence table
(tab:conf), produced by the cross-input-consistent marginal ensemble
(ENSEMBLE_GLOB); the LBCA files are the uncert_*.tsv behind that table's LBCA
rows. All carry the ten-split ensemble mean (mean_actual) and cross-split SD
(sd_actual).

Per reconstruction:
  confidence : present (mean>0.5), confident core (mean>0.9), borderline (0.4-0.6)
  noise      : mean cross-split SD, fraction of present calls flipping (SD>0.15)
  input      : near-binary input bits (<=0.02 or >=0.98) vs graded (in between)

Absolute SD is comparable within an ancestor, which shares one ensemble, and only
indicative across ancestors, since LBCA and LACA use different marginal-HQ
ensembles. The present/core/borderline counts are checked against the published
confidence table so that drift is caught.

Usage:  python3 scripts/input_noise_ranking.py
"""
import csv

# label -> (file, published (present, core, borderline) counts for the self-check,
# or None where the reconstruction has no row in the confidence table)
RECONS = [
    ("LBCA  GLD min1 (dense)",   "data/uncert_lbca_dense.tsv",        (1836, 1177, 265)),
    ("LBCA  GLD min4",           "data/uncert_lbca_min4.tsv",         None),
    ("LBCA  sparse reconcil.",   "data/uncert_lbca_sparse.tsv",       (1519,  820, 263)),
    ("LACA  COG-based uniform",  "LACA_merged_pred.tsv",              (1023,  636, 149)),
    ("LACA  Euryroot ML-orig",   "LACA_euryroot_pred.tsv",            (1321,  906, 176)),
    ("LACA  Euryroot uniform",   "LACA_euryroot_uniform_pred.tsv",    ( 909,  549, 157)),
    ("LACA  recount min1 (dense)","LACA_gld_min1_pred.tsv",           (1708, 1203, 219)),
    ("LACA  recount min4 (dense)","LACA_gld_min4_pred.tsv",           (1405,  997, 161)),
]
FLIP = 0.15  # cross-split SD above which a present call is counted as flipping


def stats(path):
    sd, inp, mean = [], [], []
    border = 0
    in_present = removed = added = 0
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ip = float(r["input_prob"])
            m = float(r["mean_actual"])
            in_p, out_p = ip > 0.5, m > 0.5
            in_present += in_p
            if in_p and not out_p:
                removed += 1              # the input proposed it, the denoiser rejected it
            if out_p and not in_p:
                added += 1                # the denoiser rescued a gene the input had lost
            if 0.4 < m < 0.6:
                border += 1               # a two-sided band, so absent calls enter it too
            if out_p:
                sd.append(float(r["sd_actual"]))
                inp.append(ip)
                mean.append(m)
    n = len(sd)
    core = sum(1 for m in mean if m > 0.9)
    mean_sd = sum(sd) / n
    flip = sum(1 for s in sd if s > FLIP)
    hard = sum(1 for x in inp if x <= 0.02 or x >= 0.98)
    return dict(present=n, core=core, border=border, mean_sd=mean_sd,
                flip=flip, flip_pct=100.0 * flip / n,
                graded_pct=100.0 * (n - hard) / n,
                in_present=in_present, removed=removed, added=added,
                removed_pct=(100.0 * removed / in_present) if in_present else 0.0)


def main():
    rows = []
    drift = 0
    print(f"{'reconstruction':27s}{'pres':>6}{'core':>6}{'bord':>6}"
          f"{'meanSD':>8}{'flip':>6}{'flip%':>7}{'graded-in%':>11}")
    for name, path, conf in RECONS:
        s = stats(path)
        rows.append((name, s))
        flag = ""
        if conf:
            exp_p, exp_c, exp_b = conf
            if (s["present"], s["core"], s["border"]) != (exp_p, exp_c, exp_b):
                flag = "  <-- DRIFT vs tab:conf"
                drift += 1
        print(f"{name:27s}{s['present']:6d}{s['core']:6d}{s['border']:6d}"
              f"{s['mean_sd']:8.4f}{s['flip']:6d}{s['flip_pct']:6.1f}%"
              f"{s['graded_pct']:10.1f}%{flag}")

    print("\n-- ranked by mean cross-split SD (noisiest -> cleanest) --")
    for name, s in sorted(rows, key=lambda r: -r[1]["mean_sd"]):
        print(f"  SD={s['mean_sd']:.3f}  flip={s['flip_pct']:4.1f}%  "
              f"core={100*s['core']//s['present']:2d}%  {name}")

    print("\n-- within-ancestor noise ratio (dense over-builder / cleanest build) --")
    lbca = {n: s for n, s in rows if n.startswith("LBCA")}
    laca = {n: s for n, s in rows if n.startswith("LACA")}
    def ratio(group):
        f = {k: v["flip_pct"] for k, v in group.items()}
        hi = max(f.values()); lo = min(f.values())
        return hi / lo, hi, lo
    rL, hL, loL = ratio(lbca)
    rA, hA, loA = ratio(laca)
    print(f"  LBCA: noisiest {hL:.1f}% / cleanest {loL:.1f}% = {rL:.1f}x")
    print(f"  LACA: noisiest {hA:.1f}% / cleanest {loA:.1f}% = {rA:.1f}x")

    # ---- input false-positive ranking
    PROPOSE = 200   # below this many confident input proposals the FP rate is not meaningful
    print("\n-- INPUT FP RANKING: removed = model's confident FP verdict (input>0.5,"
          " output<0.5); residual = split-unstable kept shell (flip%) --")
    fp = [(n, s) for n, s in rows if s["in_present"] >= PROPOSE]
    for name, s in sorted(fp, key=lambda r: -r[1]["removed_pct"]):
        print(f"  removed {s['removed']:4d}/{s['in_present']:<4d} "
              f"({s['removed_pct']:4.1f}%)   residual {s['flip_pct']:4.1f}%   {name}")
    builders = [n for n, s in rows if s["in_present"] < PROPOSE]
    print(f"  builders (no input FP, under-build instead): {', '.join(builders)}")
    print(f"\n{'DRIFT DETECTED' if drift else 'all confidence counts match tab:conf'}")


if __name__ == "__main__":
    main()
