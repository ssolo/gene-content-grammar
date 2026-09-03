#!/usr/bin/env python3.11
"""Aggregate the FP-tolerant-vs-default comparison into the three report tables:

  A. Bacterial spectra at the ancestral regime (fn = 0.9): MCC + ECE for bac-FT
     vs bac-FT-fp across the injected-FP grid (mean over 10 splits, final step).
  B. Extant recall + FP-removal, default vs FP-tolerant, per genome, at injected
     fp in {0.01, 0.05} (recall on truly-present COGs; fp_removed on injected FPs).
  C. Ancestral present-counts (COGs called present, posterior > 0.5), default vs
     FP-tolerant, at inject 0 vs 0.05, for LBCA and LACA.

Prints readable blocks plus LaTeX table bodies. Inputs:
  spectra_bacfp/*.parquet                           bac-FT / bac-FT-fp FP sweep
  data/interactome/fpcompare/recover_*.tsv          per-COG extant reconstructions
  data/interactome/fpcompare/{LBCA,LACA}_*_pred.tsv ancestral reconstructions
"""
import glob
import os
import re

import numpy as np
import pandas as pd

FPCMP = "data/interactome/fpcompare"
FP_GRID = [0.01, 0.025, 0.05, 0.1, 0.2]
GENOMES = [("ecoli", "E. coli"), ("medbac", "median bacterium"),
           ("arch", "M. jannaschii"), ("medarc", "median archaeon"),
           ("medbac_hq", "median bacterium (HQ)"), ("medarc_hq", "median archaeon (HQ)")]


def _ms(mean, sd):
    """Compact mean(SD), the SD in units of the last (third) decimal of the mean:
    _ms(0.674, 0.047) -> '0.674(47)'."""
    return f"{mean:.3f}({round(sd * 1000):d})"


def table_a():
    rows = []
    for sdir, fam, lab in [("spectra_bacfp", "nohidden_HO_T20_bacFT", "bac-FT"),
                           ("spectra_fpreal", "nohidden_HO_T20_bacFTfpreal", "bac-FT-fp")]:
        d = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{sdir}/{fam}_split*.parquet")],
                      ignore_index=True)
        d = d[d.step == d.step.max()]                       # final single-pass output
        d9 = d[np.isclose(d.fn, 0.9)]
        for fp in FP_GRID:
            g = d9[np.isclose(d9.fp, fp)]                   # one row per split
            rows.append((lab, fp, g.MCC.mean(), g.MCC.std(),
                         g.ece.mean(), g.ece.std(), g.split.nunique()))
    df = pd.DataFrame(rows, columns=["model", "fp", "MCC", "MCC_sd", "ECE", "ECE_sd", "n"])
    print("\n=== TABLE A: bacterial spectra at fn=0.9 (mean +/- SD over splits) ===")
    print(df.round(3).to_string(index=False))
    print("\n-- LaTeX body (fp & MCC_def(SD) & MCC_fp(SD) & ECE_def(SD) & ECE_fp(SD)) --")
    for fp in FP_GRID:
        md = df[(df.model == "bac-FT") & np.isclose(df.fp, fp)].iloc[0]
        mf = df[(df.model == "bac-FT-fp") & np.isclose(df.fp, fp)].iloc[0]
        print(f"${fp}$ & ${_ms(md.MCC, md.MCC_sd)}$ & ${_ms(mf.MCC, mf.MCC_sd)}$ "
              f"& ${_ms(md.ECE, md.ECE_sd)}$ & ${_ms(mf.ECE, mf.ECE_sd)}$ \\\\")


def _recall_fpr(path, fn=0.9):
    d = pd.read_csv(path, sep="\t")
    d = d[np.isclose(d.fn, fn)]
    pres = d[d.truth == 1]
    recall = float((pres.denoised_prob > 0.5).mean()) if len(pres) else float("nan")
    inj = d[(d.truth == 0) & (d.input_present == 1)]        # injected false positives
    fp_removed = float((inj.denoised_prob <= 0.5).mean()) if len(inj) else float("nan")
    return recall, fp_removed, len(inj)


def table_b(fn=0.9):
    print(f"\n=== TABLE B: extant recall + FP-removal at fn={fn} ===")
    print(f"{'genome':18} {'inj':>5} {'rec.def':>8} {'rec.fp':>8} {'fprem.def':>10} {'fprem.fp':>9} {'nInj':>6}")
    latex = []
    for key, lab in GENOMES:
        for inj in ["0.01", "0.05"]:
            rd, fd, n = _recall_fpr(f"{FPCMP}/recover_{key}_default_fp{inj}.tsv", fn)
            rf, ff, _ = _recall_fpr(f"{FPCMP}/recover_{key}_fp_fp{inj}.tsv", fn)
            print(f"{lab:18} {inj:>5} {rd*100:7.1f}% {rf*100:7.1f}% {fd*100:9.1f}% {ff*100:8.1f}% {n:6d}")
            latex.append(f"{lab} & {inj} & {rd*100:.0f}\\% & {rf*100:.0f}\\% & {fd*100:.0f}\\% & {ff*100:.0f}\\% \\\\")
    print("\n-- LaTeX body (genome & inj & rec_def & rec_fp & fprem_def & fprem_fp) --")
    print("\n".join(latex))


def table_c(cut=0.5):
    print(f"\n=== TABLE C: ancestral present-counts (posterior > {cut}) ===")
    print(f"{'node':6} {'arm':8} {'inj0':>6} {'inj0.05':>8} {'delta':>7}")
    latex = []
    for node in ["LBCA", "LACA"]:
        cells = {}
        for arm in ["default", "fp"]:
            for inj in ["0.0", "0.05"]:
                p = f"{FPCMP}/{node}_{arm}_fp{inj}_pred.tsv"
                n = int((pd.read_csv(p, sep="\t").mean_actual > cut).sum()) if os.path.exists(p) else -1
                cells[(arm, inj)] = n
            d0, d5 = cells[(arm, "0.0")], cells[(arm, "0.05")]
            print(f"{node:6} {arm:8} {d0:6d} {d5:8d} {d5-d0:+7d}")
        latex.append(f"{node} & {cells[('default','0.0')]} & {cells[('default','0.05')]} "
                     f"& {cells[('fp','0.0')]} & {cells[('fp','0.05')]} \\\\")
    print("\n-- LaTeX body (node & def_inj0 & def_inj0.05 & fp_inj0 & fp_inj0.05) --")
    print("\n".join(latex))


if __name__ == "__main__":
    table_a(); table_b(); table_c()
