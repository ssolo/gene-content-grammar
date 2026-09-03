#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver for every figure in the MBE manuscript and its Supplementary Material.

MANIFEST maps each figure to its recipe, its inputs, its provenance tier, and
where each input is staged. See REPRODUCE.md.

Provenance tiers:
  local    Rebuilt here from committed primary data (recon pred TSVs, recovery
           TSVs, summary CSVs, COG defs, the KEGG module matrix). No network, no
           GPU; a failure is an error.
  network  The iPath3 metabolic maps + the E. coli composites that embed them.
           Rebuilt only with --network (POSTs to pathways.embl.de); otherwise the
           committed PNG/PDF is the durable artifact and is presence-checked.
  cluster  Produced by a GPU sweep / corrupted-genome eval, not rebuilt here. If
           the committed summary (CSV/parquet/TSV) is present the figure is
           re-plotted locally; otherwise the committed PDF is presence-checked
           and the sweep has to be re-run (see slurm/ + REPRODUCE.md).
  external Produced outside this repo's scripts; presence-checked.

Inputs are tagged git (committed, small) or zenodo (large; fetched by
scripts/fetch_inputs.sh).

Usage
-----
  python3 scripts/build_mbe_figures.py            # rebuild local figures, presence-check the rest
  python3 scripts/build_mbe_figures.py --network  # also rebuild the iPath maps + composites
  python3 scripts/build_mbe_figures.py --only fig_ecoli_timeladder.pdf

Exit status is non-zero if any local-tier figure fails to build or a required
local input is missing.  ASCII only.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCR = REPO / "scripts"
FIG = REPO / "analysis" / "figures"
PY = sys.executable

# E. coli composite recovery TSV: the replicate whose fn=0.8 MCC is the median
# over the 500 draws of scripts/score_ecoli_replicates.py --dump-typical.
ECOLI_TSV = "analysis/leakage_audit/scratch_ecoli_split5/recover_ecoli_marginal_hq_split5_typical_fn468.tsv"


def step_ipath(node, tag=None, fn=None, sysmap=False, tsv=None):
    a = [PY, SCR / "plot_metabolic_ipath.py", "--node", node]
    if sysmap:
        a += ["--sysmap"]
    if tsv:
        a += ["--tsv", tsv]
    if fn is not None:
        a += ["--fn", str(fn)]
    if tag:
        a += ["--tag", tag]
    return a


def step_panels(fn, tag):
    return [PY, SCR / "plot_ecoli_fn_panels.py", "--tsv", ECOLI_TSV,
            "--fn", str(fn), "--tag", tag,
            "--model", "the production denoiser (marginal-HQ, split 5)"]


def recon_fig(out, pred, only, label, extra=()):
    return [PY, SCR / "plot_recon_figs.py", "--only", only,
            f"--{only}-pred", pred, f"--{only}-out", out,
            f"--{only}-label", label, "--outdir", str(FIG), *extra]


# --- the manifest --------------------------------------------------------------
# Each entry: name (figure file under analysis/figures/), doc (mbe|sm), tier,
# inputs (list of (relpath, "git"|"zenodo")), steps (argv run in order;
# [] = presence-only). An input path starting with "__" is a sentinel for a
# figure with no rebuildable input here: the input check skips it and only the
# committed figure file is checked.
MOD = ("data/module_matrix_kegg.pt", "git")
COG = ("data/cog-20.def.tab", "git")
ETSV = (ECOLI_TSV, "git")

MANIFEST = [
    # ============================ MAIN TEXT ============================
    dict(name="spectra_best_delta.pdf", doc="mbe", tier="cluster",   # Fig 1
         inputs=[("data/spectra_calibration_all.parquet", "zenodo")],
         steps=[[PY, SCR / "plot_results.py", "--spectra",
                 "data/spectra_calibration_all.parquet", "--outdir", str(FIG)]]),
    dict(name="fig_ecoli_recovery_fn08.pdf", doc="mbe", tier="network",  # Fig 2
         inputs=[ETSV, MOD, COG],
         steps=[step_ipath("ecoli", tag="_fn08", fn=0.8, tsv=ECOLI_TSV),
                step_panels(0.8, "_fn08")]),
    dict(name="fig_ecoli_timeladder.pdf", doc="mbe", tier="local",      # Fig 3
         # Point markers from the recover_*_fn468.tsv; HPD band over noise
         # instances from the reps_*_fn468.tsv (scripts/score_ecoli_replicates.py
         # via slurm/run_ecoli_zoom_replicates.sh). Both committed.
         inputs=[("analysis/ecoli_zoom/recover_ecoli_ecolizoom_phylum_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/recover_ecoli_ecolizoom_class_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/recover_ecoli_ecolizoom_order_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/recover_ecoli_ecolizoom_intermediate_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/recover_ecoli_ecolizoom_family_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/recover_ecoli_ecolizoom_species_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/reps_ecoli_ecolizoom_phylum_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/reps_ecoli_ecolizoom_class_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/reps_ecoli_ecolizoom_order_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/reps_ecoli_ecolizoom_intermediate_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/reps_ecoli_ecolizoom_family_ho3_fn468.tsv", "git"),
                 ("analysis/ecoli_zoom/reps_ecoli_ecolizoom_species_ho3_fn468.tsv", "git")],
         steps=[[PY, SCR / "plot_ecoli_timeladder.py"]]),
    dict(name="fig_lbca_profile_row.pdf", doc="mbe", tier="local",
         inputs=[("node2012_T20bacfphq_pred.tsv", "git"), MOD, COG],
         steps=[recon_fig("fig_lbca_profile_row.pdf", "node2012_T20bacfphq_pred.tsv",
                          "lbca", "bac-FT-fp-marginal-HQ, face value",
                          ["--side-by-side"])]),   # (A)|(B) in one row; (C) map full-width below
    dict(name="fig_laca_profile_row.pdf", doc="mbe", tier="local",
         inputs=[("LACA_euryroot_pred.tsv", "git"), MOD, COG],
         steps=[recon_fig("fig_laca_profile_row.pdf", "LACA_euryroot_pred.tsv",
                          "laca", "Euryroot ML origination, cons-10, expected genes",
                          ["--side-by-side"])]),
    dict(name="metabolic_lbca_recon.png", doc="mbe", tier="network",
         inputs=[("data/lbca_cog_lists.tsv", "git"), MOD],
         steps=[step_ipath("lbca", sysmap=True)]),
    dict(name="metabolic_laca_recon.png", doc="mbe", tier="network",
         inputs=[("LACA_euryroot_pred.tsv", "git"), MOD],
         steps=[step_ipath("laca", sysmap=True)]),
    dict(name="fig_laca_jaccard.png", doc="mbe", tier="local",
         inputs=[("LACA_euryroot_pred.tsv", "git"), ("LACA_merged_pred.tsv", "git"),
                 ("LACA_euryroot_uniform_pred.tsv", "git"),
                 ("LACA_gld_min1_pred.tsv", "git"), ("LACA_gld_min4_pred.tsv", "git")],
         steps=[[PY, SCR / "plot_laca_jaccard_heatmap.py"]]),
    dict(name="fig_phenotype_denoise.pdf", doc="mbe", tier="external",
         inputs=[("__external__", "git")], steps=[]),
    # ====================== SUPPLEMENTARY MATERIAL ======================
    dict(name="fig_ecoli_recovery_fn04.pdf", doc="sm", tier="network",
         inputs=[ETSV, MOD, COG],
         steps=[step_ipath("ecoli", tag="_fn04", fn=0.4, tsv=ECOLI_TSV),
                step_panels(0.4, "_fn04")]),
    dict(name="fig_ecoli_recovery_fn06.pdf", doc="sm", tier="network",
         inputs=[ETSV, MOD, COG],
         steps=[step_ipath("ecoli", tag="_fn06", fn=0.6, tsv=ECOLI_TSV),
                step_panels(0.6, "_fn06")]),
    dict(name="fig_laca_euryroot_sparse.pdf", doc="sm", tier="local",
         inputs=[("LACA_euryroot_pred.tsv", "git"), MOD, COG],
         steps=[recon_fig("fig_laca_euryroot_sparse.pdf", "LACA_euryroot_pred.tsv",
                          "laca", "Euryroot ML origination, cons-10, expected genes")]),
    dict(name="fig_marginal_fp_noise.pdf", doc="sm", tier="cluster",
         inputs=[("data/COG_train1_phylum.feather", "zenodo")],
         steps=[[PY, SCR / "plot_marginal_fp_noise.py"]]),
    dict(name="spectra_fp_varying.pdf", doc="sm", tier="cluster",
         inputs=[("data/spectra_fpreal_summary.csv", "git")],
         steps=[[PY, SCR / "plot_fp_varying.py", "--summary",
                 "data/spectra_fpreal_summary.csv"]]),
    dict(name="coherent_fp_frontier.pdf", doc="sm", tier="cluster",
         inputs=[("data/coherent_fp_mixFT.tsv", "git"), ("data/coherent_fp_bacFT.tsv", "git")],
         steps=[[PY, SCR / "plot_coherent_fp.py", "--fn", "0.9",
                 "--out", str(FIG / "coherent_fp_frontier.pdf")]]),
    dict(name="fig_lbca_softlanding.pdf", doc="sm", tier="local",
         inputs=[("data/lbca_comparison_summary.json", "git"),
                 ("data/lbca_reconciliation_removal_by_confidence.csv", "git")],
         steps=[[PY, SCR / "plot_lbca_softlanding.py"]]),
    dict(name="fig_laca_softlanding.pdf", doc="sm", tier="local",
         inputs=[("LACA_merged_pred.tsv", "git"), ("LACA_euryroot_pred.tsv", "git"),
                 ("LACA_euryroot_uniform_pred.tsv", "git"),
                 ("LACA_gld_min1_pred.tsv", "git"), ("LACA_gld_min4_pred.tsv", "git")],
         steps=[[PY, SCR / "plot_laca_softlanding.py"]]),
    dict(name="fig_interactome.pdf", doc="sm", tier="cluster",
         inputs=[("__cluster__", "zenodo")], steps=[]),
]
# Per-node extant-recovery panels (fig_recover_*). Only the clean (generalist /
# marginal) set: its recover_*.tsv are committed at repo root, whereas the
# FP-tolerant set lives under the gitignored interactome directory.
RECOVER_CLEAN = [
    ("recover_ecoli_generalist.tsv", "fig_recover_ecoli_gen.pdf"),
    ("recover_ecoli_bachard.tsv", "fig_recover_ecoli_bacmarg.pdf"),
    ("recover_archaeon_generalist.tsv", "fig_recover_archaeon_gen.pdf"),
    ("recover_archaeon_mixft.tsv", "fig_recover_archaeon_mixmarg.pdf"),
    ("recover_medbac_generalist.tsv", "fig_recover_medbac_gen.pdf"),
    ("recover_medbac_bachard.tsv", "fig_recover_medbac_bacmarg.pdf"),
    ("recover_medarc_hq_generalist.tsv", "fig_recover_medarc_hq_gen.pdf"),
    ("recover_medarc_hq_mixFT.tsv", "fig_recover_medarc_hq_mixmarg.pdf"),
    ("recover_medbac_hq_generalist.tsv", "fig_recover_medbac_hq_gen.pdf"),
    ("recover_medbac_hq_bachard.tsv", "fig_recover_medbac_hq_bacmarg.pdf"),
]
for tsv, out in RECOVER_CLEAN:
    MANIFEST.append(dict(name=out, doc="sm", tier="local",
                         inputs=[(tsv, "git"), COG],
                         steps=[[PY, SCR / "plot_recon_figs.py", "--skip-ancestral",
                                 "--recover", f"{tsv}:{out}"]]))


def have(rel):
    return (REPO / rel).exists()


def run(argv):
    return subprocess.run([str(x) for x in argv], cwd=REPO,
                          capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", action="store_true",
                    help="also rebuild the iPath maps + composites (POSTs to pathways.embl.de)")
    ap.add_argument("--list", action="store_true", help="print the manifest, build nothing")
    ap.add_argument("--inputs", action="store_true", help="print the input staging manifest")
    ap.add_argument("--only", default=None, help="build just this figure name")
    A = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    if A.inputs:
        seen = {}
        for f in MANIFEST:
            for rel, where in f["inputs"]:
                if rel.startswith("__"):
                    continue
                seen[rel] = where
        print(f"{'WHERE':7s} {'PRESENT':8s} INPUT")
        for rel in sorted(seen):
            where = seen[rel]
            print(f"{where:7s} {('yes' if have(rel) else 'MISSING'):8s} {rel}")
        git = sum(1 for w in seen.values() if w == "git")
        zen = sum(1 for w in seen.values() if w == "zenodo")
        print(f"\n{git} git-staged inputs, {zen} Zenodo inputs "
              f"(fetch with scripts/fetch_inputs.sh)")
        return

    if A.list:
        print(f"{'TIER':9s} {'DOC':4s} FIGURE")
        for f in MANIFEST:
            print(f"{f['tier']:9s} {f['doc']:4s} {f['name']}")
        print(f"\n{len(MANIFEST)} figures "
              f"({sum(f['tier']=='local' for f in MANIFEST)} local, "
              f"{sum(f['tier']=='network' for f in MANIFEST)} network, "
              f"{sum(f['tier']=='cluster' for f in MANIFEST)} cluster, "
              f"{sum(f['tier']=='external' for f in MANIFEST)} external)")
        return

    nbuilt = nskip = nfail = 0
    for f in MANIFEST:
        nm, tier, steps = f["name"], f["tier"], f["steps"]
        if A.only and nm != A.only:
            continue
        miss = [rel for rel, _ in f["inputs"]
                if not rel.startswith("__") and not have(rel)]
        presence_only = (not steps) or (tier == "network" and not A.network)
        if presence_only:
            ok = (FIG / nm).exists()
            print(f"[{'have' if ok else 'MISS'}] {nm} ({tier}): "
                  + ("committed artifact present" if ok else "ABSENT -- see REPRODUCE.md"))
            nskip += 1
            continue
        if miss:
            print(f"[skip] {nm} ({tier}): inputs absent {miss}")
            nskip += 1
            if tier == "local":
                nfail += 1     # a missing committed input is a gap, not a skip
            continue
        ok = True
        for argv in steps:
            cp = run(argv)
            if cp.returncode != 0:
                ok = False
                print(f"[FAIL] {nm} ({tier}): step failed: {' '.join(str(x) for x in argv[:3])}...")
                print((cp.stderr or "")[-600:])
                break
        if ok and (FIG / nm).exists():
            print(f"[ok]   {nm} ({tier}): built")
            nbuilt += 1
        else:
            print(f"[FAIL] {nm} ({tier}): not produced")
            nfail += 1

    print(f"\nbuilt {nbuilt}, skipped/presence {nskip}, failed {nfail}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
