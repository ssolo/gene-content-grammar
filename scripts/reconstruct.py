#!/usr/bin/env python3.11
"""Dispatcher for genome reconstruction with the Ising gene-content denoiser.

Resolves short model/node names to checkpoints and input tables, then calls
reconstruct_extant.py (extant genomes) or analyze_ancestral_node.py (ancestral
nodes and arbitrary gene-content tables). Run --list for the full registry.

Two conventions the dispatcher enforces:
  * a model is paired with the validation feathers of its own partition, so the
    held-out split is found in the partition the model trained on (no leak);
  * reconstruct_extant.py receives the {split} template and fills in the split
    that holds the genome out; analyze_ancestral_node.py receives it globbed
    ('*') and ensembles over every split.

Examples:
  python scripts/reconstruct.py denoise --input examples/ecoli_gene_content.tsv --model bac-FT --out ecoli_denoised.tsv
  python scripts/reconstruct.py extant --genome "Escherichia coli" --model bac-FT --fp 0.05 --out recover_ecoli.tsv
  python scripts/reconstruct.py extant --genome median-arc --model mix-FT-fp --out recover_medarc.tsv
  python scripts/reconstruct.py ancestral --node LBCA --model bac-FT --inject-fp 0.05 --out node_LBCA.tsv
  python scripts/reconstruct.py --list
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (checkpoint template with {split}, validation partition)
MODELS = {
    "generalist": ("gsd_results_higher_order_nohidden_T20_split{split}/model_ho3.pth", "mixed"),
    "mix-FT":     ("gsd_results_higher_order_nohidden_T20_mix_split{split}/model_ho3.pth", "mixed"),
    "bac-FT":     ("gsd_results_higher_order_nohidden_T20_bac_hard_split{split}/model_ho3.pth", "bac"),
    "arc-FT":     ("gsd_results_higher_order_nohidden_T20_arc_split{split}/model_ho3.pth", "arc"),
    # -fp: trained against coherent false positives (whole foreign modules
    # grafted in, plus module-level FP).
    "mix-FT-fp":  ("gsd_results_higher_order_nohidden_T20_mix_fp_real_split{split}/model_ho3.pth", "mixed"),
    "bac-FT-fp":  ("gsd_results_higher_order_nohidden_T20_bac_fp_real_split{split}/model_ho3.pth", "bac"),
    # -hq: the same, trained only on genomes with CheckM completeness >= 90%.
    "mix-FT-fp-hq": ("gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split{split}/model_ho3.pth", "mixed"),
    "bac-FT-fp-hq": ("gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split{split}/model_ho3.pth", "bac"),
    # -marginal-hq: HQ models fine-tuned on the dense-FP curriculum that draws
    # false positives by per-COG cross-genome marginal frequency; the LBCA (bac)
    # and LACA (mix) reconstruction models.
    "bac-FT-fp-marginal-hq": ("gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split{split}/model_ho3.pth", "bac"),
    "mix-FT-fp-marginal-hq": ("gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split{split}/model_ho3.pth", "mixed"),
    # -cons-hq: marginal-HQ fine-tuned (lambda=1, j=1) so the denoised posterior
    # agrees across the GLD/count/spectra input encodings; mix side is used for
    # LACA, bac side is the LBCA control.
    "bac-FT-fp-cons-hq": ("gsd_results_consistency_T20_bac_fp_marginal_cons_l1.0_j1.0_hq_split{split}/model_ho3.pth", "bac"),
    "mix-FT-fp-cons-hq": ("gsd_results_consistency_T20_mix_fp_marginal_cons_l1.0_j1.0_hq_split{split}/model_ho3.pth", "mixed"),
    "mix-FT-fp-margarc-hq": ("gsd_results_higher_order_nohidden_T20_mix_fp_margarc_hq_split{split}/model_ho3.pth", "mixed"),
    "mix-FT-fp-margbac-hq": ("gsd_results_higher_order_nohidden_T20_mix_fp_margbac_hq_split{split}/model_ho3.pth", "mixed"),
    "mix-FT-fp-uniform": ("gsd_results_higher_order_nohidden_T20_mix_fp_split{split}/model_ho3.pth", "mixed"),
    "bac-FT-fp-uniform": ("gsd_results_higher_order_nohidden_T20_mix_fp_bac_split{split}/model_ho3.pth", "bac"),
    "plain-T20":  ("gsd_results_nohidden_finetune_chain_T8to20_split{split}/model_T16to20_f1.pth", "mixed"),
}
VALGLOB = {  # partition -> val feather template (see the no-leak convention above)
    "mixed": "data/COG_val{split}_phylum.feather",
    "bac":   "data/COG_bac_val{split}_phylum.feather",
    "arc":   "data/COG_arc_val{split}_phylum.feather",
}
NODES = {
    "LBCA": ("data/TableAncestralRoot1.tsv", None),
    "LACA": ("data/LACA_combined_table.tsv", "LACA"),                  # combined roots; the canonical LACA input
    "LACA-MHH": ("data/LACA_MHHroot_table.tsv", "Node_GCA-000007185.1_GCA-000006805.1_0"),
    "LACA-Eury": ("data/LACA_Euryroot_table.tsv", "Node_GCA-000006805.1_AP024487.1_0"),
}
GENOMES = {"median-bac": ["--select", "median", "--domain", "d__Bacteria"],
           "median-arc": ["--select", "median", "--domain", "d__Archaea"]}
# The -hq median picks restrict the median-gene-count selection to near-complete
# genomes (CheckM completeness >= 90, contamination <= 5). Without the bar the
# archaeal median is a 75%-complete MAG, and recall-vs-truth then penalises the
# model for inferring genes the assembly lacks.
_GTDB = os.environ.get("GTDB_COGS", "$DATA_ROOT/GTDB_COGs")
_HQ = ["--min-completeness", "90", "--max-contam", "5", "--gtdb-meta"]
GENOMES["median-arc-hq"] = ["--select", "median", "--domain", "d__Archaea",
                            *_HQ, f"{_GTDB}/ar53_metadata_r220.tsv"]
GENOMES["median-bac-hq"] = ["--select", "median", "--domain", "d__Bacteria",
                            *_HQ, f"{_GTDB}/bac120_metadata_r220.tsv"]


def model_or_die(name):
    if name not in MODELS:
        sys.exit("unknown --model %r; choose from: %s" % (name, ", ".join(MODELS)))
    tmpl, part = MODELS[name]
    # The archive may ship only the uniform-FP fine-tunes; fall back to the
    # *-uniform variant so the name resolves instead of failing downstream.
    if name in ("mix-FT-fp", "bac-FT-fp") and not os.path.exists(tmpl.replace("{split}", "1")):
        alt = MODELS.get(name + "-uniform")
        if alt and os.path.exists(alt[0].replace("{split}", "1")):
            print("[reconstruct] %s: realistic-FP checkpoints not found; using the "
                  "archived uniform-FP model" % name, file=sys.stderr)
            tmpl, part = alt
    return (tmpl, part)


def run(cmd):
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


def ensure_archive_layout():
    """Symlink the unpacked archive layout (data/feathers/, <family>/split<N>/)
    into the paths the scripts expect (data/<name>.feather,
    gsd_results_<family>_split<N>/...).

    Idempotent; a no-op on the repository layout.
    """
    setup = os.path.join(HERE, "..", "setup_archive.sh")
    probe = os.path.join(HERE, "..", "data", "COG_train1_phylum.feather")
    if os.path.exists(setup) and not os.path.exists(probe):
        print("[reconstruct] normalising Zenodo-archive layout (setup_archive.sh) ...",
              file=sys.stderr)
        subprocess.call(["bash", setup])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print model/node registries and exit")
    sub = ap.add_subparsers(dest="mode")

    pe = sub.add_parser("extant", help="reconstruct a held-out extant genome (corrupt + denoise vs truth)")
    pe.add_argument("--genome", required=True, help='species name (e.g. "Escherichia coli") OR median-bac / median-arc')
    pe.add_argument("--model", required=True)
    pe.add_argument("--fp", type=float, default=0.01)
    pe.add_argument("--fp-mode", choices=["uniform", "marginal"], default="uniform",
                    help="uniform random FP (default) or marginal-frequency FP "
                         "(common-gene contamination; the deep-ancestral regime)")
    pe.add_argument("--fn", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    pe.add_argument("--device", default="auto")
    pe.add_argument("--out", required=True)

    pa = sub.add_parser("ancestral", help="reconstruct an ancestral node (LBCA/LACA) from its reconciliation input")
    pa.add_argument("--node", required=True)
    pa.add_argument("--model", required=True)
    pa.add_argument("--inject-fp", type=float, default=0.0)
    pa.add_argument("--drop-low-conf", type=float, default=0.0,
                    help="FP control: drop this fraction of lowest-confidence input families")
    pa.add_argument("--input-mode", default="raw", choices=["raw", "half", "soft", "binarize"],
                    help="headline input encoding: raw (x=p), half (x=(0.5+p)/2), soft (x=2p-1), binarize (x=+1)")
    pa.add_argument("--bin-thresholds", type=float, nargs="+", default=None,
                    help="binarize-variant thresholds (t=0 -> >0, all present -> +1); "
                         "quantifies what each threshold removes. Passed through.")
    pa.add_argument("--device", default="auto")
    pa.add_argument("--out", required=True)

    # Generic entry point; the ancestral nodes above are pre-packaged inputs to it.
    pd_ = sub.add_parser("denoise",
        help="denoise YOUR OWN gene-content table (a COG column + a per-COG value column)")
    pd_.add_argument("--input", required=True,
                     help="TAB-separated file with a 'COG' header column and one or more "
                          "per-COG value columns (copy number, or presence probability in "
                          "(0,1]; COG families you omit are treated as absent). "
                          "See examples/ecoli_gene_content.tsv.")
    pd_.add_argument("--model", required=True)
    pd_.add_argument("--column", default=None,
                     help="which value column to denoise (default: the 2nd column of --input)")
    pd_.add_argument("--input-mode", default="raw", choices=["raw", "half", "soft", "binarize"],
                     help="input encoding: raw (x=value, default), binarize (present->+1), "
                          "soft (x=2p-1), half (x=(0.5+p)/2)")
    pd_.add_argument("--inject-fp", type=float, default=0.0,
                     help="optional false-positive stress test: flip this fraction of absent families on")
    pd_.add_argument("--device", default="auto")
    pd_.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.list:
        print("MODELS (name : checkpoint template : val partition):")
        [print("  %-11s %-62s %s" % (k, t, p)) for k, (t, p) in MODELS.items()]
        print("VAL feathers:"); [print("  %-6s %s" % (k, v)) for k, v in VALGLOB.items()]
        print("NODES:"); [print("  %-5s table=%s node=%s" % (k, t, n)) for k, (t, n) in NODES.items()]
        print("GENOME shortcuts: " + ", ".join(GENOMES) + ", or any species name")
        return
    ensure_archive_layout()
    if a.mode == "extant":
        tmpl, part = model_or_die(a.model)
        sel = GENOMES.get(a.genome, ["--species", a.genome])
        cmd = [sys.executable, os.path.join(HERE, "reconstruct_extant.py"),
               *sel, "--models", tmpl, "--val-glob", VALGLOB[part],
               "--profile-glob", VALGLOB["mixed"],  # fallback profile/clade source when the partition val is subsampled
               "--model-label", a.model,
               "--fp", str(a.fp), "--fp-mode", a.fp_mode,
               "--fn", *[str(x) for x in a.fn], "--device", a.device, "--out", a.out]
        sys.exit(run(cmd))
    elif a.mode == "ancestral":
        tmpl, _ = model_or_die(a.model)
        if a.node not in NODES:
            sys.exit("unknown --node %r; choose from: %s" % (a.node, ", ".join(NODES)))
        table, nodekey = NODES[a.node]
        cmd = [sys.executable, os.path.join(HERE, "analyze_ancestral_node.py"),
               "--table", table, "--models", tmpl.replace("{split}", "*"),
               "--inject-fp", str(a.inject_fp), "--drop-low-conf", str(a.drop_low_conf),
               "--actual-mode", a.input_mode, "--device", a.device,
               "--csv-out", a.out, "--output", os.path.splitext(a.out)[0] + ".txt"]
        if a.bin_thresholds:
            cmd += ["--bin-thresholds", *[str(x) for x in a.bin_thresholds]]
        if nodekey:
            cmd += ["--node", nodekey]
        sys.exit(run(cmd))
    elif a.mode == "denoise":
        tmpl, _ = model_or_die(a.model)
        if not os.path.exists(a.input):
            sys.exit("input table not found: %r" % a.input)
        col = a.column
        if col is None:
            with open(a.input) as fh:
                hdr = fh.readline().rstrip("\n").split("\t")
            if len(hdr) < 2:
                sys.exit("--input must be TAB-separated with a 'COG' column plus "
                         ">=1 value column; got header: %r" % hdr)
            col = hdr[1]
        cmd = [sys.executable, os.path.join(HERE, "analyze_ancestral_node.py"),
               "--table", a.input, "--node", col,
               "--models", tmpl.replace("{split}", "*"),
               "--inject-fp", str(a.inject_fp), "--actual-mode", a.input_mode,
               "--device", a.device, "--csv-out", a.out,
               "--output", os.path.splitext(a.out)[0] + ".txt"]
        sys.exit(run(cmd))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
