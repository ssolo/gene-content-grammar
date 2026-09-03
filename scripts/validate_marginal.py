#!/usr/bin/env python3
"""GLD/Count trimming validation of a marginal denoiser.

The uniform-FP spectra do not test trimming; the dense GLD/Count ancestral input
does, being the over-rich reconstruction the marginal curriculum is meant to
prune. For a marginal model with a pinned split set, this denoises each dense
input and records the verdict `TRIMS` (output < input) or `INFLATES`. The split
set, glob, command and counts go to a JSON so any reported number can be
re-derived; exits non-zero if a dense input is inflated.

Requires the per-split checkpoints:
  python scripts/validate_marginal.py --model bac-FT-fp-marginal-hq --splits 1-6 --gld lbca
  python scripts/validate_marginal.py --model mix-FT-fp-marginal-hq --splits 1-3 --gld laca
  # add-hidden checkpoints: --ckpt-name model_ah3.pth
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model name -> checkpoint path template ({s} = split-glob, filled from --splits)
MODELS = {
    "bac-FT-fp-marginal-hq": "gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split{s}/model_ho3.pth",
    "mix-FT-fp-marginal-hq": "gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split{s}/model_ho3.pth",
    "bac-FT-fp-marginal-ah-hq": "gsd_results_addhidden_T20_bac_fp_marginal_ah_hq_split{s}/model_ah3.pth",
    "mix-FT-fp-marginal-ah-hq": "gsd_results_addhidden_T20_mix_fp_marginal_ah_hq_split{s}/model_ah3.pth",
}

# dense GLD/Count inputs: domain -> [(label, table, node-column)]
GLD = {
    "lbca": [("LBCA GLD min1", "LBCA_davin_sl_min1_input.tsv", "LBCA_davin_sl_min1"),
             ("LBCA GLD min4", "LBCA_davin_sl_min4_input.tsv", "LBCA_davin_sl_min4")],
    "laca": [("LACA GLD min1", "LACA_GLD_min1_input.tsv", "LACA_GLD_min1"),
             ("LACA GLD min4", "LACA_GLD_min4_input.tsv", "LACA_GLD_min4")],
}


def splits_to_glob(s):
    """'1-3' -> '[1-3]'; '1,2,3' -> '[123]'; '5' -> '[5]'. '*' is refused: the set
    of checkpoints it would match depends on what happens to be on disk."""
    s = s.strip()
    if s == "*":
        sys.exit("refuse --splits '*': pin the split set explicitly for reproducibility")
    if "-" in s:
        return "[" + s + "]"
    if "," in s:
        return "[" + s.replace(",", "") + "]"
    return "[" + s + "]"


def present(out_tsv):
    d = pd.read_csv(out_tsv, sep="\t")
    col = "mean_actual" if "mean_actual" in d.columns else d.columns[2]
    return int((d.input_prob > 0.5).sum()), int((d[col] >= 0.5).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--splits", required=True, help='PINNED, e.g. "1-3" or "1-6". Never "*".')
    ap.add_argument("--gld", choices=["lbca", "laca", "both"], default="both")
    ap.add_argument("--ckpt-name", default=None, help="override checkpoint filename")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dense-thresh", type=int, default=1800,
                    help="input present-count above which the model MUST trim; below "
                         "this, building a sparse input is correct rescue, not failure.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tmpl = MODELS[a.model]
    if a.ckpt_name:
        tmpl = os.path.join(os.path.dirname(tmpl), a.ckpt_name)
    glob_pat = os.path.join(HERE, tmpl.format(s=splits_to_glob(a.splits)))
    ckpts = sorted(glob.glob(glob_pat))
    if not ckpts:
        sys.exit("no checkpoints match %s" % glob_pat)

    targets = ["lbca", "laca"] if a.gld == "both" else [a.gld]
    rows = []
    for dom in targets:
        for label, table, col in GLD[dom]:
            tpath = os.path.join(HERE, table)
            if not os.path.exists(tpath):
                print("[skip] %s: %s missing" % (label, table))
                continue
            out_tsv = "/tmp/val_%s_%s.tsv" % (a.model, col)
            cmd = [sys.executable, os.path.join(HERE, "scripts", "analyze_ancestral_node.py"),
                   "--table", tpath, "--node", col, "--models", glob_pat,
                   "--actual-mode", "raw", "--device", a.device,
                   "--csv-out", out_tsv, "--output", os.devnull]
            print("RUN:", " ".join(cmd))
            if subprocess.run(cmd).returncode != 0:
                print("[FAIL] %s" % label)
                continue
            n_in, n_out = present(out_tsv)
            verdict = "TRIMS" if n_out < n_in else "INFLATES"
            rows.append(dict(label=label, table=table, node=col,
                             input=n_in, output=n_out, delta=n_out - n_in, verdict=verdict,
                             dense=n_in > a.dense_thresh))
            print("  %-14s %d -> %d   %s (%+d)" % (label, n_in, n_out, verdict, n_out - n_in))

    summary = dict(model=a.model, splits=a.splits, n_checkpoints=len(ckpts),
                   ckpt_name=a.ckpt_name or "model_ho3.pth", glob=glob_pat,
                   checkpoints=[os.path.relpath(c, HERE) for c in ckpts],
                   date=datetime.datetime.now().isoformat(timespec="seconds"), results=rows)
    out = a.out or os.path.join(HERE, "validation_%s_split%s.json" % (a.model, a.splits.replace("-", "to")))
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s  (model=%s splits=[%s] n_ckpt=%d)" % (out, a.model, a.splits, len(ckpts)))
    bad = [r["label"] for r in rows if r["verdict"] == "INFLATES" and r["dense"]]
    if bad:
        print("VALIDATION FAIL: model INFLATES dense (>%d) GLD input(s): %s"
              % (a.dense_thresh, ", ".join(bad)))
        sys.exit(3)
    print("VALIDATION PASS: trims every dense (>%d) GLD input." % a.dense_thresh)


if __name__ == "__main__":
    main()
