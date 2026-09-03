#!/usr/bin/env python3.11
"""Extract the learned pairwise coupling matrix J for the ProteomeLM-style
interactome benchmark.

Two coupling matrices are written, each an ensemble mean over the
phylum-held-out splits:

  * plain pairwise T20, from the NoHidden pairwise chain (no higher-order
    attention head).
  * higher-order T20, the J term that co-exists with the higher-order head.

J is read from the checkpoint key 'J', symmetrised as (J + J^T)/2 and given a
zero diagonal.  Row/column order is the COG column order of the vocabulary
feather, written out so the benchmark can map COG ids to J indices.

Outputs (data/interactome/):
  J_plain_pairwise_T20.npy   (N, N) float32
  J_ho_T20.npy               (N, N) float32
  cog_order.txt              N lines, COG id per row/col of J
  extract_J_meta.json        checkpoint counts and coupling summary statistics
"""
import glob
import json
import os

import numpy as np
import torch

VOCAB_FEATHER = "data/COG_train1_phylum.feather"
PLAIN_GLOB = "gsd_results_nohidden_finetune_chain_T8to20_split*/model_T16to20_f*.pth"
HO_GLOB = "gsd_results_higher_order_nohidden_T20_split*/model_ho3.pth"
OUT_DIR = "data/interactome"


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model", sd.get("state_dict", sd))


def mean_J(glob_pat, expect_attn):
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit("no checkpoints match %s" % glob_pat)
    acc = None
    used = []
    for p in paths:
        sd = load_sd(p)
        if "J" not in sd:
            print("  [skip] %s has no J" % p)
            continue
        has_attn = any("attn" in k for k in sd)
        if has_attn != expect_attn:
            print("  [warn] %s attn=%s (expected %s)" % (p, has_attn, expect_attn))
        J = sd["J"].float().numpy()
        acc = J.copy() if acc is None else acc + J
        used.append(p)
    J = acc / len(used)
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return J.astype(np.float32), used


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # J's gene axis is the feather's COG column order; nothing downstream
    # re-sorts it, and the benchmark joins on cog_order.txt.
    import pandas as pd
    cols = list(pd.read_feather(VOCAB_FEATHER).columns)
    cogs = [c for c in cols if c.startswith("COG")]
    print("vocab feather %s -> %d COG columns (of %d total)" % (VOCAB_FEATHER, len(cogs), len(cols)))

    meta = {"n_cog": len(cogs)}

    print("PLAIN pairwise T20:")
    J_plain, used_plain = mean_J(PLAIN_GLOB, expect_attn=False)
    print("  ensembled %d checkpoints, J shape %s" % (len(used_plain), J_plain.shape))
    meta["plain_n_ckpt"] = len(used_plain)
    meta["plain_ckpts"] = used_plain

    print("PRODUCTION HO T20:")
    J_ho, used_ho = mean_J(HO_GLOB, expect_attn=True)
    print("  ensembled %d checkpoints, J shape %s" % (len(used_ho), J_ho.shape))
    meta["ho_n_ckpt"] = len(used_ho)
    meta["ho_ckpts"] = used_ho

    assert J_plain.shape == (len(cogs), len(cogs)), "J/COG size mismatch (plain)"
    assert J_ho.shape == (len(cogs), len(cogs)), "J/COG size mismatch (ho)"

    np.save(os.path.join(OUT_DIR, "J_plain_pairwise_T20.npy"), J_plain)
    np.save(os.path.join(OUT_DIR, "J_ho_T20.npy"), J_ho)
    with open(os.path.join(OUT_DIR, "cog_order.txt"), "w") as f:
        f.write("\n".join(cogs) + "\n")

    for name, J in [("plain", J_plain), ("ho", J_ho)]:
        off = J[~np.eye(J.shape[0], dtype=bool)]
        meta["%s_J_abs_mean" % name] = float(np.abs(off).mean())
        meta["%s_J_std" % name] = float(off.std())
        meta["%s_J_frac_pos" % name] = float((off > 0).mean())
    iu = np.triu_indices(J_plain.shape[0], k=1)
    meta["corr_plain_ho_upper"] = float(np.corrcoef(J_plain[iu], J_ho[iu])[0, 1])

    with open(os.path.join(OUT_DIR, "extract_J_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved J_plain, J_ho, cog_order.txt, meta to %s" % OUT_DIR)
    print("plain<->ho upper-tri corr = %.3f" % meta["corr_plain_ho_upper"])


if __name__ == "__main__":
    main()
