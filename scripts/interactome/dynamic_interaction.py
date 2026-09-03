#!/usr/bin/env python3.11
"""Dynamic (depth-T) coupling vs the static J as a protein-interaction
predictor.

Instead of scoring a COG pair by the bare coupling J_ij, score it by the
denoiser's *response* -- how much removing gene i changes the denoised belief in
gene j, after T relaxation sweeps, in the species' own genome context:

  R_ij(T) = x_j^T(genome) - x_j^T(genome with gene i removed)

symmetrised. The response integrates J, the gates, and the INDIRECT paths
(i->k->j) the static J cannot see, and it is context-dependent. Sweeping the
relaxation depth shows how the effective interaction builds up: T=1 is the gated
J (one mean-field step), larger T folds in propagation.

The PLAIN pairwise model (no attention head) isolates the J-driven dynamics and
lets thousands of single-gene knockouts batch cheaply.

Every pair of present COGs is scored by R(T) and, as the baseline, by the static
J_ij, both evaluated (AUROC) against the same COG-pair-level STRING labels, so
the comparison to the static-J result is like for like.

Usage (needs a GPU):
  .venv/bin/python scripts/interactome/dynamic_interaction.py --taxon 511145 \
      --name "E. coli" --depths 1 2 3 5 8 12 20
Output: data/interactome/results_dyn/<taxon>.json  (AUROC vs depth, + J baseline)
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # scripts/
from analyze_ancestral_node import build_model_for_ckpt  # noqa: E402
from ising_denoiser.data import load_feathers  # noqa: E402
from ising_denoiser.modules import load_module_matrix  # noqa: E402
from ising_denoiser.training import strip_compile_prefix  # noqa: E402

STRDIR = "data/interactome/string"
OUTDIR = "data/interactome/results_dyn"
PLAIN_CKPT = "gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth"


def species_present(taxon, cog2idx):
    """Present COG indices for a species (any mapped protein) + protein->cog."""
    p2c = {}
    with open(os.path.join(STRDIR, "%s.protein_to_cog.tsv" % taxon)) as f:
        next(f)
        for line in f:
            p, cog = line.split("\t")[:2]
            if cog in cog2idx:
                p2c[p] = cog2idx[cog]
    present = sorted(set(p2c.values()))
    return present, p2c


def cogpair_labels(taxon, p2c, present, idx_pos):
    """Max STRING experimental and combined score per (present-COG) pair."""
    n = len(present)
    lab_e = np.zeros((n, n), dtype=np.int32)
    lab_c = np.zeros((n, n), dtype=np.int32)
    gz = os.path.join(STRDIR, "%s.links.detailed.txt.gz" % taxon)
    with gzip.open(gz, "rt") as f:
        next(f)
        for line in f:
            q = line.split()
            ci, cj = p2c.get(q[0]), p2c.get(q[1])
            if ci is None or cj is None or ci == cj:
                continue
            a, b = idx_pos.get(ci), idx_pos.get(cj)
            if a is None or b is None:
                continue
            e, c = int(q[6]), int(q[9])  # experimental, combined channels
            if e > lab_e[a, b]:
                lab_e[a, b] = lab_e[b, a] = e
            if c > lab_c[a, b]:
                lab_c[a, b] = lab_c[b, a] = c
    return lab_e, lab_c


def pair_metrics(score, targets):
    """AUROC/AUPRC of a score vs each {experimental, combined} target at >=700,900."""
    out = {}
    for tag, y in targets.items():
        for thr in (700, 900):
            yb = (y >= thr).astype(np.int8)
            if 0 < yb.sum() < yb.size:
                out["AUROC_%s_%d" % (tag, thr)] = float(roc_auc_score(yb, score))
                out["AUPRC_%s_%d" % (tag, thr)] = float(average_precision_score(yb, score))
    return out


@torch.no_grad()
def traj_present(model, X, M_mod, M_sizes, present, chunk=32):
    """Full relaxation trajectory at the present genes: (T, B, n_present).

    One forward with return_trajectory gives m^1..m^T, so the whole depth sweep
    comes from a single pass. The per-step gates are sized to the native T, so
    model.T must NOT be shrunk to read a shallower depth."""
    outs = []
    for s in range(0, X.size(0), chunk):
        out = model(X[s:s + chunk], M_mod=M_mod, M_sizes=M_sizes, return_trajectory=True)
        tr = out[2][:, :, present].float().cpu()   # (T, chunk, n_present)
        outs.append(tr)
    return torch.cat(outs, dim=1)                  # (T, B, n_present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20])
    ap.add_argument("--J", default="data/interactome/J_plain_pairwise_T20.npy")
    ap.add_argument("--cog-order", default="data/interactome/cog_order.txt")
    ap.add_argument("--ckpt", default=PLAIN_CKPT)
    ap.add_argument("--vocab-feather", default="data/COG_train1_phylum.feather")
    ap.add_argument("--module-matrix", default="data/module_matrix_kegg.pt")
    ap.add_argument("--chunk", type=int, default=32, help="batch chunk (use ~8 for the HO/attention model)")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _ho")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cogs = [l.strip() for l in open(a.cog_order) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    N = len(cogs)
    J = np.load(a.J).astype(np.float32)

    _, M_mod, M_sizes = load_module_matrix(a.module_matrix, dev, H=1000)
    n_modules = M_mod.shape[1]
    sd = strip_compile_prefix(torch.load(a.ckpt, map_location=dev, weights_only=True))
    model, cls_kind, T0, _ = build_model_for_ckpt(sd, dev, N, n_modules, a.ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("loaded %s (class=%s, native T=%d) on %s" % (a.ckpt, cls_kind, T0, dev))

    present, p2c = species_present(a.taxon, cog2idx)
    idx_pos = {c: a_ for a_, c in enumerate(present)}
    n = len(present)
    print("%s (taxon %s): %d present COGs -> %d COG pairs" % (a.name, a.taxon, n, n * (n - 1) // 2))

    g = -torch.ones(N, device=dev)                 # species genome, +1 present / -1 absent
    g[present] = 1.0
    iu = np.triu_indices(n, k=1)

    # Static-J baseline, on the same present-COG pairs as the response score.
    pidx = np.array(present)
    s_J = J[np.ix_(pidx, pidx)][iu]

    # Targets: the experimental channel is non-circular for a co-occurrence
    # predictor; the combined channel is the target ProteomeLM reports against.
    lab_e, lab_c = cogpair_labels(a.taxon, p2c, present, idx_pos)
    targets = {"exp": lab_e[iu], "comb": lab_c[iu]}

    # Knockout batch: row k is g with present COG k removed.
    present_t = torch.tensor(present, device=dev)
    Xko = g.unsqueeze(0).repeat(n, 1)
    Xko[torch.arange(n, device=dev), present_t] = -1.0

    res = {"taxon": a.taxon, "name": a.name, "n_present": n, "n_pairs": int(targets["exp"].size),
           "native_T": T0, "model": cls_kind, "depths": a.depths,
           "n_pos_exp_700": int((targets["exp"] >= 700).sum()),
           "n_pos_comb_900": int((targets["comb"] >= 900).sum())}
    res["J"] = pair_metrics(s_J, targets)

    trb = traj_present(model, g.unsqueeze(0), M_mod, M_sizes, present_t, chunk=a.chunk)[:, 0, :]
    trk = traj_present(model, Xko, M_mod, M_sizes, present_t, chunk=a.chunk)
    Tnat = trb.shape[0]

    res["dyn"] = {}
    for T in a.depths:
        if T > Tnat:
            continue
        xb = trb[T - 1]
        R = (xb.unsqueeze(0) - trk[T - 1]).numpy()       # (n_i, n_j): drop in j when i removed
        R = 0.5 * (R + R.T)
        res["dyn"][str(T)] = pair_metrics(R[iu], targets)
        print("  T=%2d: exp AUROC=%.3f  comb AUROC=%.3f   (J baseline: exp=%.3f comb=%.3f)"
              % (T, res["dyn"][str(T)].get("AUROC_exp_700", float("nan")),
                 res["dyn"][str(T)].get("AUROC_comb_900", float("nan")),
                 res["J"].get("AUROC_exp_700", float("nan")),
                 res["J"].get("AUROC_comb_900", float("nan"))))

    outp = os.path.join(OUTDIR, "%s%s.json" % (a.taxon, a.tag))
    json.dump(res, open(outp, "w"), indent=2)
    print("saved", outp)


if __name__ == "__main__":
    main()
