#!/usr/bin/env python3.11
"""TAP linear-response susceptibility as an interaction read-out, evaluated at a
DESATURATED operating point (never the deep attractor).

The static coupling J is the bare 2-body parameter; the depth-T knockout response
R(T) loses to it because it is read at the species' own attractor, where the
marginals saturate (m_i -> +-1, so 1 - m_i^2 -> 0) and the response is
multiplicatively suppressed. Evaluating the susceptibility at a desaturated
operating point instead -- one mean-field step m = tanh((h + J~ x)/tau) with a
temperature knob tau, not the deep-T fixed point -- removes that suppression.

At a fixed point m the TAP inverse susceptibility (Hessian of the TAP free
energy) is
    (chi^-1)_ij = delta_ij [ 1/(1-m_i^2) + sum_k J~_ik^2 (1-m_k^2) ]
                  - J~_ij - 2 J~_ij^2 m_i m_j .
Two read-outs:
  * direct  = off-diagonal of  -(chi^-1)  = J~_ij + 2 J~_ij^2 m_i m_j  (=~ J;
              the exact statistical-physics statement of the DCA/APC direct
              coupling -- removes indirect paths).
  * dressed = off-diagonal of  chi = (chi^-1)^-1  (the connected correlation /
              full propagator, all indirect i->k->j paths summed).
Both are APC-corrected and scored (AUROC) against STRING's experimental
(non-circular) and combined channels, alongside the static J baseline.

Operating point:
  --mode bare  : m = tanh((h + J~ x)/tau)        (one step, no gates; J/h from ckpt)
  --mode gated : m = model one-step output (T=1)  (gates/conditioner/tau on;
                 across a leave-clade-out ladder this separates functional
                 coupling, carried by the fixed J, from phylogenetic inertia,
                 which lives in the gates and moves m)

Needs a GPU, cog_order.txt and the STRING cache (no feather).
  .venv/bin/python scripts/interactome/susceptibility_interaction.py \
      --taxon 511145 --ckpt <model.pth> --tau 1 2 4 8 --mode bare --tag _main
Output: data/interactome/results_chi/<taxon><tag>.json
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
STRDIR = os.path.join(REPO, "data", "interactome", "string")
OUTDIR = os.path.join(REPO, "data", "interactome", "results_chi")
EPS = 1e-4   # clip |m| away from 1 to keep 1-m^2 > 0


def apc(S):
    # The APC background is taken from the OFF-diagonal only: a large diagonal
    # (e.g. 1/(1-m^2) in -H at a saturated operating point) would contaminate
    # the row/column means and inject a per-gene-confidence outer product into
    # the score.
    S = S.copy(); np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    fi = S.sum(1, keepdims=True) / (n - 1)
    fj = S.sum(0, keepdims=True) / (n - 1)
    f = S.sum() / (n * (n - 1))
    Sc = S - (fi * fj) / f if f != 0 else S.copy()
    Sc = 0.5 * (Sc + Sc.T); np.fill_diagonal(Sc, 0.0)
    return Sc


def species_present(taxon, cog2idx):
    p2c = {}
    with open(os.path.join(STRDIR, "%s.protein_to_cog.tsv" % taxon)) as f:
        next(f)
        for line in f:
            p, cog = line.split("\t")[:2]
            if cog in cog2idx:
                p2c[p] = cog2idx[cog]
    return sorted(set(p2c.values())), p2c


def cogpair_labels(taxon, p2c, present):
    idx = {c: a for a, c in enumerate(present)}
    n = len(present)
    Le = np.zeros((n, n), np.int32); Lc = np.zeros((n, n), np.int32)
    with gzip.open(os.path.join(STRDIR, "%s.links.detailed.txt.gz" % taxon), "rt") as f:
        next(f)
        for line in f:
            q = line.split()
            ci, cj = p2c.get(q[0]), p2c.get(q[1])
            if ci is None or cj is None or ci == cj:
                continue
            a, b = idx.get(ci), idx.get(cj)
            if a is None or b is None:
                continue
            e, c = int(q[6]), int(q[9])
            if e > Le[a, b]: Le[a, b] = Le[b, a] = e
            if c > Lc[a, b]: Lc[a, b] = Lc[b, a] = c
    return Le, Lc


def metrics(score, targets):
    out = {}
    for tag, y in targets.items():
        for thr in (700, 900):
            yb = (y >= thr).astype(np.int8)
            if 0 < yb.sum() < yb.size:
                out["AUROC_%s_%d" % (tag, thr)] = float(roc_auc_score(yb, score))
                out["AUPRC_%s_%d" % (tag, thr)] = float(average_precision_score(yb, score))
    return out


def tap_Hessian(Jt, m):
    """TAP inverse susceptibility (chi^-1), N x N, torch on device of Jt."""
    one_m2 = (1.0 - m * m).clamp_min(EPS)            # (N,)
    J2 = Jt * Jt
    H = -Jt - 2.0 * J2 * (m[:, None] * m[None, :])   # off-diagonal part
    diag = 1.0 / one_m2 + (J2 * one_m2[None, :]).sum(1)   # 1/(1-m^2) + sum_k J_ik^2(1-m_k^2)
    H.diagonal().copy_(diag)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxon", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--ckpt", required=True, help="model checkpoint (for h, gated m, and J unless --J given)")
    ap.add_argument("--J", default="", help="optional .npy override for J~ (e.g. the 10-split ensemble); h/gated still from --ckpt")
    ap.add_argument("--cog-order", default=os.path.join(REPO, "data/interactome/cog_order.txt"))
    ap.add_argument("--tau", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--mode", choices=["bare", "gated", "both"], default="bare")
    ap.add_argument("--ridge", type=float, default=1e-3, help="ridge added to chi^-1 before inversion")
    ap.add_argument("--module-matrix", default=os.path.join(REPO, "data/module_matrix_kegg.pt"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cogs = [l.strip() for l in open(a.cog_order) if l.strip()]
    cog2idx = {c: i for i, c in enumerate(cogs)}
    N = len(cogs)

    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    if a.J:
        J = torch.from_numpy(np.load(a.J)).float().to(dev)  # ensemble/override J~
    else:
        J = sd["J"].float().to(dev)
    Jt = 0.5 * (J + J.t()); Jt.fill_diagonal_(0.0)         # symmetric, hollow J~
    h = sd["h"].float().to(dev) if "h" in sd else torch.zeros(N, device=dev)

    present, p2c = species_present(a.taxon, cog2idx)
    Le, Lc = cogpair_labels(a.taxon, p2c, present)
    iu = np.triu_indices(len(present), k=1)
    targets = {"exp": Le[iu], "comb": Lc[iu]}
    pid = np.array(present)
    g = (-torch.ones(N, device=dev)); g[pid] = 1.0
    res = {"taxon": a.taxon, "name": a.name, "ckpt": os.path.basename(os.path.dirname(a.ckpt)) or a.ckpt,
           "n_present": len(present), "n_pairs": int(targets["exp"].size),
           "n_pos_exp_700": int((targets["exp"] >= 700).sum())}

    # Static-J baseline, on the same present-COG pairs as the susceptibility.
    sJ = apc(Jt.detach().cpu().numpy())[np.ix_(pid, pid)][iu]
    res["J"] = metrics(sJ, targets)
    print("%s (taxon %s): %d present, %d pairs | static J exp700 AUROC=%.3f"
          % (a.name, a.taxon, len(present), targets["exp"].size, res["J"].get("AUROC_exp_700", float("nan"))))

    @torch.no_grad()
    def gated_m():
        from analyze_ancestral_node import build_model_for_ckpt
        from ising_denoiser.modules import load_module_matrix
        from ising_denoiser.training import strip_compile_prefix
        _, M_mod, M_sizes = load_module_matrix(a.module_matrix, dev, H=1000)
        sdl = strip_compile_prefix(torch.load(a.ckpt, map_location=dev, weights_only=True))
        model, _, _, _ = build_model_for_ckpt(sdl, dev, N, M_mod.shape[1], a.ckpt)
        model.load_state_dict(sdl, strict=False); model.eval()
        out = model(g.unsqueeze(0), M_mod=M_mod, M_sizes=M_sizes, return_trajectory=True)
        return out[2][0, 0].float()                         # m^1 (one step), (N,)

    def readouts(m, label):
        H = tap_Hessian(Jt, m)
        S_dir = apc((-H).detach().cpu().numpy())[np.ix_(pid, pid)][iu]
        Hr = H + a.ridge * torch.eye(N, device=dev)
        chi = torch.linalg.inv(Hr)
        S_dre = apc(chi.detach().cpu().numpy())[np.ix_(pid, pid)][iu]
        # Control: the per-gene confidence outer product c_i c_j, carrying NO
        # coupling.  It measures the artifact that a diagonal-contaminated APC
        # would inject into the direct read-out.
        c = (1.0 / (1.0 - m * m).clamp_min(EPS)).detach().cpu().numpy()
        S_conf = (c[pid, None] * c[None, pid])[iu]
        mc = metrics(S_conf, targets)
        md, mr = metrics(S_dir, targets), metrics(S_dre, targets)
        res.setdefault(label, {})
        res[label]["direct"] = md; res[label]["dressed"] = mr; res[label]["conf_outer"] = mc
        res[label]["m_rms"] = float((m * m).mean().sqrt())
        res[label]["frac_saturated"] = float((m.abs() > 0.99).float().mean())
        print("  %-16s m_rms=%.3f sat=%.2f | direct=%.3f dressed=%.3f conf_outer=%.3f"
              % (label, res[label]["m_rms"], res[label]["frac_saturated"],
                 md.get("AUROC_exp_700", float("nan")), mr.get("AUROC_exp_700", float("nan")),
                 mc.get("AUROC_exp_700", float("nan"))))

    if a.mode in ("bare", "both"):
        for tau in a.tau:
            m = torch.tanh((h + Jt @ g) / tau)
            readouts(m, "bare_tau%g" % tau)
    if a.mode in ("gated", "both"):
        readouts(gated_m(), "gated")

    outp = os.path.join(OUTDIR, "%s%s.json" % (a.taxon, a.tag))
    json.dump(res, open(outp, "w"), indent=2)
    print("saved", outp)


if __name__ == "__main__":
    main()
