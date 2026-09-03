#!/usr/bin/env python3.11
"""Score protein-protein interactions from the higher-order head's attention,
the analogue of ProteomeLM reading interactions off its transformer's attention
maps.

The higher-order denoiser carries a 1-layer, 4-head self-attention over the COG
tokens (the beyond-pairwise correction applied at each relaxation step). It runs
flash SDPA, which never materialises the attention matrix, so a forward hook on
qkv_proj captures the packed QKV and A = softmax(Q K^T / sqrt(d_head)) is
recomputed per relaxation step, then averaged over steps. Attention is evaluated
on the species' own genome, so the scores are context-dependent as ProteomeLM's
are.

Each pair of present COGs is scored by the symmetrised attention, per head and
aggregated (mean/max over heads), and reported as AUROC/AUPRC against the STRING
experimental (non-circular) and combined targets, alongside the static-J
baseline.

Usage:
  python scripts/interactome/attention_interaction.py --taxon 511145 --name "E. coli"
Output: data/interactome/results_attn/<taxon>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_interaction import species_present, cogpair_labels, pair_metrics  # noqa: E402
from analyze_ancestral_node import build_model_for_ckpt  # noqa: E402
from ising_denoiser.modules import load_module_matrix  # noqa: E402
from ising_denoiser.training import strip_compile_prefix  # noqa: E402

OUTDIR = "data/interactome/results_attn"
HO_CKPT = "gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--ckpt", default=HO_CKPT)
    ap.add_argument("--J", default="data/interactome/J_plain_pairwise_T20.npy")
    ap.add_argument("--cog-order", default="data/interactome/cog_order.txt")
    ap.add_argument("--vocab-feather", default="data/COG_train1_phylum.feather")
    ap.add_argument("--module-matrix", default="data/module_matrix_kegg.pt")
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
    assert cls_kind == "nohidden_higher_order", "need the higher-order (attention) model, got %s" % cls_kind
    layer = model.attn.encoder.layers[0]
    nhead, hd = layer.nhead, layer.head_dim
    print("loaded %s (class=%s, T=%d, nhead=%d, d_head=%d)" % (a.ckpt, cls_kind, T0, nhead, hd))

    present, p2c = species_present(a.taxon, cog2idx)
    idx_pos = {c: i for i, c in enumerate(present)}
    n = len(present)
    present_t = torch.tensor(present, device=dev)
    iu = np.triu_indices(n, k=1)
    print("%s (taxon %s): %d present COGs" % (a.name, a.taxon, n))

    lab_e, lab_c = cogpair_labels(a.taxon, p2c, present, idx_pos)
    targets = {"exp": lab_e[iu], "comb": lab_c[iu]}

    g = -torch.ones(N, device=dev)
    g[present_t] = 1.0

    # The hook fires once per relaxation step, so len(captured) == T.
    captured = []
    hk = layer.qkv_proj.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    with torch.no_grad():
        model(g.unsqueeze(0), M_mod=M_mod, M_sizes=M_sizes)
    hk.remove()
    print("captured attention at %d relaxation steps" % len(captured))

    Aacc = torch.zeros(nhead, n, n, device=dev)
    with torch.no_grad():
        for qkv in captured:
            B, Nn, _ = qkv.shape
            qk = qkv.view(B, Nn, 3, nhead, hd).permute(2, 0, 3, 1, 4)   # (3,B,nhead,N,hd)
            q, k = qk[0, 0], qk[1, 0]                                   # (nhead, N, hd)
            A = torch.softmax((q @ k.transpose(-1, -2)) / (hd ** 0.5), dim=-1)  # (nhead,N,N)
            Aacc += A.index_select(1, present_t).index_select(2, present_t)
    Aacc /= len(captured)
    Asym = (0.5 * (Aacc + Aacc.transpose(-1, -2))).cpu().numpy()        # (nhead, n, n)

    res = {"taxon": a.taxon, "name": a.name, "n_present": n, "n_pairs": int(targets["exp"].size),
           "model": cls_kind, "nhead": nhead}
    res["J"] = pair_metrics(J[np.ix_(np.array(present), np.array(present))][iu], targets)
    for h in range(nhead):
        res["head_%d" % h] = pair_metrics(Asym[h][iu], targets)
    res["mean_heads"] = pair_metrics(Asym.mean(0)[iu], targets)
    res["max_heads"] = pair_metrics(Asym.max(0)[iu], targets)

    best_exp = max([res["head_%d" % h]["AUROC_exp_700"] for h in range(nhead)]
                   + [res["mean_heads"]["AUROC_exp_700"], res["max_heads"]["AUROC_exp_700"]])
    print("  J(exp700)=%.3f | attn best(exp700)=%.3f mean-heads=%.3f | J(comb900)=%.3f attn mean(comb900)=%.3f"
          % (res["J"]["AUROC_exp_700"], best_exp, res["mean_heads"]["AUROC_exp_700"],
             res["J"]["AUROC_comb_900"], res["mean_heads"]["AUROC_comb_900"]))
    for h in range(nhead):
        print("    head %d: exp700=%.3f comb900=%.3f" % (h, res["head_%d" % h]["AUROC_exp_700"],
                                                         res["head_%d" % h]["AUROC_comb_900"]))
    json.dump(res, open(os.path.join(OUTDIR, "%s.json" % a.taxon), "w"), indent=2)
    print("saved", os.path.join(OUTDIR, "%s.json" % a.taxon))


if __name__ == "__main__":
    main()
