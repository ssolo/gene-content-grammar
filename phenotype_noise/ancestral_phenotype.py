#!/usr/bin/env python3
"""Denoise the C65 ancestral gene-content reconstruction before predicting an
ancestral phenotype, instead of thresholding the reconstruction directly.

The reconstruction (TableAncestralRoot1_aggregated_byCOG.tsv, 2852 COGs x 2013
nodes = leaves plus internal ancestral nodes, root 2012) is a noisy per-node
gene-content estimate. The raw arm thresholds it (--thr), as the reference
implementation does; the denoised arm pushes the same thresholded presence
through the HQ-marginal cross-input-consistent ensemble. Both arms are scored by
the same aerobicity predictor.

Denoising and prediction are decoupled so only the denoise pass needs a GPU:
  python3 phenotype_noise/ancestral_phenotype.py --nodes all --device cuda \
              --save-features data/ancestral_feats.npz
  python3 phenotype_noise/ancestral_phenotype.py --load-features data/ancestral_feats.npz

Validation on the labeled leaves is leak-free: the predictor is fit on extant
genomes with those leaf accessions held out.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PH = os.path.join(REPO, 'phenotype_noise')
ENS = 'gsd_results_consistency_T20_mix_fp_marginal_cons_l1.0_j1.0_hq_split{s}/model_ho3.pth'


def denoise_nodes(A):
    """Raw and denoised feature presence for every requested C65 node.

    Returns (Xraw, Xden, nodes, short, label): two (nodes, 2677) presence
    matrices in the predictor's feature-COG order, the node names, their short
    codes, and the leaf label (1 aerobe, 0 anaerobe, -1 where no oxytolerance is
    known, which covers every internal ancestral node).
    """
    sys.path.insert(0, os.path.join(REPO, 'scripts')); sys.path.insert(0, PH)
    from analyze_ancestral_node import build_model_for_ckpt
    from ising_denoiser.modules import load_module_matrix
    from ising_denoiser.training import strip_compile_prefix
    dev = torch.device(A.device)
    mm = torch.load(os.path.join(REPO, 'data/module_matrix_kegg.pt'), weights_only=False)
    cog_names = list(mm['cog_names']); pos = {c: i for i, c in enumerate(cog_names)}; N = len(cog_names)
    feat = [l.strip() for l in open(os.path.join(PH, 'x_feature_cogs.txt')) if l.strip()]
    feat_slot = np.array([pos.get(c, -1) for c in feat]); fvalid = feat_slot >= 0

    df = pd.read_csv(os.path.join(PH, 'ancestral/TableAncestralRoot1_aggregated_byCOG.tsv'),
                     sep='\t', index_col='COG')
    gi = pd.read_csv(os.path.join(PH, 'ancestral/GenomesInfo.csv'))
    gi = gi[gi['oxytolerance'].isin(['aerobe', 'anaerobe'])]
    leaf_lab = dict(zip(gi['ShortCode'], (gi['oxytolerance'] == 'aerobe').astype(int)))
    nodes = list(df.columns); short = [c.split('(')[0] for c in nodes]
    if A.nodes == 'labeled':
        keep = [i for i, s in enumerate(short) if s in leaf_lab]
        nodes = [nodes[i] for i in keep]; short = [short[i] for i in keep]
    print(f'denoising {len(nodes)} nodes ({A.nodes}) on {dev}', flush=True)

    # The denoiser has no missing-data channel: COGs outside the reconstruction
    # table stay 0 and so enter the model as absent.
    anc_cogs = [c for c in df.index if c in pos]
    anc_slot = np.array([pos[c] for c in anc_cogs])
    rec = np.zeros((len(nodes), N), dtype=np.float32)
    rec[:, anc_slot] = df.loc[anc_cogs, nodes].to_numpy().T
    raw_pres = (rec > A.thr).astype(np.float32)
    x4789 = 2.0 * raw_pres - 1.0                               # (nodes, N) in {-1, +1}

    _, M_mod, M_sizes = load_module_matrix(os.path.join(REPO, 'data/module_matrix_kegg.pt'), dev, 1000)
    n_mod = M_mod.shape[1]
    models = []
    for s in range(1, A.ensemble + 1):
        ck = os.path.join(REPO, ENS.format(s=s))
        sd = strip_compile_prefix({k.replace('module.', ''): v
                                   for k, v in torch.load(ck, map_location=dev, weights_only=True).items()})
        m, _, _, _ = build_model_for_ckpt(sd, dev, N, n_mod, ck); m.load_state_dict(sd, strict=False); m.eval()
        models.append(m)
    post = np.zeros((len(nodes), N), dtype=np.float32); xt = torch.from_numpy(x4789)
    with torch.no_grad():
        for b in range(0, len(nodes), A.batch):
            xb = xt[b:b + A.batch].to(dev)
            acc = torch.zeros(xb.shape[0], N, device=dev)
            # Magnetisation m in (-1, 1) -> presence probability (m + 1)/2,
            # averaged over the ensemble.
            for m in models:
                acc += ((m(xb, M_mod, M_sizes)[0] + 1) * 0.5).clamp(0, 1)
            post[b:b + A.batch] = (acc / len(models)).cpu().numpy()
            print(f'  denoised {min(b+A.batch,len(nodes))}/{len(nodes)}', flush=True)

    # Feature COGs outside the denoiser's COG universe have no denoised value;
    # both arms carry absence there, so they differ only where the denoiser acts.
    Xraw = raw_pres[:, feat_slot.clip(min=0)].copy(); Xraw[:, ~fvalid] = 0.0
    Xden = (post[:, feat_slot.clip(min=0)] > 0.5).astype(np.float32); Xden[:, ~fvalid] = Xraw[:, ~fvalid]
    label = np.array([leaf_lab.get(s, -1) for s in short])
    return Xraw, Xden, np.array(nodes), np.array(short), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', choices=['labeled', 'all'], default='labeled')
    ap.add_argument('--ensemble', type=int, default=10)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--save-features', default=None, help='denoise-only: write features npz and exit')
    ap.add_argument('--load-features', default=None, help='predict from a saved features npz')
    A = ap.parse_args()
    os.makedirs(os.path.join(PH, 'results'), exist_ok=True)

    if A.load_features:
        Z = np.load(A.load_features, allow_pickle=True)
        Xraw, Xden, nodes, short, label = Z['Xraw'], Z['Xden'], Z['nodes'], Z['short'], Z['label']
        print(f'loaded {Xraw.shape[0]} nodes from {A.load_features}')
    else:
        Xraw, Xden, nodes, short, label = denoise_nodes(A)
        if A.save_features:
            np.savez_compressed(A.save_features, Xraw=Xraw, Xden=Xden,
                                nodes=nodes, short=short, label=label)
            print(f'saved features -> {A.save_features} ({Xraw.shape[0]} nodes)')
            return

    # Predict, with the labeled leaf genomes held out of the training set.
    from xgboost import XGBClassifier
    from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score
    feat = [l.strip() for l in open(os.path.join(PH, 'x_feature_cogs.txt')) if l.strip()]
    gi0 = pd.read_csv(os.path.join(PH, 'ancestral/GenomesInfo.csv'))
    leaf_acc = set(gi0[gi0['oxytolerance'].isin(['aerobe', 'anaerobe'])]['accession'])
    ann = pd.read_csv(os.path.join(PH, 'all_gene_annotations.tsv'), sep='\t')
    aer = pd.read_csv(os.path.join(PH, 'aerob_annot_with_taxonomy.csv')).drop_duplicates('accession')
    lab = dict(zip(aer['accession'], (aer['oxytolerance'] == 'aerobe').astype(int)))
    tr = ann[ann['accession'].isin(lab) & ~ann['accession'].isin(leaf_acc)]
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.7, colsample_bytree=0.7, tree_method='hist')
    clf.fit((tr[feat].to_numpy() > 0).astype(np.float32), tr['accession'].map(lab).to_numpy())
    print(f'predictor trained on {len(tr)} extant genomes (leaves held out)')

    p_raw = clf.predict_proba(Xraw)[:, 1]; p_den = clf.predict_proba(Xden)[:, 1]
    out = pd.DataFrame({'node': nodes, 'short': short, 'label': label,
                        'p_raw': p_raw, 'p_den': p_den,
                        'pred_raw': (p_raw > 0.5).astype(int), 'pred_den': (p_den > 0.5).astype(int)})
    out.to_csv(os.path.join(PH, 'results/ancestral_pred.csv'), index=False)

    m = label >= 0
    if m.any():
        yl = label[m]
        print(f'\n=== VALIDATION on {int(m.sum())} labeled leaves '
              f'({int(yl.sum())} aerobe / {int((1-yl).sum())} anaerobe), leak-free ===')
        for nm, yh in [('raw (thresholded)', out.loc[m, 'pred_raw'].to_numpy()),
                       ('DENOISED', out.loc[m, 'pred_den'].to_numpy())]:
            print(f'  {nm:18s} acc={accuracy_score(yl,yh):.3f} '
                  f'bal_acc={balanced_accuracy_score(yl,yh):.3f} MCC={matthews_corrcoef(yl,yh):.3f}')
    anc = out[out['label'] < 0]
    if len(anc):
        flip = (anc['pred_raw'] != anc['pred_den']).mean()
        print(f'\nancestral (internal) nodes: {len(anc)}, raw->denoised flip rate = {flip:.3f}')
        for r in ['2012', '2011', '2010', '2009', '2008']:
            row = out[out['short'] == r]
            if len(row):
                print(f'  node {r}: p_raw={row.p_raw.iloc[0]:.2f} -> p_den={row.p_den.iloc[0]:.2f}')


if __name__ == '__main__':
    main()
