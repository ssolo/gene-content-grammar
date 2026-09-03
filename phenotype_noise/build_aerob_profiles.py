#!/usr/bin/env python3
"""Build full 4789-COG presence profiles for the aerobicity-dataset genomes, in
the denoiser's COG order, so the denoiser can be used as a front-end to the
phenotype task.

The aerobicity predictor exposes only 2677 COG features (all_gene_annotations.tsv
/ x_feature_cogs.txt). The other ~2112 COGs of each genome are taken from the
denoiser training feathers (COG_train1 + COG_val1), which carry the full 4789-COG
counts per GTDB accession, so the denoiser sees a real genome context rather than
a depleted subset.

Outputs phenotype_noise/aerob_profiles.npz:
  acc       (G,)       accession strings
  prof      (G, 4789)  presence in {0, 1}, denoiser COG order (mm['cog_names'])
  covered   (G,)       bool; False -> the genome is in neither feather, so only
                       the 2677 feature COGs are known and the rest are 0
  cog_names (4789,)    the denoiser COG order
  feat_idx  (2677,)    index into cog_names for each feature COG (X column order)
"""
import os
import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATHERS = [f'{REPO}/data/COG_train1_phylum.feather', f'{REPO}/data/COG_val1_phylum.feather']
MM = f'{REPO}/data/module_matrix_kegg.pt'


def main():
    mm = torch.load(MM, weights_only=False)
    cog_names = list(mm['cog_names'])           # (4789,) denoiser COG order
    pos = {c: i for i, c in enumerate(cog_names)}
    N = len(cog_names)

    feat_cogs = [l.strip() for l in open(f'{REPO}/phenotype_noise/x_feature_cogs.txt') if l.strip()]
    # feat_idx[j] = denoiser slot for X-column j, or -1 if that feature COG is
    # outside the denoiser universe (2 of 2677); those columns pass through undenoised.
    feat_idx = np.array([pos.get(c, -1) for c in feat_cogs], dtype=np.int64)
    n_out = int((feat_idx < 0).sum())
    print(f'denoiser COGs={N}  feature COGs={len(feat_cogs)}  '
          f'outside-universe (passthrough)={n_out}')

    aer = pd.read_csv(f'{REPO}/phenotype_noise/aerob_annot_with_taxonomy.csv')
    acc = list(dict.fromkeys(aer['accession']))                       # unique, order-stable
    aidx = {a: i for i, a in enumerate(acc)}
    G = len(acc)
    prof = np.zeros((G, N), dtype=np.int8)
    covered = np.zeros(G, dtype=bool)

    for fp in FEATHERS:
        df = pd.read_feather(fp)
        cogcols = [c for c in df.columns if c.startswith('COG')]
        col_slot = np.array([pos[c] for c in cogcols], dtype=np.int64)  # feather col -> denoiser slot
        want = df['accession'].isin(aidx)
        sub = df[want]
        vals = (sub[cogcols].to_numpy() > 0).astype(np.int8)           # counts -> presence
        rows = np.array([aidx[a] for a in sub['accession']], dtype=np.int64)
        prof[rows[:, None], col_slot[None, :]] = vals
        covered[rows] = True
        print(f'{fp.split("/")[-1]}: matched {len(rows)} aerob genomes  (cumulative covered {covered.sum()}/{G})')

    np.savez_compressed(f'{REPO}/phenotype_noise/aerob_profiles.npz',
                        acc=np.array(acc), prof=prof, covered=covered,
                        cog_names=np.array(cog_names), feat_idx=feat_idx)
    print(f'wrote aerob_profiles.npz  covered={covered.sum()}/{G} ({100*covered.mean():.1f}%)  '
          f'mean genome mass (covered) = {prof[covered].sum(1).mean():.0f} COGs')


if __name__ == '__main__':
    main()
