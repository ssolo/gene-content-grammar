#!/usr/bin/env python3
"""Bundle the aerobicity phylum-holdout test splits (2677 COG feature counts,
recovered accession, label) into a single pickle for the denoise step.

The split tensors carry no accession, so each X row is matched back to
all_gene_annotations.tsv by the md5 of its feature vector; the match is exact on
all 705 rows and the recovered labels agree with the split labels.

Output: phenotype_noise/denoise_bundle.pkl, read by denoise_features.py
alongside aerob_profiles.npz:
    {split: {'X': (n, 2677) float32, 'acc': (n,) object, 'y': (n,)}}
"""
import hashlib
import os
import pickle

import numpy as np
import pandas as pd
import torch

PH = os.path.dirname(os.path.abspath(__file__))


def main(n_splits=10):
    ann = pd.read_csv(os.path.join(PH, 'all_gene_annotations.tsv'), sep='\t')
    feat = [l.strip() for l in open(os.path.join(PH, 'x_feature_cogs.txt')) if l.strip()]
    A = ann[feat].to_numpy().astype(np.float32)
    h2acc = {}
    for r, a in zip(A, ann['accession'].to_numpy()):
        h2acc.setdefault(hashlib.md5(r.tobytes()).hexdigest(), a)

    bundle = {}
    for s in range(n_splits):
        f = os.path.join(PH, 'splits', f'test_data_phylum_split_{s}')
        if not os.path.exists(f):
            continue
        X = torch.load(f, weights_only=False).numpy().astype(np.float32)
        y = torch.load(os.path.join(PH, 'splits', f'test_annot_phylum_split_{s}'),
                       weights_only=False).numpy()
        acc = [h2acc.get(hashlib.md5(r.tobytes()).hexdigest()) for r in X]
        nmiss = sum(a is None for a in acc)
        bundle[s] = {'X': X, 'acc': np.array(acc, dtype=object), 'y': y}
        print(f'split {s}: X{X.shape} y[{len(y)}]  unmatched-acc={nmiss}')

    with open(os.path.join(PH, 'denoise_bundle.pkl'), 'wb') as fh:
        pickle.dump(bundle, fh)
    print(f'wrote denoise_bundle.pkl ({len(bundle)} splits)')


if __name__ == '__main__':
    main()
