#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_archaeal_feathers.py -- archaea-only train/val splits for fine-tuning a
joint-trained model toward the archaeal domain.

The union of the mixed-domain feathers COG_train1_phylum.feather and
COG_val1_phylum.feather is the full presplit pool (113k genomes); this filters
it to d__Archaea (5,869 genomes across 21 GTDB r220 phyla) and runs whole-phylum
holdout under 10 seeded shuffles. Reusing the pivoted matrix avoids a second
pass over the 4 GB eggNOG CSV.

Output (one pair per replicate):
  data/COG_arc_train{r}_phylum.feather   (~4700 genomes, typically 20 phyla)
  data/COG_arc_val{r}_phylum.feather     (~1170 genomes, 1-4 phyla)

Column structure matches the mixed-domain feathers (COG columns + accession +
7 taxonomy ranks + 4 QC columns), so the trainer reads them unchanged.

With only 21 archaeal phyla the 20% val target is met by holding out 1-4 phyla,
so the splits are lumpier than the bacterial-dominant ones (100+ phyla allow
finer packing).

Usage:
  python3 scripts/build_archaeal_feathers.py \\
      [--source-train data/COG_train1_phylum.feather] \\
      [--source-val   data/COG_val1_phylum.feather] \\
      [--outdir data] [--val-frac 0.2] [--seed 42] [--replicates 10]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Duplicated from scripts/build_feathers.py so this script stays standalone;
# the two copies must stay in step or the splits stop being reproducible.
def clade_holdout_split(df, split_rank, val_frac, seed):
    """Hold out whole clades at `split_rank` until val_frac of genomes is reached.

    Clades are visited in a seeded shuffle and packed greedily; one that would
    overshoot the target is skipped rather than ending the packing.
    """
    rng = np.random.RandomState(seed)
    clade_counts = df[split_rank].value_counts()
    clades = clade_counts.index.tolist()
    rng.shuffle(clades)

    n_total = len(df)
    n_target = int(round(n_total * val_frac))

    val_clades = []
    n_val = 0
    for clade in clades:
        n_clade = clade_counts[clade]
        if n_val > 0 and abs(n_val + n_clade - n_target) > abs(n_val - n_target):
            continue
        val_clades.append(clade)
        n_val += n_clade
        if n_val >= n_target:
            break

    val_mask = df[split_rank].isin(val_clades)
    return df[~val_mask], df[val_mask], sorted(val_clades)


def main():
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--source-train', default='COG_train1_phylum.feather',
                    help='Source mixed-domain train feather (used only to '
                         'reconstruct the full presplit pool).')
    pa.add_argument('--source-val', default='COG_val1_phylum.feather',
                    help='Source mixed-domain val feather (paired with '
                         '--source-train to reconstruct the full pool).')
    pa.add_argument('--outdir', default='data',
                    help='Where to write COG_arc_{train,val}{r}_phylum.feather.')
    pa.add_argument('--split-rank', default='phylum',
                    choices=['phylum', 'class', 'order', 'family'],
                    help='GTDB rank at which to hold out clades (default phylum).')
    pa.add_argument('--val-frac', type=float, default=0.2,
                    help='Target fraction of archaea in val (default 0.2).')
    pa.add_argument('--replicates', type=int, default=10,
                    help='Number of independent splits (default 10).')
    pa.add_argument('--seed', type=int, default=42,
                    help='Base random seed (replicate r uses seed + r).')
    pa.add_argument('--mix-bacteria-ratio', type=float, default=0.0,
                    help='If > 0, ALSO write mixed-domain feathers that '
                         'preserve the bacterial regulariser of the '
                         'pretraining distribution: each archaea train set '
                         'is concatenated with R x |archaea| random '
                         'bacterial genomes (R = this flag).  R=1 gives '
                         '50/50 archaea/bacteria with archaea oversampled '
                         '~10x vs their natural 5%% rate.  Outputs go to '
                         'data/COG_mix_{train,val}{r}_phylum.feather.  Val '
                         'is the same archaea-only held-out set (so val '
                         'still measures archaeal generalisation, not '
                         'bacterial -- bacteria are training-only).')
    pa.add_argument('--mix-bacteria-seed', type=int, default=4242,
                    help='Random seed for the bacterial subsample.')
    pa.add_argument('--also-full', action='store_true',
                    help='Additionally write a NO-HOLDOUT pair: '
                         'train = all archaea (5,869 genomes), val = '
                         'random 200-genome sample of train (monitor '
                         'only -- val MCC during training will be '
                         'inflated since val is a subset of train).  '
                         'Output: data/COG_arc_train_full_phylum.feather '
                         'and data/COG_arc_val_full_phylum.feather.  '
                         'Use for an extra inference-only "joint" model '
                         'on all data, as an 11th member of the 10-split '
                         'ensemble (or as a standalone).')
    pa.add_argument('--full-val-n', type=int, default=200,
                    help='Size of the monitor val sample for --also-full.')
    args = pa.parse_args()

    # Accept a source feather given as a path, under --outdir, or in the
    # working directory: build_feathers.py writes to data/, but an unpacked
    # archive may leave the pair at the repository root.
    def resolve(name):
        for cand in (Path(name), Path(args.outdir) / name, Path('.') / name):
            if cand.exists():
                return cand
        raise FileNotFoundError(f'Cannot find {name} in cwd, outdir, or as-is')

    train_path = resolve(args.source_train)
    val_path   = resolve(args.source_val)

    print(f'Loading {train_path} ...')
    train = pd.read_feather(train_path)
    print(f'  {len(train):,} genomes')
    print(f'Loading {val_path} ...')
    val   = pd.read_feather(val_path)
    print(f'  {len(val):,} genomes')

    pool = pd.concat([train, val], ignore_index=True)
    print(f'\nPresplit pool: {len(pool):,} genomes')
    print(f'  domain counts:')
    for d, n in pool['domain'].value_counts().items():
        print(f'    {d:14s}  {n:7,d}')

    arch = pool[pool['domain'] == 'd__Archaea'].copy().reset_index(drop=True)
    print(f'\nFiltered to archaea: {len(arch):,} genomes')

    if args.split_rank not in arch.columns:
        raise SystemExit(f'No column "{args.split_rank}" in source feather')

    clade_counts = arch[args.split_rank].value_counts()
    print(f'\nArchaeal {args.split_rank} groups ({len(clade_counts)}):')
    for ph, n in clade_counts.items():
        print(f'  {ph:38s}  {n:5d}')

    arch = arch.dropna(subset=[args.split_rank])
    print(f'\nAfter dropping rows missing {args.split_rank}: {len(arch):,} genomes')

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*72}')
    print(f'  Building {args.replicates} archaea-only splits '
          f'(target val_frac={args.val_frac})')
    print(f'{"="*72}')

    for r in range(1, args.replicates + 1):
        seed_r = args.seed + r
        train_df, val_df, val_clades = clade_holdout_split(
            arch, args.split_rank, args.val_frac, seed_r)

        train_out = outdir / f'COG_arc_train{r}_{args.split_rank}.feather'
        val_out   = outdir / f'COG_arc_val{r}_{args.split_rank}.feather'

        train_df.reset_index(drop=True).to_feather(train_out)
        val_df.reset_index(drop=True).to_feather(val_out)

        actual = len(val_df) / (len(train_df) + len(val_df))
        print(f'\nReplicate {r} (seed={seed_r}):')
        print(f'  train: {len(train_df):5,d}  -> {train_out}')
        print(f'  val:   {len(val_df):5,d}  -> {val_out}')
        print(f'  val fraction: {actual:.1%}  '
              f'({len(val_clades)} {args.split_rank}{"s" if len(val_clades) != 1 else ""} held out)')
        print(f'  held-out: {", ".join(val_clades)}')

    print(f'\nDone.  {args.replicates} archaea-only replicate splits written to {outdir}/')

    if args.mix_bacteria_ratio > 0:
        print(f'\n{"="*72}')
        print(f'  --mix-bacteria-ratio={args.mix_bacteria_ratio}: writing '
              f'archaea+bacteria mixed train feathers (val unchanged)')
        print(f'{"="*72}')
        bact = pool[pool['domain'] == 'd__Bacteria'].copy().reset_index(drop=True)
        print(f'  bacterial pool: {len(bact):,} genomes')
        rng_b = np.random.RandomState(args.mix_bacteria_seed)
        for r in range(1, args.replicates + 1):
            # Re-derived from the same seed as the main loop, so the archaeal
            # half of the mixed split is identical to the archaea-only split.
            seed_r = args.seed + r
            train_df, val_df, _ = clade_holdout_split(
                arch, args.split_rank, args.val_frac, seed_r)
            n_arc = len(train_df)
            n_bac = int(round(args.mix_bacteria_ratio * n_arc))
            n_bac = min(n_bac, len(bact))
            idx = sorted(rng_b.choice(len(bact), size=n_bac, replace=False).tolist())
            bact_sample = bact.iloc[idx]
            mix_train = pd.concat([train_df, bact_sample], ignore_index=True)
            # Shuffle so a minibatch sees both domains rather than a
            # contiguous block of one.
            mix_train = mix_train.sample(frac=1.0, random_state=seed_r) \
                                 .reset_index(drop=True)
            train_out = outdir / f'COG_mix_train{r}_{args.split_rank}.feather'
            val_out   = outdir / f'COG_mix_val{r}_{args.split_rank}.feather'
            mix_train.to_feather(train_out)
            val_df.reset_index(drop=True).to_feather(val_out)
            print(f'  rep {r}: mix train = {n_arc:,} archaea + {n_bac:,} '
                  f'bacteria ({n_arc + n_bac:,} total)  ->  {train_out}')
            print(f'           val (archaea-only held-out) = {len(val_df):,} '
                  f'-> {val_out}')

    if args.also_full:
        print(f'\n{"="*72}')
        print(f'  --also-full: writing no-holdout pair (all {len(arch):,} '
              f'archaea in train; {args.full_val_n}-genome random sample as val)')
        print(f'{"="*72}')
        train_full = outdir / f'COG_arc_train_full_{args.split_rank}.feather'
        val_full   = outdir / f'COG_arc_val_full_{args.split_rank}.feather'
        arch.reset_index(drop=True).to_feather(train_full)
        rng = np.random.RandomState(args.seed)
        n_mon = min(args.full_val_n, len(arch))
        idx = sorted(rng.choice(len(arch), size=n_mon, replace=False).tolist())
        arch.iloc[idx].reset_index(drop=True).to_feather(val_full)
        print(f'  train_full ({len(arch):,} genomes) -> {train_full}')
        print(f'  val_full   ({n_mon} monitor sample) -> {val_full}')
        print(f'  NOTE: val_full is a subset of train_full -- the per-epoch '
              f'val MCC during training is INFLATED.  This is a monitor, '
              f'not a held-out test set.  The model trains on all archaea.')


if __name__ == '__main__':
    main()
