#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bacterial_feathers.py -- 10 bacteria-only train/val splits for fine-tuning
a joint-trained model toward the bacterial domain (the LBCA specialist).  The
bacterial mirror of build_archaeal_feathers.py.

The union of the mixed-domain feathers (COG_train1_phylum.feather +
COG_val1_phylum.feather) is the full presplit pool of ~113k genomes; this is
filtered to d__Bacteria (~107k genomes across 100+ GTDB r220 phyla), then split
by whole-phylum holdout under 10 seed shuffles.  Val therefore holds only phyla
absent from train -- the LBCA reconstruction setting, where a deep ancestor's
descendant lineages are only partially observed.  With 100+ bacterial phyla the
20% val target packs finely, each replicate holding out several phyla.

The --max-train / --max-val caps hold each fine-tune at mix-FT compute scale:
the mix-FT recipe used ~10k genomes/split against the ~107k bacterial pool, so
the defaults subsample the holdout train set to 15k and val to 3k.  The uncapped
arc-only fine-tune became de-facto retraining and regressed.  --max-train 0 /
--max-val 0 disable the caps.  Val is still drawn only from the held-out phyla
after subsampling, so an ECE measured on it is still a held-out-lineage number,
just from fewer genomes.

Output (one pair per replicate):
  data/COG_bac_train{r}_phylum.feather   (<= --max-train genomes)
  data/COG_bac_val{r}_phylum.feather     (<= --max-val genomes, held-out phyla)

Both carry the same column structure as the mixed-domain feathers (COG columns +
accession + 7 taxonomy ranks + 4 QC columns), so the trainer reads them unchanged
via --train-feather / --val-feather.

Usage:
  python3 scripts/build_bacterial_feathers.py \\
      [--source-train data/COG_train1_phylum.feather] \\
      [--source-val   data/COG_val1_phylum.feather] \\
      [--outdir data] [--val-frac 0.2] [--seed 42] [--replicates 10] \\
      [--max-train 15000] [--max-val 3000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---- Whole-clade holdout, duplicated from build_archaeal_feathers.py /
#      build_feathers.py so this script stays standalone.

def clade_holdout_split(df, split_rank, val_frac, seed):
    """Hold out entire clades at `split_rank` until val_frac is reached.

    Greedy packing in a shuffled clade order: a clade joins val unless adding it
    overshoots the target by more than the current shortfall.  Returns
    (train_df, val_df, sorted held-out clade names).
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


def maybe_subsample(df, cap, seed):
    """Randomly subsample df to at most `cap` rows; cap <= 0 disables the cap."""
    if cap and cap > 0 and len(df) > cap:
        return df.sample(n=cap, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


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
                    help='Where to write COG_bac_{train,val}{r}_phylum.feather.')
    pa.add_argument('--split-rank', default='phylum',
                    choices=['phylum', 'class', 'order', 'family'],
                    help='GTDB rank at which to hold out clades (default phylum).')
    pa.add_argument('--val-frac', type=float, default=0.2,
                    help='Target fraction of bacteria in val (default 0.2).')
    pa.add_argument('--replicates', type=int, default=10,
                    help='Number of independent splits (default 10).')
    pa.add_argument('--seed', type=int, default=42,
                    help='Base random seed (replicate r uses seed + r).')
    pa.add_argument('--max-train', type=int, default=15000,
                    help='Cap on train genomes per split (0 = no cap; use the '
                         'full whole-phylum-holdout train set).  Default 15000 '
                         'keeps the fine-tune at mix-FT compute scale.')
    pa.add_argument('--max-val', type=int, default=3000,
                    help='Cap on val genomes per split (0 = no cap).  Default '
                         '3000 is ample for ECE/Brier/NLL estimation and keeps '
                         'the calibration eval fast.')
    pa.add_argument('--subsample-seed', type=int, default=4242,
                    help='Base seed for the train/val subsample (replicate r '
                         'uses this + r).')
    pa.add_argument('--also-full', action='store_true',
                    help='Additionally write a NO-HOLDOUT pair from ALL '
                         'bacteria (train = every bacterium, capped by '
                         '--max-train; val = random --full-val-n monitor '
                         'sample of train).  val is a SUBSET of train, so the '
                         'per-epoch val MCC is inflated -- a monitor, not a '
                         'held-out test.  Output: '
                         'data/COG_bac_train_full_phylum.feather and '
                         'data/COG_bac_val_full_phylum.feather.')
    pa.add_argument('--full-val-n', type=int, default=200,
                    help='Size of the monitor val sample for --also-full.')
    args = pa.parse_args()

    # The source feathers may sit at the repo root or under data/.
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

    bact = pool[pool['domain'] == 'd__Bacteria'].copy().reset_index(drop=True)
    print(f'\nFiltered to bacteria: {len(bact):,} genomes')

    if args.split_rank not in bact.columns:
        raise SystemExit(f'No column "{args.split_rank}" in source feather')

    bact = bact.dropna(subset=[args.split_rank])
    print(f'After dropping rows missing {args.split_rank}: {len(bact):,} genomes')

    clade_counts = bact[args.split_rank].value_counts()
    print(f'\nBacterial {args.split_rank} groups: {len(clade_counts)}')
    print(f'  largest 10:')
    for ph, n in clade_counts.head(10).items():
        print(f'    {ph:42s}  {n:6,d}')

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*72}')
    print(f'  Building {args.replicates} bacteria-only splits '
          f'(target val_frac={args.val_frac}; '
          f'caps train<={args.max_train or "all"} val<={args.max_val or "all"})')
    print(f'{"="*72}')

    for r in range(1, args.replicates + 1):
        seed_r = args.seed + r
        train_df, val_df, val_clades = clade_holdout_split(
            bact, args.split_rank, args.val_frac, seed_r)

        n_train_full, n_val_full = len(train_df), len(val_df)
        sub_seed = args.subsample_seed + r
        train_df = maybe_subsample(train_df, args.max_train, sub_seed)
        val_df   = maybe_subsample(val_df,   args.max_val,   sub_seed)

        train_out = outdir / f'COG_bac_train{r}_{args.split_rank}.feather'
        val_out   = outdir / f'COG_bac_val{r}_{args.split_rank}.feather'

        train_df.to_feather(train_out)
        val_df.to_feather(val_out)

        actual = n_val_full / (n_train_full + n_val_full)
        print(f'\nReplicate {r} (seed={seed_r}):')
        print(f'  holdout: {n_train_full:6,d} train / {n_val_full:6,d} val '
              f'({actual:.1%} val, {len(val_clades)} '
              f'{args.split_rank}{"s" if len(val_clades) != 1 else ""} held out)')
        print(f'  written: {len(train_df):6,d} train  -> {train_out}')
        print(f'           {len(val_df):6,d} val    -> {val_out}')
        print(f'  held-out {args.split_rank}s: {", ".join(val_clades[:8])}'
              f'{" ..." if len(val_clades) > 8 else ""}')

    print(f'\nDone.  {args.replicates} bacteria-only replicate splits written '
          f'to {outdir}/')

    if args.also_full:
        print(f'\n{"="*72}')
        print(f'  --also-full: no-holdout pair from all {len(bact):,} bacteria '
              f'(train capped by --max-train; {args.full_val_n}-genome monitor val)')
        print(f'{"="*72}')
        train_full_df = maybe_subsample(bact, args.max_train, args.subsample_seed)
        train_full = outdir / f'COG_bac_train_full_{args.split_rank}.feather'
        val_full   = outdir / f'COG_bac_val_full_{args.split_rank}.feather'
        train_full_df.to_feather(train_full)
        rng = np.random.RandomState(args.seed)
        n_mon = min(args.full_val_n, len(train_full_df))
        idx = sorted(rng.choice(len(train_full_df), size=n_mon, replace=False).tolist())
        train_full_df.iloc[idx].reset_index(drop=True).to_feather(val_full)
        print(f'  train_full ({len(train_full_df):,} genomes) -> {train_full}')
        print(f'  val_full   ({n_mon} monitor sample) -> {val_full}')
        print(f'  NOTE: val_full is a subset of train_full -- the per-epoch '
              f'val MCC during training is INFLATED.  This is a monitor, not a '
              f'held-out test set.')


if __name__ == '__main__':
    main()
