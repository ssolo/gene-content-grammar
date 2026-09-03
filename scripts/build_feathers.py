#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_feathers.py -- build the train/val COG feathers with whole-clade holdout.

Genomes are split by holding out entire GTDB clades at a chosen rank (phylum,
class, order, family, genus or species), so no clade appears in both train and
val. Replicates differ only in the seeded shuffle of the clade order before
greedy packing.

Input files:
  - eggNOG CSV with columns: acc, eggNOG_OGs, count
  - GTDB archaeal metadata TSV (ar53_metadata_r220.tsv)
  - GTDB bacterial metadata TSV (bac120_metadata_r220.tsv)

Output files (per replicate):
  data/COG_train{r}_{rank}.feather  -- training genomes (whole clades)
  data/COG_val{r}_{rank}.feather    -- held-out genomes (whole clades)

Usage:
  python scripts/build_feathers.py \\
      --eggnog filtered_all_eggnog.csv \\
      --ar-meta ar53_metadata_r220.tsv \\
      --bac-meta bac120_metadata_r220.tsv \\
      --split-rank phylum --val-frac 0.2 --replicates 3 --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd


GTDB_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

METADATA_COLUMNS = [
    "accession",
    "gtdb_taxonomy",
    "checkm_completeness",
    "checkm_contamination",
    "coding_bases",
    "genome_size",
    "gc_percentage",
]


def load_eggnog(path):
    """Load the eggNOG CSV and pivot the COG rows to a genome x family matrix."""
    print(f"Loading eggNOG annotations from {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} total records")

    # COG only: arCOGs are a separate namespace with no KEGG mapping in
    # data/module_matrix_kegg.pt and ~2.7x the column count, so keeping them
    # would both break the module matrix and inflate feather-load RAM.
    mask = df["eggNOG_OGs"].str.startswith("COG")
    df = df[mask]
    print(f"  {len(df):,} COG records after filtering")

    pivot = (
        df.pivot_table(
            index="acc", columns="eggNOG_OGs", values="count",
            aggfunc="sum", fill_value=0,
        )
        .reset_index()
        .rename(columns={"acc": "accession"})
    )
    n_cogs = pivot.shape[1] - 1
    print(f"  Pivoted: {len(pivot):,} genomes x {n_cogs:,} gene families")
    return pivot


def load_metadata(ar_path, bac_path):
    """Concatenate the archaeal and bacterial GTDB metadata tables."""
    print(f"Loading archaeal metadata from {ar_path} ...")
    df_ar = pd.read_csv(ar_path, sep="\t", low_memory=False, usecols=METADATA_COLUMNS)
    print(f"  {len(df_ar):,} archaeal genomes")

    print(f"Loading bacterial metadata from {bac_path} ...")
    df_bac = pd.read_csv(bac_path, sep="\t", low_memory=False, usecols=METADATA_COLUMNS)
    print(f"  {len(df_bac):,} bacterial genomes")

    df = pd.concat([df_ar, df_bac], ignore_index=True)
    print(f"  {len(df):,} total genomes after combining")
    return df


def parse_gtdb_taxonomy(series):
    """Split 'd__Bacteria;p__Firmicutes;...' into [domain, phylum, ..., species]."""
    splits = series.fillna("").str.split(";", expand=True)
    for i in range(splits.shape[1], len(GTDB_RANKS)):
        splits[i] = None
    splits = splits.iloc[:, :len(GTDB_RANKS)]
    splits.columns = GTDB_RANKS
    splits.replace("", None, inplace=True)
    return splits


def build_merged_table(eggnog_path, ar_meta_path, bac_meta_path):
    """Merge the gene counts with the GTDB metadata and parse the taxonomy.

    Returns (merged_df, sorted COG column names).
    """
    pivot = load_eggnog(eggnog_path)
    meta = load_metadata(ar_meta_path, bac_meta_path)

    print("Merging gene counts with metadata ...")
    merged = pd.merge(pivot, meta, how="left", on="accession")
    missing = merged["gtdb_taxonomy"].isna().sum()
    if missing:
        print(f"  WARNING: {missing:,} genomes have no GTDB taxonomy -- will be dropped")

    tax = parse_gtdb_taxonomy(merged["gtdb_taxonomy"])
    merged = pd.concat([merged, tax], axis=1)

    cog_columns = sorted(c for c in pivot.columns if c != "accession")
    print(f"  Final table: {len(merged):,} genomes x {len(cog_columns):,} COG families")
    return merged, cog_columns


def clade_holdout_split(df, split_rank, val_frac, seed):
    """Hold out whole clades at `split_rank` until val_frac of genomes is reached.

    Clades are visited in a seeded shuffle and packed greedily.
    Returns (train_df, val_df, sorted val_clades).
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
        # Skip rather than stop, so a single huge clade cannot end the packing
        # while smaller ones still fit.
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
        description="Build train/val COG feather files with whole-clade holdout.")
    pa.add_argument("--eggnog", default="filtered_all_eggnog.csv",
                    help="eggNOG CSV (acc, eggNOG_OGs, count)")
    pa.add_argument("--ar-meta", default="ar53_metadata_r220.tsv",
                    help="GTDB archaeal metadata TSV")
    pa.add_argument("--bac-meta", default="bac120_metadata_r220.tsv",
                    help="GTDB bacterial metadata TSV")
    pa.add_argument("--outdir", default=".",
                    help="Output directory for feather files")
    pa.add_argument("--split-rank", default="phylum", choices=GTDB_RANKS[1:],
                    help="GTDB rank at which to hold out entire clades (default: phylum)")
    pa.add_argument("--replicates", type=int, default=3,
                    help="Number of independent train/val splits")
    pa.add_argument("--val-frac", type=float, default=0.2,
                    help="Target fraction of genomes in validation")
    pa.add_argument("--seed", type=int, default=42,
                    help="Base random seed (replicate r uses seed + r)")
    args = pa.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    merged, cog_columns = build_merged_table(
        args.eggnog, args.ar_meta, args.bac_meta
    )

    merged = merged.dropna(subset=[args.split_rank])
    print(f"\n{len(merged):,} genomes with valid {args.split_rank}")

    clade_counts = merged[args.split_rank].value_counts()
    print(f"{len(clade_counts)} unique {args.split_rank} groups")
    print(f"  Largest:  {clade_counts.iloc[0]:>7,}  {clade_counts.index[0]}")
    print(f"  Smallest: {clade_counts.iloc[-1]:>7,}  {clade_counts.index[-1]}")

    keep_cols = cog_columns + [
        "accession", "domain", "phylum", "class", "order",
        "family", "genus", "species",
        "checkm_completeness", "checkm_contamination",
        "genome_size", "gc_percentage",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]

    rank_tag = args.split_rank

    for r in range(1, args.replicates + 1):
        seed_r = args.seed + r
        train_df, val_df, val_clades = clade_holdout_split(
            merged, args.split_rank, args.val_frac, seed_r
        )

        train_path = os.path.join(args.outdir, f"data/COG_train{r}_{rank_tag}.feather")
        val_path = os.path.join(args.outdir, f"data/COG_val{r}_{rank_tag}.feather")

        train_df[keep_cols].reset_index(drop=True).to_feather(train_path)
        val_df[keep_cols].reset_index(drop=True).to_feather(val_path)

        actual_frac = len(val_df) / (len(train_df) + len(val_df))
        print(f"\nReplicate {r} (seed={seed_r}):")
        print(f"  Train: {len(train_df):>7,} genomes  -> {train_path}")
        print(f"  Val:   {len(val_df):>7,} genomes  -> {val_path}")
        print(f"  Val fraction: {actual_frac:.1%}  ({len(val_clades)} {rank_tag} groups held out)")
        print(f"  Held-out {rank_tag}s: {', '.join(val_clades[:10])}"
              + (f" ... (+{len(val_clades)-10} more)" if len(val_clades) > 10 else ""))

    print("\nDone.")


if __name__ == "__main__":
    main()
