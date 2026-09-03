#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_ecoli_zoom_feathers.py -- nested leave-clade-out ladder around E. coli.

Re-partitions the split-5 HQ bacterial corpus into a ladder of holdouts. At each
rank in {phylum, class, order, family} E. coli's whole clade at that rank goes to
val and everything else to train, so each successive train set adds progressively
closer relatives of E. coli.

The Figure 2 genome (E. coli RS_GCF_003697165.2) is itself absent from the HQ
corpus and is scored separately via reconstruct_extant against COG_val5; it is
never in any train set by construction, since its clade is always the held-out one.

Source corpus = the split-5 HQ bacterial feathers (train5 + val5), the genome
universe the marginal-HQ split-5 model trained on. Rows are a strict subset and
columns (including the 4789 COG columns) are copied verbatim, so vocabulary
alignment with data/module_matrix_kegg.pt is guaranteed.

E. coli's lineage is read from the GTDB metadata rather than hardcoded. Asserted
before anything is written: train/val disjoint, the full clade in val, E. coli
absent from the corpus, column parity with the source. Provenance for every rung
is merged into data/ecolizoom_provenance.json.

Reading .feather needs pyarrow, and holding the corpus needs ~32 GB of RAM.

  python3 scripts/build_ecoli_zoom_feathers.py
"""
import argparse
import json
import os
import sys

import pandas as pd

ECOLI_ACC = "RS_GCF_003697165.2"          # the Figure 2 genome
ECOLI_KEY = "GCF_003697165"               # accession substring (RS_/version-agnostic)
# Buildable rungs. phylum..family are leave-clade-out of the HQ corpus. The
# closest rung, "species", is leave-the-Fig-2-genome-out: the HQ corpus holds no
# Escherichia, so it is augmented with the HQ-quality Enterobacteriaceae from the
# full root holdout COG_val5 (including E. coli's 10 Escherichia congeners) and
# only the Fig-2 genome is held out.
RANKS = ["phylum", "class", "order", "family", "species", "intermediate"]
LINEAGE_RANKS = ["phylum", "class", "order", "family", "genus", "species"]
# The "intermediate" rung holds out E. coli's ~1.4 Ga Gammaproteobacteria
# sub-clade, a block of 6 GTDB orders including Enterobacterales (node ~1436 Ma,
# stem ~1563 Ma in the Davin et al. dated tree, so 2x divergence ~3125 Ma). The
# nearest training relative is then a more distant Gammaproteobacteria order,
# which fills the order-to-class gap on the recovery-vs-divergence-time curve.
# Order names shift between GTDB r202 and r220; isin() tolerates absent or
# renamed ones, and the major members (Enterobacterales, Pseudomonadales) are
# stable across releases.
INTERMEDIATE_ORDERS = ["o__Enterobacterales", "o__Pseudomonadales",
                       "o__Francisellales", "o__Piscirickettsiales",
                       "o__HP12", "o__SAR86"]
SRC_TRAIN = "data/COG_bac_hq_train5_phylum.feather"
SRC_VAL = "data/COG_bac_hq_val5_phylum.feather"
SRC_FULL_VAL = "data/COG_val5_phylum.feather"   # full root holdout (has the Escherichia)
META_TSV = "data/genome_metadata.tsv"
OUT_TMPL = "data/COG_bac_hq_ecolizoom_{rank}_{split}.feather"


def is_cog(col):
    return col.startswith("COG") and col[3:].isdigit()


def build_intermediate_rung(corpus):
    """Hold out E. coli's ~1.4 Ga Gammaproteobacteria sub-clade (INTERMEDIATE_ORDERS).

    E. coli is not in the corpus and is held out at this depth through its order.
    Returns (train, val, info)."""
    mask = corpus["order"].isin(INTERMEDIATE_ORDERS)
    val = corpus[mask].reset_index(drop=True)
    train = corpus[~mask].reset_index(drop=True)
    present = sorted(set(val["order"]))
    assert len(val) > 0, "no INTERMEDIATE_ORDERS genomes in the HQ corpus"
    assert list(train.columns) == list(corpus.columns)
    info = {"clade": "ecoli ~1.4Ga Gammaproteobacteria sub-clade",
            "held_out_orders_present": present,
            "stem_age_Ma": 1563, "divergence_2x_Ma": 3125,
            "note": "Davin Figure3 node ~1436 Ma, stem ~1563 Ma; hold out 6 orders"}
    return train, val, info


def build_species_rung(corpus, lineage):
    """Leave-the-Fig-2-genome-out, the closest rung.

    The HQ corpus contains no Escherichia, so it is augmented with the HQ-quality
    Enterobacteriaceae from the full root holdout (COG_val5), including E. coli's
    10 Escherichia congeners. Only the Fig-2 genome is held out; the monitor val
    is a random 5% drawn from outside E. coli's order, so every order/family/genus
    relative stays in training. Returns (train, val, info)."""
    fam, gen, order = lineage["family"], lineage["genus"], lineage["order"]
    v5 = pd.read_feather(SRC_FULL_VAL)
    assert list(v5.columns) == list(corpus.columns), "vocab mismatch with root val5"
    extra = v5[(v5["family"] == fam)
               & (v5["checkm_completeness"] >= 90) & (v5["checkm_contamination"] <= 5)
               & (~v5["accession"].astype(str).str.contains(ECOLI_KEY, regex=False, na=False))]
    combined = (pd.concat([corpus, extra], ignore_index=True)
                .drop_duplicates("accession").reset_index(drop=True))
    outside = combined[combined["order"] != order]
    val_idx = outside.sample(frac=0.05, random_state=42).index
    val = combined.loc[val_idx].reset_index(drop=True)
    train = combined.drop(val_idx).reset_index(drop=True)
    n_fig2 = int(train["accession"].astype(str).str.contains(ECOLI_KEY, regex=False).sum())
    n_cong = int((train["genus"] == gen).sum())
    assert n_fig2 == 0, "Fig-2 genome leaked into species-rung train"
    assert n_cong >= 1, "no Escherichia congeners in species-rung train"
    assert list(train.columns) == list(corpus.columns)
    info = {"clade": "leave-Fig2-out (s__Escherichia coli; 1 genome in dataset = Fig2)",
            "n_extra_family_added": int(len(extra)),
            "n_congeners_in_train": n_cong,
            "n_family_in_train": int((train["family"] == fam).sum()),
            "note": "HQ corpus + HQ Enterobacteriaceae from COG_val5, hold out only the "
                    "Fig-2 genome; monitor val = random 5% outside Enterobacterales"}
    return train, val, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the partition plan without writing feathers")
    ap.add_argument("--ranks", default=",".join(RANKS),
                    help="comma-separated subset of ranks to (re)build; "
                         "default all. Provenance JSON is merged, not clobbered.")
    A = ap.parse_args()
    want = [r.strip() for r in A.ranks.split(",") if r.strip()]
    bad = [r for r in want if r not in RANKS]
    if bad:
        sys.exit(f"unknown rank(s) {bad}; choose from {RANKS}")

    corpus = pd.concat(
        [pd.read_feather(SRC_TRAIN), pd.read_feather(SRC_VAL)],
        ignore_index=True,
    )
    cog_cols = [c for c in corpus.columns if is_cog(c)]
    print(f"HQ corpus (train5 + val5): {len(corpus)} genomes, "
          f"{len(cog_cols)} COG columns")

    meta = pd.read_csv(META_TSV, sep="\t", low_memory=False)
    hit = meta[meta["accession"] == ECOLI_ACC]
    if not len(hit):
        sys.exit(f"E. coli {ECOLI_ACC} not found in {META_TSV}")
    erow = hit.iloc[0]
    lineage = {r: str(erow[r]) for r in LINEAGE_RANKS}
    print(f"E. coli {ECOLI_ACC} lineage: "
          + " / ".join(f"{r}={lineage[r]}" for r in LINEAGE_RANKS))

    in_corpus = corpus["accession"].astype(str).str.contains(
        "GCF_003697165", na=False, regex=False).any()
    assert not in_corpus, "E. coli is unexpectedly present in the HQ corpus"

    prov_path = "data/ecolizoom_provenance.json"
    prov = {"source": {}, "rungs": {}}
    if os.path.exists(prov_path):
        with open(prov_path) as fh:
            prov = json.load(fh)
    prov["source"] = {"train": SRC_TRAIN, "val": SRC_VAL, "corpus_n": int(len(corpus))}
    prov["ecoli_accession"] = ECOLI_ACC
    prov["ecoli_lineage"] = lineage
    prov["n_cog_cols"] = len(cog_cols)
    prov.setdefault("rungs", {})
    print("\nrung      held-out clade                  val   train   feathers")
    for rank in want:
        if rank == "species":
            train, val, info = build_species_rung(corpus, lineage)
            clade = info["clade"]
            entry = info
        elif rank == "intermediate":
            train, val, info = build_intermediate_rung(corpus)
            clade = info["clade"]
            entry = info
        else:
            clade = lineage[rank]
            val = corpus[corpus[rank] == clade].reset_index(drop=True)
            train = corpus[corpus[rank] != clade].reset_index(drop=True)
            assert len(val) + len(train) == len(corpus)
            assert (val[rank] == clade).all()
            assert not (train[rank] == clade).any()
            entry = {"clade": clade}
        assert list(train.columns) == list(corpus.columns)
        assert list(val.columns) == list(corpus.columns)
        tf = OUT_TMPL.format(rank=rank, split="train")
        vf = OUT_TMPL.format(rank=rank, split="val")
        print(f"{rank:8s}  {clade:28.28s}  {len(val):5d}  {len(train):6d}  {tf}")
        entry.update({"n_val": int(len(val)), "n_train": int(len(train)),
                      "train_feather": tf, "val_feather": vf})
        prov["rungs"][rank] = entry
        if not A.dry_run:
            train.to_feather(tf)
            val.to_feather(vf)

    if not A.dry_run:
        with open(prov_path, "w") as fh:
            json.dump(prov, fh, indent=2)
        print(f"\nwrote {prov_path}")
    else:
        print("\n[dry-run] no feathers written")


if __name__ == "__main__":
    main()
