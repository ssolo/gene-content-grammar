#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_validation_data.py — Download STRING PPI data and COG mappings.

Builds the COG-pair reference used to ask whether the learned coupling matrix
J encodes known protein–protein interactions.  Per species: download STRING
links above a combined-score threshold, map STRING proteins to COG families by
gene name against the NCBI COG table, aggregate protein-level scores to COG
pairs, and save binary labels at two confidence thresholds for AUC-ROC / AUPR.

Target species (prokaryotes with well-characterised interactomes):

    E. coli K-12 MG1655     NCBI taxid 511145
    B. subtilis 168          NCBI taxid 224308
    M. tuberculosis H37Rv    NCBI taxid  83332
    S. aureus N315           NCBI taxid 158879

Data sources:
    STRING v12        https://string-db.org
    NCBI COG2024      https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/

Output per species:
    validation_data/{species}/
        protein_links.tsv        STRING links above threshold
        protein_to_cog.tsv       protein ID → COG family mapping
        cog_pair_scores.pt       (N, N) tensor of aggregated STRING scores
        cog_pair_labels.pt       (N, N) binary labels at confidence threshold

Usage:
    python scripts/download_validation_data.py --outdir validation_data
    python scripts/download_validation_data.py --outdir validation_data --species ecoli
    python scripts/download_validation_data.py --outdir validation_data --min-score 700
"""
import argparse, gzip, io, os, sys, time, urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---- Species definitions

SPECIES = {
    'ecoli': {
        'name': 'Escherichia coli K-12 MG1655',
        'taxid': '511145',
        'string_id': '511145',
    },
    'bsubtilis': {
        'name': 'Bacillus subtilis 168',
        'taxid': '224308',
        'string_id': '224308',
    },
    'mtb': {
        'name': 'Mycobacterium tuberculosis H37Rv',
        'taxid': '83332',
        'string_id': '83332',
    },
    'saureus': {
        'name': 'Staphylococcus aureus N315',
        'taxid': '158879',
        'string_id': '158879',
    },
}

STRING_VERSION = "12.0"
STRING_BASE = "https://stringdb-downloads.org/download"

# ---- Download helpers

def fetch_url(url, desc="", max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"  Fetching {desc or url}...", flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Python/research'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            print(f"    {len(data):,} bytes", flush=True)
            return data
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    print(f"  ERROR: Failed to download {url}", flush=True)
    return None


def download_string_links(taxid, outdir, min_score=400):
    """Download STRING protein links for a species.

    Columns are protein1 protein2 combined_score, the score on 0-1000.
    """
    outpath = outdir / 'protein_links.tsv'
    if outpath.exists():
        print(f"  Already exists: {outpath}", flush=True)
        return outpath

    url = (f"{STRING_BASE}/protein.links.v{STRING_VERSION}/"
           f"{taxid}.protein.links.v{STRING_VERSION}.txt.gz")
    data = fetch_url(url, f"STRING links for taxid {taxid}")
    if data is None:
        url2 = (f"https://stringdb-downloads.org/download/"
                f"protein.links.v{STRING_VERSION}/{taxid}.protein.links.v{STRING_VERSION}.txt.gz")
        data = fetch_url(url2, f"STRING links (alt URL)")
    if data is None:
        print(f"  WARNING: Could not download STRING links for {taxid}", flush=True)
        return None

    text = gzip.decompress(data).decode('utf-8')
    lines = text.strip().split('\n')
    header = lines[0]

    kept = 0
    with open(outpath, 'w') as f:
        f.write("protein1\tprotein2\tcombined_score\n")
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                score = int(parts[2])
                if score >= min_score:
                    f.write(f"{parts[0]}\t{parts[1]}\t{score}\n")
                    kept += 1

    print(f"  {kept:,} links above score {min_score} (from {len(lines)-1:,} total)", flush=True)
    return outpath


def download_string_cog_annotations(taxid, outdir):
    """Download the STRING protein-info file for one species.

    Returns the path to the saved file, not to a COG mapping; the COG
    assignment is built downstream by build_protein_cog_mapping.
    """
    outpath = outdir / 'protein_to_cog_string.tsv'
    if outpath.exists():
        print(f"  Already exists: {outpath}", flush=True)
        return outpath

    url = (f"{STRING_BASE}/protein.info.v{STRING_VERSION}/"
           f"{taxid}.protein.info.v{STRING_VERSION}.txt.gz")
    data = fetch_url(url, f"STRING protein info for {taxid}")
    if data is None:
        return None

    text = gzip.decompress(data).decode('utf-8')
    lines = text.strip().split('\n')

    # protein.info fields: protein_external_id, preferred_name, protein_size,
    # annotation.  Kept whole: build_protein_cog_mapping matches on
    # preferred_name.
    info_path = outdir / 'protein_info.tsv'
    with open(info_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"  Saved protein info: {len(lines)-1} proteins", flush=True)
    return info_path


def download_ncbi_cog_mapping(outdir):
    """Download the NCBI COG2024 gene→COG table (cog-24.csv.gz)."""
    outpath = outdir / 'cog-24.csv'
    if outpath.exists():
        print(f"  Already exists: {outpath}", flush=True)
        return outpath

    url = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.csv.gz"
    data = fetch_url(url, "NCBI COG2024 gene→COG mapping (cog-24.csv.gz)")
    if data is None:
        url2 = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.csv"
        data = fetch_url(url2, "cog-24.csv (uncompressed)")
        if data is not None:
            with open(outpath, 'wb') as f:
                f.write(data)
            return outpath
        return None

    text = gzip.decompress(data).decode('utf-8')
    with open(outpath, 'w') as f:
        f.write(text)
    print(f"  {len(text.splitlines()):,} lines", flush=True)
    return outpath


def build_protein_cog_mapping(string_info_path, ncbi_cog_path, taxid, outdir):
    """Map STRING proteins to COGs by gene name.

    STRING's preferred_name is matched case-insensitively against the gene
    name in the NCBI COG table.  Name mismatches make this imperfect; it
    covers ~70-80% of genes.
    """
    outpath = outdir / 'protein_to_cog.tsv'
    if outpath.exists():
        n = sum(1 for _ in open(outpath)) - 1
        print(f"  Already exists: {outpath} ({n} mappings)", flush=True)
        return outpath

    string_proteins = {}  # preferred_name → string_id
    if string_info_path and Path(string_info_path).exists():
        with open(string_info_path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    string_id = parts[0]
                    name = parts[1].lower()
                    string_proteins[name] = string_id

    gene_to_cog = {}  # gene_name → COG_id
    if ncbi_cog_path and Path(ncbi_cog_path).exists():
        with open(ncbi_cog_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    # cog-24.csv fields: genome_name, genome_id, protein_id,
                    # protein_len, domain_start, domain_end, COG_id,
                    # membership_class
                    gene_name = parts[2].strip().lower()
                    cog_id = parts[6].strip()
                    if cog_id.startswith('COG'):
                        gene_to_cog[gene_name] = cog_id

    matched = {}
    for name, string_id in string_proteins.items():
        if name in gene_to_cog:
            matched[string_id] = gene_to_cog[name]

    with open(outpath, 'w') as f:
        f.write("string_id\tcog_id\tgene_name\n")
        for string_id, cog_id in sorted(matched.items()):
            name = [n for n, s in string_proteins.items() if s == string_id][0]
            f.write(f"{string_id}\t{cog_id}\t{name}\n")

    print(f"  Mapped {len(matched)}/{len(string_proteins)} STRING proteins to COGs", flush=True)
    return outpath


def build_cog_pair_scores(links_path, mapping_path, cog_vocab, outdir,
                          high_conf=700, physical_only=False):
    """Aggregate STRING protein-pair scores to COG-pair scores.

    For a pair of COG families (i, j) the score is the maximum STRING combined
    score over all protein pairs with protein1 ∈ COG_i and protein2 ∈ COG_j;
    the number of supporting protein pairs is kept alongside it.  Self-pairs
    (i == j) are excluded.

    Saves cog_pair_scores.pt with 'scores' (N_cog, N_cog) float, max STRING
    score / 1000; 'counts' (N_cog, N_cog) int, supporting protein pairs;
    'labels_700' / 'labels_900' (N_cog, N_cog) bool at score ≥ 700 / ≥ 900;
    plus 'cog_names' and the positive-pair counts.
    """
    if not HAS_TORCH:
        print("  WARNING: torch not available, skipping .pt output", flush=True)
        return None

    prot2cog = {}
    with open(mapping_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                prot2cog[parts[0]] = parts[1]

    cog_idx = {c: i for i, c in enumerate(cog_vocab)}
    N = len(cog_vocab)

    scores = np.zeros((N, N), dtype=np.float32)
    counts = np.zeros((N, N), dtype=np.int32)

    n_mapped = 0
    n_total = 0
    with open(links_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            n_total += 1
            p1, p2 = parts[0], parts[1]
            sc = int(parts[2]) / 1000.0

            c1 = prot2cog.get(p1)
            c2 = prot2cog.get(p2)
            if c1 is None or c2 is None:
                continue
            i1 = cog_idx.get(c1)
            i2 = cog_idx.get(c2)
            if i1 is None or i2 is None:
                continue
            if i1 == i2:
                continue  # skip self-interactions

            n_mapped += 1
            if sc > scores[i1, i2]:
                scores[i1, i2] = sc
                scores[i2, i1] = sc
            counts[i1, i2] += 1
            counts[i2, i1] += 1

    labels_700 = scores >= 0.7
    labels_900 = scores >= 0.9

    result = {
        'scores': torch.tensor(scores),
        'counts': torch.tensor(counts),
        'labels_700': torch.tensor(labels_700),
        'labels_900': torch.tensor(labels_900),
        'cog_names': cog_vocab,
        'n_pos_700': int(labels_700.sum()) // 2,  # symmetric, count once
        'n_pos_900': int(labels_900.sum()) // 2,
        'n_mapped_links': n_mapped,
        'n_total_links': n_total,
    }

    outpath = outdir / 'cog_pair_scores.pt'
    torch.save(result, outpath)
    print(f"  Mapped {n_mapped:,}/{n_total:,} links to COG pairs", flush=True)
    print(f"  Positive pairs: {result['n_pos_700']:,} (score≥700), "
          f"{result['n_pos_900']:,} (score≥900)", flush=True)
    print(f"  Saved: {outpath}", flush=True)
    return outpath


def main():
    pa = argparse.ArgumentParser(
        description="Download STRING PPI data and build COG-pair validation sets.")
    pa.add_argument('--outdir', default='validation_data')
    pa.add_argument('--species', nargs='*', default=None,
                    help=f'Species to download (default: all). Options: {list(SPECIES.keys())}')
    pa.add_argument('--min-score', type=int, default=400,
                    help='Minimum STRING combined score to keep (0-1000)')
    pa.add_argument('--feather', default='data/COG_train1_phylum.feather',
                    help='Feather file for COG vocabulary')
    A = pa.parse_args()

    od = Path(A.outdir)
    od.mkdir(exist_ok=True)

    # COG vocabulary = sorted COG column order of the training feather, the
    # order every model and evaluation assumes.
    cog_vocab = None
    if Path(A.feather).exists():
        import pandas as pd
        df = pd.read_feather(A.feather)
        cog_vocab = sorted([c for c in df.columns if c.startswith('COG')])
        print(f"COG vocabulary: {len(cog_vocab)} families from {A.feather}\n")
    else:
        print(f"WARNING: {A.feather} not found — will skip .pt generation")
        print(f"  (provide --feather to enable COG-pair score tensors)\n")

    # One NCBI COG table, shared by every species.
    ncbi_cog = download_ncbi_cog_mapping(od)

    species_list = A.species if A.species else list(SPECIES.keys())
    for sp_key in species_list:
        if sp_key not in SPECIES:
            print(f"\nUnknown species: {sp_key}. Options: {list(SPECIES.keys())}")
            continue

        sp = SPECIES[sp_key]
        print(f"\n{'='*60}")
        print(f"  {sp['name']} (taxid {sp['taxid']})")
        print(f"{'='*60}")

        sp_dir = od / sp_key
        sp_dir.mkdir(exist_ok=True)

        links = download_string_links(sp['string_id'], sp_dir, A.min_score)
        info = download_string_cog_annotations(sp['string_id'], sp_dir)
        mapping = build_protein_cog_mapping(info, ncbi_cog, sp['taxid'], sp_dir)
        if links and mapping and cog_vocab:
            build_cog_pair_scores(links, mapping, cog_vocab, sp_dir)

    print(f"\n{'='*60}")
    print(f"  Done. Validation data in: {od}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
