#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# KEGG LICENCE NOTICE -- READ BEFORE RUNNING
#
# This script queries the KEGG REST API (rest.kegg.jp) to build the KEGG layer of
# the module-membership matrix. KEGG is a copyrighted database. Academic use of the
# website and the REST API is free for individuals, but bulk or automated download,
# redistribution of KEGG-derived content, and any commercial use require a licence
# from Pathway Solutions (https://www.kegg.jp/kegg/legal.html).
#
# Running this script is your responsibility, under whatever KEGG terms apply to
# you. Check them before you run it, and keep --delay at a courteous value.
#
# For this reason no KEGG-derived data is redistributed with this repository: the
# built matrix (data/module_matrix_kegg.pt) is not included and must be rebuilt
# locally with this script. The COG-only matrix (data/module_matrix.pt, 26 COG
# functional categories + 66 COG pathway groupings) contains no KEGG content and
# is included.
#
"""
build_module_matrix_kegg.py — Build the COG → module membership matrix.

Writes the M = 419-column binary membership matrix used for module-
completeness conditioning and the auxiliary loss.  Its columns are three
annotation layers concatenated in this order:

    26  COG functional categories  (NCBI cog-24.def.tab)
     66 COG pathway groupings      (NCBI cog-24.def.tab)
   327 KEGG metabolic modules     (KEGG REST API)

The KEGG layer is COG → KO → Module.  link/ko/cog is deprecated, so the
COG cross-references are scraped from the DBLINKS section of the
individual KO entries: list/ko for the ~27 000 KO IDs, then
get/ko:K00001+... in batches, then link/module/ko.

Every KO that has been looked at is appended to --cache, including the
ones with no COG link (as a bare line), so --resume can skip them after
an interrupted run.

Usage (add --resume to continue an interrupted fetch from --cache):
    python scripts/build_module_matrix_kegg.py \\
        --feather data/COG_train1_phylum.feather --output data/module_matrix_kegg.pt
"""
import argparse, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict
import urllib.request
import numpy as np, pandas as pd
import torch

COG_CATEGORIES = {
    'J': 'Translation, ribosomal structure and biogenesis',
    'A': 'RNA processing and modification',
    'K': 'Transcription',
    'L': 'Replication, recombination and repair',
    'B': 'Chromatin structure and dynamics',
    'D': 'Cell cycle control, cell division, chromosome partitioning',
    'Y': 'Nuclear structure',
    'V': 'Defense mechanisms',
    'T': 'Signal transduction mechanisms',
    'M': 'Cell wall/membrane/envelope biogenesis',
    'N': 'Cell motility',
    'Z': 'Cytoskeleton',
    'W': 'Extracellular structures',
    'U': 'Intracellular trafficking, secretion, and vesicular transport',
    'O': 'Posttranslational modification, protein turnover, chaperones',
    'X': 'Mobilome: prophages, transposons',
    'C': 'Energy production and conversion',
    'G': 'Carbohydrate transport and metabolism',
    'E': 'Amino acid transport and metabolism',
    'F': 'Nucleotide transport and metabolism',
    'H': 'Coenzyme transport and metabolism',
    'I': 'Lipid transport and metabolism',
    'P': 'Inorganic ion transport and metabolism',
    'Q': 'Secondary metabolites biosynthesis, transport and catabolism',
    'R': 'General function prediction only',
    'S': 'Function unknown',
}

def fetch(url, delay=0.35, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(delay)
            req = urllib.request.Request(url, headers={'User-Agent': 'Python/research'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode('utf-8')
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 * (attempt + 1))

def get_all_ko_ids():
    print("  Step 1: Fetching all KO IDs...", flush=True)
    text = fetch("https://rest.kegg.jp/list/ko", delay=0.5)
    kos = []
    for line in text.strip().split('\n'):
        if line.strip():
            kid = line.split('\t')[0].strip().replace('ko:', '')
            kos.append(kid)
    print(f"    {len(kos)} KO IDs", flush=True)
    return kos

def parse_ko_entries(text):
    """Extract KO → [COG, ...] from a multi-entry KEGG /get response.

    Records end at a '///' line and a field continues onto any following
    line indented by 12 spaces, so the COG: sub-field may sit either on
    the DBLINKS line itself or on one of its continuations.
    """
    ko2cog = {}
    current_ko = None
    in_dblinks = False
    for line in text.split('\n'):
        if line.startswith('ENTRY'):
            m = re.search(r'(K\d{5})', line)
            if m: current_ko = m.group(1)
            in_dblinks = False
        elif line.startswith('DBLINKS'):
            in_dblinks = True
            # Anchor COG: at a field boundary; a bare 'COG:' also matches
            # inside the CCOG: and ECOG: database tags.
            m = re.search(r'(?:^|\s)COG:\s+(.*)', line)
            if m and current_ko:
                cogs = [c.strip() for c in m.group(1).split()
                        if re.match(r'^COG\d{4,5}$', c.strip())]
                if cogs:
                    ko2cog[current_ko] = cogs
        elif in_dblinks and line.startswith('            '):
            m = re.search(r'(?:^|\s)COG:\s+(.*)', line.strip())
            if m and current_ko:
                cogs = [c.strip() for c in m.group(1).split()
                        if re.match(r'^COG\d{4,5}$', c.strip())]
                if cogs:
                    ko2cog.setdefault(current_ko, []).extend(cogs)
        elif not line.startswith(' '):
            in_dblinks = False
        if line.startswith('///'):
            current_ko = None
            in_dblinks = False
    return ko2cog

def _fetch_one_batch(batch):
    """Fetch one batch of KO entries; returns {KO: [COG, ...]}."""
    query = '+'.join(f'ko:{k}' for k in batch)
    text = fetch(f"https://rest.kegg.jp/get/{query}", delay=0.05)
    return parse_ko_entries(text)

def batch_fetch_ko_cog(ko_ids, cache_file=None, resume=False):
    """Fetch every KO entry and return {KO: [COG, ...]}.

    A batch that fails all its retries is dropped rather than raised, so the
    returned mapping can be incomplete; rerun with resume=True to pick up
    whatever the cache is missing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    batch_size = 10
    n_workers = 3  # KEGG limit: 3 requests/sec

    ko2cog_all = {}
    done_kos = set()
    if resume and cache_file and Path(cache_file).exists():
        with open(cache_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ko2cog_all[parts[0]] = parts[1].split(',')
                    done_kos.add(parts[0])
                elif len(parts) == 1:
                    done_kos.add(parts[0])
        print(f"    Resumed: {len(done_kos)} KOs already fetched, "
              f"{len(ko2cog_all)} with COG links", flush=True)

    remaining = [k for k in ko_ids if k not in done_kos]
    batches = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
    est_min = len(batches) * 0.35 / n_workers / 60
    print(f"  Step 2: Fetching {len(remaining)} KO entries "
          f"({len(batches)} batches, {n_workers} threads, ~{est_min:.0f} min)...",
          flush=True)

    cache_f = open(cache_file, 'a') if cache_file else None
    lock = threading.Lock()
    n_found = len(ko2cog_all)
    n_done = 0
    t0 = time.time()

    def process_batch(batch):
        nonlocal n_found, n_done
        try:
            results = _fetch_one_batch(batch)
        except Exception as e:
            return
        with lock:
            for ko in batch:
                if ko in results:
                    ko2cog_all[ko] = results[ko]
                    n_found += 1
                    if cache_f:
                        cache_f.write(f"{ko}\t{','.join(results[ko])}\n")
                else:
                    if cache_f:
                        cache_f.write(f"{ko}\n")
            n_done += len(batch)
            if n_done % 1000 < batch_size:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(remaining) - n_done) / max(rate, 0.01)
                print(f"    {n_done:>6}/{len(remaining)} fetched, "
                      f"{n_found} with COG, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA", flush=True)
                if cache_f: cache_f.flush()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(process_batch, b) for b in batches]
        for f in as_completed(futs):
            f.result()

    if cache_f: cache_f.close()
    print(f"    Done: {n_found} KOs have COG cross-references "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ko2cog_all

def get_ko_module():
    print("  Step 3: Fetching KO→Module links...", flush=True)
    text = fetch("https://rest.kegg.jp/link/module/ko", delay=0.5)
    ko2mod = defaultdict(set)
    for line in text.strip().split('\n'):
        if not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) == 2:
            ko = parts[0].replace('ko:', '')
            mod = parts[1].replace('md:', '')
            ko2mod[ko].add(mod)
    print(f"    {len(ko2mod)} KOs → Modules", flush=True)
    return ko2mod

def get_module_names():
    print("  Step 3b: Fetching module descriptions...", flush=True)
    text = fetch("https://rest.kegg.jp/list/module", delay=0.5)
    names = {}
    for line in text.strip().split('\n'):
        if not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            names[parts[0].strip()] = parts[1].strip()
    print(f"    {len(names)} module descriptions", flush=True)
    return names

def get_cog_categories():
    print("  Step 4: Fetching COG functional categories...", flush=True)
    try:
        text = fetch("https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.def.tab", delay=0)
        cog2cats = {}; cog2path = {}
        for line in text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                cog2cats[parts[0].strip()] = list(parts[1].strip())
                if len(parts) > 4 and parts[4].strip():
                    cog2path[parts[0].strip()] = parts[4].strip()
        print(f"    {len(cog2cats)} COGs with categories, {len(cog2path)} with pathways", flush=True)
        return cog2cats, cog2path
    except Exception as e:
        print(f"    NCBI fetch failed: {e}", flush=True)
        return {}, {}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--feather', default='data/COG_train1_phylum.feather')
    pa.add_argument('--output', default='data/module_matrix_kegg.pt')
    pa.add_argument('--cache', default='ko_cog_cache.tsv',
                    help='Cache file for KO→COG fetching (for resume)')
    pa.add_argument('--resume', action='store_true',
                    help='Resume from cache file')
    pa.add_argument('--min-module-size', type=int, default=3)
    A = pa.parse_args()

    print(f"Loading vocabulary from {A.feather}...", flush=True)
    df = pd.read_feather(A.feather)
    # Rows of M follow the lexicographic COG order that load_feathers gives
    # the models, so row i of M is visible spin i.
    cog_names = sorted([c for c in df.columns if c.startswith('COG')])
    N = len(cog_names)
    cog_idx = {c: i for i, c in enumerate(cog_names)}
    print(f"  {N} COGs in vocabulary\n", flush=True)

    ko_ids = get_all_ko_ids()
    ko2cog = batch_fetch_ko_cog(ko_ids, cache_file=A.cache, resume=A.resume)

    cog2ko = defaultdict(set)
    for ko, cogs in ko2cog.items():
        for cog in cogs:
            cog2ko[cog].add(ko)
    print(f"\n  COG→KO mapping: {len(cog2ko)} COGs mapped to KOs", flush=True)
    in_vocab = sum(1 for c in cog_names if c in cog2ko)
    print(f"  In our vocabulary: {in_vocab}/{N}", flush=True)

    ko2mod = get_ko_module()
    mod_names = get_module_names()

    cog2mod = defaultdict(set)
    for cog in cog_names:
        for ko in cog2ko.get(cog, set()):
            for mod in ko2mod.get(ko, set()):
                cog2mod[cog].add(mod)

    mod_counts = defaultdict(int)
    for cog in cog_names:
        for mod in cog2mod.get(cog, set()):
            mod_counts[mod] += 1

    valid_mods = sorted([m for m, c in mod_counts.items() if c >= A.min_module_size])
    mod_idx = {m: i for i, m in enumerate(valid_mods)}
    n_mod = len(valid_mods)

    M_kegg = torch.zeros(N, n_mod)
    for cog in cog_names:
        ci = cog_idx[cog]
        for mod in cog2mod.get(cog, set()):
            if mod in mod_idx:
                M_kegg[ci, mod_idx[mod]] = 1.0

    kegg_sizes = M_kegg.sum(dim=0)
    kegg_mapped = int((M_kegg.sum(dim=1) > 0).sum())
    print(f"\n  KEGG modules: {n_mod} with >= {A.min_module_size} COGs", flush=True)
    print(f"  {kegg_mapped}/{N} COGs assigned to >= 1 KEGG module", flush=True)
    print(f"  {int(M_kegg.sum())} total memberships", flush=True)

    sz = kegg_sizes.numpy()
    print(f"\n  KEGG module size distribution:", flush=True)
    for lo, hi in [(3,5), (5,10), (10,20), (20,50), (50,200)]:
        n = int(((sz >= lo) & (sz < hi)).sum())
        if n > 0: print(f"    {lo:>3d}-{hi:<3d}: {n} modules", flush=True)

    print(f"\n  Top 20 KEGG modules:", flush=True)
    for i in kegg_sizes.argsort(descending=True)[:20]:
        i = i.item()
        desc = mod_names.get(valid_mods[i], valid_mods[i])
        print(f"    {valid_mods[i]:>8s} ({int(kegg_sizes[i]):>3d} COGs): {desc[:55]}", flush=True)

    cog2cats, cog2path = get_cog_categories()

    all_cats = sorted(COG_CATEGORIES.keys())
    cat_idx = {c: i for i, c in enumerate(all_cats)}
    n_cat = len(all_cats)
    M_cat = torch.zeros(N, n_cat)
    for cog in cog_names:
        for cat in cog2cats.get(cog, []):
            if cat in cat_idx:
                M_cat[cog_idx[cog], cat_idx[cat]] = 1.0
    cat_sizes = M_cat.sum(dim=0)
    cat_mapped = int((M_cat.sum(dim=1) > 0).sum())
    print(f"\n  Functional categories: {n_cat}, {cat_mapped}/{N} mapped", flush=True)

    path_counts = defaultdict(int)
    for cog in cog_names:
        pw = cog2path.get(cog, '')
        if pw: path_counts[pw] += 1
    valid_paths = sorted([p for p, c in path_counts.items() if c >= A.min_module_size])
    path_idx = {p: i for i, p in enumerate(valid_paths)}
    n_path = len(valid_paths)
    M_path = torch.zeros(N, n_path)
    for cog in cog_names:
        pw = cog2path.get(cog, '')
        if pw in path_idx:
            M_path[cog_idx[cog], path_idx[pw]] = 1.0
    path_sizes = M_path.sum(dim=0)

    # Trained models expect the column order [26 categories | COG pathways |
    # KEGG modules]; sizes and module_names are concatenated to match.
    M = torch.cat([M_cat, M_path, M_kegg], dim=1)
    sizes = torch.cat([cat_sizes, path_sizes, kegg_sizes])
    module_names = all_cats + valid_paths + valid_mods
    n_total = M.shape[1]

    print(f"\n  Combined matrix: {N} x {n_total} "
          f"({n_cat} cats + {n_path} COG pathways + {n_mod} KEGG modules)", flush=True)
    print(f"  Total COGs with any assignment: "
          f"{int((M.sum(dim=1) > 0).sum())}/{N}", flush=True)

    result = dict(
        cog_names=cog_names,
        M=M, sizes=sizes, module_names=module_names,
        n_cat=n_cat, n_path=n_path, n_kegg=n_mod,
        M_cat=M_cat, cat_sizes=cat_sizes,
        cat_names=all_cats,
        cat_descs=[COG_CATEGORIES[c] for c in all_cats],
        M_path=M_path, path_sizes=path_sizes, path_names=valid_paths,
        M_kegg=M_kegg, kegg_sizes=kegg_sizes,
        kegg_names=valid_mods,
        kegg_descs=[mod_names.get(m, m) for m in valid_mods],
    )
    torch.save(result, A.output)
    print(f"\n  Saved: {A.output}", flush=True)
    print(f"  M:      {M.shape} ({n_cat} cat + {n_path} COG path + {n_mod} KEGG mod)", flush=True)

if __name__ == '__main__':
    main()
