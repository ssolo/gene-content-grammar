#!/usr/bin/env python3.11
"""Assign a STRING proteome to COG families with diamond.

For one STRING taxon: download its protein sequences (STRING v12.0), run
diamond blastp against the COG-2020 reference DB built by setup_cog_diamond.py,
and write a protein -> COG table (best-bitscore hit, e-value <= 1e-3).

The assignment is at COG *family* level (cross-species orthology), consistent
with how eggNOG assigns the COGs the Ising model was trained on.

Needs outbound network access for the sequence download.

  .venv/bin/python scripts/interactome/map_proteome_to_cog.py --taxon 511145

Output: data/interactome/string/<taxon>.protein_to_cog.tsv
        columns: string_protein, cog, pident, evalue, bitscore
"""
import argparse
import gzip
import os
import pickle
import subprocess

COGDIR = "data/interactome/cog2020"
STRDIR = "data/interactome/string"
SEQ_URL = "https://stringdb-downloads.org/download/protein.sequences.v12.0/{t}.protein.sequences.v12.0.fa.gz"


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    subprocess.run(["curl", "-sSL", "--fail", "--max-time", "600", "-o", dst, url], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", required=True)
    ap.add_argument("--evalue", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(STRDIR, exist_ok=True)
    t = a.taxon

    gz = os.path.join(STRDIR, "%s.sequences.fa.gz" % t)
    faa = os.path.join(STRDIR, "%s.sequences.faa" % t)
    print("fetch sequences for taxon", t)
    fetch(SEQ_URL.format(t=t), gz)
    if not os.path.exists(faa):
        with gzip.open(gz, "rt") as fi, open(faa, "w") as fo:
            fo.write(fi.read())
    n_prot = sum(1 for line in open(faa) if line.startswith(">"))
    print("  %d proteins" % n_prot)

    out = os.path.join(STRDIR, "%s.diamond.tsv" % t)
    dmnd = os.path.join(COGDIR, "diamond")
    db = os.path.join(COGDIR, "cog2020.dmnd")
    if not os.path.exists(out):
        print("diamond blastp vs COG-2020 ...")
        subprocess.run([
            dmnd, "blastp", "--query", faa, "--db", db,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
            "--max-target-seqs", "5", "--evalue", str(a.evalue),
            "--more-sensitive", "--threads", str(a.threads), "--quiet",
            "--out", out,
        ], check=True)

    # Reference proteins with no COG are skipped, so the best COG-bearing hit
    # wins even when a higher-scoring hit is unannotated.
    with open(os.path.join(COGDIR, "protein_to_cog.pkl"), "rb") as f:
        p2c = pickle.load(f)
    best = {}
    with open(out) as f:
        for line in f:
            q, s, pid, ev, bit = line.rstrip("\n").split("\t")
            cog = p2c.get(s)
            if cog is None:
                continue
            bit = float(bit)
            if q not in best or bit > best[q][4]:
                best[q] = (cog, float(pid), float(ev), bit, bit)

    res = os.path.join(STRDIR, "%s.protein_to_cog.tsv" % t)
    with open(res, "w") as f:
        f.write("string_protein\tcog\tpident\tevalue\tbitscore\n")
        for q, (cog, pid, ev, bit, _) in best.items():
            f.write("%s\t%s\t%.1f\t%.2e\t%.1f\n" % (q, cog, pid, ev, bit))
    n_mapped = len(best)
    n_cogs = len(set(v[0] for v in best.values()))
    print("mapped %d/%d proteins (%.1f%%) to %d distinct COGs -> %s"
          % (n_mapped, n_prot, 100.0 * n_mapped / max(n_prot, 1), n_cogs, res))


if __name__ == "__main__":
    main()
