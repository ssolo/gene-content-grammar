#!/usr/bin/env python3.11
"""Set up the diamond + NCBI COG-2020 reference for assigning pathogen
proteins to COG families (the same COG namespace as the Ising J).

Downloads (idempotent) into data/interactome/cog2020/:
  - diamond linux64 binary (v2.1.9)
  - cog-20.fa.gz       (COG-2020 member protein sequences; headers = protein
                        accession with '.'->'_')
  - cog-20.cog.csv     (per-protein COG assignment; col2 = protein accession,
                        col6 = COG id)
  - cog-20.def.tab     (COG id -> name/category)

Builds:
  - cog-20.fa          (gunzipped)
  - cog2020.dmnd       (diamond protein DB; subject ids = fa headers)
  - protein_to_cog.pkl (dict: normalized protein accession -> COG id)

Mapping (done later by map_proteome_to_cog.py): diamond blastp a STRING
proteome against cog2020.dmnd, take the best-bitscore hit per query with
e-value <= 1e-3, and resolve its subject id to a COG via protein_to_cog.pkl.
This is a COG *family* assignment (cross-species orthology), the same kind of
call eggNOG makes for the training COGs -- NOT an exact-protein 90% match.

Run on a machine with outbound HTTPS access:
  python scripts/interactome/setup_cog_diamond.py --outdir data/interactome/cog2020
"""
import gzip
import os
import pickle
import subprocess
import sys
import argparse

DIR = "data/interactome/cog2020"
URLS = {
    "diamond.tgz": "https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz",
    "cog-20.fa.gz": "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.fa.gz",
    "cog-20.cog.csv": "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.cog.csv",
    "cog-20.def.tab": "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.def.tab",
}


def fetch(name, url, outdir, timeout):
    dst = os.path.join(outdir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("  have %s (%d bytes)" % (name, os.path.getsize(dst)))
        return dst
    print("  downloading %s ..." % name)
    subprocess.run(["curl", "-sSL", "--fail", "--max-time", str(timeout), "-o", dst, url], check=True)
    print("    -> %d bytes" % os.path.getsize(dst))
    return dst


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download COG-2020 references and build the DIAMOND database."
    )
    p.add_argument("--outdir", default=DIR, help="Output/setup directory.")
    p.add_argument(
        "--curl-timeout",
        type=int,
        default=1800,
        help="Maximum seconds per curl download.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    for name, url in URLS.items():
        fetch(name, url, outdir, args.curl_timeout)

    dmnd = os.path.join(outdir, "diamond")
    if not os.path.exists(dmnd):
        subprocess.run(["tar", "xzf", os.path.join(outdir, "diamond.tgz"), "-C", outdir, "diamond"], check=True)
    os.chmod(dmnd, 0o755)
    print("diamond:", subprocess.run([dmnd, "version"], capture_output=True, text=True).stdout.strip())

    fa = os.path.join(outdir, "cog-20.fa")
    if not os.path.exists(fa) or os.path.getsize(fa) == 0:
        print("gunzip cog-20.fa.gz ...")
        with gzip.open(os.path.join(outdir, "cog-20.fa.gz"), "rb") as fi, open(fa, "wb") as fo:
            while True:
                b = fi.read(1 << 22)
                if not b:
                    break
                fo.write(b)
    print("cog-20.fa:", os.path.getsize(fa), "bytes")

    pkl = os.path.join(outdir, "protein_to_cog.pkl")
    if not os.path.exists(pkl):
        print("building protein_to_cog map from cog-20.cog.csv ...")
        p2c = {}
        with open(os.path.join(outdir, "cog-20.cog.csv")) as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 7:
                    continue
                acc = parts[2].replace(".", "_")  # cog-20.fa headers write '.' as '_'
                cog = parts[6]
                # cog-20.cog.csv gives the best assignment per protein REGION, so a
                # multi-domain protein has several rows; the first is kept as the
                # protein's COG.
                if acc not in p2c:
                    p2c[acc] = cog
        with open(pkl, "wb") as f:
            pickle.dump(p2c, f)
        print("  %d protein->COG entries" % len(p2c))
    else:
        print("have protein_to_cog.pkl")

    db = os.path.join(outdir, "cog2020.dmnd")
    if not os.path.exists(db):
        print("diamond makedb ...")
        subprocess.run([dmnd, "makedb", "--in", fa, "-d", os.path.join(outdir, "cog2020"),
                        "--threads", str(os.cpu_count() or 4)], check=True)
    print("diamond DB:", db, os.path.getsize(db) if os.path.exists(db) else "MISSING", "bytes")
    print("SETUP COMPLETE")


if __name__ == "__main__":
    main()
