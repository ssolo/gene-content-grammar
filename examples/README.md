# Example input -- bring your own gene content

`reconstruct.py denoise` takes a plain TSV of per-COG gene content and returns
the model's denoised per-COG posterior. This is the generic way to run **your
own** genome (or any per-COG presence / copy-number profile) through the
denoiser. The pre-packaged ancestral nodes (`reconstruct.py ancestral --node
LBCA/LACA`) are just specific input tables in this same format.

## Input format

A TAB-separated file with:

- a first column headed `COG`, holding **COG family IDs** (NCBI COG-2020, e.g.
  `COG0001`; sub-family suffixes like `COG0001_0` are accepted and summed per
  bare COG);
- one or more **value columns**. Pick which one to denoise with `--column`
  (default: the first value column).

Each value is the evidence that the family is present in the genome:

- `1` for a gene you observe (plain presence), **or**
- a **copy number** (`2`, `3`, ... ; capped at 1.0 internally), **or**
- a **presence probability** in `(0, 1]` (e.g. a gene-tree/species-tree
  reconciliation posterior).

Families you **omit are treated as absent** -- you list only what is present (or
has nonzero evidence). The model's fixed COG vocabulary defines the universe of
families; IDs outside it are ignored.

`ecoli_gene_content.tsv` is a worked example: the 2,178 COG families present in
*Escherichia coli* (GTDB `RS_GCF_003697165.2`), each with value `1`:

```
COG	Escherichia_coli
COG0001	1
COG0002	1
COG0004	1
...
```

## Run it

```bash
python scripts/reconstruct.py denoise \
    --input examples/ecoli_gene_content.tsv \
    --model bac-FT \
    --out ecoli_denoised.tsv
```

Useful flags:

- `--model`   which denoiser: `bac-FT` (bacterial specialist), `mix-FT`
  (mixed-domain generalist), `arc-FT` (archaeal), the false-positive-tolerant
  `bac-FT-fp` / `mix-FT-fp`, or `plain-T20`. `reconstruct.py --list` lists all.
- `--column NAME`   which value column to denoise (default: the first one).
- `--input-mode`   `raw` (default; use the value as given), `binarize` (treat
  any listed family as fully present), or the softer `soft` / `half` encodings.
- `--inject-fp FRAC`   optional: inject a fraction of random false positives
  first, to watch the model reject them.

## Output

A per-COG TSV (`--out`) with columns `COG_ID`, `input_prob`, and
`mean_<mode>` / `sd_<mode>`: the denoised posterior (mean +/- s.d.) **ensembled
over the 10 cross-validation splits**. A family is called present at
`mean_* > 0.5`. A human-readable summary (present / rescued / silenced counts
and a module breakdown) is written alongside as `<out>.txt`.
