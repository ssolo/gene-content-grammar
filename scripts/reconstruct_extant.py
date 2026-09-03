#!/usr/bin/env python3
"""reconstruct_extant.py -- ground-truth recovery test on a real genome.

An extant genome has known gene content, so recovery can be scored directly.  A
real genome is corrupted with false-negative (gene loss) and false-positive
(spurious gain) noise at several FN levels and denoised with the model from the
cross-validation split that holds its phylum out.  Per FN level and COG family
the output TSV carries:

    truth            1 if the COG is really in the genome
    input_present    1 if it survived corruption (the noisy model input)
    denoised_prob    recovered present-probability in [0, 1]

separating genes erased by FN and switched back on from spurious genes injected
by FP and switched back off.

The split is found by scanning each split's val feather for the requested
species.  --val-glob must name the data partition the --models glob was trained
on, or the "held-out" genome may have been in training (generalist ->
COG_val{split}; bacterial specialist -> COG_bac_val{split}).

Needs torch; a single genome runs on CPU.  Example:
  python3 scripts/reconstruct_extant.py \
      --species 'Escherichia coli' --domain d__Bacteria \
      --val-glob data/COG_val{split}_phylum.feather \
      --models 'gsd_results_higher_order_nohidden_T20_split{split}/model_ho3.pth' \
      --model-label 'T20 generalist (held-out split)' \
      --fn 0.5 0.75 0.9 --fp 0.05 \
      --module-matrix data/module_matrix_kegg.pt \
      --out recover_ecoli_generalist.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ancestral_node import build_model_for_ckpt, run_model  # noqa: E402
from ising_denoiser.modules import load_module_matrix  # noqa: E402
from ising_denoiser.training import strip_compile_prefix  # noqa: E402


def load_vocab(module_matrix):
    """COG vocabulary in the model's own column order, from the module matrix."""
    meta = torch.load(module_matrix, weights_only=False)
    return list(meta['cog_names'])


def _profile_from_row(row, cols, cog_names):
    present = np.zeros(len(cog_names), dtype=np.float32)
    for i, c in enumerate(cog_names):
        if c in cols:
            try:
                if float(row[c]) > 0:
                    present[i] = 1.0
            except (ValueError, TypeError):
                pass
    return present


def find_held_out_genome(species, domain, val_glob, cog_names, profile_glob=''):
    """Return (split, present_vec, accession, species, n_genes) from a split that
    held the genome out of training.

    A split whose val feather (`val_glob`) contains `species` is a clean test.
    Failing that, and given `profile_glob`, fall back to a cross-partition test:
    profile and phylum from `profile_glob` (the full, unsubsampled feathers),
    model from the `val_glob` split whose held-out set contains that phylum.
    Needed because a subsampled partition drops most genomes from val (the
    bacterial specialist caps val at 3k/split) even where their phylum is held
    out.
    """
    is_acc = ('GCF_' in species) or ('GCA_' in species)
    spec_n = species.replace('RS_', '').replace('GB_', '')

    def _sel(d):
        # A GCF_/GCA_ accession pins a single genome. GTDB splits Escherichia
        # coli across species clusters (coli, coli_A ... coli_F), so a loose
        # species substring resolves to a different accession in each split.
        if is_acc:
            a = (d['accession'].astype(str).str.replace('RS_', '', regex=False)
                                           .str.replace('GB_', '', regex=False))
            return a == spec_n
        return d['species'].astype(str).str.contains(species, case=False, na=False, regex=False)

    for split in range(1, 11):
        p = val_glob.format(split=split)
        if not Path(p).exists():
            continue
        df = pd.read_feather(p)
        sel = _sel(df)
        if domain and 'domain' in df.columns:
            sel = sel & (df['domain'].astype(str) == domain)
        m = df[sel]
        if len(m) == 0:
            continue
        row = m.iloc[0]
        present = _profile_from_row(row, set(df.columns), cog_names)
        return split, present, str(row.get('accession', '?')), \
            str(row.get('species', species)), int(present.sum())

    # Cross-partition fallback.
    if profile_glob:
        for split in range(1, 11):
            p = profile_glob.format(split=split)
            if not Path(p).exists():
                continue
            df = pd.read_feather(p)
            sel = _sel(df)
            if domain and 'domain' in df.columns:
                sel = sel & (df['domain'].astype(str) == domain)
            m = df[sel]
            if len(m) == 0:
                continue
            row = m.iloc[0]
            present = _profile_from_row(row, set(df.columns), cog_names)
            acc = str(row.get('accession', '?'))
            spname = str(row.get('species', species))
            clade = str(row.get('phylum', '?'))
            for s in range(1, 11):
                pv = val_glob.format(split=s)
                if not Path(pv).exists():
                    continue
                dfv = pd.read_feather(pv)
                if not ('phylum' in dfv.columns and (dfv['phylum'].astype(str) == clade).any()):
                    continue
                # Splits are independent whole-clade shuffles: presence in val{s}
                # does not imply absence from train{s}, so verify before trusting
                # the model as leak-free.
                ptr = val_glob.replace('val', 'train').format(split=s)
                if Path(ptr).exists():
                    try:
                        dtr = pd.read_feather(ptr, columns=['phylum'])
                    except (TypeError, ValueError):
                        dtr = pd.read_feather(ptr)
                    n_train = int((dtr['phylum'].astype(str) == clade).sum())
                    if n_train > 0:
                        print(f'  split {s}: phylum {clade} in val but ALSO {n_train} in '
                              f'train -- skipping (would leak)', file=sys.stderr)
                        continue
                    verified = 'train verified clade-free'
                else:
                    verified = 'train feather absent; relying on whole-clade holdout'
                print(f'cross-partition test: {spname} is absent from {val_glob} val '
                      f'(subsampled); profile from {profile_glob}; phylum {clade} held out '
                      f'in split {s} ({verified}) -- that split never trained on the clade.',
                      file=sys.stderr)
                return s, present, acc, spname, int(present.sum())
            raise SystemExit(f'phylum {clade!r} of {species!r} is not cleanly held out in '
                             f'any val split of {val_glob} (in val but also in train); no '
                             f'leak-free model available')

    # Nothing matched: list held-out species so the caller can fix --species.
    for split in range(1, 11):
        p = val_glob.format(split=split)
        if not Path(p).exists():
            continue
        df = pd.read_feather(p)
        if domain and 'domain' in df.columns:
            df = df[df['domain'].astype(str) == domain]
        ex = sorted(df['species'].astype(str).unique())[:20]
        print(f'[split {split}] sample held-out species ({domain}): {ex}',
              file=sys.stderr)
    raise SystemExit(f'species {species!r} ({domain}) not found in any '
                     f'val split of {val_glob} (and no --profile-glob fallback)')


def load_quality_map(meta_paths):
    """accession -> (checkm_completeness, checkm_contamination) from the GTDB
    metadata TSV(s).  Read with the csv module so pyarrow is not required."""
    import csv as _csv
    q = {}
    for p in [x for x in meta_paths.split(',') if x]:
        if not Path(p).exists():
            print(f'[warn] --gtdb-meta path not found: {p}', file=sys.stderr)
            continue
        with open(p) as fh:
            for row in _csv.DictReader(fh, delimiter='\t'):
                try:
                    q[row['accession']] = (float(row['checkm_completeness']),
                                           float(row['checkm_contamination']))
                except (KeyError, ValueError):
                    pass
    return q


def find_median_genome(domain, val_glob, cog_names,
                       qmap=None, min_compl=0.0, max_contam=100.0):
    """Pick the held-out genome whose true gene count is closest to the median
    over all held-out genomes of `domain`, across every val split.

    Returns the same tuple as find_held_out_genome.  With `qmap` and a quality
    bar set, only genomes with CheckM completeness >= min_compl and
    contamination <= max_contam are eligible: recall-vs-truth against a
    fragmentary MAG penalises the model for inferring genes the assembly lacks.
    """
    vocab = set(cog_names)
    recs = []           # (gene_count, split, row_index, accession) per genome
    dfs = {}
    for split in range(1, 11):
        p = val_glob.format(split=split)
        if not Path(p).exists():
            continue
        df = pd.read_feather(p)
        dfs[split] = df
        mask = (df['domain'].astype(str) == domain) if (domain and 'domain'
                in df.columns) else pd.Series(True, index=df.index)
        cols = [c for c in df.columns if c in vocab]
        counts = (df[cols] > 0).sum(axis=1)
        for idx in df.index[mask]:
            recs.append((int(counts[idx]), split, idx,
                         str(df.loc[idx].get('accession', '?'))))
    if not recs:
        raise SystemExit(f'no {domain} genomes found in any split of {val_glob}')
    if qmap and (min_compl > 0.0 or max_contam < 100.0):
        n0 = len(recs)
        recs = [r for r in recs if r[3] in qmap
                and qmap[r[3]][0] >= min_compl and qmap[r[3]][1] <= max_contam]
        if not recs:
            raise SystemExit(f'no {domain} genomes pass the quality filter '
                             f'(completeness>={min_compl}, contam<={max_contam})')
        print(f'quality filter: {len(recs)}/{n0} {domain} genomes pass '
              f'(completeness>={min_compl}, contam<={max_contam})',
              file=sys.stderr)
    counts = np.array([r[0] for r in recs])
    med = float(np.median(counts))
    j = int(np.argmin(np.abs(counts - med)))
    cnt, split, idx, _acc = recs[j]
    df = dfs[split]
    row = df.loc[idx]
    cols = set(df.columns)
    present = np.zeros(len(cog_names), dtype=np.float32)
    for i, c in enumerate(cog_names):
        if c in cols:
            try:
                if float(row[c]) > 0:
                    present[i] = 1.0
            except (ValueError, TypeError):
                pass
    acc = str(row.get('accession', '?'))
    spname = str(row.get('species', '?'))
    print(f'median {domain} gene count = {med:.0f} over {len(recs)} held-out '
          f'genomes; picked {spname} (acc={acc}, {cnt} genes, split {split})',
          file=sys.stderr)
    return split, present, acc, spname, int(present.sum())


def corrupt(clean_pm1, fn, fp, seed, fp_mode='uniform', marg=None):
    """Corrupt clean_pm1, a (N,) tensor in {-1,+1}, under a deterministic seed.

    Present COGs flip to absent with probability fn (the FN model of
    ising_denoiser.metrics.eval_spectra_2d); false positives are then added among
    the absent COGs:

    uniform   each absent COG flips present w.p. fp; the independent-flip null.
    marginal  the same expected count, round(fp * n_absent), drawn in proportion
              to the cross-genome marginal frequency p_c (`marg`) -- common-gene,
              module-incoherent contamination.
    """
    g = torch.Generator().manual_seed(int(seed))
    ny = clean_pm1.clone()
    # Draw on CPU so a given seed gives the same flips on any device.
    r = torch.rand(ny.shape, generator=g).to(ny.device)
    ny[(clean_pm1 == 1) & (r < fn)] = -1.0
    absent = (clean_pm1 == -1)
    if fp_mode == 'marginal' and marg is not None:
        n_fp = int(round(fp * int(absent.sum().item())))
        if n_fp > 0:
            w = marg.clone().cpu()
            w[~absent.cpu()] = 0.0
            nz = int((w > 0).sum().item())
            if nz > 0:
                pick = torch.multinomial(w, min(n_fp, nz), replacement=False,
                                         generator=g)
                ny[pick.to(ny.device)] = 1.0
    else:
        r2 = torch.rand(ny.shape, generator=g).to(ny.device)
        ny[absent & (r2 < fp)] = 1.0
    return ny


def accounting(truth, inp, pred):
    """Recovery confusion counts from three boolean (N,) arrays."""
    t, i, p = truth.astype(bool), inp.astype(bool), pred.astype(bool)
    return dict(
        truth_total=int(t.sum()),
        input_total=int(i.sum()),
        kept=int((t & i & p).sum()),            # true gene, survived, kept on
        recovered=int((t & ~i & p).sum()),      # true gene erased by FN, restored
        missed=int((t & ~p).sum()),             # true gene left off
        removed_fp=int((~t & i & ~p).sum()),    # injected FP correctly removed
        retained_fp=int((~t & i & p).sum()),    # injected FP wrongly kept
        hallucinated=int((~t & ~i & p).sum()),  # spurious gene invented
        injected_fp=int((~t & i).sum()),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--select', choices=['species', 'median'], default='species',
                    help="how to choose the genome: by --species name, or the "
                         "median-gene-count held-out genome of --domain")
    ap.add_argument('--species', default='',
                    help="substring match on the genome's species name "
                         "(required when --select species)")
    ap.add_argument('--domain', default='',
                    help="GTDB domain filter, e.g. d__Bacteria / d__Archaea")
    ap.add_argument('--min-completeness', type=float, default=0.0,
                    help="(--select median) keep only held-out genomes with GTDB "
                         "CheckM completeness >= this %% (needs --gtdb-meta). "
                         "Use to pick a HIGH-QUALITY median rather than a "
                         "fragmentary MAG.")
    ap.add_argument('--max-contam', type=float, default=100.0,
                    help="(--select median) keep only genomes with GTDB CheckM "
                         "contamination <= this %% (needs --gtdb-meta).")
    ap.add_argument('--gtdb-meta', default='',
                    help="comma-separated GTDB metadata TSV(s) "
                         "(ar53_/bac120_metadata_r220.tsv) for the completeness "
                         "filter; maps accession -> checkm_completeness/contamination. "
                         "Plain TSV, so it reads on any node.")
    ap.add_argument('--profile-glob', default='',
                    help='fallback feather glob ({split}) for the true profile + '
                         'phylum when the genome is absent from --val-glob (e.g. '
                         'the subsampled bacterial val); the model split is then '
                         'the one whose held-out set contains the genome phylum.')
    ap.add_argument('--val-glob', default='data/COG_val{split}_phylum.feather',
                    help="held-out feather glob with a {split} placeholder")
    ap.add_argument('--models', required=True,
                    help="model checkpoint glob with a {split} placeholder")
    ap.add_argument('--model-label', default='model')
    ap.add_argument('--fn', type=float, nargs='+', default=[0.5, 0.75, 0.9])
    ap.add_argument('--fp', type=float, default=0.05)
    ap.add_argument('--fp-mode', choices=['uniform', 'marginal'], default='uniform',
                    help="how false positives are injected. uniform: each absent "
                         "COG flipped w.p. fp (random rare junk). marginal: the "
                         "same expected count drawn by per-COG cross-genome "
                         "marginal frequency p_c (common-gene, module-incoherent "
                         "contamination -- the deep-ancestral failure mode).")
    ap.add_argument('--marginal-freq', default='data/cog_marginal_frequency.tsv',
                    help="COG_ID<TAB>p_c table for --fp-mode marginal "
                         "(scripts/compute_marginal_frequency.py).")
    ap.add_argument('--module-matrix', default='data/module_matrix_kegg.pt')
    ap.add_argument('--device', default='auto',
                    help="auto (default) = CUDA if available, else CPU.")
    ap.add_argument('--out', required=True, help='per-COG long-form TSV')
    A = ap.parse_args()

    if A.device == 'auto':
        A.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(A.device if (A.device != 'cuda' or
                                    torch.cuda.is_available()) else 'cpu')

    cog_names = load_vocab(A.module_matrix)
    N = len(cog_names)
    marg = None
    if A.fp_mode == 'marginal':
        fr = pd.read_csv(A.marginal_freq, sep='\t')
        m = dict(zip(fr.COG_ID, fr.p_c))
        marg = torch.tensor([float(m.get(c, 0.0)) for c in cog_names], dtype=torch.float32)
        print(f'[fp-mode marginal] loaded p_c for {int((marg>0).sum())}/{N} COGs '
              f'(mean {float(marg.mean()):.3f})')
    _, M_mod, M_sizes = load_module_matrix(A.module_matrix, dev, H=1000)
    n_modules = M_mod.shape[1]

    if A.select == 'median':
        qmap = load_quality_map(A.gtdb_meta) if A.gtdb_meta else None
        split, present, acc, spname, ngenes = find_median_genome(
            A.domain, A.val_glob, cog_names,
            qmap=qmap, min_compl=A.min_completeness, max_contam=A.max_contam)
    else:
        if not A.species:
            raise SystemExit('--species is required when --select species')
        split, present, acc, spname, ngenes = find_held_out_genome(
            A.species, A.domain, A.val_glob, cog_names, profile_glob=A.profile_glob)
    print(f'genome: {spname}  acc={acc}  held-out in split {split}  '
          f'({ngenes} COG families present)', file=sys.stderr)

    clean = torch.from_numpy(2.0 * present - 1.0).to(dev)  # {-1,+1}

    ckpt = A.models.format(split=split)
    if not Path(ckpt).exists():
        raise SystemExit(f'model checkpoint not found: {ckpt}')
    sd = torch.load(ckpt, map_location=dev, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = strip_compile_prefix(sd)
    model, cls_kind, T, onsager = build_model_for_ckpt(
        sd, dev, N, n_modules, ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'model: {cls_kind} T={T} onsager={onsager}  '
          f'(missing={len(missing)} unexpected={len(unexpected)})  '
          f'<- {ckpt}', file=sys.stderr)

    rows = []
    print(f'\n=== {spname}  |  {A.model_label} (split {split})  |  '
          f'truth={ngenes}, FP={A.fp:g} ===', file=sys.stderr)
    hdr = (f"{'fn':>5} {'input':>6} {'kept':>5} {'recov':>6} {'missed':>6} "
           f"{'rmFP':>5} {'keptFP':>6} {'hallu':>6} {'injFP':>6}")
    print(hdr, file=sys.stderr)
    for fn in A.fn:
        # Deterministic seed per (fn, fp) cell; the prime multiplier on fp keeps
        # the fn and fp contributions from colliding, so the same genes are
        # flipped across models and the recovery counts stay comparable.
        seed = 42 + int(fn * 1000) + 7919 * int(A.fp * 1000)
        ny = corrupt(clean, fn, A.fp, seed, fp_mode=A.fp_mode, marg=marg)
        x_out = run_model(model, ny.unsqueeze(0), M_mod, M_sizes)
        prob = ((x_out + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()
        inp = (ny.cpu().numpy() > 0).astype(np.float32)
        pred = (prob > 0.5).astype(np.float32)
        a = accounting(present, inp, pred)
        print(f"{fn:5.2f} {a['input_total']:6d} {a['kept']:5d} "
              f"{a['recovered']:6d} {a['missed']:6d} {a['removed_fp']:5d} "
              f"{a['retained_fp']:6d} {a['hallucinated']:6d} {a['injected_fp']:6d}",
              file=sys.stderr)
        for i, c in enumerate(cog_names):
            rows.append((c, f'{fn:g}', int(present[i]), int(inp[i]),
                         f'{prob[i]:.4f}'))

    out = Path(A.out)
    with open(out, 'w') as fh:
        fh.write('COG_ID\tfn\ttruth\tinput_present\tdenoised_prob\n')
        for r in rows:
            fh.write('\t'.join(str(x) for x in r) + '\n')
    # Sidecar key/value TSV the plotters read for panel titles and provenance.
    with open(out.with_suffix('.meta.tsv'), 'w') as fh:
        fh.write('key\tvalue\n')
        fh.write(f'species\t{spname}\naccession\t{acc}\nsplit\t{split}\n')
        fh.write(f'domain\t{A.domain}\nmodel_label\t{A.model_label}\n')
        fh.write(f'truth_total\t{ngenes}\nfp\t{A.fp:g}\n')
        fh.write(f'fn_levels\t{",".join(f"{f:g}" for f in A.fn)}\n')
    print(f'\nwrote {out}  ({len(rows)} rows; {len(A.fn)} FN x {N} COGs)',
          file=sys.stderr)


if __name__ == '__main__':
    main()
