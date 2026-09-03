#!/usr/bin/env python3
"""Build high-completeness (HQ) train/val feathers by a per-domain CheckM filter.

Incomplete genomes inject false-negative label noise into the training target
(genes truly present but unassembled are scored "absent"), biasing the model to
under-call.  Same reason the recovery benchmark uses high-completeness median
genomes.

The thresholds differ by domain because a uniform >=90% cut decimates the
MAG-dominated archaea (3.4k -> 1.1k genomes, 25 phyla lost) while leaving bacteria
at 52k, well above the 15k/split training cap:
    bacteria  completeness >= 90,  contamination <= 5
    archaea   completeness >= 80,  contamination <= 5   (preserve deep diversity)

Writes data/COG_<fam>_hq_<train|val><split>_phylum.feather for fam in --fams,
filtering the existing COG_<fam>_<train|val><split>_phylum.feather row-wise.
"""
import argparse
import pandas as pd

MINCOMPL = {'d__Bacteria': 90.0, 'd__Archaea': 80.0}
MAXCONTAM = 5.0


def hq_mask(df):
    c = pd.to_numeric(df['checkm_completeness'], errors='coerce')
    x = pd.to_numeric(df['checkm_contamination'], errors='coerce')
    minc = df['domain'].map(MINCOMPL).fillna(90.0)   # default 90 for any other domain
    return (c >= minc) & (x <= MAXCONTAM)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fams', nargs='+', default=['bac', 'mix'],
                    help='feather families to filter (COG_<fam>_...).')
    ap.add_argument('--splits', type=int, default=10)
    a = ap.parse_args()
    for fam in a.fams:
        for s in range(1, a.splits + 1):
            for kind in ('train', 'val'):
                src = f'data/COG_{fam}_{kind}{s}_phylum.feather'
                dst = f'data/COG_{fam}_hq_{kind}{s}_phylum.feather'
                try:
                    df = pd.read_feather(src)
                except FileNotFoundError:
                    print(f'  skip (missing): {src}')
                    continue
                out = df[hq_mask(df)].reset_index(drop=True)
                out.to_feather(dst)
                nb = int((out['domain'] == 'd__Bacteria').sum())
                na = int((out['domain'] == 'd__Archaea').sum())
                print(f'  {dst}: {len(out)}/{len(df)} kept '
                      f'({nb} bac, {na} arc, {out["phylum"].nunique()} phyla)')


if __name__ == '__main__':
    main()
