#!/usr/bin/env python3
"""Turn data/stratified_eval.tsv into two LaTeX supplementary tables.

Input (long): dim, stratum, fn, n_genomes, MCC, ECE
  dim in {completeness, phylum}.

Output:
  --compl-out    MCC + ECE by CheckM completeness bin, as a table
  --phylum-out   MCC + ECE by GTDB phylum, as a longtable

Each row: stratum | n | MCC@fn(0.5/0.75/0.9) | ECE@fn(0.5/0.75/0.9). n is the
per-stratum genome count, the same genomes at every fn, so the max over fn is
taken. Phyla are grouped Bacteria-then-Archaea (GTDB archaeal-phyla set below)
and sorted by n within each group; phyla with fewer than --min-n genomes are
dropped, since the pooled MCC/ECE is unstable on a handful of genomes.
"""
import argparse
import pandas as pd

# GTDB archaeal phyla (p__ prefix). Anything not listed is treated as Bacteria
# for the table's domain grouping (the overwhelming majority of phyla/genomes).
ARCHAEAL = {
    'p__Halobacteriota', 'p__Methanobacteriota', 'p__Methanobacteriota_A',
    'p__Methanobacteriota_B', 'p__Thermoproteota', 'p__Thermoplasmatota',
    'p__Asgardarchaeota', 'p__Nanoarchaeota', 'p__Aenigmatarchaeota',
    'p__Altiarchaeota', 'p__Hadarchaeota', 'p__Hydrothermarchaeota',
    'p__Iainarchaeota', 'p__Micrarchaeota', 'p__Nanohaloarchaeota',
    'p__Undinarchaeota', 'p__Huberarchaeota', 'p__EX4484-52',
    'p__B1Sed10-29', 'p__SpSt-1190', 'p__QMZS01', 'p__PWEA01',
}
FNS = [0.5, 0.75, 0.9]


def pivot(df):
    """stratum -> {n, mcc<fn>, ece<fn>}; a stratum missing an fn gets NaN there."""
    out = {}
    for stratum, g in df.groupby('stratum'):
        rec = {'n': int(g['n_genomes'].max())}
        for fn in FNS:
            r = g[abs(g['fn'] - fn) < 1e-6]
            rec[f'mcc{fn}'] = float(r['MCC'].iloc[0]) if len(r) else float('nan')
            rec[f'ece{fn}'] = float(r['ECE'].iloc[0]) if len(r) else float('nan')
        out[stratum] = rec
    return out


def cell(v):
    return '--' if v != v else f'{v:.3f}'      # NaN -> dash


def row(label, rec):
    mcc = ' & '.join(cell(rec[f'mcc{fn}']) for fn in FNS)
    ece = ' & '.join(cell(rec[f'ece{fn}']) for fn in FNS)
    return f'{label} & {rec["n"]:,} & {mcc} & {ece} \\\\'


HDR = (r'\textbf{%s} & $n$ & \multicolumn{3}{c}{MCC} & \multicolumn{3}{c}{ECE} \\'
       + '\n' + r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}'
       + '\n' + r' & & 0.5 & 0.75 & 0.9 & 0.5 & 0.75 & 0.9 \\')


def compl_table(piv):
    order = ['0-80', '80-90', '90-95', '95-99', '99-100', 'NA']
    lines = [r'\begin{table}[ht]\centering\small',
             r'\caption{Recovery (MCC) and calibration (ECE) stratified by CheckM '
             r'genome completeness, pooled over held-out validation genomes at '
             r'false-negative rates $\mathrm{fn}=0.5/0.75/0.9$ (fixed '
             r'$\mathrm{fp}=0.01$), generalist HO-$T{=}20$. $n$ = genomes in bin '
             r'(summed across CV splits).}',
             r'\label{tab:strat_compl}',
             r'\begin{tabular}{lrrrrrrr}\toprule',
             HDR % 'Completeness',
             r'\midrule']
    for b in order:
        if b in piv:
            lab = b if b == 'NA' else f'{b}\\%'
            lines.append(row(lab, piv[b]))
    lines += [r'\bottomrule\end{tabular}\end{table}']
    return '\n'.join(lines)


def phylum_table(piv, min_n):
    items = [(s, r) for s, r in piv.items() if r['n'] >= min_n]
    arc = sorted([x for x in items if x[0] in ARCHAEAL], key=lambda x: -x[1]['n'])
    bac = sorted([x for x in items if x[0] not in ARCHAEAL], key=lambda x: -x[1]['n'])
    dropped = len(piv) - len(items)
    cap = (r'\caption{Recovery (MCC) and calibration (ECE) stratified by GTDB '
           r'phylum, pooled over held-out validation genomes at '
           r'$\mathrm{fn}=0.5/0.75/0.9$ (fixed $\mathrm{fp}=0.01$), generalist '
           r'HO-$T{=}20$. Whole-phylum holdout: each phylum is scored by a model '
           r'that never trained on it. Phyla with $<%d$ genomes omitted '
           r'(%d phyla). $n$ summed across CV splits.}'
           r'\label{tab:strat_phylum}\\' % (min_n, dropped))
    lines = [r'\begin{longtable}{lrrrrrrr}', cap,
             r'\toprule', HDR % 'Phylum', r'\midrule\endfirsthead',
             r'\toprule', HDR % 'Phylum (cont.)', r'\midrule\endhead',
             r'\midrule\multicolumn{8}{r}{\small\itshape continued on next page}\\ \endfoot',
             r'\bottomrule\endlastfoot']
    if bac:
        lines.append(r'\multicolumn{8}{l}{\textit{Bacteria}}\\')
        lines += [row(s.replace('p__', '').replace('_', r'\_'), r) for s, r in bac]
    if arc:
        lines.append(r'\midrule\multicolumn{8}{l}{\textit{Archaea}}\\')
        lines += [row(s.replace('p__', '').replace('_', r'\_'), r) for s, r in arc]
    lines.append(r'\end{longtable}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='data/stratified_eval.tsv')
    ap.add_argument('--min-n', type=int, default=30,
                    help='drop phyla with fewer genomes (unstable MCC/ECE).')
    ap.add_argument('--compl-out', default='analysis/tab_strat_compl.tex')
    ap.add_argument('--phylum-out', default='analysis/tab_strat_phylum.tex')
    a = ap.parse_args()
    df = pd.read_csv(a.inp, sep='\t')
    pc = pivot(df[df['dim'] == 'completeness'])
    pp = pivot(df[df['dim'] == 'phylum'])
    open(a.compl_out, 'w').write(compl_table(pc) + '\n')
    open(a.phylum_out, 'w').write(phylum_table(pp, a.min_n) + '\n')
    kept = sum(1 for r in pp.values() if r['n'] >= a.min_n)
    print(f'{a.inp}: {len(pc)} completeness bins, {len(pp)} phyla '
          f'({kept} kept at min-n={a.min_n}) -> {a.compl_out}, {a.phylum_out}')


if __name__ == '__main__':
    main()
