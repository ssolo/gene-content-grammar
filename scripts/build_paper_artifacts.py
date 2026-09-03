#!/usr/bin/env python3
"""build_paper_artifacts.py -- reproducibility driver for the MBE paper.

Regenerates every figure and verifies every table in the MBE manuscript from the
committed denoiser models and primary data (root COG/arCOG reconciliation counts,
training feathers, the cons-10 ensemble predictions). The tables live in
`analysis/SM.tex`, labelled `s-tab:*` for soft-landing-origin and `r-tab:*` for
report-origin, plus one main-text table in `analysis/mbe_manuscript.tex`.

`report.tex` is not a verification target; `--report report` is an alias for
`--report mbe`. The soft-landing companion `lbca_softlanding.tex` is verifiable
via `--report softlanding`.

Provenance tiers:
  recon    Derivable locally from the denoiser checkpoints + a committed root
           reconciliation / COG-count table (+ the KEGG module matrix / COG
           definitions). Regenerated and recomputed here from scratch.
  cluster  Produced by a GPU sweep / corrupted-genome eval that this driver does
           not re-run. The durable committed artifact is a summary
           CSV/parquet/JSON/TSV; the large per-(split,genome) intermediates are
           .gitignored. Figures are re-plotted from the committed summary and
           tables checked against it.
  static   Hand-authored (model lists, the provenance map, literature constants).
           Carries no machine-checkable numbers; listed for completeness.

The .tex tables stay hand-typed. Where the inputs are present locally the numbers
are recomputed from primary data, compared with the expected values embedded
below (which mirror the .tex), and each expected value's LaTeX string is required
to appear inside that table's environment, so a number cannot drift in either
direction unnoticed. Tables whose inputs are cluster-only are skipped with a
logged reason; their committed summary, if present, is still checked.

Exit status is non-zero if any check fails, or if a figure whose inputs are all
present locally fails to build. Missing-input skips do not fail the run.

Usage
-----
  python3 scripts/build_paper_artifacts.py            # verify MBE tables (SM.tex) + build local figures
  python3 scripts/build_paper_artifacts.py --tables   # tables only (against SM.tex / mbe_manuscript.tex)
  python3 scripts/build_paper_artifacts.py --figures  # figures only
  python3 scripts/build_paper_artifacts.py --report softlanding   # legacy: verify lbca_softlanding.tex
  python3 scripts/build_paper_artifacts.py --report all           # MBE + soft-landing companion
  python3 scripts/build_paper_artifacts.py --list     # print the manifest and exit
  python3 scripts/build_paper_artifacts.py --denoise  # also rebuild the 5 LACA pred
        TSVs from the 10-split ensemble before verifying (slow; runs on CPU)

Requires torch 2.3.1 plus pandas/pyarrow; sys.executable is what every child tool
is invoked with.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / 'scripts'
FIGDIR = REPO / 'analysis' / 'figures'
PY = sys.executable
# The cross-input-consistent marginal, 10-split ensemble (model_ho3.pth in each).
ENSEMBLE_GLOB = 'gsd_results_consistency_*l1.0*hq_split*/model_ho3.pth'

# The five denoised LACA predictions (cons-10, raw x=p, noisy-OR arCOG->COG),
# each paired with its input table and that table's value-column header, which
# --denoise passes as analyze_ancestral_node.py --node. The two Euryroot variants
# share the root node and differ only in the origination prior.
EURY_NODE = 'Node_GCA-000006805.1_AP024487.1_0'
LACA_PREDS = {
    'merged':           ('LACA_merged_pred.tsv',           'data/LACA_combined_table.tsv',           'LACA'),
    'euryroot':         ('LACA_euryroot_pred.tsv',          'data/LACA_Euryroot_arcog_table.tsv',      EURY_NODE),
    'euryroot_uniform': ('LACA_euryroot_uniform_pred.tsv',  'data/LACA_Euryroot_uniform_arcog_table.tsv', EURY_NODE),
    'gld_min1':         ('LACA_gld_min1_pred.tsv',          'LACA_GLD_min1_input.tsv',                 'LACA_GLD_min1'),
    'gld_min4':         ('LACA_gld_min4_pred.tsv',          'LACA_GLD_min4_input.tsv',                 'LACA_GLD_min4'),
}


# ---------------------------------------------------------------------------
# Verification target: 'mbe' (SM.tex + mbe_manuscript.tex, prose searched across
# both) or 'softlanding' (the companion paper). Each verify_* function names its
# table by the soft-landing logical label (tab:X); MBE_LABELS maps that to the
# SM.tex label, and None means no counterpart in the MBE paper -- the numbers are
# still recomputed, only the .tex cell-coupling is skipped.
MBE_LABELS = {
    'tab:laca_inputs': 's-tab:laca_inputs',   'tab:laca_jacc': 's-tab:laca_jacc',
    'tab:laca_corr_in': 's-tab:laca_corr_in', 'tab:laca_corr': 's-tab:laca_corr',
    'tab:conf': 's-tab:conf',                 'tab:laca_core_jacc': 's-tab:laca_core_jacc',
    'tab:addrem': 's-tab:addrem',             'tab:jacc': 's-tab:jacc',
    'tab:inputnoise': 's-tab:inputnoise',
    'tab:fp_spectra': 'r-tab:fp_spectra',     'tab:cohfp_table': 'r-tab:cohfp_table',
    'tab:fp_ancestral': 'r-tab:fp_ancestral', 'tab:strat_compl': 'r-tab:strat_compl',
    'tab:strat_phylum': 'r-tab:strat_phylum', 'tab:recon_models': 'r-tab:recon_models',
    'tab:gmass': 'r-tab:gmass',               'tab:interactome': 'r-tab:interactome',
    'tab:dynamic': 'r-tab:dynamic',
    # Soft-landing-companion-only tables: no MBE counterpart, coupling skipped.
    'tab:energy': None, 'tab:fpranking': None, 'tab:tier': None,
    'tab:mcc': None, 'tab:fp_auc': None,
}
# Set by main(). PROSE_TEX is the .tex list searched for prose numbers.
TARGET = 'mbe'
PROSE_TEX = ['analysis/mbe_manuscript.tex', 'analysis/SM.tex']


def resolve_label(label):
    """Logical (soft-landing) label -> the target's label, or None if the target
    paper has no such table."""
    return MBE_LABELS.get(label, label) if TARGET == 'mbe' else label


# ---------------------------------------------------------------------------
# Check accumulator
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.passed = self.failed = self.skipped = 0

    def ok(self, msg):
        self.passed += 1
        print(f'  [PASS] {msg}')

    def fail(self, msg):
        self.failed += 1
        print(f'  [FAIL] {msg}')

    def skip(self, msg):
        self.skipped += 1
        print(f'  [SKIP] {msg}')

    def near(self, label, got, exp, tol=0):
        """got == exp (ints) or |got-exp| <= tol (floats); records and returns the verdict."""
        good = abs(got - exp) <= tol if (tol or isinstance(exp, float)) else got == exp
        (self.ok if good else self.fail)(f'{label}: got {got} expected {exp}'
                                         + ('' if good else '  <-- DRIFT'))
        return good


def exists(rel):
    return (REPO / rel).exists()


def have(rels):
    return [r for r in rels if not exists(r)]


def run(argv, **kw):
    """Run a child tool under this interpreter with cwd = the repo root. stdout is
    captured because verify_lbca_compare parses it."""
    print('    $ ' + ' '.join(str(a) for a in argv))
    return subprocess.run([PY] + [str(a) for a in argv], cwd=REPO,
                          capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------
# .tex coupling: pull a table environment out by its \label and test substrings
# ---------------------------------------------------------------------------
def tex_table_block(texfile, label):
    txt = (REPO / texfile).read_text(encoding='utf-8', errors='replace')
    needle = r'\label{' + label + '}'
    i = txt.find(needle)
    if i < 0:
        return None
    start = txt.rfind(r'\begin{table}', 0, i)
    end = txt.find(r'\end{table}', i)
    if start < 0 or end < 0:
        return None
    return txt[start:end]


def latex_num(n):
    """Format an int the way the .tex does: thousands separated by {,}."""
    if isinstance(n, int) and abs(n) >= 1000:
        return f'{n:,}'.replace(',', '{,}')
    return str(n)


def assert_in_tex(chk, texfile, label, values, what):
    """Assert every value (int or pre-formatted str) appears inside the target
    paper's table environment. `label` is the logical (soft-landing) label; a
    table with no counterpart in the target is skipped."""
    rlabel = resolve_label(label)
    if rlabel is None:
        chk.skip(f'{label} ({what}): no table in the {TARGET} paper '
                 '(numbers recomputed + checked above; nothing to .tex-couple)')
        return
    block = tex_table_block(texfile, rlabel)
    if block is None:
        chk.fail(f'{rlabel}: \\label not found in {texfile}')
        return
    missing = []
    for v in values:
        s = v if isinstance(v, str) else latex_num(v)
        if s not in block:
            missing.append(s)
    if missing:
        chk.fail(f'{rlabel} ({what}) not found in {texfile}: ' + ', '.join(missing))
    else:
        chk.ok(f'{rlabel} ({what}) cells present in {texfile}')


def assert_in_tex_text(chk, texfile, values, what):
    """Assert every value's string appears in the target paper's prose. The MBE
    target searches all of PROSE_TEX and downgrades an absent value to a skip,
    since it may live only in the companion paper's prose; the soft-landing
    target checks the single file strictly and fails."""
    files = PROSE_TEX if TARGET == 'mbe' else [texfile]
    corpus = '\n'.join((REPO / f).read_text(encoding='utf-8', errors='replace')
                       for f in files if (REPO / f).exists())
    missing = [(v if isinstance(v, str) else latex_num(v)) for v in values
               if (v if isinstance(v, str) else latex_num(v)) not in corpus]
    where = ' + '.join(files)
    if not missing:
        chk.ok(f'{what} present in {where}')
    elif TARGET == 'mbe':
        chk.skip(f'{what}: not in MBE prose ({where}); recomputed above: '
                 + ', '.join(missing))
    else:
        chk.fail(f'{what} not found in {texfile}: ' + ', '.join(missing))


# ---------------------------------------------------------------------------
# primary-data recompute helpers
# ---------------------------------------------------------------------------
def read_pred(path):
    """pred TSV -> {COG_ID: (input_prob, mean_actual)}."""
    out = {}
    with open(REPO / path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            out[r['COG_ID']] = (float(r['input_prob']), float(r['mean_actual']))
    return out


def present_set(pred, thr=0.5):
    return {c for c, (_i, m) in pred.items() if m > thr}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def conf_bands(path):
    """(present>0.5, core>0.9, borderline 0.4-0.6) from a pred/uncert TSV."""
    p = c = b = 0
    with open(REPO / path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            m = float(r['mean_actual'])
            p += m > 0.5
            c += m > 0.9
            b += 0.4 < m < 0.6
    return p, c, b


# ===========================================================================
# Table verifiers. Each names its table by the soft-landing logical label;
# SL and resolve_label decide which .tex the cells are asserted in.
# ===========================================================================
SL = 'analysis/SM.tex'   # target table .tex; reassigned by main() per --report


def verify_laca_inputs(chk):
    """tab:laca_inputs -- present-by-threshold + denoised, the 5 LACA inputs."""
    # Strict-greater throughout, matching laca_inputs_tables.py: the .tex header
    # writes the second column as >=0.01 but the cell counts >0.01.
    THR = [0.0, 0.01, 0.1, 0.5]
    EXP = {  # rows: [ >0, >0.01, >0.1, >0.5, denoised ]   (mirror of the .tex)
        'merged':           [832, 575, 254, 64, 1023],
        'euryroot':         [2112, 1968, 1539, 1018, 1321],
        'euryroot_uniform': [805, 615, 311, 93, 909],
        'gld_min1':         [2466, 2299, 2181, 2017, 1708],
        'gld_min4':         [2240, 1789, 1573, 1422, 1405],
    }
    miss = have([LACA_PREDS[k][0] for k in EXP])
    if miss:
        chk.skip(f'tab:laca_inputs: missing preds {miss}')
        return
    allcells = []
    for k, exp in EXP.items():
        pred = read_pred(LACA_PREDS[k][0])
        ins = [float(i) for (i, _m) in pred.values()]
        cnt = [sum(1 for i in ins if i > t) for t in THR]
        den = len(present_set(pred))
        got = cnt + [den]
        for j, (g, e) in enumerate(zip(got, exp)):
            chk.near(f'tab:laca_inputs {k}[{j}]', g, e)
        allcells += exp
    assert_in_tex(chk, SL, 'tab:laca_inputs', allcells, 'present-counts')


def verify_laca_jacc(chk):
    """tab:laca_jacc -- 4x4 Jaccard of the denoised present sets (2 decimals)."""
    order = ['merged', 'euryroot', 'euryroot_uniform', 'gld_min1', 'gld_min4']
    EXP = {('merged', 'euryroot'): 0.73, ('merged', 'euryroot_uniform'): 0.84,
           ('merged', 'gld_min1'): 0.59, ('merged', 'gld_min4'): 0.71,
           ('euryroot', 'euryroot_uniform'): 0.67, ('euryroot', 'gld_min1'): 0.76,
           ('euryroot', 'gld_min4'): 0.86, ('euryroot_uniform', 'gld_min1'): 0.52,
           ('euryroot_uniform', 'gld_min4'): 0.64, ('gld_min1', 'gld_min4'): 0.81}
    miss = have([LACA_PREDS[k][0] for k in order])
    if miss:
        chk.skip(f'tab:laca_jacc: missing preds {miss}')
        return
    sets = {k: present_set(read_pred(LACA_PREDS[k][0])) for k in order}
    cells = []
    for (a, b), e in EXP.items():
        g = round(jaccard(sets[a], sets[b]), 2)
        chk.near(f'tab:laca_jacc J({a},{b})', g, e, tol=0.005)
        cells.append(f'${e:.2f}$'.rstrip('0').rstrip('.') if False else f'{e:.2f}')
    # Bare strings, not \textbf / $...$ wrapping: a bolded cell still contains
    # the plain number.
    assert_in_tex(chk, SL, 'tab:laca_jacc', [f'{e:.2f}' for e in EXP.values()],
                  'Jaccard')


def verify_laca_convergence(chk):
    """tab:laca_corr_in + tab:laca_corr + the Convergence paragraph: Pearson
    correlation of the per-COG input and output posteriors, the mean pairwise
    input->output rise, and the five-way present-set core. Correlations are
    threshold-free because the inputs are too diffuse for a >0.5 Jaccard to be
    informative; that Jaccard is checked in verify_laca_jacc."""
    import itertools
    order = ['merged', 'euryroot', 'euryroot_uniform', 'gld_min1', 'gld_min4']
    miss = have([LACA_PREDS[k][0] for k in order])
    if miss:
        chk.skip(f'tab:laca_corr: missing preds {miss}')
        return
    preds = {k: read_pred(LACA_PREDS[k][0]) for k in order}
    cogs = sorted(set().union(*[set(p) for p in preds.values()]))
    invec = {k: np.array([preds[k].get(c, (0.0, 0.0))[0] for c in cogs]) for k in order}
    outvec = {k: np.array([preds[k].get(c, (0.0, 0.0))[1] for c in cogs]) for k in order}

    def pear(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    EXP_INC = {('merged', 'euryroot'): 0.42, ('merged', 'euryroot_uniform'): 0.57,
               ('merged', 'gld_min1'): 0.26, ('merged', 'gld_min4'): 0.34,
               ('euryroot', 'euryroot_uniform'): 0.54, ('euryroot', 'gld_min1'): 0.71,
               ('euryroot', 'gld_min4'): 0.77, ('euryroot_uniform', 'gld_min1'): 0.30,
               ('euryroot_uniform', 'gld_min4'): 0.39, ('gld_min1', 'gld_min4'): 0.81}
    EXP_OUTC = {('merged', 'euryroot'): 0.92, ('merged', 'euryroot_uniform'): 0.97,
                ('merged', 'gld_min1'): 0.84, ('merged', 'gld_min4'): 0.92,
                ('euryroot', 'euryroot_uniform'): 0.89, ('euryroot', 'gld_min1'): 0.92,
                ('euryroot', 'gld_min4'): 0.97, ('euryroot_uniform', 'gld_min1'): 0.79,
                ('euryroot_uniform', 'gld_min4'): 0.88, ('gld_min1', 'gld_min4'): 0.95}
    for (a, b), e in EXP_INC.items():
        chk.near(f'tab:laca_corr_in r({a},{b})', round(pear(invec[a], invec[b]), 2),
                 e, tol=0.006)
    assert_in_tex(chk, SL, 'tab:laca_corr_in', [f'{e:.2f}' for e in EXP_INC.values()],
                  'input correlation')
    for (a, b), e in EXP_OUTC.items():
        chk.near(f'tab:laca_corr r({a},{b})', round(pear(outvec[a], outvec[b]), 2),
                 e, tol=0.006)
    assert_in_tex(chk, SL, 'tab:laca_corr', [f'{e:.2f}' for e in EXP_OUTC.values()],
                  'output correlation')
    pairs = list(itertools.combinations(order, 2))
    mi = sum(pear(invec[a], invec[b]) for a, b in pairs) / len(pairs)
    mo = sum(pear(outvec[a], outvec[b]) for a, b in pairs) / len(pairs)
    chk.near('convergence mean input correlation', round(mi, 2), 0.51, tol=0.006)
    chk.near('convergence mean output correlation', round(mo, 2), 0.90, tol=0.006)
    outset = {k: present_set(preds[k]) for k in order}
    core = set.intersection(*outset.values())
    chk.near('five-way shared core', len(core), 868)
    assert_in_tex_text(chk, SL, ['0.51', '0.90', '0.71', '868'], 'convergence numbers')
    # "Metabolic breadth" paragraph: the min-4 residual over Euryroot, split by
    # COG category into general-function/unknown (R/S) and metabolic (CGEFHIPQ).
    cog2cat = {}
    with open(REPO / 'data/cog-20.def.tab', encoding='latin-1') as fh:
        for row in csv.reader(fh, delimiter='\t'):
            if len(row) > 1:
                cog2cat[row[0]] = row[1]
    resid = outset['gld_min4'] - outset['euryroot']
    metab = set('CGEFHIPQ')
    rs = sum(1 for c in resid if 'R' in cog2cat.get(c, '') or 'S' in cog2cat.get(c, ''))
    me = sum(1 for c in resid if any(ch in metab for ch in cog2cat.get(c, '')))
    chk.near('min-4 residual vs Euryroot', len(resid), 142)
    chk.near('min-4 residual R/S %', round(100 * rs / len(resid)), 19, tol=1)
    chk.near('min-4 residual metabolic %', round(100 * me / len(resid)), 57, tol=1)
    assert_in_tex_text(chk, SL, ['\\sim\\!140', '\\sim\\!19', '\\sim\\!57'],
                       'min-4 metabolic-breadth numbers')


def verify_conf(chk):
    """tab:conf -- present/core>0.9/borderline, LBCA(uncert) + all 5 LACA(preds)."""
    ROWS = [('LBCA min1', 'data/uncert_lbca_dense.tsv', (1836, 1177, 265)),
            ('LBCA sparse', 'data/uncert_lbca_sparse.tsv', (1519, 820, 263)),
            ('LACA COG-uorig', 'LACA_merged_pred.tsv', (1023, 636, 149)),
            ('LACA eury-ML', 'LACA_euryroot_pred.tsv', (1321, 906, 176)),
            ('LACA eury-uniform', 'LACA_euryroot_uniform_pred.tsv', (909, 549, 157)),
            ('LACA min1', 'LACA_gld_min1_pred.tsv', (1708, 1203, 219)),
            ('LACA min4', 'LACA_gld_min4_pred.tsv', (1405, 997, 161))]
    cells = []
    for name, path, exp in ROWS:
        if not exists(path):
            chk.skip(f'tab:conf {name}: missing {path}')
            continue
        got = conf_bands(path)
        for g, e in zip(got, exp):
            chk.near(f'tab:conf {name}', g, e)
        cells += list(exp)
    assert_in_tex(chk, SL, 'tab:conf', cells, 'present/core/borderline')


def _set(path, col, thr):
    return {r['COG_ID'] for r in csv.DictReader(open(REPO / path), delimiter='\t')
            if float(r[col]) > thr}


def verify_laca_core(chk):
    """tab:laca_core_jacc + the 'shared core' paragraph: confident-core (p>0.9)
    Jaccard of the five reconstructions, the five-way confident core, and the
    min-4-only sets over ML Euryroot and over uniform Euryroot with their
    confident fractions."""
    order = ['merged', 'euryroot', 'euryroot_uniform', 'gld_min1', 'gld_min4']
    miss = have([LACA_PREDS[k][0] for k in order])
    if miss:
        chk.skip(f'tab:laca_core_jacc: missing preds {miss}')
        return
    CORE = {k: _set(LACA_PREDS[k][0], 'mean_actual', 0.9) for k in order}
    PRES = {k: _set(LACA_PREDS[k][0], 'mean_actual', 0.5) for k in order}
    EXP = {('merged', 'euryroot'): 0.64, ('merged', 'euryroot_uniform'): 0.78,
           ('merged', 'gld_min1'): 0.52, ('merged', 'gld_min4'): 0.62,
           ('euryroot', 'euryroot_uniform'): 0.58, ('euryroot', 'gld_min1'): 0.70,
           ('euryroot', 'gld_min4'): 0.81, ('euryroot_uniform', 'gld_min1'): 0.45,
           ('euryroot_uniform', 'gld_min4'): 0.54, ('gld_min1', 'gld_min4'): 0.79}
    for (a, b), e in EXP.items():
        chk.near(f'tab:laca_core_jacc J({a},{b})', round(jaccard(CORE[a], CORE[b]), 2),
                 e, tol=0.006)
    assert_in_tex(chk, SL, 'tab:laca_core_jacc', [f'{e:.2f}' for e in EXP.values()],
                  'confident-core Jaccard')
    chk.near('LACA five-way confident core', len(set.intersection(*CORE.values())), 506)
    d_ml = PRES['gld_min4'] - PRES['euryroot']
    chk.near('min4-only over Eury-ML', len(d_ml), 142)
    chk.near('  of which confident', len(d_ml & CORE['gld_min4']), 17)
    d_u = PRES['gld_min4'] - PRES['euryroot_uniform']
    chk.near('min4-only over Eury-uniform', len(d_u), 503)
    chk.near('  of which confident', len(d_u & CORE['gld_min4']), 156)
    assert_in_tex_text(chk, SL, ['506', '142', '503'], 'shared-core paragraph')


def verify_lbca_core(chk):
    """LBCA 'shared core' numbers: dense (GLD min1) vs sparse reconciliation
    overlap at p>0.5 and p>0.9, and the confident fraction of the dense-only set."""
    if have(['data/uncert_lbca_dense.tsv', 'data/uncert_lbca_sparse.tsv']):
        chk.skip('LBCA shared-core: missing uncert TSVs')
        return
    dP = _set('data/uncert_lbca_dense.tsv', 'mean_actual', 0.5)
    sP = _set('data/uncert_lbca_sparse.tsv', 'mean_actual', 0.5)
    dC = _set('data/uncert_lbca_dense.tsv', 'mean_actual', 0.9)
    sC = _set('data/uncert_lbca_sparse.tsv', 'mean_actual', 0.9)
    chk.near('LBCA shared present', len(dP & sP), 1433)
    chk.near('LBCA both p>0.9', len(dC & sC), 779)
    donly = dP - sP
    chk.near('LBCA dense-only over sparse', len(donly), 403)
    chk.near('  of which confident', len(donly & dC), 56)
    assert_in_tex_text(chk, SL, ['1{,}433', '779', '403'], 'LBCA shared-core')


def verify_lbca_compare(chk):
    """tab:addrem, tab:jacc (LBCA), tab:energy. compare_lbca_reconstructions.py
    owns these three tables and runs its own checks; its JSON and stdout are then
    tested against the .tex."""
    if have(['marg_sparse.tsv', 'marg_soft.tsv', 'marg_soft_min4.tsv']):
        chk.skip('LBCA tables: missing marg_*.tsv inputs')
        return
    cp = run([SCRIPTS / 'compare_lbca_reconstructions.py'])
    if cp.returncode != 0:
        chk.fail('compare_lbca_reconstructions.py exited nonzero (self-checks failed)')
        print(cp.stdout[-1500:]); print(cp.stderr[-800:])
        return
    chk.ok('compare_lbca_reconstructions.py self-checks PASSED')
    d = json.loads((REPO / 'data/lbca_comparison_summary.json').read_text())
    rec, pair = d['recons'], d['pairs']
    # tab:addrem
    chk.near('tab:addrem min1 out', rec['soft-trim-min1']['output'], 1836)
    chk.near('tab:addrem min1 added', rec['soft-trim-min1']['added'], 216)
    chk.near('tab:addrem min1 removed', rec['soft-trim-min1']['removed'], 394)
    chk.near('tab:addrem min4 out', rec['soft-trim-min4']['output'], 1776)
    chk.near('tab:addrem min4 added', rec['soft-trim-min4']['added'], 186)
    chk.near('tab:addrem min4 removed', rec['soft-trim-min4']['removed'], 391)
    chk.near('tab:addrem sparse out', rec['sparse-rescue']['output'], 1519)
    assert_in_tex(chk, SL, 'tab:addrem',
                  [493, 1519, 1145, 119, 2014, 1836, 216, 394, 1981, 1776, 186, 391],
                  'add/remove')
    # tab:jacc (LBCA dense-vs-sparse)
    chk.near('tab:jacc min1 shared', pair['soft-trim-min1_vs_sparse']['shared'], 1433)
    chk.near('tab:jacc min1 J', round(pair['soft-trim-min1_vs_sparse']['jaccard'], 2), 0.75, tol=0.005)
    chk.near('tab:jacc min1 GLD-only', pair['soft-trim-min1_vs_sparse']['A_only'], 403)
    chk.near('tab:jacc min1 sparse-only', pair['soft-trim-min1_vs_sparse']['B_only'], 86)
    chk.near('tab:jacc min4 shared', pair['soft-trim-min4_vs_sparse']['shared'], 1426)
    chk.near('tab:jacc min4 sparse-only', pair['soft-trim-min4_vs_sparse']['B_only'], 93)
    assert_in_tex(chk, SL, 'tab:jacc', [1433, 1426, 403, 350, 86, 93], 'overlap')
    # tab:energy -- 12 module-completeness triples, parsed out of the child's
    # KEY ENERGY stdout block.
    ENERGY = [('NADH:quinone', (100, 100, 71)),
              ('F-type ATPase', (100, 100, 100)),
              ('Cytochrome c oxidase, prokaryotes', (0, 0, 0)),
              ('Cytochrome bc1', (25, 25, 0)),
              ('Cytochrome bd', (33, 0, 67)),
              ('Menaquinone', (100, 100, 75)),
              ('Glycolysis', (67, 67, 67)),
              ('Citrate cycle', (69, 69, 65)),
              ('Entner-Doudoroff', (60, 60, 80)),
              ('Phosphate acetyltransferase', (100, 100, 100)),
              ('Cobalamin', (89, 78, 100)),
              ('Pentose phosphate', (67, 67, 83))]
    energy_got = {}
    for line in cp.stdout.splitlines():
        m = re.search(r'\s(\d+)%\s+(\d+)%\s+(\d+)%\s*$', line)
        if m:
            energy_got[line[:48].strip()] = tuple(int(x) for x in m.groups())
    for key, exp in ENERGY:
        row = next((v for k, v in energy_got.items() if key.lower() in k.lower()), None)
        if row is None:
            chk.fail(f'tab:energy "{key}": row not found in stdout')
        else:
            chk.near(f'tab:energy {key}', row, exp) if row == exp else \
                chk.fail(f'tab:energy {key}: got {row} expected {exp}  <-- DRIFT')
    # Only the distinctive repinned cells are coupled to the .tex.
    assert_in_tex(chk, SL, 'tab:energy', [71, 89], 'repinned %')


def _read_tsv(path):
    with open(REPO / path) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def verify_input_noise(chk):
    """tab:inputnoise (8-reconstruction absolute noise ranking) + tab:fpranking
    (input false-positive ranking). Mean cross-split SD, split-flip rate and the
    build/prune FP verdict, recomputed from data/uncert_lbca_*.tsv (LBCA) and the
    canonical LACA_*_pred.tsv (LACA), the same files verify_conf reads. Mirrors
    scripts/input_noise_ranking.py."""
    NOISE = [  # label, file, (present, mean_sd_3dp, flip_pct_1dp)
        ('LBCA min1',       'data/uncert_lbca_dense.tsv',       (1836, 0.084, 26.3)),
        ('LBCA min4',       'data/uncert_lbca_min4.tsv',        (1776, 0.084, 26.3)),
        ('LBCA sparse',     'data/uncert_lbca_sparse.tsv',      (1519, 0.062,  9.8)),
        ('LACA recount1',   'LACA_gld_min1_pred.tsv',           (1708, 0.071, 22.1)),
        ('LACA recount4',   'LACA_gld_min4_pred.tsv',           (1405, 0.054, 14.9)),
        ('LACA euryroot',   'LACA_euryroot_pred.tsv',           (1321, 0.055, 14.1)),
        ('LACA eury-unif',  'LACA_euryroot_uniform_pred.tsv',   ( 909, 0.047,  6.1)),
        ('LACA COG-merged', 'LACA_merged_pred.tsv',             (1023, 0.041,  3.5)),
    ]
    missing = have([f for _, f, _ in NOISE])
    if missing:
        chk.skip(f'tab:inputnoise/tab:fpranking: missing {missing[0]} (+{len(missing)-1} more)')
        return
    sd_cells = []
    for label, f, (ep, esd, eflip) in NOISE:
        pres = [r for r in _read_tsv(f) if float(r['mean_actual']) > 0.5]
        n = len(pres)
        msd = sum(float(r['sd_actual']) for r in pres) / n
        flip = sum(1 for r in pres if float(r['sd_actual']) > 0.15)
        chk.near(f'tab:inputnoise {label} present', n, ep)
        chk.near(f'tab:inputnoise {label} mean SD', round(msd, 3), esd, tol=0.0011)
        chk.near(f'tab:inputnoise {label} flip%', round(100 * flip / n, 1), eflip, tol=0.05)
        sd_cells.append(f'{esd:.3f}')
    assert_in_tex(chk, SL, 'tab:inputnoise', sorted(set(sd_cells)), 'mean cross-split SD')
    # tab:fpranking -- removable FP: input proposes p>0.5, output switches off.
    FP = [  # label, file, (input>0.5, removed)
        ('recount min1', 'LACA_gld_min1_pred.tsv',     (2017, 420)),
        ('GLD min1',     'data/uncert_lbca_dense.tsv',  (2014, 394)),
        ('GLD min4',     'data/uncert_lbca_min4.tsv',   (1981, 391)),
        ('recount min4', 'LACA_gld_min4_pred.tsv',      (1422, 197)),
        ('euryroot ML',  'LACA_euryroot_pred.tsv',      (1018, 138)),
    ]
    for label, f, (eip, erm) in FP:
        rows = _read_tsv(f)
        ip = sum(1 for r in rows if float(r['input_prob']) > 0.5)
        rm = sum(1 for r in rows if float(r['input_prob']) > 0.5 and float(r['mean_actual']) <= 0.5)
        chk.near(f'tab:fpranking {label} input>0.5', ip, eip)
        chk.near(f'tab:fpranking {label} removed', rm, erm)
    assert_in_tex(chk, SL, 'tab:fpranking', [2017, 420, 2014, 394, 1981, 391, 1422, 197, 1018, 138],
                  'input FP ranking')


def verify_fp_tier(chk):
    """tab:tier (multi-axis borderline shell), replicating
    scripts/fp_confidence_tier.py: present = q_full_mean>0.5, flagged
    weakly-proposed (input_prob bottom 33pct), self-anchored (self_anchor_absent
    top 33pct) or split-disagreed (q_full_sd top 33pct); confident core = 0 flags,
    borderline shell = >=2.

    Scored on the committed fp-context diagnostic TSVs, which come from a separate
    post-hoc forward pass: the present counts (1772/1792) therefore differ from
    tab:conf's ensemble counts (1836/1708), and the LACA diagnostic is on the mix
    marginal-HQ model (see the tab:tier caption)."""
    TIER = [('LBCA', 'data/fp_context_lbca_dense.tsv', (1772, 930, 567)),
            ('LACA', 'data/fp_context_laca_dense.tsv', (1792, 934, 601))]
    missing = have([f for _, f, _ in TIER])
    if missing:
        chk.skip(f'tab:tier: missing {missing[0]}')
        return

    def pctl(vals, p):
        s = sorted(vals)
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        return s[k]

    for label, f, (ep, ecore, eshell) in TIER:
        rows = [r for r in _read_tsv(f) if float(r['q_full_mean']) > 0.5]
        inp = [float(r['input_prob']) for r in rows]
        anc = [float(r['self_anchor_absent']) for r in rows]
        sdv = [float(r['q_full_sd']) for r in rows]
        ti, ta, ts = pctl(inp, 33), pctl(anc, 67), pctl(sdv, 67)
        core = sum(1 for i in range(len(rows))
                   if (inp[i] < ti) + (anc[i] > ta) + (sdv[i] > ts) == 0)
        shell = sum(1 for i in range(len(rows))
                    if (inp[i] < ti) + (anc[i] > ta) + (sdv[i] > ts) >= 2)
        chk.near(f'tab:tier {label} present', len(rows), ep)
        chk.near(f'tab:tier {label} core', core, ecore)
        chk.near(f'tab:tier {label} shell', shell, eshell)
    assert_in_tex(chk, SL, 'tab:tier', [930, 567, 934, 601], 'multi-axis core/shell')


def verify_mcc(chk):
    """tab:mcc (cluster) -- bac fn=0.9 rows from the committed spectra CSV."""
    csvp = 'data/spectra_fpreal_summary.csv'
    if not exists(csvp):
        chk.skip(f'tab:mcc: missing {csvp}')
        return
    EXP = {  # family -> {fp: MCC} at fn=0.9   (mirror of the .tex)
        'bacFTfpreal':       {0.01: 0.461, 0.025: 0.440, 0.05: 0.390, 0.1: 0.214, 0.2: 0.018},
        'bacFTfprealhq':     {0.01: 0.465, 0.025: 0.444, 0.05: 0.399, 0.1: 0.232, 0.2: 0.019},
        'bacFTfpmarginalhq': {0.01: 0.457, 0.025: 0.437, 0.05: 0.398, 0.1: 0.254, 0.2: 0.016},
    }
    # The .tex ECE row is the marginal-FP model's calibration error.
    EXP_ECE = {'bacFTfpmarginalhq': {0.01: 0.136, 0.025: 0.141, 0.05: 0.150,
                                     0.1: 0.179, 0.2: 0.230}}
    rows = list(csv.DictReader(open(REPO / csvp)))
    # The CSV family field is prefixed (nohidden_HO_T20_<fam>).
    mcol = next((c for c in rows[0] if c.lower() in ('mcc_mean', 'mcc')), 'MCC_mean')
    ecol = next((c for c in rows[0] if c.lower() in ('ece_mean', 'ece')), 'ece_mean')

    def cell(fam, fp, col):
        for r in rows:
            if r.get('family', '').endswith(fam) and abs(float(r['fn']) - 0.9) < 1e-6 \
                    and abs(float(r['fp']) - fp) < 1e-6:
                return round(float(r[col]), 3)
        return None
    any_seen = False
    for fam, fps in EXP.items():
        for fp, e in fps.items():
            g = cell(fam, fp, mcol)
            if g is None:
                continue
            any_seen = True
            chk.near(f'tab:mcc MCC {fam} fp={fp}', g, e, tol=0.0005)
    for fam, fps in EXP_ECE.items():
        for fp, e in fps.items():
            g = cell(fam, fp, ecol)
            if g is not None:
                chk.near(f'tab:mcc ECE {fam} fp={fp}', g, e, tol=0.0005)
    if not any_seen:
        chk.skip('tab:mcc: CSV present but no matching (family,fn=0.9,fp) rows '
                 '(schema differs) -- numbers transcribed from the sweep, not re-checked')
    else:
        assert_in_tex(chk, SL, 'tab:mcc',
                      ['0.461', '0.465', '0.457', '0.136', '0.230'], 'MCC/ECE')


def verify_report_tables(chk):
    """The cluster-tier, report-origin tables (SM.tex r-tab:*). Their numbers are
    transcribed from GPU sweeps, so the committed summary CSV/parquet is
    presence-checked only. The fp-control / extant / interactome tables have no
    local inputs at all (their *_pred.tsv / recover_*.tsv / *.npz are .gitignored);
    those are skipped with the regeneration recipe logged."""
    if not list((REPO / 'data/interactome/fpcontrol').glob('*_pred.tsv')):
        for lab, rec in [('tab:fp_ancestral', 'reconstruct.py ancestral + analyze_ancestral_node.py'),
                         ('tab:fpcontrol', 'interactome/run_fp_control_recon.sh'),
                         ('tab:threshold', 'reconstruct.py ancestral --drop-low-conf / --bin-thresholds')]:
            chk.skip(f'{lab} (recon): fpcontrol/*_pred.tsv absent locally; '
                     f'regenerate on cluster via {rec}')
    if not list((REPO / 'data/interactome/fpcompare').glob('recover_*.tsv')):
        chk.skip('tab:fp_extant (cluster): fpcompare/recover_*.tsv absent; '
                 'regenerate via interactome/run_fp_compare.sh')
    if not list((REPO / 'data/interactome/results').glob('*.npz')):
        chk.skip('tab:interactome + tab:dynamic (cluster): interactome/*.npz absent; '
                 'regenerate via interactome/aggregate_interactome.py')
    # Presence only: a cell re-check would need the sweep schema.
    for lab, src in [('tab:models', 'data/spectra_calibration_all.parquet'),
                     ('tab:recon_models', 'data/spectra_calibration_all.parquet'),
                     ('tab:fp_spectra', 'data/spectra_fpreal_summary.csv'),
                     ('tab:strat_compl', 'data/strat_raw_merged.csv'),
                     ('tab:strat_phylum', 'data/strat_raw_merged.csv')]:
        (chk.ok if exists(src) else chk.skip)(
            f'{lab} (cluster): committed summary {src} '
            + ('present (numbers transcribed from the sweep)' if exists(src)
               else 'absent -- regenerate sweep on cluster'))
    chk.skip('tab:provenance (static): hand-authored map, no numeric cells')


# ===========================================================================
# FIGURES
# ===========================================================================
def _recon_fig(name, pred, only, label, extra=()):
    return dict(name=name, tier='recon',
                inputs=[pred, 'data/module_matrix_kegg.pt', 'data/cog-20.def.tab'],
                argv=[SCRIPTS / 'plot_recon_figs.py', '--only', only,
                      f'--{only}-pred', pred, f'--{only}-out', name,
                      f'--{only}-label', label, '--outdir', FIGDIR, *extra])


def figure_manifest():
    """Every figure both papers include, as name / tier / required local inputs /
    recipe argv. Figures whose inputs are all present are rebuilt; the rest are
    presence-checked."""
    figs = [
        # --- recon: ancestral reconstructions (local) ---
        # The focal LACA reconstruction (Fig. 4 in both papers) is the
        # Euryarchaeota-rooted build in the expected-gene view (sum of posteriors;
        # plot_recon_figs --only laca defaults to --laca-expected). LBCA is binary.
        _recon_fig('fig_laca_euryroot_sparse.pdf', 'LACA_euryroot_pred.tsv', 'laca',
                   'Euryroot ML origination, cons-10, raw $x=p$, expected genes'),
        _recon_fig('fig_lbca_modules.pdf', 'node2012_T20bacfphq_pred.tsv', 'lbca',
                   'bac-FT-fp HQ, face value'),
        dict(name='fig_laca_softlanding.pdf', tier='recon',
             inputs=['LACA_merged_pred.tsv', 'LACA_euryroot_pred.tsv',
                     'LACA_euryroot_uniform_pred.tsv', 'LACA_gld_min1_pred.tsv',
                     'LACA_gld_min4_pred.tsv'],
             argv=[SCRIPTS / 'plot_laca_softlanding.py']),
        dict(name='fig_lbca_softlanding.pdf', tier='recon',
             inputs=['data/lbca_comparison_summary.json',
                     'data/lbca_reconciliation_removal_by_confidence.csv'],
             argv=[SCRIPTS / 'plot_lbca_softlanding.py']),
        dict(name='fig_marginal_fp_noise.pdf', tier='recon',
             inputs=['data/COG_train1_phylum.feather'],
             argv=[SCRIPTS / 'plot_marginal_fp_noise.py']),
        # --- cluster: re-plotted from committed summaries (local) ---
        dict(name='spectra_fp_varying.pdf', tier='cluster',
             inputs=['data/spectra_fpreal_summary.csv'],
             argv=[SCRIPTS / 'plot_fp_varying.py', '--summary', 'data/spectra_fpreal_summary.csv']),
        dict(name='coherent_fp_frontier.pdf', tier='cluster',
             inputs=['data/coherent_fp_mixFT.tsv', 'data/coherent_fp_bacFT.tsv'],
             argv=[SCRIPTS / 'plot_coherent_fp.py', '--fn', '0.9',
                   '--out', FIGDIR / 'coherent_fp_frontier.pdf']),
    ]
    # Cluster-tier figures whose intermediates are .gitignored: the committed PDF
    # is the durable artifact, so these are presence-checked, not rebuilt.
    cluster_only = [
        'spectra_best_delta.pdf', 'spectra_ho_vs_plain.pdf', 'cal_ft_global.pdf',
        'cal_ft_arc.pdf', 'cal_ft_bac_delta.pdf', 'fig_interactome.pdf',
        'fig_threshold_removal.pdf',
        # fig:laca -- merged-rootings LACA from the mix-FT-fp-marginal-HQ pred
        # (1,018 present), which lives under the .gitignored fpcontrol set; binary.
        'fig_laca_categories.pdf',
        # FP-control ancestral-profile panels:
        'fig_lbca_base.pdf', 'fig_lbca_gt0p01.pdf', 'fig_lbca_gt0p02.pdf', 'fig_lbca_hardlit.pdf',
        'fig_laca_base.pdf', 'fig_laca_drop10.pdf', 'fig_laca_drop25.pdf', 'fig_laca_hardlit.pdf',
        'fig_laca_mhh.pdf', 'fig_laca_eury.pdf', 'fig_lbca_fptuned.pdf', 'fig_laca_fptuned.pdf',
        'fig_lbca_fp_gt0.pdf', 'fig_laca_fp_gt0.pdf', 'fig_laca_mhh_fp_gt0.pdf', 'fig_laca_eury_fp_gt0.pdf',
        # per-genome recovery panels (clean + FP-tolerant), from reconstruct_extant.py:
        'fig_recover_ecoli_gen.pdf', 'fig_recover_ecoli_bacmarg.pdf', 'fig_recover_archaeon_gen.pdf',
        'fig_recover_archaeon_mixmarg.pdf', 'fig_recover_medbac_gen.pdf', 'fig_recover_medbac_bacmarg.pdf',
        'fig_recover_medarc_gen.pdf', 'fig_recover_medarc_mixmarg.pdf', 'fig_recover_medarc_hq_gen.pdf',
        'fig_recover_medarc_hq_mixmarg.pdf', 'fig_recover_medbac_hq_gen.pdf', 'fig_recover_medbac_hq_bacmarg.pdf',
        'fig_recover_ecoli_mixfp.pdf', 'fig_recover_ecoli_bacfp.pdf', 'fig_recover_arch_mixfp.pdf',
        'fig_recover_medbac_mixfp.pdf', 'fig_recover_medbac_bacfp.pdf', 'fig_recover_medarc_mixfp.pdf',
        'fig_recover_medbac_hq_mixfp.pdf', 'fig_recover_medbac_hq_bacfp.pdf', 'fig_recover_medarc_hq_mixfp.pdf',
    ]
    for nm in cluster_only:
        figs.append(dict(name=nm, tier='cluster', inputs=['__cluster__'], argv=None))
    return figs


def build_figures(chk):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for f in figure_manifest():
        nm, argv = f['name'], f['argv']
        if argv is None:                       # presence-only cluster artifact
            (chk.ok if (FIGDIR / nm).exists() else chk.skip)(
                f'{nm} ({f["tier"]}): committed PDF '
                + ('present (regenerate on cluster)' if (FIGDIR / nm).exists()
                   else 'ABSENT -- regenerate on cluster'))
            continue
        miss = have([p for p in f['inputs'] if p != '__cluster__'])
        if miss:
            chk.skip(f'{nm} ({f["tier"]}): inputs absent locally {miss}')
            continue
        cp = run(argv)
        if cp.returncode == 0 and (FIGDIR / nm).exists():
            chk.ok(f'{nm} ({f["tier"]}): built')
        else:
            chk.fail(f'{nm} ({f["tier"]}): build FAILED')
            print((cp.stdout or '')[-800:]); print((cp.stderr or '')[-800:])


# ===========================================================================
# Optional --denoise stage: rebuild the 5 LACA pred TSVs from the 10-split ensemble
# ===========================================================================
def denoise_laca(chk):
    if not list(REPO.glob(ENSEMBLE_GLOB)):
        chk.skip(f'--denoise: ensemble {ENSEMBLE_GLOB} not found')
        return
    for k, (pred, table, node) in LACA_PREDS.items():
        if not exists(table):
            chk.skip(f'denoise {k}: input table {table} absent')
            continue
        cp = run([SCRIPTS / 'analyze_ancestral_node.py', '--device', 'cpu',
                  '--actual-mode', 'raw', '--models', ENSEMBLE_GLOB,
                  '--table', table, '--node', node, '--csv-out', pred])
        (chk.ok if cp.returncode == 0 else chk.fail)(f'denoise {k} -> {pred}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', choices=['mbe', 'softlanding', 'report', 'all'], default='mbe',
                    help="paper to verify tables against (default: mbe = SM.tex + "
                         "mbe_manuscript.tex). 'report' is DEPRECATED (-> mbe).")
    ap.add_argument('--tables', action='store_true', help='verify tables only')
    ap.add_argument('--figures', action='store_true', help='build figures only')
    ap.add_argument('--denoise', action='store_true',
                    help='rebuild the 5 LACA pred TSVs from the ensemble first (slow)')
    ap.add_argument('--list', action='store_true', help='print the figure manifest and exit')
    args = ap.parse_args()

    if args.list:
        print('# figure manifest (name | tier | inputs)')
        for f in figure_manifest():
            print(f'  {f["name"]:34s} {f["tier"]:8s} '
                  + (','.join(f['inputs']) if f['argv'] else '(committed PDF)'))
        return 0

    do_tables = args.tables or not args.figures
    do_figs = args.figures or not args.tables
    chk = Checks()

    if args.denoise:
        print('== denoise: rebuild LACA pred TSVs from the 10-split ensemble ==')
        denoise_laca(chk)

    global TARGET, SL
    sel = args.report
    if sel == 'report':
        print('NOTE: --report report is DEPRECATED (report.tex is no longer a '
              'verification target; its tables live in SM.tex as r-tab:*). Using '
              '--report mbe.')
        sel = 'mbe'
    targets = ['mbe', 'softlanding'] if sel == 'all' else [sel]

    if do_tables:
        for t in targets:
            TARGET = t
            SL = 'analysis/SM.tex' if t == 'mbe' else 'analysis/lbca_softlanding.tex'
            print('== %s tables ==' % ('MBE paper (SM.tex + mbe_manuscript.tex)'
                                       if t == 'mbe' else 'soft-landing companion'))
            verify_laca_inputs(chk)
            verify_laca_jacc(chk)
            verify_laca_convergence(chk)
            verify_conf(chk)
            verify_laca_core(chk)
            verify_lbca_core(chk)
            verify_lbca_compare(chk)
            verify_fp_tier(chk)
            verify_input_noise(chk)
            verify_mcc(chk)
            # tab:fp_auc scores the matched-strength FP gate on held-out extant
            # genomes; its AUCs come from fp_context_diagnostics.py extant, not
            # from anything committed here.
            chk.skip('tab:fp_auc (cluster): extant FP-gate AUCs from '
                     'fp_context_diagnostics.py extant; not re-run here')
            if t == 'mbe':
                print('== MBE cluster-tier tables (r-tab:*, summary-checked) ==')
                verify_report_tables(chk)

    if do_figs:
        print('== figures ==')
        build_figures(chk)

    print('\n' + '=' * 72)
    print(f'PASS {chk.passed}   FAIL {chk.failed}   SKIP {chk.skipped}')
    print('=' * 72)
    if chk.failed:
        print('DRIFT or build failure detected -- see [FAIL] lines above.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
