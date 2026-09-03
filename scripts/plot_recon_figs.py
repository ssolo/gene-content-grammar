#!/usr/bin/env python3
"""Ancestral-reconstruction and extant-recovery figures.

The two ancestral figures (LBCA, LACA) share one two-panel layout: (A) per-COG
functional-category profile, (B) curated key KEGG systems.  In both, a bar is
the denoised present set, the hatched part was already present in the input,
and bars left of zero were silenced by the denoiser.  Values are taken at face
value from a single model, with no cross-root or cross-variant intersection.

The --recover figures show, per functional category, what the denoiser does to
a corrupted real genome against the known truth.

Style matches scripts/plot_results.py (Okabe-Ito, serif + cm mathtext,
constrained_layout, despined, 300 dpi vector PDF).

Needs torch, to load the KEGG module matrix:
  python3 scripts/plot_recon_figs.py --outdir analysis/figures

Defaults: LBCA via bac-FT, LACA via mix-FT, both from the 10-split-ensemble
tables in data/interactome/fpcontrol/.  The node2012_T20mix table is a mix-FT
relaxation and is the wrong model for the bacterial node.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent

# ----  Palette + style, kept in sync with scripts/plot_results.py
OKABE = {
    'black':  '#000000',
    'orange': '#E69F00',
    'sky':    '#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'verm':   '#D55E00',
    'purple': '#CC79A7',
    'grey':   '#999999',
}


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10.5,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
    })


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ----  Data loaders
def load_present(path, var='actual', thresh=0.5, inp_thresh=0.0):
    """COG_ID -> bool present (denoised mean_var > thresh) and bool in input
    (input_prob > inp_thresh).  inp_thresh=0.0 takes every family the ancestral
    reconciliation proposed at all as the 'present in input' set."""
    present = {}
    inp = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            c = row['COG_ID']
            present[c] = float(row[f'mean_{var}']) > thresh
            inp[c] = float(row['input_prob']) > inp_thresh
    return present, inp


def load_raw(path, var='actual'):
    """COG_ID -> (denoised posterior mean_var, input_prob) as raw floats, for the
    expected-gene-count view (summed per category/module, not thresholded)."""
    out, inp = {}, {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            c = row['COG_ID']
            out[c] = float(row[f'mean_{var}'])
            inp[c] = float(row['input_prob'])
    return out, inp


def load_cog_cats(path):
    out = {}
    with open(path, encoding='latin-1') as fh:
        for row in csv.reader(fh, delimiter='\t'):
            if row:
                out[row[0]] = row[1] if len(row) > 1 else ''
    return out


CAT_DESC = {
    'J': 'Translation, ribosome',     'A': 'RNA processing',
    'K': 'Transcription',             'L': 'Replication & repair',
    'B': 'Chromatin',                 'D': 'Cell cycle / division',
    'V': 'Defense',                   'T': 'Signal transduction',
    'M': 'Cell wall / membrane',      'N': 'Cell motility',
    'U': 'Trafficking, secretion',    'O': 'PTM, chaperones',
    'C': 'Energy production',         'G': 'Carbohydrate metab.',
    'E': 'Amino-acid metab.',         'F': 'Nucleotide metab.',
    'H': 'Coenzyme metab.',           'I': 'Lipid metab.',
    'P': 'Inorganic-ion transport',   'Q': 'Secondary metabolites',
    'R': 'General function pred.',     'S': 'Function unknown',
    'X': 'Mobilome',                  'W': 'Extracellular',
    'Y': 'Nuclear structure',         'Z': 'Cytoskeleton',
}

# Canonical COG functional-category order (information storage | cellular
# processes | metabolism | poorly characterised).  Shared by every per-category
# figure so panels are directly comparable.
CAT_ORDER = ['J', 'A', 'K', 'L', 'B',
             'D', 'Y', 'V', 'T', 'M', 'N', 'Z', 'W', 'U', 'O', 'X',
             'C', 'G', 'E', 'F', 'H', 'I', 'P', 'Q',
             'R', 'S']
CAT_RANK = {c: i for i, c in enumerate(CAT_ORDER)}


# Curated key systems for the LBCA panel: (module_id, short_label, group).
# Completeness is computed from the reconstruction, not hard-coded.
LBCA_CURATED = [
    ('M00023', 'Trp biosynthesis',            'biosynth'),
    ('M00016', 'Lys biosynthesis (DAP)',      'biosynth'),
    ('M00026', 'His biosynthesis',            'biosynth'),
    ('M00017', 'Met biosynthesis',            'biosynth'),
    ('M00052', 'Pyrimidine ribonucleotide',   'biosynth'),
    ('M00048', 'Purine de novo => IMP',       'biosynth'),
    ('M00157', 'F-type ATPase',               'energy'),
    ('M00144', 'Complex I (NADH:quinone)',    'energy'),
    ('M00009', 'TCA cycle',                   'energy'),
    ('M00001', 'Glycolysis (EMP)',            'energy'),
    ('M00004', 'Pentose phosphate',           'energy'),
    ('M00866', 'KDO2-lipid A (Raetz)',        'envelope'),
    ('M00063', 'CMP-KDO biosynthesis',        'envelope'),
    ('M00930', 'Menaquinone (futalosine)',    'cofactor'),
    ('M00125', 'Riboflavin biosynthesis',     'cofactor'),
    ('M00926', 'Heme biosynthesis',           'cofactor'),
    ('M00120', 'Coenzyme A biosynthesis',     'cofactor'),
    ('M00155', 'Cytochrome c oxidase',        'aerobic'),
    ('M00151', 'Cytochrome bc1 complex',      'aerobic'),
    ('M00564', 'H. pylori pathogenicity',     'lineage'),
]
GROUP_COLOR = {
    'biosynth': OKABE['blue'],
    'energy':   OKABE['green'],
    'envelope': OKABE['orange'],
    'cofactor': OKABE['purple'],
    'aerobic':  OKABE['verm'],
    'lineage':  OKABE['grey'],
    'bacterial': OKABE['verm'],
}
GROUP_LABEL = {
    'biosynth': 'Amino-acid / nucleotide biosynthesis',
    'energy':   'Energy / central carbon',
    'envelope': 'Cell envelope (lipid A)',
    'cofactor': 'Cofactor biosynthesis',
    'aerobic':  'Aerobic respiration (ABSENT)',
    'lineage':  'Pathogenicity (ABSENT)',
}

# Curated key systems for the LACA panel.  The two bacterial systems
# (F-type ATPase, lipid A) are expected absences.
LACA_CURATED = [
    ('M00159', 'V/A-type ATPase',            'energy'),
    ('M00567', 'Methanogenesis (CO2)',       'energy'),
    ('M00357', 'Methanogenesis (acetate)',   'energy'),
    ('M00422', 'Acetyl-CoA pathway',         'energy'),
    ('M00377', 'Wood-Ljungdahl',             'energy'),
    ('M00378', 'F420 biosynthesis',          'cofactor'),
    ('M00935', 'Methanofuran biosynthesis',  'cofactor'),
    ('M00358', 'Coenzyme M biosynthesis',    'cofactor'),
    ('M00896', 'Thiamine (archaea)',         'cofactor'),
    ('M00125', 'Riboflavin biosynthesis',    'cofactor'),
    ('M00052', 'Pyrimidine ribonucleotide',  'biosynth'),
    ('M00048', 'Purine de novo => IMP',      'biosynth'),
    ('M00023', 'Trp biosynthesis',           'biosynth'),
    ('M00026', 'His biosynthesis',           'biosynth'),
    ('M00157', 'F-type ATPase',              'bacterial'),
    ('M00866', 'KDO2-lipid A (Raetz)',       'bacterial'),
]
LBCA_GROUP_LABEL = GROUP_LABEL
LACA_GROUP_LABEL = {
    'energy':    'Archaeal energy / C-fixation',
    'cofactor':  'Archaeal cofactor biosynthesis',
    'biosynth':  'Amino-acid / nucleotide biosynthesis',
    'bacterial': 'Bacterial systems (declined)',
}


def module_completeness(present, mm):
    """Return dict module_id -> (k_present, n_total) over KEGG modules."""
    import torch
    cog_names = mm['cog_names']
    Mk = mm['M_kegg']
    if hasattr(Mk, 'numpy'):
        Mk = Mk.numpy()
    knames = mm['kegg_names']
    pres_vec = np.array([1.0 if present.get(c, False) else 0.0
                         for c in cog_names])
    out = {}
    for j, nm in enumerate(knames):
        mem = Mk[:, j] > 0
        n = int(mem.sum())
        if n == 0:
            continue
        out[nm] = (int(pres_vec[mem].sum()), n)
    return out


def module_decomp(present, inp, mm):
    """module_id -> (kept, recovered, removed, n) over KEGG modules:
    kept      = member present in both input and denoised output,
    recovered = present in the output but not the input,
    removed   = present in the input but silenced by the denoiser,
    so denoised present = kept+recovered and input present = kept+removed."""
    import torch  # noqa: F401  (mm already loaded by caller)
    cog_names = mm['cog_names']
    Mk = mm['M_kegg']
    if hasattr(Mk, 'numpy'):
        Mk = Mk.numpy()
    knames = mm['kegg_names']
    p = np.array([1 if present.get(c, False) else 0 for c in cog_names])
    q = np.array([1 if inp.get(c, False) else 0 for c in cog_names])
    out = {}
    for j, nm in enumerate(knames):
        mem = Mk[:, j] > 0
        n = int(mem.sum())
        if n == 0:
            continue
        pm, qm = p[mem], q[mem]
        out[nm] = (int(((pm == 1) & (qm == 1)).sum()),
                   int(((pm == 1) & (qm == 0)).sum()),
                   int(((pm == 0) & (qm == 1)).sum()), n)
    return out


def module_decomp_expected(out_p, in_p, mm):
    """Expected-gene-count analogue of module_decomp, summing posteriors per
    COG instead of thresholding them:
      kept      = sum( min(in_p, out_p) )
      recovered = sum( max(0, out_p - in_p) )
      removed   = sum( max(0, in_p - out_p) )
    so kept+recovered = E[output present] and kept+removed = E[input present].
    n = #COGs in the module."""
    import torch  # noqa: F401  (mm already loaded by caller)
    cog_names = mm['cog_names']
    Mk = mm['M_kegg']
    if hasattr(Mk, 'numpy'):
        Mk = Mk.numpy()
    knames = mm['kegg_names']
    po = np.array([float(out_p.get(c, 0.0)) for c in cog_names])
    qi = np.array([float(in_p.get(c, 0.0)) for c in cog_names])
    out = {}
    for j, nm in enumerate(knames):
        mem = Mk[:, j] > 0
        n = int(mem.sum())
        if n == 0:
            continue
        pm, qm = po[mem], qi[mem]
        out[nm] = (float(np.minimum(pm, qm).sum()),
                   float(np.clip(pm - qm, 0, None).sum()),
                   float(np.clip(qm - pm, 0, None).sum()), n)
    return out


# ----  Figure: LBCA KEGG-module completeness
def fig_lbca_modules(lbca_pred, module_pt, out_pdf, model_label):
    import torch
    present, inp = load_present(lbca_pred, var='actual')
    mm = torch.load(module_pt, map_location='cpu', weights_only=False)
    dec = module_decomp(present, inp, mm)

    fr_den = np.array([(k + r) / n for (k, r, rm, n) in dec.values()])
    fr_in = np.array([(k + rm) / n for (k, r, rm, n) in dec.values()])
    n_tot = len(fr_den)
    n_full = int((fr_den >= 0.999).sum()); n_half = int((fr_den >= 0.5).sum())
    n_full_i = int((fr_in >= 0.999).sum()); n_half_i = int((fr_in >= 0.5).sum())

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.4, 8.6), constrained_layout=True,
        gridspec_kw={'height_ratios': [0.95, 2.7]})

    # Panel A: completeness distribution, input vs denoised.
    bins = np.linspace(0, 1, 21)
    cnt, _, _ = axA.hist(
        fr_den, bins=bins, color=OKABE['sky'], edgecolor=OKABE['blue'],
        linewidth=0.6, zorder=3,
        label=f'denoised: {n_full} full, {n_half} $\\geq$half (of {n_tot})')
    cnt_in, _, _ = axA.hist(
        fr_in, bins=bins, histtype='step', color='0.20', linewidth=1.5,
        zorder=4, label=f'input: {n_full_i} full, {n_half_i} $\\geq$half')
    axA.set_xlabel('KEGG-module completeness')
    axA.set_ylabel('number of modules')
    axA.set_xlim(0, 1)
    axA.set_ylim(0, max(cnt.max(), cnt_in.max(), 1) * 1.30)
    axA.set_title('(A)', loc='left', fontweight='bold')
    axA.legend(loc='upper center', fontsize=8.4, frameon=False)
    despine(axA)

    # Panel B: curated key systems, input / recovered / removed.
    rows = []
    for mid, label, grp in LBCA_CURATED:
        if mid not in dec:
            continue
        k, r, rm, n = dec[mid]
        rows.append((label, grp, k, r, rm, n))
    rows.sort(key=lambda z: (z[2] + z[3]) / z[5])   # by denoised fraction
    labels = [z[0] for z in rows]
    cols = [GROUP_COLOR[z[1]] for z in rows]
    y = np.arange(len(rows))
    den = np.array([(k + r) / n for (_, _, k, r, rm, n) in rows])
    kep = np.array([k / n for (_, _, k, r, rm, n) in rows])
    rem = np.array([rm / n for (_, _, k, r, rm, n) in rows])
    # Full bar (group colour) = denoised present; the hatch over [0,kept] is
    # the part already in the input, so the un-hatched extension is recovery.
    axB.barh(y, den, color=cols, edgecolor='white', linewidth=0.5,
             height=0.74, zorder=3)
    axB.barh(y, kep, facecolor='none', edgecolor='0.12', hatch='////',
             linewidth=0.0, height=0.74, zorder=5)
    # Input families the denoiser silenced go left of zero.
    if rem.max() > 0:
        axB.barh(y, -rem, color='#b2182b', edgecolor='white', linewidth=0.5,
                 height=0.74, zorder=3)
    for yi, (_, _, k, r, rm, n) in zip(y, rows):
        axB.text((k + r) / n + 0.015, yi, f'{k + r}/{n}', va='center',
                 ha='left', fontsize=8.4, color='0.15', zorder=6)
    axB.set_yticks(y)
    axB.set_yticklabels(labels, fontsize=9)
    axB.set_ylim(-0.6, len(rows) - 0.4)
    axB.set_xlim(min(-0.06, -(rem.max() + 0.03)), 1.18)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xlabel('fraction of module COGs  (left of 0: removed from input)')
    axB.set_title('(B)', loc='left', fontweight='bold')
    axB.axvline(0, color='0.4', lw=0.8, zorder=2)
    axB.axvline(1.0, color='0.6', lw=0.7, ls=':', zorder=1)
    axB.text(0.985, 0.025, model_label, transform=axB.transAxes, ha='right',
             va='bottom', fontsize=7.2, color='0.5')
    despine(axB)
    groups_present = [g for g in GROUP_LABEL if any(z[1] == g for z in rows)]
    handles = [Patch(fc=GROUP_COLOR[g], ec='white', label=GROUP_LABEL[g])
               for g in groups_present]
    handles += [Patch(facecolor='white', edgecolor='0.12', hatch='////',
                      label='present in input (hatched)'),
                Patch(facecolor='#b2182b', label='removed from input')]
    axB.legend(handles=handles, loc='lower right', fontsize=7.3,
               frameon=True, framealpha=0.96, edgecolor='0.8',
               labelspacing=0.30, handlelength=1.3, borderaxespad=0.5)

    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'wrote {out_pdf}  (denoised {n_full}/{n_tot} full; '
          f'input {n_full_i}/{n_tot} full)')


# ----  Figure: LACA functional-category present fraction
def fig_laca_categories(laca_pred, cog_def, out_pdf, model_label):
    """Per functional category: the bar is the denoised present fraction, the
    hatched portion the fraction already present in the input, and families
    silenced from the input are drawn left of zero.  Untitled."""
    present, inp = load_present(laca_pred, var='actual', inp_thresh=0.0)
    cog2cat = load_cog_cats(cog_def)
    tot = defaultdict(int); kept = defaultdict(int)
    recov = defaultdict(int); removed = defaultdict(int)
    for c, is_p in present.items():
        i = inp.get(c, False)
        for ch in cog2cat.get(c, ''):
            if not ch.isalpha():
                continue
            tot[ch] += 1
            if i and is_p:         kept[ch] += 1
            elif (not i) and is_p: recov[ch] += 1
            elif i and (not is_p): removed[ch] += 1
    cats = [ch for ch in tot if tot[ch] >= 10]
    cats.sort(key=lambda ch: (kept[ch] + recov[ch]) / tot[ch])
    y = np.arange(len(cats))
    den = np.array([(kept[ch] + recov[ch]) / tot[ch] for ch in cats])
    kep = np.array([kept[ch] / tot[ch] for ch in cats])
    rem = np.array([removed[ch] / tot[ch] for ch in cats])

    fig, ax = plt.subplots(figsize=(7.4, 8.2), constrained_layout=True)
    ax.barh(y, den, color=OKABE['sky'], edgecolor='white', linewidth=0.5,
            height=0.72, zorder=3)
    ax.barh(y, kep, facecolor='none', edgecolor='0.12', hatch='////',
            linewidth=0.0, height=0.72, zorder=5)
    if rem.max() > 0:
        ax.barh(y, -rem, color='#b2182b', edgecolor='white', linewidth=0.5,
                height=0.72, zorder=3)
    for yi, ch in zip(y, cats):
        ax.text((kept[ch] + recov[ch]) / tot[ch] + 0.008, yi,
                f'{kept[ch] + recov[ch]}/{tot[ch]}', va='center', ha='left',
                fontsize=8.2, color='0.15', zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{CAT_DESC.get(ch, ch)}  [{ch}]' for ch in cats],
                       fontsize=9)
    ax.set_ylim(-0.6, len(cats) - 0.4)
    ax.set_xlim(min(-0.05, -(rem.max() + 0.02)), max(den) * 1.18)
    ax.set_xlabel('fraction of COGs present  (left of 0: removed from input)')
    ax.axvline(0, color='0.4', lw=0.8, zorder=2)
    ax.text(0.985, 0.02, model_label, transform=ax.transAxes, ha='right',
            va='bottom', fontsize=7.2, color='0.5')
    handles = [Patch(facecolor=OKABE['sky'], edgecolor='white',
                     label='denoised present'),
               Patch(facecolor='white', edgecolor='0.12', hatch='////',
                     label='present in input (hatched)'),
               Patch(facecolor='#b2182b', label='removed from input')]
    ax.legend(handles=handles, loc='lower right', frameon=True,
              framealpha=0.96, edgecolor='0.8', fontsize=8.2)
    despine(ax)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'wrote {out_pdf}')


# ----  Figure: ground-truth recovery on a real genome (reconstruct_extant.py)
def load_recover_tsv(path):
    """Long TSV -> {fn: {COG_ID: (truth, input_present, pred_bool)}}, sorted fns."""
    by_fn = defaultdict(dict)
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            by_fn[float(row['fn'])][row['COG_ID']] = (
                int(row['truth']), int(row['input_present']),
                float(row['denoised_prob']) > 0.5)
    return by_fn, sorted(by_fn)


def load_recover_meta(path):
    d = {}
    p = Path(path)
    if p.exists():
        with open(p) as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                d[row['key']] = row['value']
    return d


# Fallbacks so a recovery figure self-titles when its .meta.tsv sidecar is
# absent.  Keyed by the tag/model in fig_recover_<tag>_<model>.pdf.
RECOVER_SPECIES = {
    'ecoli': 'Escherichia coli',
    'archaeon': 'Methanocaldococcus jannaschii',
    'medbac': 'Kryptonium thompsonii',
    'medarc': 'SM1-50 sp002506745',
    'medbac_hq': 'UBA7675 sp002483085',
    'medarc_hq': 'Methanocorpusculum sp017387505',
}
RECOVER_MODEL = {'generalist': 'generalist', 'mixft': 'mix-FT', 'bachard': 'bac-FT',
                 'mixfp': 'mix-FT-fp', 'bacfp': 'bac-FT-fp',
                 'gen': 'generalist (HO-T20)', 'bacmarg': 'bac-FT-fp-marginal',
                 'mixmarg': 'mix-FT-fp-marginal'}


def fig_extant_recovery(tsv, cog_def, out_pdf, min_cat=8):
    """Per functional category, a diverging bar per false-negative level of
    what the denoiser does to a corrupted real genome against the known truth.
    All fractions are normalised to the category's true gene count.

    Right of zero: true genes in the output (hatched where they survived
    corruption in the input), true genes lost to FN and not recovered, then
    false positives left in the output.  Left of zero: injected false positives
    the denoiser removed, and true genes it wrongly removed.  Untitled."""
    by_fn, fns = load_recover_tsv(tsv)
    cog2cat = load_cog_cats(cog_def)

    def cats_of(cog):
        return [ch for ch in cog2cat.get(cog, '') if ch.isalpha()]

    # Categories with at least min_cat genes in this genome's true content.
    truth_by_cat = defaultdict(int)
    for cog, (t, i, p) in by_fn[fns[0]].items():
        if t:
            for ch in cats_of(cog):
                truth_by_cat[ch] += 1
    cats = [c for c in truth_by_cat if truth_by_cat[c] >= min_cat]

    def col_counts(fn):
        # Per category: kept/recov/fpOut on the output (right); rmfp/remT
        # removed from the input (left).  tot: genome-wide FP accounting.
        kept = defaultdict(int); recov = defaultdict(int); fpOut = defaultdict(int)
        rmfp = defaultdict(int); remT = defaultdict(int); missed = defaultdict(int)
        rm_fp = inj_fp = 0
        for cog, (t, i, p) in by_fn[fn].items():
            chs = cats_of(cog)
            if t:
                for ch in chs:
                    if i and p:           kept[ch] += 1
                    elif (not i) and p:   recov[ch] += 1
                    elif i and (not p):   remT[ch] += 1   # true gene removed (error)
                    else:                 missed[ch] += 1 # dropped by FN, not recovered
            else:  # not in truth: false-positive territory
                if i:
                    inj_fp += 1
                    if not p:
                        rm_fp += 1
                        for ch in chs:
                            rmfp[ch] += 1               # injected FP removed
                if p:
                    for ch in chs:
                        fpOut[ch] += 1                  # FP left in output (error)
        return kept, recov, fpOut, rmfp, remT, missed, dict(rm_fp=rm_fp, inj_fp=inj_fp)

    ok, orc, *_ = col_counts(fns[0])
    # Canonical order (top to bottom, J..S), identical across panels.
    cats.sort(key=lambda c: CAT_RANK.get(c, 99), reverse=True)

    fig, axes = plt.subplots(1, len(fns), figsize=(3.9 * len(fns), 7.8),
                             sharey=True, constrained_layout=True)
    if len(fns) == 1:
        axes = [axes]
    # Title from the run's metadata sidecar, so stacked rows are self-labelled.
    meta = load_recover_meta(str(tsv).replace('.tsv', '.meta.tsv'))
    if meta:
        sp = meta.get('species', '').replace('s__', '')
        ml = meta.get('model_label', '')
        tt = meta.get('truth_total', '?')
        fpv = meta.get('fp', '?')
    else:
        # No sidecar: derive species and model from the output filename.
        stem = Path(out_pdf).stem.replace('fig_recover_', '')
        tag, _, modk = stem.rpartition('_')
        sp = RECOVER_SPECIES.get(tag, tag)
        ml = RECOVER_MODEL.get(modk, modk)
        tt = sum(1 for _c, (_t, _i, _p) in by_fn[fns[0]].items() if _t)
        fpv = '0.01'
    fig.suptitle(f'{sp}  --  {ml}   (truth {tt} genes, FP={fpv})',
                 fontsize=11, fontweight='bold', y=1.02)
    y = np.arange(len(cats))
    C_TRUE = OKABE['sky']
    C_FP, C_RMFP, C_REMT = OKABE['verm'], OKABE['green'], OKABE['yellow']
    C_MISS = '0.82'   # true gene dropped by FN and not recovered
    for ax, fn in zip(axes, fns):
        kept, recov, fpOut, rmfp, remT, missed, tot = col_counts(fn)
        Tc = np.array([max(truth_by_cat[c], 1) for c in cats], dtype=float)
        kp = np.array([kept[c] for c in cats]) / Tc
        rc = np.array([recov[c] for c in cats]) / Tc
        fp = np.array([fpOut[c] for c in cats]) / Tc
        rf = np.array([rmfp[c] for c in cats]) / Tc
        rt = np.array([remT[c] for c in cats]) / Tc
        ms = np.array([missed[c] for c in cats]) / Tc
        ax.axvline(0, color='0.4', lw=0.8, zorder=2)
        ax.axvline(1.0, color='0.7', lw=0.7, ls=':', zorder=2)
        # Grey fills out to the true category size (dotted 1.0 line); false
        # positives left in the output stack beyond it.
        ax.barh(y, kp + rc, color=C_TRUE, height=0.74, zorder=3)
        ax.barh(y, kp, facecolor='none', edgecolor='0.12', hatch='////',
                linewidth=0.0, height=0.74, zorder=5)
        ax.barh(y, ms, left=kp + rc, color=C_MISS, height=0.74, zorder=3)
        ax.barh(y, fp, left=kp + rc + ms, color=C_FP, height=0.74, zorder=3)
        # Left of 0: injected false positives cleaned out, then true genes
        # wrongly removed.
        ax.barh(y, -rf, color=C_RMFP, height=0.74, zorder=3)
        ax.barh(y, -rt, left=-rf, color=C_REMT, height=0.74, zorder=3)
        ax.set_title(f"$f_N = {fn:g}$", fontsize=10)
        ax.set_xlabel('fraction of true category size')
        despine(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'{CAT_DESC.get(c, c)}  [{c}]' for c in cats],
                            fontsize=8.5)
    axes[0].set_ylim(-0.6, len(cats) - 0.4)
    handles = [Patch(facecolor=C_TRUE, edgecolor='white',
                     label='true gene present in output'),
               Patch(facecolor='white', edgecolor='0.12', hatch='////',
                     label='present in input (hatched)'),
               Patch(facecolor=C_MISS, label='true gene not recovered (missed FN)'),
               Patch(facecolor=C_FP, label='false positive left in output'),
               Patch(facecolor=C_RMFP, label='injected false positive removed'),
               Patch(facecolor=C_REMT, label='true gene removed (error)')]
    # 'outside' makes constrained_layout reserve a strip below the panels, so
    # the legend never overlaps the bars or the x-labels.
    fig.legend(handles=handles, loc='outside lower center', ncol=3,
               fontsize=8.0, frameon=False)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'wrote {out_pdf}')


def fig_ancestral_profile(pred, cog_def, module_pt, out_pdf, model_label,
                          curated, group_label, var='actual', inp_thresh=0.0,
                          expected=False, side_by_side=False):
    """Two-panel ancestral figure: (A) per-COG functional-category profile,
    (B) curated key KEGG systems.

    expected=False (LBCA): a COG is present if its posterior clears 0.5, and
      kept/recovered/removed are integer family counts.
    expected=True (LACA): posteriors are summed per category/module, so the
      output bar is E[#present]=sum(mean_var), the hatched input is
      sum(min(input,output)) per COG, and removed (left of 0) is
      sum(max(0, input-output)).  No input threshold is needed, since the
      diffuse arCOG->COG tail contributes only its small probability mass."""
    import torch
    cog2cat = load_cog_cats(cog_def)
    mm = torch.load(module_pt, map_location='cpu', weights_only=False)
    if expected:
        out_p, in_p = load_raw(pred, var=var)
        dec = module_decomp_expected(out_p, in_p, mm)
    else:
        present, inp = load_present(pred, var=var, inp_thresh=inp_thresh)
        dec = module_decomp(present, inp, mm)

    if side_by_side:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 6.6),
                                       constrained_layout=True,
                                       gridspec_kw={'width_ratios': [1.0, 0.95]})
    else:
        fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.6, 10.2),
                                       constrained_layout=True,
                                       gridspec_kw={'height_ratios': [1.15, 1.0]})

    # Panel A: per-COG functional-category profile.
    tot = defaultdict(float); kept = defaultdict(float)
    recov = defaultdict(float); removed = defaultdict(float)
    if expected:
        for c, po in out_p.items():
            pi = in_p.get(c, 0.0)
            for ch in cog2cat.get(c, ''):
                if not ch.isalpha():
                    continue
                tot[ch] += 1
                kept[ch] += min(pi, po)
                recov[ch] += max(0.0, po - pi)
                removed[ch] += max(0.0, pi - po)
    else:
        for c, is_p in present.items():
            i = inp.get(c, False)
            for ch in cog2cat.get(c, ''):
                if not ch.isalpha():
                    continue
                tot[ch] += 1
                if i and is_p:         kept[ch] += 1
                elif (not i) and is_p: recov[ch] += 1
                elif i and (not is_p): removed[ch] += 1
    cats = [ch for ch in tot if tot[ch] >= 10]
    cats.sort(key=lambda ch: (kept[ch] + recov[ch]) / tot[ch])
    yA = np.arange(len(cats))
    denA = np.array([(kept[ch] + recov[ch]) / tot[ch] for ch in cats])
    kepA = np.array([kept[ch] / tot[ch] for ch in cats])
    remA = np.array([removed[ch] / tot[ch] for ch in cats])
    axA.barh(yA, denA, color=OKABE['sky'], edgecolor='white', linewidth=0.5,
             height=0.72, zorder=3)
    axA.barh(yA, kepA, facecolor='none', edgecolor='0.12', hatch='////',
             linewidth=0.0, height=0.72, zorder=5)
    if remA.max() > 0:
        axA.barh(yA, -remA, color='#b2182b', edgecolor='white', linewidth=0.5,
                 height=0.72, zorder=3)
    for yi, ch in zip(yA, cats):
        val = kept[ch] + recov[ch]
        lab = f'{val:.0f}/{int(tot[ch])}' if expected else \
              f'{int(val)}/{int(tot[ch])}'
        axA.text(val / tot[ch] + 0.008, yi, lab, va='center', ha='left',
                 fontsize=7.4, color='0.15', zorder=6)
    axA.set_yticks(yA)
    axA.set_yticklabels([f'{CAT_DESC.get(ch, ch)}  [{ch}]' for ch in cats],
                        fontsize=8)
    axA.set_ylim(-0.6, len(cats) - 0.4)
    axA.set_xlim(min(-0.05, -(remA.max() + 0.02)), max(denA) * 1.18)
    axA.set_xlabel('expected fraction of COGs present  (left of 0: removed)'
                   if expected else
                   'fraction of COGs present  (left of 0: removed)')
    axA.set_title('(A)  functional-category profile', loc='left',
                  fontweight='bold')
    axA.axvline(0, color='0.4', lw=0.8, zorder=2)
    despine(axA)
    _legA = [
        Patch(facecolor=OKABE['sky'], edgecolor='white',
              label='expected present' if expected else 'denoised present'),
        Patch(facecolor='white', edgecolor='0.12', hatch='////',
              label='present in input'),
        Patch(facecolor='#b2182b', label='removed from input')]
    if side_by_side:
        axA.legend(handles=_legA, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   ncol=3, fontsize=7.8, frameon=False)
    else:
        axA.legend(handles=_legA, loc='lower right', fontsize=7.3, frameon=True,
                   framealpha=0.96, edgecolor='0.8')

    # Panel B: curated key systems.
    rows = []
    for mid, label, grp in curated:
        if mid not in dec:
            continue
        k, r, rm, n = dec[mid]
        rows.append((label, grp, k, r, rm, n))
    rows.sort(key=lambda z: (z[2] + z[3]) / z[5])
    yB = np.arange(len(rows))
    cols = [GROUP_COLOR[z[1]] for z in rows]
    denB = np.array([(k + r) / n for (_, _, k, r, rm, n) in rows])
    kepB = np.array([k / n for (_, _, k, r, rm, n) in rows])
    remB = np.array([rm / n for (_, _, k, r, rm, n) in rows])
    axB.barh(yB, denB, color=cols, edgecolor='white', linewidth=0.5,
             height=0.74, zorder=3)
    axB.barh(yB, kepB, facecolor='none', edgecolor='0.12', hatch='////',
             linewidth=0.0, height=0.74, zorder=5)
    if rows and remB.max() > 0:
        axB.barh(yB, -remB, color='#b2182b', edgecolor='white', linewidth=0.5,
                 height=0.74, zorder=3)
    for yi, (_, _, k, r, rm, n) in zip(yB, rows):
        lab = f'{k + r:.1f}/{n}' if expected else f'{int(k + r)}/{n}'
        axB.text((k + r) / n + 0.015, yi, lab, va='center',
                 ha='left', fontsize=8.0, color='0.15', zorder=6)
    axB.set_yticks(yB)
    axB.set_yticklabels([z[0] for z in rows], fontsize=8.5)
    axB.set_ylim(-0.6, len(rows) - 0.4)
    axB.set_xlim(min(-0.06, -((remB.max() if len(rows) else 0) + 0.03)), 1.18)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xlabel('expected fraction of module COGs  (left of 0: removed)'
                   if expected else
                   'fraction of module COGs  (left of 0: removed)')
    axB.set_title('(B)  curated key systems', loc='left', fontweight='bold')
    axB.axvline(0, color='0.4', lw=0.8, zorder=2)
    axB.axvline(1.0, color='0.6', lw=0.7, ls=':', zorder=1)
    despine(axB)
    groups_present = [g for g in group_label if any(z[1] == g for z in rows)]
    handlesB = [Patch(fc=GROUP_COLOR[g], ec='white', label=group_label[g])
                for g in groups_present]
    handlesB += [Patch(facecolor='white', edgecolor='0.12', hatch='////',
                       label='present in input'),
                 Patch(facecolor='#b2182b', label='removed from input')]
    if side_by_side:
        axB.legend(handles=handlesB, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   ncol=3, fontsize=7.6, frameon=False)
    else:
        axB.legend(handles=handlesB, loc='upper center', bbox_to_anchor=(0.5, -0.11),
                   ncol=3, fontsize=7.2, frameon=False)
    # Node inferred from the output filename.
    _stem = Path(out_pdf).stem.lower()
    node = ('LBCA' if 'lbca' in _stem else 'LACA' if 'laca' in _stem else '')
    suptitle = f'{node} reconstruction  --  {model_label}' if node else model_label
    fig.suptitle(suptitle, fontsize=12.5, fontweight='bold')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'wrote {out_pdf}')


def main():
    ap = argparse.ArgumentParser()
    # LBCA via the bacterial specialist (bac-FT), LACA via the mixed-domain
    # generalist (mix-FT), from the same 10-split-ensemble pred tables the
    # FP-control baselines use.  node2012_T20mix is a mix-FT relaxation of the
    # LBCA node, the wrong model for it, and is not used here.
    ap.add_argument('--lbca-pred',
                    default=str(REPO / 'data/interactome/fpcontrol/LBCA_default_baseline_pred.tsv'))
    ap.add_argument('--laca-pred',
                    default=str(REPO / 'data/interactome/fpcontrol/LACA_default_baseline_pred.tsv'))
    ap.add_argument('--module-pt', default=str(REPO / 'data/module_matrix_kegg.pt'))
    ap.add_argument('--cog-def', default=str(REPO / 'data/cog-20.def.tab'))
    ap.add_argument('--outdir', default=str(REPO / 'analysis/figures'))
    ap.add_argument('--lbca-label', default='bac-FT, face value')
    ap.add_argument('--laca-label', default='mix-FT, face value')
    ap.add_argument('--lbca-var', default='actual',
                    help="pred column to plot: actual (raw x=p) or a binarize "
                         "variant, e.g. '>0' (most liberal), '>0.04'.")
    ap.add_argument('--laca-var', default='actual')
    ap.add_argument('--lbca-input-thr', type=float, default=0.0,
                    help='input_prob threshold for the "present in input" hatch '
                         '(default 0.0 = any support, faithful for a reconciliation '
                         'posterior; use 0.5 for sum-aggregated inputs whose '
                         'tiny-value tail otherwise inflates the input set and '
                         'hides the added/rescued genes).')
    ap.add_argument('--laca-input-thr', type=float, default=0.0,
                    help='as --lbca-input-thr, for the LACA figure.')
    ap.add_argument('--recover', nargs='*', default=[],
                    help='extant ground-truth recovery figures; each entry is '
                         '"recover_tsv:out_pdf_name" (reconstruct_extant.py output)')
    ap.add_argument('--skip-ancestral', action='store_true',
                    help='only build the --recover figures (skip LBCA/LACA)')
    ap.add_argument('--lbca-out', default='fig_lbca_modules.pdf',
                    help='LBCA output pdf name (default fig_lbca_modules.pdf)')
    ap.add_argument('--laca-out', default='fig_laca_categories.pdf',
                    help='LACA output pdf name (default fig_laca_categories.pdf)')
    ap.add_argument('--only', choices=['lbca', 'laca'], default=None,
                    help='build only one of the two ancestral figures')
    ap.add_argument('--laca-expected', dest='laca_expected',
                    action='store_true', default=True,
                    help='LACA focal reconstruction: bars are the SUM OF '
                         'EXPECTED GENES (sum of posteriors) for input and '
                         'output rather than thresholded present-counts '
                         '(default; LBCA always stays binary).')
    ap.add_argument('--no-laca-expected', dest='laca_expected',
                    action='store_false',
                    help='revert the LACA figure to binary present-counts.')
    ap.add_argument('--side-by-side', action='store_true',
                    help='lay the (A) functional-category profile and (B) curated '
                         'key systems panels side by side in one row (default: '
                         'stacked vertically).')
    args = ap.parse_args()

    setup_style()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_ancestral:
        if args.only != 'laca':
            fig_ancestral_profile(args.lbca_pred, args.cog_def, args.module_pt,
                                  out / args.lbca_out, args.lbca_label,
                                  LBCA_CURATED, LBCA_GROUP_LABEL, var=args.lbca_var,
                                  inp_thresh=args.lbca_input_thr, expected=False,
                                  side_by_side=args.side_by_side)
        if args.only != 'lbca':
            fig_ancestral_profile(args.laca_pred, args.cog_def, args.module_pt,
                                  out / args.laca_out, args.laca_label,
                                  LACA_CURATED, LACA_GROUP_LABEL, var=args.laca_var,
                                  inp_thresh=args.laca_input_thr,
                                  expected=args.laca_expected,
                                  side_by_side=args.side_by_side)
    for spec in args.recover:
        tsv, _, name = spec.partition(':')
        fig_extant_recovery(Path(tsv), args.cog_def, out / name)


if __name__ == '__main__':
    main()
