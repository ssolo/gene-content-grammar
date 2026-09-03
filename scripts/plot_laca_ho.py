#!/usr/bin/env python3
"""plot_laca_ho.py -- diagnostic figures for the LACA (last archaeal
common ancestor) denoising run of the tied-attention higher-order
ensemble.

Reads the per-root prediction tables for the two alternative archaeal
root placements, Euryroot and MHHroot, plus the canonical LACA_pred.tsv
whose input side averages the two.  Each table gives, per COG, the input
copy number from the reconciliation and the denoised mean and s.d. over
the five input variants in VARIANTS.  Six PNGs are written next to them:

  LACA_HO_input_vs_denoised.png             input copy number vs denoised
                                            mean, one panel per root
  LACA_HO_per_variant_counts.png            COGs called present under each
                                            input variant, per root
  LACA_HO_func_categories.png               functional categories of the
                                            cross-root consensus sets
  LACA_HO_set_overlap.png                   overlap of the Eury, MHH and
                                            LBCA all-variant consensus sets
  LACA_HO_top_rescued_silenced_heatmap.png  the strongest rescued and
                                            silenced COGs over 2 roots x
                                            5 variants
  LACA_canonical_input_vs_denoised.png      the first panel again for the
                                            canonical report
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPORTS = {
    'Euryroot': REPO / 'LACA_Euryroot_HO_tied_pred.tsv',
    'MHHroot':  REPO / 'LACA_MHHroot_HO_tied_pred.tsv',
}
# Canonical LACA report (HO_tied, both roots averaged on input side)
LACA_PRED = REPO / 'LACA_pred.tsv'
LACA_ANALYSIS = REPO / 'LACA_analysis.txt'
CONS_TXT = {
    'Euryroot': REPO / 'LACA_Euryroot_HO_tied_analysis.txt',
    'MHHroot':  REPO / 'LACA_MHHroot_HO_tied_analysis.txt',
    'LBCA':     REPO / 'node2012_analysis.txt',
}
COG_DEF = REPO / 'data/cog-20.def.tab'

CAT_DESC = {
    'J': 'Translation, ribosomal',     'A': 'RNA processing',
    'K': 'Transcription',              'L': 'Replication, recomb, repair',
    'B': 'Chromatin',                  'D': 'Cell cycle / division',
    'V': 'Defense',                    'T': 'Signal transduction',
    'M': 'Cell wall/membrane',         'N': 'Cell motility',
    'U': 'Trafficking, secretion',     'O': 'PTM, chaperones',
    'C': 'Energy production',          'G': 'Carbohydrate metab.',
    'E': 'Amino acid metab.',          'F': 'Nucleotide metab.',
    'H': 'Coenzyme metab.',            'I': 'Lipid metab.',
    'P': 'Inorganic ion transport',    'Q': 'Secondary metabolites',
    'R': 'General fn pred only',       'S': 'Function unknown',
    'X': 'Mobilome',                   'W': 'Extracellular',
    'Y': 'Nuclear structure',          'Z': 'Cytoskeleton',
}

VARIANTS = ['actual', '>0', '>0.01', '>0.04', '>0.099']


def load_pred(path):
    """Return list of (cog, input_prob, [mean_v for v in VARIANTS], [sd_v])."""
    rows = []
    with open(path) as fh:
        r = csv.DictReader(fh, delimiter='\t')
        for row in r:
            cog = row['COG_ID']
            inp = float(row['input_prob'])
            means = [float(row[f'mean_{v}']) for v in VARIANTS]
            sds   = [float(row[f'sd_{v}'])   for v in VARIANTS]
            rows.append((cog, inp, means, sds))
    return rows


def classify(rows):
    """Return dict cog -> status.

    A COG is called present under a variant when its denoised mean
    exceeds 0.50.  Present under every variant and absent from the input
    is "rescued", absent under every variant while present in the input
    is "silenced", agreement in either direction is "confirmed_present" /
    "confirmed_absent", and anything variant-dependent is "partial".
    """
    out = {}
    for cog, inp, means, _ in rows:
        all_present = all(m > 0.50 for m in means)
        all_absent  = all(m <= 0.50 for m in means)
        if inp == 0.0 and all_present:    out[cog] = 'rescued'
        elif inp > 0.0  and all_absent:   out[cog] = 'silenced'
        elif inp > 0.0  and all_present:  out[cog] = 'confirmed_present'
        elif inp == 0.0 and all_absent:   out[cog] = 'confirmed_absent'
        else:                              out[cog] = 'partial'
    return out


def load_cog_cats(path):
    out = {}
    with open(path, encoding='latin-1') as fh:
        for row in csv.reader(fh, delimiter='\t'):
            if not row: continue
            out[row[0]] = row[1] if len(row) > 1 else ''
    return out


def parse_consensus_set(path):
    """Pull the all-5-variant intersection COG IDs from an analysis report.

    The IDs sit in an indented block two lines below the "COGs in the
    all-5-variant intersection" header and run until the first blank or
    rule line.  A report without that header yields the empty set.
    """
    text = Path(path).read_text().splitlines()
    start = None
    for i, line in enumerate(text):
        if 'COGs in the all-5-variant intersection' in line:
            start = i + 2; break
    if start is None: return set()
    rx = re.compile(r'^\s+(COG\d{4,5})\s+')
    out = set()
    for line in text[start:]:
        if not line.strip() or line.lstrip().startswith(('=', '-')):
            break
        m = rx.match(line)
        if m: out.add(m.group(1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build HO_tied-only LACA diagnostic plots.')
    parser.parse_args(argv)

    data = {}
    for root, path in REPORTS.items():
        rows = load_pred(path)
        status = classify(rows)
        data[root] = {'rows': rows, 'status': status}

    cog2cat = load_cog_cats(COG_DEF)

    consensus = {k: parse_consensus_set(v) for k, v in CONS_TXT.items()}


    COLORS = {
        'rescued':            '#d62728',   # red
        'silenced':           '#1f77b4',   # blue
        'confirmed_present':  '#2ca02c',   # green
        'confirmed_absent':   '#888888',   # grey
        'partial':            '#ffbf00',   # amber
    }


    # ---- Plot 1: input vs denoised mean, one panel per root, symlog x
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (root, d) in zip(axes, data.items()):
        rows = d['rows']
        status = d['status']
        order = ['confirmed_absent', 'partial', 'confirmed_present', 'rescued', 'silenced']
        for st in order:
            xs = []; ys = []
            for cog, inp, means, _ in rows:
                if status[cog] != st: continue
                xs.append(inp)
                ys.append(np.mean(means))
            label = f'{st.replace("_", " ")}  (n={len(xs)})'
            if st == 'confirmed_absent':
                ax.scatter(xs, ys, c=COLORS[st], s=5, alpha=0.18, label=label, edgecolors='none')
            elif st == 'partial':
                ax.scatter(xs, ys, c=COLORS[st], s=14, alpha=0.55, label=label, edgecolors='none')
            elif st == 'rescued':
                ax.scatter(xs, ys, c=COLORS[st], s=22, alpha=0.85, label=label, edgecolors='none')
            elif st == 'silenced':
                ax.scatter(xs, ys, c=COLORS[st], s=28, alpha=0.95, label=label, edgecolors='black', linewidths=0.5)
            else:  # confirmed_present
                ax.scatter(xs, ys, c=COLORS[st], s=14, alpha=0.55, label=label, edgecolors='none')
        ax.axhline(0.50, color='k', linewidth=0.7, linestyle=':')
        ax.axvline(0.5,  color='k', linewidth=0.5, linestyle=':', alpha=0.3)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlim(-0.005, 1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Input copy number  (recon; symlog scale, linear below 0.01)')
        ax.set_title(f'{root}  HO_tied   (n={len(rows)} COGs in vocab)')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
                  markerscale=1.5)
        ax.grid(alpha=0.25, linestyle='--')
    axes[0].set_ylabel('Denoised mean (averaged over 5 input variants)')
    fig.suptitle('LACA HO_tied: input copy number vs denoised mean   '
                 '(red=rescued, blue=silenced, green=confirmed present, '
                 'grey=confirmed absent, amber=partial)',
                 fontsize=12)
    fig.tight_layout()
    out = REPO / 'LACA_HO_input_vs_denoised.png'
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')
    plt.close(fig)


    # ---- Plot 2: predicted-present counts per variant, grouped by root
    counts = {root: [] for root in data}
    for root, d in data.items():
        for i, v in enumerate(VARIANTS):
            n = sum(1 for _, _, means, _ in d['rows'] if means[i] > 0.50)
            counts[root].append(n)

    x = np.arange(len(VARIANTS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w/2, counts['Euryroot'], w, label='Euryroot', color='#1f77b4')
    b2 = ax.bar(x + w/2, counts['MHHroot'],  w, label='MHHroot',  color='#ff7f0e')
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 8, f'{int(h)}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
    ax.set_xlabel('Input variant')
    ax.set_ylabel('COGs predicted present (denoised mean > 0.50)')
    ax.set_title('LACA HO_tied: COGs predicted present per variant')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    out = REPO / 'LACA_HO_per_variant_counts.png'
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')
    plt.close(fig)


    # ---- Plot 3: functional categories of the cross-root consensus.
    # A COG counts only if it carries the same status under both roots,
    # so a category's bars are an intersection, not an average.
    def status_sets(d):
        s = d['status']
        return {
            'rescued':           {c for c, st in s.items() if st == 'rescued'},
            'silenced':          {c for c, st in s.items() if st == 'silenced'},
            'confirmed_present': {c for c, st in s.items() if st == 'confirmed_present'},
        }

    ss_e = status_sets(data['Euryroot'])
    ss_m = status_sets(data['MHHroot'])

    joint = {
        'rescued':           ss_e['rescued']           & ss_m['rescued'],
        'silenced':          ss_e['silenced']          & ss_m['silenced'],
        'confirmed_present': ss_e['confirmed_present'] & ss_m['confirmed_present'],
    }
    print(f'\nHO_tied cross-root joint:')
    for k, s in joint.items():
        print(f'  {k}: {len(s)}')

    def cat_counts(cog_set):
        c = Counter()
        for cog in cog_set:
            cats = cog2cat.get(cog, '')
            for ch in cats: c[ch] += 1
        return c

    cc = {k: cat_counts(s) for k, s in joint.items()}
    total = Counter()
    for c in cc.values():
        for cat, n in c.items(): total[cat] += n
    top_cats = [cat for cat, _ in total.most_common(18) if cat in CAT_DESC]

    x = np.arange(len(top_cats))
    w = 0.27
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(x - w, [cc['confirmed_present'].get(c, 0) for c in top_cats], w,
           label=f'Confirmed present  (n={len(joint["confirmed_present"])})',
           color=COLORS['confirmed_present'])
    ax.bar(x,     [cc['rescued'].get(c, 0)           for c in top_cats], w,
           label=f'Rescued (added back) (n={len(joint["rescued"])})',
           color=COLORS['rescued'])
    ax.bar(x + w, [cc['silenced'].get(c, 0)          for c in top_cats], w,
           label=f'Silenced (removed)   (n={len(joint["silenced"])})',
           color=COLORS['silenced'])
    ax.set_xticks(x)
    labels = [f'{c}: {CAT_DESC.get(c,"?")}' for c in top_cats]
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('COG count (cross-root HO_tied joint)')
    ax.set_title('LACA HO_tied: functional categories of confirmed / rescued / silenced (both roots agree)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    out = REPO / 'LACA_HO_func_categories.png'
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')
    plt.close(fig)


    # ---- Plot 4: overlap of the Eury, MHH and LBCA consensus cores
    A = consensus['Euryroot']           # Eury HO_tied 5-variant intersection
    B = consensus['MHHroot']            # MHH  HO_tied 5-variant intersection
    C = consensus['LBCA']               # LBCA nohidden 5-variant intersection
    only_A = A - B - C
    only_B = B - A - C
    only_C = C - A - B
    AB     = (A & B) - C
    AC     = (A & C) - B
    BC     = (B & C) - A
    ABC    = A & B & C

    print(f'\nSet sizes:')
    print(f'  Eury HO_tied core (A):  {len(A)}')
    print(f'  MHH  HO_tied core (B):  {len(B)}')
    print(f'  LBCA nohidden core (C): {len(C)}')
    print(f'  ABC: {len(ABC)}, AB-only: {len(AB)}, AC-only: {len(AC)}, BC-only: {len(BC)}')
    print(f'  A-only: {len(only_A)}, B-only: {len(only_B)}, C-only: {len(only_C)}')

    fig, ax = plt.subplots(figsize=(9, 7))
    # Circles are fixed in place and equal in radius, not area-proportional;
    # the printed counts carry the sizes.
    from matplotlib.patches import Circle
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.0, 3.5); ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Circle((-1.0,  0.8), 1.9, alpha=0.32, fc='#1f77b4', ec='black', lw=1.2))
    ax.add_patch(Circle(( 1.0,  0.8), 1.9, alpha=0.32, fc='#ff7f0e', ec='black', lw=1.2))
    ax.add_patch(Circle(( 0.0, -1.2), 1.9, alpha=0.32, fc='#2ca02c', ec='black', lw=1.2))
    def lbl(x, y, n, fs=14, bold=False, color='black'):
        ax.text(x, y, str(n), ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', color=color)
    lbl(-2.4,  1.2, len(only_A))                       # Eury only
    lbl( 2.4,  1.2, len(only_B))                       # MHH  only
    lbl( 0.0, -2.4, len(only_C))                       # LBCA only
    lbl( 0.0,  1.8, len(AB))                           # Eury n MHH (not LBCA)
    lbl(-1.4, -0.7, len(AC))                           # Eury n LBCA (not MHH)
    lbl( 1.4, -0.7, len(BC))                           # MHH n LBCA (not Eury)
    lbl( 0.0,  0.0, len(ABC), fs=16, bold=True)       # ABC triple
    ax.text(-2.0,  2.8, f'Euryroot HO_tied core\n(n={len(A)})',
            ha='center', fontsize=11, fontweight='bold', color='#1f77b4')
    ax.text( 2.0,  2.8, f'MHHroot HO_tied core\n(n={len(B)})',
            ha='center', fontsize=11, fontweight='bold', color='#ff7f0e')
    ax.text( 0.0, -3.0, f'LBCA node2012 core\n(n={len(C)})',
            ha='center', fontsize=11, fontweight='bold', color='#2ca02c')
    ax.set_title('Cross-variant consensus overlap: Eury / MHH / LBCA',
                 fontsize=13, pad=10)
    fig.tight_layout()
    out = REPO / 'LACA_HO_set_overlap.png'
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')
    plt.close(fig)


    # ---- Plot 5: heatmap of the 30 cross-root-consensus COGs rescued most
    # strongly and the 12 silenced most strongly, over 2 roots x 5 variants.
    def mean_across_reports(cog):
        vals = []
        for d in data.values():
            for c, _, means, _ in d['rows']:
                if c == cog:
                    vals.extend(means); break
        return float(np.mean(vals)) if vals else 0.0

    rescued_top = sorted(joint['rescued'],
                         key=lambda c: -mean_across_reports(c))[:30]
    silenced_top = sorted(joint['silenced'],
                          key=lambda c: mean_across_reports(c))[:12]
    rows = rescued_top + silenced_top

    # (n_rows, 10): the 5 variants of the first root, then of the second.
    col_labels = []
    for root in ['Euryroot', 'MHHroot']:
        for v in VARIANTS:
            col_labels.append(f'{root[0]}-{v}')

    mat = np.zeros((len(rows), 10))
    for i, cog in enumerate(rows):
        for j, (root, d) in enumerate(data.items()):
            for c, _, means, _ in d['rows']:
                if c == cog:
                    for k, m in enumerate(means):
                        mat[i, j*5 + k] = m
                    break

    fig, ax = plt.subplots(figsize=(11, 0.30*len(rows) + 2.5))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows)))
    labels = []
    for cog in rows:
        cat = cog2cat.get(cog, '?')
        labels.append(f'{cog}  [{cat:3s}]')
    ax.set_yticklabels(labels, fontsize=7)
    ax.axhline(len(rescued_top) - 0.5, color='black', linewidth=1.6)
    ax.text(-0.5, len(rescued_top)/2 - 0.5, 'RESCUED', rotation=90,
            va='center', ha='right', fontsize=11, fontweight='bold', color='#d62728')
    ax.text(-0.5, len(rescued_top) + len(silenced_top)/2 - 0.5, 'SILENCED', rotation=90,
            va='center', ha='right', fontsize=11, fontweight='bold', color='#1f77b4')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Denoised mean')
    ax.set_title('LACA HO_tied: top 30 RESCUED (added) + 12 SILENCED (removed) cross-root consensus COGs')
    fig.tight_layout()
    out = REPO / 'LACA_HO_top_rescued_silenced_heatmap.png'
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')
    plt.close(fig)


    # ---- Plot 6: Plot 1 for the canonical report, whose input copy number
    # is the average over both roots.
    laca_rows = load_pred(LACA_PRED)
    laca_status = classify(laca_rows)
    fig, ax = plt.subplots(figsize=(11, 7))
    order = ['confirmed_absent', 'partial', 'confirmed_present', 'rescued', 'silenced']
    for st in order:
        xs = []; ys = []
        for cog, inp, means, _ in laca_rows:
            if laca_status[cog] != st: continue
            xs.append(inp)
            ys.append(np.mean(means))
        label = f'{st.replace("_", " ")}  (n={len(xs)})'
        if st == 'confirmed_absent':
            ax.scatter(xs, ys, c=COLORS[st], s=5, alpha=0.18, label=label, edgecolors='none')
        elif st == 'partial':
            ax.scatter(xs, ys, c=COLORS[st], s=14, alpha=0.55, label=label, edgecolors='none')
        elif st == 'rescued':
            ax.scatter(xs, ys, c=COLORS[st], s=22, alpha=0.85, label=label, edgecolors='none')
        elif st == 'silenced':
            ax.scatter(xs, ys, c=COLORS[st], s=28, alpha=0.95, label=label,
                       edgecolors='black', linewidths=0.5)
        else:
            ax.scatter(xs, ys, c=COLORS[st], s=14, alpha=0.55, label=label, edgecolors='none')
    ax.axhline(0.50, color='k', linewidth=0.7, linestyle=':')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_xlim(-0.005, 1.2); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Input copy number (avg of Eury + MHH recon; symlog, linear below 0.01)')
    ax.set_ylabel('Denoised mean (averaged over 5 input variants)')
    ax.set_title('LACA canonical HO_tied report  --  '
                 f'input vs denoised  (n={len(laca_rows)} COGs in vocab)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, markerscale=1.5)
    ax.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    out = REPO / 'LACA_canonical_input_vs_denoised.png'
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')
    plt.close(fig)
    print('\nCanonical LACA HO_tied (input = avg of both roots):')
    from collections import Counter as _C
    ct = _C(laca_status.values())
    for k, n in ct.most_common():
        print(f'  {k}: {n}')


if __name__ == "__main__":
    main()
