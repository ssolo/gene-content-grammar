#!/usr/bin/env python3
"""LACA-inputs comparison: over the four canonical denoised reconstructions,
print the input set size by probability threshold beside the denoised present-set
size, and the Jaccard matrix of the denoised present sets.

Every input is a noisy-OR arCOG->COG projection built by
extract_laca_arcog_node.py or build_laca_gld_input.py, then denoised by the
consistency ensemble at raw x=p into the LACA_<name>_pred.tsv files read here:

  python scripts/analyze_ancestral_node.py --actual-mode raw \
      --models 'gsd_results_consistency_*l1.0*hq_split*/model_ho3.pth' \
      --table <input> --node <node> --csv-out LACA_<name>_pred.tsv
"""
import csv, itertools, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECONS = [('merged sparse', 'LACA_merged_pred.tsv'),
          ('Euryroot',      'LACA_euryroot_pred.tsv'),
          ('GLD min1',      'LACA_gld_min1_pred.tsv'),
          ('GLD min4',      'LACA_gld_min4_pred.tsv')]


def load(f):
    """(COG_ID -> input probability, denoised present set at posterior > 0.5)."""
    rows = list(csv.DictReader(open(os.path.join(HERE, f)), delimiter='\t'))
    ip = {r['COG_ID']: float(r['input_prob']) for r in rows}
    pres = {r['COG_ID'] for r in rows if float(r['mean_actual']) > 0.5}
    return ip, pres


def main():
    D = {n: load(f) for n, f in RECONS}
    names = [n for n, _ in RECONS]
    print("=== input by threshold  ->  denoised present (cons-10, raw x=p, noisy-OR) ===")
    print("%-14s %6s %7s %6s %6s | %8s" % ('input', '>0', '>0.01', '>0.1', '>0.5', 'denoised'))
    for n in names:
        v = list(D[n][0].values())
        print("%-14s %6d %7d %6d %6d | %8d" % (
            n, sum(x > 0 for x in v), sum(x > 0.01 for x in v),
            sum(x > 0.1 for x in v), sum(x > 0.5 for x in v), len(D[n][1])))
    print("\n=== Jaccard of denoised present sets ===")
    print("%-14s" % "" + "".join("%10s" % n for n in names))
    for a in names:
        A = D[a][1]
        print("%-14s" % a + "".join("%10.3f" % (len(A & D[b][1]) / len(A | D[b][1])) for b in names))
    print("\n=== shared / sparse-only / dense-only ===")
    for sp, dn in itertools.product(['merged sparse', 'Euryroot'], ['GLD min1', 'GLD min4']):
        A, B = D[sp][1], D[dn][1]
        print("  %-13s vs %-8s: shared=%4d  %s-only=%4d  %s-only=%4d  J=%.3f"
              % (sp, dn, len(A & B), sp, len(A - B), dn, len(B - A), len(A & B) / len(A | B)))


if __name__ == '__main__':
    main()
