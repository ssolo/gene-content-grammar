#!/usr/bin/env python3
"""
compare_lbca_reconstructions.py - compare the marginal-FP-HQ denoiser's LBCA
reconstructions from two upstream inputs at root node 2012.

  soft-trim      denoise of the Count/softlanding LBCA (a dense, confident
                 posterior).  Two variants: min1 and min4 gene-occurrence cuts.
  sparse-rescue  denoise of the ALE/reconciliation LBCA at node2012 (a diffuse
                 posterior; almost nothing is confident).

"Apparent FN/FP vs the input" requires binarising the input, and the two inputs
have completely different confidence structure, so a single shared threshold is
either fine (Count) or meaningless (reconciliation).  The policy is therefore
fixed here rather than left to the caller:

  output presence            : mean_actual >= 0.5      (denoiser posterior call)
  Count/softlanding input    : input_prob  >  0.5      (dense; no ties at 0.5)
  reconciliation input       : not binarised at one threshold.  Removal and
                               retention are qualified across a confidence grid
                               p (input_prob >= p), p in {0.01, 0.02, ...}.

Inputs are re-derived from their original provenance files and checked against
the input_prob column recorded in the reconstruction TSVs.

    python scripts/compare_lbca_reconstructions.py

Outputs: a printed report, data/lbca_comparison_summary.json,
data/lbca_reconciliation_removal_by_confidence.csv and data/lbca_cog_lists.tsv.
Exits non-zero if any check fails.
"""
import csv, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(HERE, *a)

PRESENT = 0.5                         # output presence threshold (mean_actual >= PRESENT)
COUNT_INPUT_THR = 0.5                 # softlanding/Count input present (input_prob > COUNT_INPUT_THR)
CONF_GRID = [0.01,0.02,0.03,0.04,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50]

RECONS = {
    'soft-trim-min1': dict(tsv='marg_soft.tsv', kind='count',
                           prov=('davin_softlanding_min1_capped_out/davin.families.csv', '1886')),
    'soft-trim-min4': dict(tsv='marg_soft_min4.tsv', kind='count',
                           prov=('davin_softlanding_min4_capped_out/davin.families.csv', '1886')),
    'sparse-rescue':  dict(tsv='marg_sparse.tsv', kind='reconciliation',
                           prov=('data/LBCA_node2012_input_aggregated.tsv', None)),
}

FAILURES = []
def check(cond, msg):
    ok = bool(cond)
    print(f"   [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok: FAILURES.append(msg)
    return ok

# ---- loading
def load_recon(tsv):
    """COG_ID -> (input_prob, mean_actual) for all rows."""
    d = {}
    with open(P(tsv)) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            d[r['COG_ID']] = (float(r['input_prob']), float(r['mean_actual']))
    return d

def rederive_count_input(famcsv, node_index):
    """Re-derive per-COG input from softlanding families.csv at the root node.

    Sub-families are aggregated by max of P_present and capped at 1.0, matching
    the original extract.  Returns COG -> probability."""
    out = {}
    with open(P(famcsv)) as f:
        header = f.readline().rstrip('\n').split(',')
        i_ni, i_pp, i_fam = header.index('node_index'), header.index('P_present'), header.index('family_name')
        for line in f:
            c = line.rstrip('\n').split(',')
            if c[i_ni] != node_index:
                continue
            cog = c[i_fam].split('_')[0]
            v = min(1.0, float(c[i_pp]))
            if v > out.get(cog, -1.0):
                out[cog] = v
    return out

def rederive_recon_input(aggtsv):
    """Per-COG aggregated reconciliation input from the provenance TSV."""
    out = {}
    with open(P(aggtsv)) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out[r['COG_ID']] = float(r['input_prob'])
    return out

# ---- module / category machinery
def load_modules():
    m = torch.load(P('data/module_matrix_kegg.pt'), map_location='cpu', weights_only=False)
    cogs = list(m['cog_names'])
    Mk = m['M_kegg'].numpy().astype(float)
    ksz = m['kegg_sizes'].numpy().astype(float)
    names = list(m.get('kegg_descs', m.get('kegg_names')))
    idx = {c: i for i, c in enumerate(cogs)}
    return cogs, Mk, ksz, names, idx

def load_cats():
    cat = {}
    with open(P('data/cog-20.def.tab'), encoding='latin-1') as f:
        for ln in f:
            fld = ln.rstrip('\n').split('\t')
            if len(fld) >= 2:
                cat[fld[0]] = fld[1]
    return cat

CATNAME = {'J':'Translation','K':'Transcription','L':'Replication','C':'Energy',
 'E':'AminoAcid','G':'Carb','H':'Coenzyme','F':'Nucleotide','I':'Lipid','M':'CellWall',
 'N':'Motility','O':'PTM','P':'InorgIon','T':'Signal','U':'Secretion','V':'Defense',
 'R':'GenFunc','S':'Unknown','X':'Mobilome','D':'CellCycle','Q':'2ndary','W':'Extracell'}

def present_vec(present_set, cogs, idx):
    v = np.zeros(len(cogs))
    for c in present_set:
        if c in idx: v[idx[c]] = 1.0
    return v

def module_frac(present_set, cogs, Mk, ksz, idx):
    return (present_vec(present_set, cogs, idx) @ Mk) / np.maximum(ksz, 1)

def cat_breakdown(cog_set, cat, topn=7):
    from collections import Counter
    cc = Counter()
    for c in cog_set:
        for ch in cat.get(c, ''):
            if ch.isalpha(): cc[ch] += 1
    # Count descending, then label ascending: most_common() breaks ties by Counter
    # insertion order, which follows set-iteration order (PYTHONHASHSEED), so
    # equal-count categories would otherwise swap run to run and change which one
    # survives the topn cut.
    items = [(CATNAME.get(k, k), n) for k, n in cc.items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[:topn]

# ---- main
def main():
    print("="*78)
    print("LBCA reconstruction comparison -- self-verifying")
    print(f"policy: output present mean_actual>={PRESENT}; count input input_prob>{COUNT_INPUT_THR};")
    print(f"        reconciliation removal qualified across p={CONF_GRID}")
    print("="*78)

    data = {k: load_recon(v['tsv']) for k, v in RECONS.items()}
    cogs, Mk, ksz, knames, idx = load_modules()
    cat = load_cats()
    summary = {'policy': dict(output_present=PRESENT, count_input='>%.2f' % COUNT_INPUT_THR,
                              reconciliation='qualified by confidence p'), 'recons': {}, 'pairs': {}}

    print("\n[0] PROVENANCE -- re-derive input from source vs input_prob recorded in TSV")
    for name, cfg in RECONS.items():
        rec = data[name]
        src, node = cfg['prov']
        if not os.path.exists(P(src)):
            print(f"   [SKIP] {name}: provenance source not present ({src}); "
                  f"input_prob trusted as recorded in TSV")
            continue
        if cfg['kind'] == 'count':
            der = rederive_count_input(src, node)
        else:
            der = rederive_recon_input(src)
        # Every COG in the reconstruction; a COG absent from the source counts as 0.
        mism = max_abs = 0
        for cog, (ip, _) in rec.items():
            d = der.get(cog, 0.0)
            if abs(d - ip) > 1.5e-2:            # input_prob stored to ~2 dp
                mism += 1; max_abs = max(max_abs, abs(d - ip))
        check(mism == 0, f"{name}: input_prob matches {os.path.basename(src)} "
                         f"(mismatches={mism}, max|d|={max_abs:.4f})")

    print("\n[1] INPUT -> OUTPUT")
    for name, cfg in RECONS.items():
        rec = data[name]
        out_set = {c for c, (ip, mo) in rec.items() if mo >= PRESENT}
        rinfo = dict(output=len(out_set), kind=cfg['kind'])
        if cfg['kind'] == 'count':
            in_set = {c for c, (ip, mo) in rec.items() if ip > COUNT_INPUT_THR}
            added = out_set - in_set            # apparent FN: model added (input absent)
            removed = in_set - out_set          # apparent FP: model removed (input present)
            kept = in_set & out_set
            check(len(kept) + len(removed) == len(in_set), f"{name}: kept+removed == |input|")
            check(len(kept) + len(added) == len(out_set), f"{name}: kept+added == |output|")
            print(f"   {name}: input(>{COUNT_INPUT_THR})={len(in_set)}  output(>={PRESENT})={len(out_set)}  "
                  f"net {len(out_set)-len(in_set):+d}")
            print(f"      added (apparent FN) = {len(added)}")
            print(f"      removed(apparent FP)= {len(removed)}  ({len(removed)/len(in_set):.0%} of input)")
            print(f"      kept                = {len(kept)}")
            rinfo.update(input=len(in_set), added=len(added), removed=len(removed), kept=len(kept))
        else:
            print(f"   {name}: DIFFUSE input -> removal qualified by confidence (output={len(out_set)})")
            print(f"      {'p>=':>6} {'input_n':>8} {'kept':>6} {'removed':>8} {'removed/input':>14}")
            rows = []
            for p in CONF_GRID:
                in_p = {c for c, (ip, mo) in rec.items() if ip >= p}
                kept = in_p & out_set
                removed = in_p - out_set
                check(len(kept) + len(removed) == len(in_p), f"{name}: kept+removed==|input>={p}|")
                frac = (len(removed) / len(in_p)) if in_p else 0.0
                print(f"      {p:>6.2f} {len(in_p):>8d} {len(kept):>6d} {len(removed):>8d} {frac:>13.0%}")
                rows.append(dict(p=p, input_n=len(in_p), kept=len(kept), removed=len(removed), removed_frac=frac))
            # Additions measured against any upstream support at all (input_prob > 0).
            in_any = {c for c, (ip, mo) in rec.items() if ip > 0}
            added_vs_any = out_set - in_any     # output present with no upstream support
            print(f"      => output {len(out_set)}; of these {len(added_vs_any)} had ZERO upstream support "
                  f"(pure additions); removal is <={rows[0]['removed']} even counting trace weight")
            rinfo.update(removal_by_confidence=rows, added_zero_support=len(added_vs_any))
        summary['recons'][name] = rinfo

    # ---- pairwise comparisons: each soft-trim against the sparse-rescue
    sp = data['sparse-rescue']
    sp_out = {c for c, (ip, mo) in sp.items() if mo >= PRESENT}
    sp_frac = module_frac(sp_out, cogs, Mk, ksz, idx)
    for soft in ['soft-trim-min1', 'soft-trim-min4']:
        rec = data[soft]
        a_out = {c for c, (ip, mo) in rec.items() if mo >= PRESENT}            # soft-trim output
        a_in = {c for c, (ip, mo) in rec.items() if ip > COUNT_INPUT_THR}      # its dense input
        shared = a_out & sp_out; a_only = a_out - sp_out; b_only = sp_out - a_out
        union = a_out | sp_out
        jac = len(shared) / len(union)
        check(len(shared)+len(a_only) == len(a_out), f"{soft} vs sparse: shared+Aonly==|A|")
        check(len(shared)+len(b_only) == len(sp_out), f"{soft} vs sparse: shared+Bonly==|B|")
        check(0.0 <= jac <= 1.0, f"{soft} vs sparse: Jaccard in [0,1]")

        a_frac = module_frac(a_out, cogs, Mk, ksz, idx)
        a_full = int((a_frac >= 0.999).sum()); a_half = int((a_frac >= 0.5).sum())
        sp_full = int((sp_frac >= 0.999).sum()); sp_half = int((sp_frac >= 0.5).sum())
        # modules one reconstruction completes and the other leaves gappy (< 0.85)
        a_closes = [knames[i] for i in range(len(ksz)) if a_frac[i] >= 0.999 and sp_frac[i] < 0.85]
        b_closes = [knames[i] for i in range(len(ksz)) if sp_frac[i] >= 0.999 and a_frac[i] < 0.85]
        # capability erosion in the prune: complete in the input, gappy in the output
        in_frac = module_frac(a_in, cogs, Mk, ksz, idx)
        eroded = sorted([(knames[i], in_frac[i], a_frac[i]) for i in range(len(ksz))
                         if ksz[i] >= 4 and in_frac[i] - a_frac[i] >= 0.20], key=lambda x: -(x[1]-x[2]))
        in_full = int((in_frac >= 0.999).sum())

        print(f"\n[2] {soft}  vs  sparse-rescue   (outputs at >={PRESENT})")
        print(f"   sizes: {soft}={len(a_out)}  sparse={len(sp_out)}  shared={len(shared)}  "
              f"Jaccard={jac:.3f}  {soft}-only={len(a_only)}  sparse-only={len(b_only)}")
        print(f"   {soft}-only function: " + ", ".join(f"{n} {c}" for n, c in cat_breakdown(a_only, cat)))
        print(f"   sparse-only  function: " + ", ".join(f"{n} {c}" for n, c in cat_breakdown(b_only, cat)))
        print(f"[3] module completeness (full=100% / >=50%)")
        print(f"   {soft}: {a_full} / {a_half}     sparse: {sp_full} / {sp_half}")
        print(f"   {soft} closes (sparse gappy): {len(a_closes)}   sparse closes ({soft} gappy): {len(b_closes)}")
        print(f"[4] prune phenotype: input full modules {in_full} -> output full {a_full} "
              f"(net genes {len(a_out)-len(a_in):+d})")
        print(f"   biggest capability EROSION input->output (n>=4, drop>=20pp):")
        for n, vi, vo in eroded[:10]:
            print(f"      - {n[:50]:52s} {vi*100:3.0f}% -> {vo*100:3.0f}%")
        summary['pairs'][f'{soft}_vs_sparse'] = dict(
            A=len(a_out), B=len(sp_out), shared=len(shared), jaccard=round(jac, 4),
            A_only=len(a_only), B_only=len(b_only),
            A_only_function=cat_breakdown(a_only, cat), B_only_function=cat_breakdown(b_only, cat),
            A_full=a_full, A_half=a_half, B_full=sp_full, B_half=sp_half,
            A_closes_vs_B=len(a_closes), B_closes_vs_A=len(b_closes),
            input_full=in_full, eroded=[[n, round(vi,3), round(vo,3)] for n, vi, vo in eroded[:15]])

    print("\n[5] KEY ENERGY / CENTRAL-CARBON MODULES (percent complete)")
    KEY = ['NADH:quinone oxidoreductase, prokaryotes','F-type ATPase, prokaryotes',
           'Cytochrome c oxidase, prokaryotes','Cytochrome bc1','Cytochrome bd ubiquinol',
           'Cytochrome o ubiquinol','Menaquinone biosynthesis, futalosine',
           'Glycolysis (Embden','Citrate cycle (TCA','Entner-Doudoroff pathway, glucose-6P',
           'Phosphate acetyltransferase-acetate kinase','Cobalamin biosynthesis, cobyrinate',
           'Pentose phosphate pathway, oxidative']
    m1 = module_frac({c for c,(ip,mo) in data['soft-trim-min1'].items() if mo>=PRESENT}, cogs,Mk,ksz,idx)
    m4 = module_frac({c for c,(ip,mo) in data['soft-trim-min4'].items() if mo>=PRESENT}, cogs,Mk,ksz,idx)
    print(f"   {'module':46s} {'min1':>5} {'min4':>5} {'sparse':>6}")
    seen=set()
    for key in KEY:
        for i,n in enumerate(knames):
            if key in n and i not in seen:
                seen.add(i)
                print(f"   {n[:44]:46s} {m1[i]*100:4.0f}% {m4[i]*100:4.0f}% {sp_frac[i]*100:5.0f}%")
                break

    # ---- machine-readable outputs
    with open(P('data/lbca_comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(P('data/lbca_reconciliation_removal_by_confidence.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['p_ge','input_n','kept','removed','removed_frac'])
        for row in summary['recons']['sparse-rescue']['removal_by_confidence']:
            w.writerow([row['p'], row['input_n'], row['kept'], row['removed'], f"{row['removed_frac']:.4f}"])
    # per-COG input/output lists, one shared format for all three reconstructions
    cogname = {}
    with open(P('data/cog-20.def.tab'), encoding='latin-1') as f:
        for ln in f:
            fld = ln.rstrip('\n').split('\t')
            if len(fld) >= 3:
                cogname[fld[0]] = fld[2]
    order = [('sparse', 'sparse-rescue'), ('GLD_min1', 'soft-trim-min1'),
             ('GLD_min4', 'soft-trim-min4')]
    all_cogs = sorted(set().union(*[set(data[k]) for _, k in order]))
    with open(P('data/lbca_cog_lists.tsv'), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        hdr = ['COG_ID', 'category', 'name']
        for lab, _ in order:
            hdr += [f'{lab}_input', f'{lab}_out_post', f'{lab}_present']
        w.writerow(hdr)
        for c in all_cogs:
            row = [c, cat.get(c, ''), cogname.get(c, '')]
            for _, k in order:
                ip, mo = data[k].get(c, (0.0, 0.0))
                row += [f'{ip:.4f}', f'{mo:.4f}', 1 if mo >= PRESENT else 0]
            w.writerow(row)
    print("wrote data/lbca_comparison_summary.json + data/lbca_reconciliation_removal_by_confidence.csv "
          "+ data/lbca_cog_lists.tsv")

    print("\n" + "="*78)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} CHECK(S):")
        for m in FAILURES: print("  - " + m)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("="*78)

if __name__ == '__main__':
    main()
