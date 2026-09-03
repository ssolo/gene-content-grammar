#!/usr/bin/env python3.11
"""Estimate the apparent false-negative (fN) and false-positive (fP) rate of the
reconciliation input at an ancestral node (LBCA, LACA), calibrated against the
extant-genome validation baselines.

The reconciliation posterior handed to the denoiser has an unknown noise level
and the true ancestral genome is unobservable; the calibration comes from the
same denoiser applied to held-out real genomes corrupted at controlled noise
(reconstruct_extant.py).

At a node, from the denoiser's input/output:
    I        = input families present (input_prob > 0)
    D        = denoised families present (posterior mean > 0.5)
    kept     = present in both input and output
    silenced = present in input, absent in output
    rescued  = absent in input, present in output
    rho      = D / I  (expansion),  sigma = silenced / I

On the matching held-out genomes (same domain and model: bac-FT for the
bacterial LBCA, mix-FT for the archaeal LACA), corrupted at known fN with
fP = 0.01, the same statistics give the calibration curves:
    rho_val(fN) = denoised_present / noisy_input_present  (monotone in fN, as
                  required to invert it by interpolation)
    recall(fN)  = true_recovered / true_total
    s(fN)       = false-silence rate: true genes present in the input that the
                  denoiser drops, over true genes present in the input

Apparent fN is rho_val inverted at the node's observed rho; rho_val folds in the
denoiser's own imperfect recall, so the estimate is baseline-corrected.  The
implied true ancestral size is G = D / recall(fN*).

Apparent fP attributes silencing in excess of the baseline false-silence rate
s(fN*) to genuine false positives in the input:
    spurious_in_input = max(0, silenced - s(fN*) * kept)
    apparent fP       = spurious_in_input / (N - G)   (per truly-absent family)
and is also reported as spurious_in_input / I.

Limitations (printed with the result): rho_val is interpolated piecewise-linearly
and the node is flagged when it falls outside the sampled range; a node fP above
the 0.01 used in validation deflates rho and under-estimates fN.
"""
import argparse
import numpy as np
import pandas as pd

N_FAM = 4789


def node_counts(pred_tsv, in_thresh=0.0, out_thresh=0.5):
    d = pd.read_csv(pred_tsv, sep="\t")
    I = d["input_prob"] > in_thresh
    D = d["mean_actual"] > out_thresh
    return {
        "I": int(I.sum()),
        "D": int(D.sum()),
        "kept": int((I & D).sum()),
        "silenced": int((I & ~D).sum()),
        "rescued": int((~I & D).sum()),
    }


def val_curves(recover_tsvs):
    """Pool recover TSVs from one domain and model into {fn: {I, D, rho, recall, s}}."""
    df = pd.concat([pd.read_csv(f, sep="\t") for f in recover_tsvs], ignore_index=True)
    out = {}
    for fn, g in df.groupby("fn"):
        dp = g["denoised_prob"] > 0.5
        tr = g["truth"] == 1
        t_in = tr & (g["input_present"] == 1)
        I = int(g["input_present"].sum())
        D = int(dp.sum())
        out[float(fn)] = {
            "I": I,
            "D": D,
            "rho": D / I,
            "recall": float((tr & dp).sum()) / float(tr.sum()),
            "s": float((t_in & ~dp).sum()) / float(t_in.sum()),
        }
    return out


def estimate(tag, pred_tsv, recover_tsvs):
    n = node_counts(pred_tsv)
    c = val_curves(recover_tsvs)
    fns = sorted(c)
    rhos = [c[f]["rho"] for f in fns]
    rho_node = n["D"] / n["I"]
    sigma_node = n["silenced"] / n["I"]
    fN = float(np.interp(rho_node, rhos, fns))
    extrap = rho_node < rhos[0] or rho_node > rhos[-1]
    recall = float(np.interp(fN, fns, [c[f]["recall"] for f in fns]))
    s = float(np.interp(fN, fns, [c[f]["s"] for f in fns]))
    G = n["D"] / recall
    spurious = max(0.0, n["silenced"] - s * n["kept"])
    fP = spurious / (N_FAM - G)

    print("=== %s ===" % tag)
    print("  node: I=%d D=%d kept=%d silenced=%d rescued=%d  rho=%.3f sigma=%.3f"
          % (n["I"], n["D"], n["kept"], n["silenced"], n["rescued"], rho_node, sigma_node))
    print("  validation noise-meter (fN: rho, recall, s):")
    for f in fns:
        print("    fN=%.2f  rho=%.2f  recall=%.2f  s=%.3f" % (f, c[f]["rho"], c[f]["recall"], c[f]["s"]))
    flag = "  [EXTRAPOLATED beyond curve]" if extrap else ""
    print("  -> apparent fN = %.2f%s   (recall* = %.2f, implied true size G = %.0f)"
          % (fN, flag, recall, G))
    print("  -> apparent fP = %.3f   (%.0f spurious input families among %.0f truly absent;"
          % (fP, spurious, N_FAM - G))
    print("                          %.1f%% of the input is spurious after baseline correction)"
          % (100.0 * spurious / n["I"]))
    return {"tag": tag, "fN": fN, "fP": fP, "G": G, "recall": recall, **n}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lbca-pred", default="node2012_T20bachard_pred.tsv")
    ap.add_argument("--lbca-recover", nargs="+",
                    default=["recover_ecoli_bachard.tsv", "recover_medbac_bachard.tsv"])
    ap.add_argument("--laca-pred", default="LACA_combined_T20mix_pred.tsv")
    ap.add_argument("--laca-recover", nargs="+",
                    default=["recover_archaeon_mixft.tsv", "recover_medarc_mixft.tsv"])
    a = ap.parse_args()
    estimate("LBCA / bac-FT (bacterial)", a.lbca_pred, a.lbca_recover)
    estimate("LACA / mix-FT (archaeal)", a.laca_pred, a.laca_recover)


if __name__ == "__main__":
    main()
