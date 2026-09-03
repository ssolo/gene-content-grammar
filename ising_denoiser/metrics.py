"""
Evaluation metrics and FN/FP-spectrum computation.

The headline metric is MCC, over false-negative rates FN = 0.75-0.90 (the
reconciliation operating range at LBCA and LUCA).  F1 is reported alongside
but is misleading here: only ~30 % of the 4 789 COG families are present in
a typical prokaryotic genome, fewer still under high FN, and F1 ignores the
true negatives.
"""
import torch
from torch.amp import autocast


FN_GRID = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]

# FP grid for the 2D (FN, FP) scans.  Includes the fp = 0.01 held fixed by
# the 1D scan; the upper cells run well past the fp_max = 0.1 of the FP
# fine-tune curriculum.
FP_GRID = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]


def metrics(pred, tgt):
    """F1/precision/recall/MCC for a continuous prediction thresholded at 0.

    pred, tgt: (B, N); tgt in +/-1 encoding.
    """
    p = pred.sign()
    tp = ((p == 1) & (tgt == 1)).sum().float()
    fp_ = ((p == 1) & (tgt == -1)).sum().float()
    fn_ = ((p == -1) & (tgt == 1)).sum().float()
    tn = ((p == -1) & (tgt == -1)).sum().float()
    # Empty positive marginals: return 0, matching sklearn's zero_division=0.
    pr = tp / (tp + fp_) if (tp + fp_) > 0 else torch.tensor(0.0, device=tp.device)
    rc = tp / (tp + fn_) if (tp + fn_) > 0 else torch.tensor(0.0, device=tp.device)
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else torch.tensor(0.0, device=tp.device)
    # MCC is undefined when any of the four marginals vanishes; NaN rather
    # than an epsilon-guarded ~0 that would slip unnoticed into an average.
    denom_sq = (tp + fp_) * (tp + fn_) * (tn + fp_) * (tn + fn_)
    if denom_sq > 0:
        mcc = ((tp * tn - fp_ * fn_) / torch.sqrt(denom_sq)).item()
    else:
        mcc = float('nan')
    return dict(F1=f1.item(), prec=pr.item(), rec=rc.item(), MCC=mcc)


def mstr(m):
    """Format metrics dict as compact string (MCC-first)."""
    return f"MCC={m['MCC']:.4f} F1={m['F1']:.4f} P={m['prec']:.3f} R={m['rec']:.3f}"


# ---- FN-spectrum evaluation

def eval_spectra(model, vt, dev, amp, fn_grid=None, fp=0.01, n=2048,
                 M_mod=None, M_sizes=None, eval_batch=256):
    """Corrupt the validation set at each FN, denoise, and score.

    The seed depends only on FN, so the same genes are flipped for every
    checkpoint and world size and spectra from different runs are comparable.

    vt is (N_val, N) in +/-1 encoding on CPU; the first n rows are used.
    M_mod (N, n_mod) / M_sizes (n_mod,) are the module matrix and per-module
    COG counts, or None for unconditioned models.  eval_batch bounds memory:
    the transformer variants build an N-token attention map per genome.

    Returns one dict per FN: [{fn, F1, prec, rec, MCC}, ...]
    """
    if fn_grid is None:
        fn_grid = FN_GRID
    model.eval()
    vc = vt[:min(n, vt.size(0))].to(dev)
    rows = []
    for fn in fn_grid:
        torch.manual_seed(42 + int(fn * 1000))
        ny = vc.clone()
        r = torch.rand_like(ny)
        ny[(vc == 1) & (r < fn)] = -1
        r2 = torch.rand_like(ny)
        ny[(vc == -1) & (r2 < fp)] = 1
        with torch.no_grad():
            xd_parts = []
            for i in range(0, ny.size(0), eval_batch):
                batch = ny[i:i+eval_batch]
                with autocast('cuda', enabled=amp):
                    if M_mod is not None:
                        out, _ = model(batch, M_mod, M_sizes)
                    else:
                        out, _ = model(batch)
                xd_parts.append(out)
            xd = torch.cat(xd_parts, dim=0)
            m = metrics(xd, vc)
        rows.append(dict(fn=fn, **m))
    return rows


def eval_spectra_2d(model, vt, dev, amp, fn_grid=None, fp_grid=None, n=2048,
                    M_mod=None, M_sizes=None, eval_batch=256):
    """As eval_spectra, over a 2D grid of (FN, FP) rates.

    Returns one dict per cell with fp varying fastest:
    [{fn, fp, F1, prec, rec, MCC, fp_removed}, ...], where fp_removed is
    the fraction of the injected false positives (genes flipped -1 -> +1
    in the input) that come back out negative.  NaN at fp == 0.
    """
    if fn_grid is None:
        fn_grid = FN_GRID
    if fp_grid is None:
        fp_grid = FP_GRID
    model.eval()
    vc = vt[:min(n, vt.size(0))].to(dev)
    rows = []
    for fn in fn_grid:
        for fp in fp_grid:
            # Prime multiplier keeps the fn and fp contributions from colliding.
            torch.manual_seed(42 + int(fn * 1000) + 7919 * int(fp * 1000))
            ny = vc.clone()
            r = torch.rand_like(ny)
            ny[(vc == 1) & (r < fn)] = -1
            r2 = torch.rand_like(ny)
            ny[(vc == -1) & (r2 < fp)] = 1
            with torch.no_grad():
                xd_parts = []
                for i in range(0, ny.size(0), eval_batch):
                    batch = ny[i:i+eval_batch]
                    with autocast('cuda', enabled=amp):
                        if M_mod is not None:
                            out, _ = model(batch, M_mod, M_sizes)
                        else:
                            out, _ = model(batch)
                    xd_parts.append(out)
                xd = torch.cat(xd_parts, dim=0)
                m = metrics(xd, vc)
                # Reuse r2 so inj is the injected set itself; "removed" uses
                # the same sign binarisation as metrics().
                inj = (vc == -1) & (r2 < fp)
                n_inj = int(inj.sum())
                if n_inj > 0:
                    fp_removed = float((xd[inj].sign() == -1).float().mean())
                else:
                    fp_removed = float('nan')
            rows.append(dict(fn=fn, fp=fp, fp_removed=fp_removed, **m))
    return rows


@torch.no_grad()
def eval_spectra_multistep(model, vt, dev, amp, M_mod, M_sizes,
                           n_steps=3, fn_grid=None, fp=0.01, n=2048):
    """Multi-step refinement evaluation (module-conditioned models only).

    Applies the denoiser n_steps times; the module-completeness profile
    f_obs is recomputed inside the model from the latest reconstruction at
    each step.  Corruption and seeding follow eval_spectra, at fixed fp.
    """
    if fn_grid is None:
        fn_grid = FN_GRID
    model.eval()
    vc = vt[:min(n, vt.size(0))].to(dev)
    rows = []
    for fn in fn_grid:
        torch.manual_seed(42 + int(fn * 1000))
        ny = vc.clone()
        r = torch.rand_like(ny)
        ny[(vc == 1) & (r < fn)] = -1
        r2 = torch.rand_like(ny)
        ny[(vc == -1) & (r2 < fp)] = 1
        x = ny
        for step in range(n_steps):
            with autocast('cuda', enabled=amp):
                x, _ = model(x, M_mod, M_sizes)
        m = metrics(x, vc)
        rows.append(dict(fn=fn, steps=n_steps, **m))
    return rows


# ---- Comparison printing

def print_three_way(new_rows, old_rows, v7_rows, log_fn, label=""):
    """Tabulate one spectra list against two references.

    Rows are matched positionally, so all three lists must come from the
    same FN grid.  The gain column is new - v7.
    """
    log_fn(f"\n    {label}")
    log_fn(f"      {'FN':>6} {'new MCC':>8} {'OLD MCC':>7} {'v7 MCC':>7} {'new-v7':>7} "
           f"{'new F1':>7} {'v7 F1':>7} {'new P/R':>11}")
    log_fn(f"    {'~'*82}")
    for rn, ro, r7 in zip(new_rows, old_rows, v7_rows):
        g7 = rn['MCC'] - r7['MCC']
        log_fn(f"      {rn['fn']:>6.2f} {rn['MCC']:>8.4f} {ro['MCC']:>7.4f} {r7['MCC']:>7.4f} "
               f"{g7:>+7.4f} {rn['F1']:>7.4f} {r7['F1']:>7.4f} "
               f"{rn['prec']:.3f}/{rn['rec']:.3f}")


def print_four_way(v10_rows, v10ms_rows, old_rows, v7_rows, log_fn, label=""):
    """Tabulate 1-step and multi-step spectra against two references.

    Rows are matched positionally; both gain columns are against v7_rows.
    """
    log_fn(f"\n    {label}")
    log_fn(f"      {'FN':>6} {'v10 MCC':>8} {'v10ms':>7} {'v7 MCC':>7} {'v10-v7':>7} "
           f"{'ms-v7':>7} {'OLD':>7} {'v10 P/R':>11}")
    log_fn(f"    {'~'*85}")
    for r10, rms, ro, r7 in zip(v10_rows, v10ms_rows, old_rows, v7_rows):
        g7 = r10['MCC'] - r7['MCC']
        gms = rms['MCC'] - r7['MCC']
        log_fn(f"      {r10['fn']:>6.2f} {r10['MCC']:>8.4f} {rms['MCC']:>7.4f} "
               f"{r7['MCC']:>7.4f} {g7:>+7.4f} {gms:>+7.4f} {ro['MCC']:>7.4f} "
               f"{r10['prec']:.3f}/{r10['rec']:.3f}")


def print_comparison(new_rows, ref_dict, log_fn, label=""):
    """Tabulate one spectra list against an arbitrary set of references.

    ref_dict maps name -> spectra list, one column each; every list must be
    on the same FN grid as new_rows (rows are matched positionally).
    """
    names = list(ref_dict.keys())
    log_fn(f"\n    {label}")
    hdr = f"      {'FN':>6} {'This':>8}"
    for n in names: hdr += f" {n[:8]:>8}"
    for n in names: hdr += f" {'d'+n[:5]:>7}"
    log_fn(hdr)
    log_fn(f"    {'~'*(14 + 16*len(names))}")
    for i, rn in enumerate(new_rows):
        line = f"      {rn['fn']:>6.2f} {rn['MCC']:>8.4f}"
        for n in names: line += f" {ref_dict[n][i]['MCC']:>8.4f}"
        for n in names: line += f" {rn['MCC']-ref_dict[n][i]['MCC']:>+7.4f}"
        log_fn(line)
