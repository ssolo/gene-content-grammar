# -*- coding: utf-8 -*-
"""
Evaluation figures and a standalone LaTeX report.

From an ordered dict of model spectra (name -> per-FN metric dicts), writes
precision_recall_vs_fn.pdf, f1_vs_fn.pdf, mcc_vs_fn.pdf, results_report.tex
and spectra.json into one output directory. Figures carry the null
expectation (the noisy input returned unchanged) as a dashed grey floor and
shade the LBCA/LUCA range FN = 0.70-0.95.
"""
import json, math
from pathlib import Path
from collections import OrderedDict
import numpy as np

# torch is used only by eval_null and matplotlib only by the plots, so both
# are imported lazily.
_plt = None
_MultipleLocator = None

def _import_mpl():
    global _plt, _MultipleLocator
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        _plt = plt
        _MultipleLocator = MultipleLocator
    return _plt, _MultipleLocator


# ---- Null expectation

def eval_null(val_t, dev, fn_grid, fp=0.01, n=2048):
    """Score the corrupted input against the clean genomes, with no model.

    Returns rows in the eval_spectra format.
    """
    import torch
    from .metrics import metrics
    vc = val_t[:min(n, val_t.size(0))].to(dev)
    rows = []
    for fn in fn_grid:
        # Seeded as in eval_spectra, so the null flips the genes the models saw.
        torch.manual_seed(42 + int(fn * 1000))
        ny = vc.clone()
        r = torch.rand_like(ny); ny[(vc == 1) & (r < fn)] = -1
        r2 = torch.rand_like(ny); ny[(vc == -1) & (r2 < fp)] = 1
        m = metrics(ny, vc)  # noisy input in the prediction slot
        rows.append(dict(fn=fn, **m))
    return rows


# ---- Plot styling (Okabe-Ito colourblind-safe)

# Ordered by model complexity, simplest first.
MODEL_STYLES = OrderedDict([
    ('Noisy input',           dict(color='#999999', marker='x',  ls='--', lw=1.2, zorder=1)),
    ('Baseline denoiser',     dict(color='#E69F00', marker='s',  ls='-',  lw=1.5, zorder=2)),
    ('Gated denoiser',        dict(color='#0072B2', marker='o',  ls='-',  lw=1.5, zorder=3)),
    ('Extended iterations',   dict(color='#009E73', marker='^',  ls='-',  lw=1.5, zorder=4)),
    ('Module-conditioned',    dict(color='#D55E00', marker='D',  ls='-',  lw=1.8, zorder=5)),
    ('Conditioned (3-step)',  dict(color='#CC79A7', marker='v',  ls=':',  lw=1.8, zorder=5)),
])

def _get_style(name):
    for key, style in MODEL_STYLES.items():
        if key.lower() in name.lower():
            return style
    return dict(color='black', marker='o', ls='-', lw=1.3, zorder=2)


def _setup_style():
    plt, _ = _import_mpl()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.8,
        'lines.markersize': 5,
    })


def _shade_lbca(ax):
    yl = ax.get_ylim()
    ax.axvspan(0.70, 0.95, alpha=0.07, color='#D55E00', zorder=0)
    ax.text(0.825, yl[0] + 0.03 * (yl[1] - yl[0]),
            'LBCA / LUCA', fontsize=7, color='#8B0000',
            ha='center', style='italic', zorder=0)


def _plot_lines(ax, spectra_dict, key):
    for name, rows in spectra_dict.items():
        fns = [r['fn'] for r in rows]
        vals = [r[key] for r in rows]
        sty = _get_style(name)
        ax.plot(fns, vals, label=name, **sty)


def plot_precision_recall(spectra_dict, outpath):
    """Two-panel figure: Precision (a) and Recall (b) vs FN."""
    plt, Loc = _import_mpl()
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax1.set_xlabel('False-negative rate')
    ax1.set_ylabel('Precision')
    ax1.set_title('(a) Precision')
    _plot_lines(ax1, spectra_dict, 'prec')
    ax1.set_xlim(-0.02, 0.95); ax1.set_ylim(0.55, 1.02)
    ax1.xaxis.set_major_locator(Loc(0.25))
    _shade_lbca(ax1)
    ax1.legend(loc='lower left', framealpha=0.9)

    ax2.set_xlabel('False-negative rate')
    ax2.set_ylabel('Recall')
    ax2.set_title('(b) Recall')
    _plot_lines(ax2, spectra_dict, 'rec')
    ax2.set_xlim(-0.02, 0.95); ax2.set_ylim(0.05, 1.02)
    ax2.xaxis.set_major_locator(Loc(0.25))
    _shade_lbca(ax2)
    ax2.legend(loc='lower left', framealpha=0.9)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(str(outpath))
    plt.close(fig)


def plot_f1(spectra_dict, outpath):
    plt, Loc = _import_mpl()
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.set_xlabel('False-negative rate')
    ax.set_ylabel('F1 score')
    ax.set_title('F1 score vs.[ht] noise level')
    _plot_lines(ax, spectra_dict, 'F1')
    ax.set_xlim(-0.02, 0.95); ax.set_ylim(0.05, 1.02)
    ax.xaxis.set_major_locator(Loc(0.25))
    _shade_lbca(ax)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(outpath))
    plt.close(fig)


def plot_mcc(spectra_dict, outpath):
    plt, Loc = _import_mpl()
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.set_xlabel('False-negative rate')
    ax.set_ylabel('Matthews Correlation Coefficient (MCC)')
    ax.set_title('MCC vs.[ht] noise level')
    _plot_lines(ax, spectra_dict, 'MCC')
    ax.set_xlim(-0.02, 0.95); ax.set_ylim(-0.05, 1.02)
    ax.xaxis.set_major_locator(Loc(0.25))
    _shade_lbca(ax)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(outpath))
    plt.close(fig)


# ---- LaTeX report

def _fmt(val, bold_val=None, fmt=".4f"):
    """Format a metric value, bolding it if it ties bold_val to 1e-6."""
    s = f"{val:{fmt}}"
    if bold_val is not None and abs(val - bold_val) < 1e-6:
        return r"\textbf{" + s + "}"
    return s


def _table_mcc(spectra_dict, fn_grid):
    """MCC for every model at every FN, best model value bolded per row."""
    names = list(spectra_dict.keys())
    cols = "r" + "r" * len(names)
    lines = []
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = "FN & " + " & ".join(n for n in names) + r" [ht]"
    lines.append(header)
    lines.append(r"\midrule")
    for i, fn in enumerate(fn_grid):
        vals = [spectra_dict[n][i]['MCC'] for n in names]
        # 'noisy' marks the null baseline; it never wins the bolding.
        model_vals = [v for n, v in zip(names, vals) if 'noisy' not in n.lower()]
        best = max(model_vals) if model_vals else None
        cells = [f"{fn:.2f}"]
        for n, v in zip(names, vals):
            if 'noisy' not in n.lower() and best is not None:
                cells.append(_fmt(v, best))
            else:
                cells.append(_fmt(v))
        lines.append(" & ".join(cells) + r" [ht]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _table_full(spectra_dict, fn_grid, fn_val):
    """All metrics at a single FN value; falls back to fn_grid[0] if absent."""
    idx = fn_grid.index(fn_val) if fn_val in fn_grid else 0
    names = list(spectra_dict.keys())
    lines = []
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & MCC & F1 & Precision & Recall [ht]")
    lines.append(r"\midrule")
    mccs = [spectra_dict[n][idx]['MCC'] for n in names if 'noisy' not in n.lower()]
    best_mcc = max(mccs) if mccs else None
    for n in names:
        r = spectra_dict[n][idx]
        mcc_s = _fmt(r['MCC'], best_mcc if 'noisy' not in n.lower() else None)
        lines.append(f"  {n} & {mcc_s} & {r['F1']:.4f} & {r['prec']:.3f} & {r['rec']:.3f} " + r"[ht]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_latex(spectra_dict, fn_grid, outpath, fig_dir="."):
    """Write the LaTeX document to outpath.

    Sections follow insertion order: problem setting, null expectation, one
    section per model with the MCC table for the models so far, then the
    summary figures and tables. fig_dir prefixes the includegraphics paths.
    """
    names = list(spectra_dict.keys())

    # Spectra up to and including last_name, in insertion order.
    def subset_to(last_name):
        d = OrderedDict()
        for n in names:
            d[n] = spectra_dict[n]
            if n == last_name:
                break
        return d

    tex = []
    tex.append(r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}

\title{Ising Denoiser --- Evaluation Report}
\author{Generated by \texttt{generate\_report.py}}
\date{\today}

\begin{document}
\maketitle
""")

    tex.append(r"""
\section{Problem Setting}

Gene tree--species tree reconciliation infers ancestral gene content but
suffers from high false-negative rates at deep phylogenetic nodes.
At the Last Bacterial Common Ancestor (LBCA), $\text{FN} \approx 0.75$;
at the Last Universal Common Ancestor (LUCA), $\text{FN} \approx 0.85\text{--}0.90$.
False positives are rare ($\text{FP} \approx 0.01$).

We evaluate each model across a grid of false-negative rates
$\text{FN} \in \{0, 0.1, 0.25, 0.5, 0.75, 0.9\}$ on a held-out
validation set of 2{,}048 extant genomes with deterministic corruption
(fixed seed per FN value). The primary metric is MCC
(Matthews Correlation Coefficient), which properly accounts for the
heavy class imbalance (most genes absent in any given genome).

The shaded region in all figures marks the LBCA/LUCA operating range.
""")

    # --- Null expectation
    has_null = any('noisy' in n.lower() for n in names)
    if has_null:
        null_name = [n for n in names if 'noisy' in n.lower()][0]
        tex.append(r"""
\section{Null Expectation: No Denoising}

The simplest baseline is to return the noisy reconciliation output unchanged.
This establishes the floor: any model must beat this to be useful.
At $\text{FN} = 0$, the null achieves perfect metrics. As FN increases,
recall drops as $1 - \text{FN}$ (by definition), while precision remains
high because false positives are rare.

\begin{table}[ht]
\centering
\caption{Metrics of the noisy input (no denoising).}
""")
        tex.append(_table_full(subset_to(null_name), fn_grid, 0.75))
        tex.append(r"""
\end{table}

At $\text{FN} = 0.90$, the null has recall $\approx 0.10$ --- only 10\% of
present genes are detected. Any useful denoiser must dramatically improve recall
while maintaining precision.
""")

    # --- One section per model
    model_names = [n for n in names if 'noisy' not in n.lower()]

    section_text = {
        'baseline': r"""
\section{Baseline Denoiser}

The baseline architecture uses an iterative mean-field Ising model with pairwise
couplings $\bm{J}$ over $N = 4{,}789$ COG families, a hidden state $\bm{z} \in \mathbb{R}^{1000}$,
and scalar skip connections $\alpha_t$ providing direct access to the noisy input at
each of $T = 20$ iterations:
\[
  \bm{x}^{t+1} = \tanh\!\big(\bm{h} + \bm{x}^t \tilde{\bm{J}} + \bm{z}^t \bm{A}^\top + \alpha_t \cdot \bm{x}_0\big)
\]
The coupling matrix $\bm{J}$ is pre-trained via masked language modelling (pseudolikelihood)
on ${\sim}90{,}000$ extant prokaryotic genomes, then the full model is trained as a denoiser.
""",
        'gated': r"""
\section{Gated Denoiser}

The gated architecture replaces the scalar skip $\alpha_t$ with input-dependent,
per-gene gates that modulate both the skip connection (trust in $\bm{x}_0$) and the
coupling strength:
\[
  \bm{x}^{t+1} = \tanh\!\Big(\bm{h} + \big(1 + \phi_t(\bm{x}_0)\big) \cdot \text{coupling} + \sigma_t(\bm{x}_0) \cdot \bm{x}_0\Big)
\]
where $\sigma_t(\bm{x}_0) = \alpha_t + \bm{w}_t \odot \bm{x}_0 + \bm{x}_0 \bm{V}_t \bm{V}_t^\top$
(scalar + diagonal + low-rank skip gate) and
$\phi_t(\bm{x}_0) = \bm{u}_t \odot \bm{x}_0 + \bm{d}_t + \bm{x}_0 \bm{P}_t \bm{P}_t^\top$
(diagonal + low-rank field gate). Gates are trained via a progressive complexity ladder
with $T = 8$ iterations (${\sim}35.5$M parameters).
""",
        'extended': r"""
\section{Extended Iterations}

Extending from $T = 8$ to $T = 12$ gives the coupling matrix more ``hops'' to propagate
information, which is critical at high FN where few genes survive and long-range
co-occurrence patterns must be exploited. Extra timestep parameters are initialised by
interpolation from the last trained step toward neutral values.

Training uses $\text{FN} \sim \text{Beta}(2,1) \times 0.9$ (mean ${\approx}0.6$) to bias
toward the high-FN regime, and includes a hard-mining stage (S7) with
$\text{FN} \sim \text{Beta}(5,1) \times 0.9$ (mean ${\approx}0.83$) for gate-only refinement.
""",
        'conditioned': r"""
\section{Module-Conditioned Denoiser}

The conditioned model receives the observable module completeness profile
$\bm{f}_{\text{obs}} = \bm{M}^\top (\bm{x}_0+1)/2 \,/\, |\bm{M}| \in [0,1]^{419}$
as an explicit input, where $\bm{M}$ maps COGs to 419 functional modules
(COG categories, COG pathways, KEGG modules). A small MLP maps $\bm{f}_{\text{obs}}$
to per-timestep gate modulations:
\[
  \sigma_t^{\text{cond}} = (1 + \gamma_t^{\text{skip}}) \cdot \sigma_t(\bm{x}_0), \qquad
  \phi_t^{\text{cond}} = (1 + \gamma_t^{\text{field}}) \cdot \phi_t(\bm{x}_0)
\]
This tells the model \emph{which specific pathways are intact or degraded},
enabling noise-adaptive denoising without estimating the noise level.
The conditioning replaces the tempered curriculum with a single continuous training
stage over all noise levels.
""",
        'multistep': r"""
\subsection{Multi-Step Refinement}

At inference, the conditioned model supports iterative refinement:
denoise $\to$ recompute $\bm{f}_{\text{obs}}$ $\to$ denoise $\to$ \ldots\
Module fractions improve per step, leading to better conditioning and better
reconstruction. Convergence is typically reached in 2--3 steps.
""",
    }

    for mn in model_names:
        ml = mn.lower()
        if 'baseline' in ml:
            tex.append(section_text['baseline'])
        elif 'gated' in ml:
            tex.append(section_text['gated'])
        elif 'extended' in ml:
            tex.append(section_text['extended'])
        elif 'step' in ml:
            tex.append(section_text['multistep'])
        elif 'conditioned' in ml or 'module' in ml:
            tex.append(section_text['conditioned'])

        sub = subset_to(mn)
        tex.append(r"""
\begin{table}[ht]
\centering
\caption{MCC comparison (best model result per FN in bold).}
\label{tab:mcc_""" + mn.replace(' ', '_').replace('(', '').replace(')', '') + r"""}
""")
        tex.append(_table_mcc(sub, fn_grid))
        tex.append(r"\end{table}")
        tex.append("")

    tex.append(r"""
\section{Summary}

\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{""" + fig_dir + r"""/precision_recall_vs_fn.pdf}
\caption{Precision (a) and Recall (b) as a function of the false-negative rate.
The dashed grey line shows the null expectation (noisy input returned unchanged).
The shaded region marks the LBCA/LUCA operating range (FN~$= 0.75\text{--}0.90$).}
\label{fig:prec-rec}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.65\textwidth]{""" + fig_dir + r"""/f1_vs_fn.pdf}
\caption{F1 score vs.\ false-negative rate. At FN~$= 0.90$, the null expectation
gives F1~$\approx 0.17$ (almost all present genes missed); the best model recovers
F1~$> 0.7$.}
\label{fig:f1}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.65\textwidth]{""" + fig_dir + r"""/mcc_vs_fn.pdf}
\caption{Matthews Correlation Coefficient (MCC) vs.\ false-negative rate.
MCC is the primary evaluation metric: it accounts for the heavy class imbalance
(most genes absent) by incorporating all four entries of the confusion matrix.}
\label{fig:mcc}
\end{figure}
""")

    tex.append(r"""
\begin{table}[ht]
\centering
\caption{Detailed metrics at FN~$= 0.75$ (LBCA operating point).}
\label{tab:detail-075}
""")
    tex.append(_table_full(spectra_dict, fn_grid, 0.75))
    tex.append(r"\end{table}")

    tex.append(r"""
\begin{table}[ht]
\centering
\caption{Detailed metrics at FN~$= 0.90$ (LUCA operating point).}
\label{tab:detail-090}
""")
    tex.append(_table_full(spectra_dict, fn_grid, 0.90))
    tex.append(r"\end{table}")

    tex.append(r"""
\end{document}
""")

    Path(outpath).write_text("\n".join(tex))


def generate_report(spectra_dict, fn_grid, outdir):
    """Write the figures, the LaTeX document and spectra.json into outdir.

    spectra_dict maps model name -> metric dicts, one per FN in fn_grid and in
    the same order. Names are load-bearing: one containing a MODEL_STYLES key
    picks up that style, one containing 'noisy' is treated as the null
    baseline and is excluded from best-value bolding.
    """
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating report in {od}/", flush=True)

    plot_precision_recall(spectra_dict, od / 'precision_recall_vs_fn.pdf')
    plot_f1(spectra_dict, od / 'f1_vs_fn.pdf')
    plot_mcc(spectra_dict, od / 'mcc_vs_fn.pdf')

    generate_latex(spectra_dict, fn_grid, od / 'results_report.tex', fig_dir='.')
    print(f"  Saved: {od / 'results_report.tex'}")

    # Raw metrics, so figures can be redrawn without re-running eval.
    json_data = {
        'fn_grid': fn_grid,
        'models': {name: rows for name, rows in spectra_dict.items()}
    }
    with open(od / 'spectra.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {od / 'spectra.json'}")

    print(f"  Done. Compile with: cd {od} && pdflatex results_report.tex\n")
