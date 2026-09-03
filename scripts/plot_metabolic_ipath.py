#!/usr/bin/env python3
"""Metabolic reconstruction figures (pre- vs post-denoising) via iPath3.

Paints an ancestral gene-content reconstruction onto the iPath3 global metabolic
map (pathways.embl.de), which accepts COG identifiers natively. Views per node:

    pre       COGs the reconciliation proposed (the input)
    post      COGs present after denoising (posterior > 0.5)
    overlay   kept      present in input AND output            (grey)
              recovered absent in input, present in output     (blue)
              silenced  present in input, absent in output     (red)

Nodes:
    lbca  data/lbca_cog_lists.tsv  -- sparse bacterial-root reconciliation
          (Davin et al. 2025); the input is near-empty, so pre = input copy
          number > 0, post = denoised present call.
    laca  LACA_euryroot_pred.tsv   -- Euryarchaeota-rooted reconciliation
          (Huang et al. 2025); the input is graded, so pre = input probability
          > 0.5, post = posterior > 0.5.

Output SVGs (+ PDFs via rsvg-convert) in analysis/figures/metabolic_<node>_*.

Usage:
    python3 scripts/plot_metabolic_ipath.py                 # both nodes, all views
    python3 scripts/plot_metabolic_ipath.py --node laca     # one node
    python3 scripts/plot_metabolic_ipath.py --no-post       # reuse SVGs, just convert

iPath3 exports SVG only; SVG->PDF conversion is local, via rsvg-convert with a
high-resolution PNG fallback through qlmanage.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "analysis", "figures")
IPATH = "https://pathways.embl.de/mapping.cgi"

# colour-blind-safe: grey kept, blue recovered/present, red silenced
GREY, BLUE, RED = "#9e9e9e", "#2166ac", "#b2182b"


def _read_tsv(path):
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ci = {k: i for i, k in enumerate(header)}
        for line in fh:
            yield ci, line.rstrip("\n").split("\t")


def load_lbca():
    """(cog, pre, post, p) for the sparse bacterial-root build; p = ensemble posterior."""
    rows = []
    for ci, f in _read_tsv(os.path.join(REPO, "data", "lbca_cog_lists.tsv")):
        cog = f[ci["COG_ID"]]
        pre = float(f[ci["sparse_input"]]) > 0.0
        p = float(f[ci["sparse_out_post"]])
        post = int(f[ci["sparse_present"]]) == 1
        rows.append((cog, pre, post, p))
    return rows


def load_laca():
    """(cog, pre, post, p) for the focal Euryarchaeota-rooted build; p = posterior."""
    rows = []
    for ci, f in _read_tsv(os.path.join(REPO, "LACA_euryroot_pred.tsv")):
        cog = f[ci["COG_ID"]]
        pre = float(f[ci["input_prob"]]) > 0.5
        p = float(f[ci["mean_actual"]])
        post = p > 0.5
        rows.append((cog, pre, post, p))
    return rows


NODES = {"lbca": load_lbca, "laca": load_laca}


def selection_overlay(rows):
    # Recovered genes get the thickest stroke, so a net-building node reads as
    # blue even where it also prunes.
    out = []
    for cog, pre, post, *_ in rows:
        if pre and post:
            out.append(f"{cog} {GREY} W8")         # kept
        elif (not pre) and post:
            out.append(f"{cog} {BLUE} W18")        # recovered (built)
        elif pre and (not post):
            out.append(f"{cog} {RED} W10")         # silenced (pruned)
    return "\n".join(out)


def selection_single(rows, which):
    out = []
    for cog, pre, post, *_ in rows:
        on = pre if which == "pre" else post
        if on:
            out.append(f"{cog} {BLUE} W14")
    return "\n".join(out)


# ---- E. coli ground-truth recovery (validation node)
# The extant benchmark has a ground truth, so present calls are coloured by
# correctness against it: true positive blue, false positive red.
GREEN = "#1a9850"   # truth reference

def load_ecoli(fn=0.9, tsv="recover_ecoli_bachard.tsv"):
    """(cog, truth, input_present, denoised_present, posterior_p) for E. coli
    under the bacterial specialist at the given false-negative level."""
    rows = []
    for ci, f in _read_tsv(os.path.join(REPO, tsv)):
        if abs(float(f[ci["fn"]]) - fn) > 1e-9:
            continue
        prob = float(f[ci["denoised_prob"]])
        rows.append((
            f[ci["COG_ID"]],
            int(f[ci["truth"]]) == 1,
            int(f[ci["input_present"]]) == 1,
            prob > 0.5,
            prob,
        ))
    return rows


def selection_ecoli(rows, which, tp_w=18, fp_w=12):
    """Paint the present set of `which` in {input, recon}, coloured by truth:
    true positive blue, false positive red. Fully opaque; the confidence-graded
    recon map is selection_ecoli_recon."""
    out = []
    for cog, truth, inp, post, _p in rows:
        on = inp if which == "input" else post
        if not on:
            continue
        out.append(f"{cog} {BLUE} W{tp_w}" if truth else f"{cog} {RED} W{fp_w}")
    return "\n".join(out)


def selection_ecoli_recon(rows):
    """Recon map in three correctness states: a true gene kept or recovered is blue,
    a true gene not recovered is grey (the gap), a spurious present call is red.
    Blue genes that survived the corrupted input carry a thin black centreline (see
    _style_recon_paths); opacity is the posterior confidence.

    Blue is emitted last: where one reaction is covered by several COGs iPath keeps
    the last entry, so a correct call is not masked by a false positive or a gap."""
    out = []
    def rank(r):                       # red/grey drawn first, blue last
        truth, post = r[1], r[3]
        return 2 if (truth and post) else (1 if (truth and not post) else 0)
    for cog, truth, inp, post, p in sorted(rows, key=rank):
        if truth and post:
            out.append(f"{cog} {_rgba(BLUE, _conf_alpha(p))} W{_W_KEPT if inp else _W_ADDED}")
        elif truth and not post:
            out.append(f"{cog} {_rgba(_GREY_MISSED, 1.0)} W{_W_ADDED}")
        elif post:
            out.append(f"{cog} {_rgba(RED, _conf_alpha(p))} W{_W_ADDED}")
        # truth=0, post=0 -> correctly absent, not drawn
    return "\n".join(out)


def selection_ecoli_input(rows):
    """The corrupted input, in the same scheme and colours as the reconstruction:
    surviving true genes blue, injected false positives red. Every input gene rides
    the _W_KEPT width, so it carries the black input centreline and reads identically
    here and in the reconstruction. The input is a binary observation with no
    posterior, hence full opacity."""
    out = []
    for cog, truth, inp, post, p in sorted(rows, key=lambda r: r[1]):  # FPs first, true genes last
        if not inp:
            continue
        color = BLUE if truth else RED
        out.append(f"{cog} {_rgba(color, 1.0)} W{_W_KEPT}")
    return "\n".join(out)


def selection_ecoli_truth(rows):
    """The real E. coli metabolism (ground truth) as a reference panel."""
    return "\n".join(f"{cog} {GREEN} W14" for cog, truth, *_ in rows if truth)


# iPath draws KEGG pathway-category labels as bright rounded-rect "bubbles"; remap
# those default colours to the Okabe palette used for the genes. None of these
# source colours is used by a selection stroke.
_BUBBLE_RECOLOR = {
    "#00CC33": "#009E73",  # carbohydrate / energy -> Okabe green
    "#0000EE": "#0072B2",  # amino-acid          -> Okabe blue
    "#3399FF": "#E69F00",  # glycan biosynthesis -> Okabe orange (envelope)
    "#9933CC": "#CC79A7",  # cofactors/vitamins  -> Okabe purple
    "#FF0000": "#D55E00",  # nucleotide          -> Okabe vermilion
    "#FF6600": "#E69F00",  # -> Okabe orange
    "#FF9933": "#E69F00",  # -> Okabe orange
    "#009999": "#999999",  # lipid (not a panel-B key system) -> neutral grey
    "#CC3366": "#999999",  # secondary metabolites -> grey
    "#FF7899": "#CC79A7",  # -> Okabe purple
    "#CCAA99": "#999999",  # xenobiotics         -> grey
}


def ipath_post(selection, out_svg, map_name="metabolic",
               default_color="#e6e6e6", default_opacity="0.35", default_width="2",
               recon_style=False, styler=None):
    data = urllib.parse.urlencode({
        "selection": selection,
        "map": map_name,
        "export_type": "svg",
        "default_color": default_color,
        "default_opacity": default_opacity,
        "default_width": default_width,
        "background_color": "#ffffff",
        "keep_colors": "0",
    }).encode()
    req = urllib.request.Request(IPATH, data=data,
                                 headers={"User-Agent": "ising-denoiser/1.0 (metabolic figure)"})
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read()
    if not body.lstrip().startswith(b"<?xml") and b"<svg" not in body[:400]:
        sys.exit(f"iPath did not return SVG for {out_svg}: {body[:200]!r}")
    # iPath emits both upper- and lower-case hex fills.
    text = body.decode("utf-8", "replace")
    for src, dst in _BUBBLE_RECOLOR.items():
        text = text.replace(src, dst).replace(src.lower(), dst)
    if recon_style:
        text = _style_recon_paths(text)
    elif styler is not None:
        text = styler(text)
    body = text.encode("utf-8")
    with open(out_svg, "wb") as fh:
        fh.write(body)
    print(f"  wrote {os.path.relpath(out_svg, REPO)} ({len(body)} bytes)")


# The manuscript embeds the PNG: large iPath vector PDFs render in pdflatex but
# show blank in some viewers. The vector PDF and SVG are still emitted.
PNG_WIDTH = 4000


def svg_to_pdf(svg):
    if shutil.which("rsvg-convert"):
        png = svg[:-4] + ".png"
        subprocess.run(["rsvg-convert", "-f", "png", "-w", str(PNG_WIDTH), "-o", png, svg], check=True)
        print(f"  -> {os.path.relpath(png, REPO)} (raster, {PNG_WIDTH}px; embedded by the manuscript)")
    pdf = svg[:-4] + ".pdf"
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", pdf, svg], check=True)
        print(f"  -> {os.path.relpath(pdf, REPO)} (vector, for production)")
        return pdf
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", svg, "--export-type=pdf", f"--export-filename={pdf}"], check=True)
        return pdf
    if shutil.which("qlmanage"):
        subprocess.run(["qlmanage", "-t", "-s", "3000", svg, "-o", os.path.dirname(svg)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        png = svg + ".png"
        if os.path.exists(png):
            final = svg[:-4] + ".png"
            os.replace(png, final)
            print(f"  -> {os.path.relpath(final, REPO)} (raster fallback; install librsvg for vector)")
            return final
    print(f"  !! no SVG converter; {os.path.relpath(svg, REPO)} left as SVG")
    return None


def run_ecoli(no_post, fn=0.9, tsv="recover_ecoli_bachard.tsv", tag=""):
    """Before/after maps for the extant E. coli benchmark: corrupted input vs
    denoised reconstruction, both coloured against the ground truth, plus the
    truth itself as a reference panel."""
    rows = load_ecoli(fn=fn, tsv=tsv)
    truth = sum(1 for _, t, _, _, _ in rows if t)
    n_in = sum(1 for _, _, i, _, _ in rows if i)
    n_rec = sum(1 for _, _, _, post, _ in rows if post)
    tp_in = sum(1 for _, t, i, _, _ in rows if t and i)
    tp_rec = sum(1 for _, t, _, post, _ in rows if t and post)
    fp_rec = sum(1 for _, t, _, post, _ in rows if (not t) and post)
    print(f"E. coli (fn={fn}, {os.path.basename(tsv)}): truth {truth}; input {n_in} "
          f"(recall {tp_in/truth:.0%}); reconstruction {n_rec} "
          f"(recall {tp_rec/truth:.0%}, {fp_rec} false positives)")
    # The truth panel stays solid (styler None).
    jobs = [
        (f"metabolic_ecoli_input{tag}", selection_ecoli_input(rows), _style_recon_paths),
        (f"metabolic_ecoli_recon{tag}", selection_ecoli_recon(rows), _style_recon_paths),
        (f"metabolic_ecoli_truth{tag}", selection_ecoli_truth(rows), None),
    ]
    # The faint default colour ghosts the full metabolic atlas under every panel.
    for name, sel, styler in jobs:
        svg = os.path.join(FIGDIR, name + ".svg")
        if not no_post:
            print(f"iPath POST: {name}")
            ipath_post(sel, svg, default_color="#c8ccd0",
                       default_opacity="0.30", default_width="1.4", styler=styler)
        if os.path.exists(svg):
            svg_to_pdf(svg)


# ---- LBCA/LACA sysmap mode: recon + before/after, coloured by curated key system.
# KEGG modules and their panel-B group colours (mirrors plot_recon_figs.py).
_OKABE = {"blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
          "purple": "#CC79A7", "verm": "#D55E00", "grey": "#999999"}
_GROUP_COLOR = {"biosynth": _OKABE["blue"], "energy": _OKABE["green"],
                "envelope": _OKABE["orange"], "cofactor": _OKABE["purple"],
                "aerobic": _OKABE["verm"], "lineage": _OKABE["grey"],
                "bacterial": _OKABE["verm"]}
_LBCA_CURATED = [("M00023", "biosynth"), ("M00016", "biosynth"), ("M00026", "biosynth"),
    ("M00017", "biosynth"), ("M00052", "biosynth"), ("M00048", "biosynth"),
    ("M00157", "energy"), ("M00144", "energy"), ("M00009", "energy"),
    ("M00001", "energy"), ("M00004", "energy"), ("M00866", "envelope"),
    ("M00063", "envelope"), ("M00930", "cofactor"), ("M00125", "cofactor"),
    ("M00926", "cofactor"), ("M00120", "cofactor"), ("M00155", "aerobic"),
    ("M00151", "aerobic"), ("M00564", "lineage")]
_LACA_CURATED = [("M00159", "energy"), ("M00567", "energy"), ("M00357", "energy"),
    ("M00422", "energy"), ("M00377", "energy"), ("M00378", "cofactor"),
    ("M00935", "cofactor"), ("M00358", "cofactor"), ("M00896", "cofactor"),
    ("M00125", "cofactor"), ("M00052", "biosynth"), ("M00048", "biosynth"),
    ("M00023", "biosynth"), ("M00026", "biosynth"), ("M00157", "bacterial"),
    ("M00866", "bacterial")]
_CURATED = {"lbca": _LBCA_CURATED, "laca": _LACA_CURATED}
_GREY_KEEP = "#c3c7ca"
_RED_REMOVED = "#b2182b"

# Reconstruction-map styling, rebuilt in _style_recon_paths. The kept/added split
# is carried in the selection WIDTH (_W_KEPT vs _W_ADDED), since kept and added
# genes of the same curated system share a colour. Opacity is the posterior
# confidence, fading to nothing at the p=0.5 call boundary.
_BLACK = "#000000"
_GREY_ADDED = "#a6a6a6"   # present gene outside any curated key system (!= lineage grey)
_GREY_MISSED = "#9aa0a6"  # E. coli: a true gene not recovered (the gap)
_RECON_OUTLINE_W = 16     # outer stroke, in the gene's own colour
_RECON_INNER_W = 5        # black centreline marking input-present (kept) genes
_RECON_THIN_W = 7         # thin stroke: red negatives, grey missed genes
_W_KEPT = 30
_W_ADDED = 31
_SYS_BODIES = {v.lstrip("#").lower() for v in _OKABE.values()}  # panel-B system hexes


def curated_cog_groups(node, mm_path="data/module_matrix_kegg.pt"):
    """COG -> panel-B group colour of its curated key system (first match)."""
    import torch
    import numpy as np
    mm = torch.load(os.path.join(REPO, mm_path), map_location="cpu", weights_only=False)
    kn = list(mm["kegg_names"]); cn = list(mm["cog_names"])
    Mk = mm["M_kegg"]; Mk = Mk.numpy() if hasattr(Mk, "numpy") else np.asarray(Mk)
    col = {}
    for mid, grp in _CURATED[node]:
        if mid not in kn:
            continue
        j = kn.index(mid)
        for i in np.where(Mk[:, j] > 0)[0]:
            col.setdefault(cn[i], _GROUP_COLOR[grp])
    return col


def _lighten(hexc, f=0.5):
    r, g, b = (int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(round(v + f * (255 - v))) for v in (r, g, b))
    return "#%02x%02x%02x" % (r, g, b)


def selection_sysmap(rows, cog_col, which):
    """which='input': present input genes coloured by curated key system, with the
    genes the denoiser will remove in red. 'output': present output genes coloured
    by system, with the added genes thick and lightened. Present genes outside any
    curated system are neutral grey."""
    out = []
    for cog, pre, post, *_ in rows:
        sysc = cog_col.get(cog)
        if which == "input":
            if not pre:
                continue
            if pre and not post:                          # silenced
                out.append(f"{cog} {_RED_REMOVED} W16")
            else:
                out.append(f"{cog} {sysc or _GREY_KEEP} W{10 if sysc else 6}")
        else:                                             # output
            if not post:
                continue
            if post and not pre:                          # added
                out.append(f"{cog} {_lighten(sysc or '#2166ac', 0.5)} W18")
            else:
                out.append(f"{cog} {sysc or _GREY_KEEP} W{10 if sysc else 6}")
    return "\n".join(out)


def _conf_alpha(p):
    """Opacity from posterior confidence: full at p=1.0, zero at the p=0.5 boundary."""
    return max(0.0, min(1.0, (p - 0.5) / 0.5))


def _rgba(hexc, a):
    """6-digit hex + an alpha byte -> 8-digit rgba hex (iPath passes it to the SVG)."""
    return "%s%02x" % (hexc, int(round(a * 255)))


def selection_recon_sysmap(rows, cog_col):
    """Single reconstruction map: present genes in their curated key-system colour
    (neutral grey outside any curated system), silenced genes in red. Opacity is the
    ensemble posterior confidence, solid at p=1.0 and fading to nothing at the p=0.5
    call boundary; for silenced genes the confidence is that of absence, 1 - p. The
    kept/added split rides in the selection width for _style_recon_paths to rebuild."""
    out = []
    for cog, pre, post, p in rows:
        sysc = cog_col.get(cog)
        color = sysc if sysc else _GREY_ADDED
        if post and pre:                # kept
            out.append(f"{cog} {_rgba(color, _conf_alpha(p))} W{_W_KEPT}")
        elif post and not pre:          # added
            out.append(f"{cog} {_rgba(color, _conf_alpha(p))} W{_W_ADDED}")
        elif pre and not post:          # silenced
            out.append(f"{cog} {_rgba(_RED_REMOVED, _conf_alpha(1.0 - p))} W{_W_ADDED}")
    return "\n".join(out)


def _style_recon_paths(text):
    """Post-process a recon-map SVG into the input-marker / colour-added encoding.
    Keys on the selection width sentinel rather than colour, so it serves both the
    curated-key-system maps (LBCA/LACA) and the E. coli truth-coloured map.

    iPath draws every selected reaction as <path ... stroke='#RRGGBBAA' stroke-width=W>
    at a constant style opacity of 0.30. (RRGGBB, AA, W) is recovered and rebuilt
    without that 0.30 cap, so the AA confidence alpha governs visibility:
      * width _W_KEPT  (present in the input) -> outline in the gene's colour plus a
        thin black centreline;
      * width _W_ADDED (added / silenced)     -> one stroke in the gene's colour.
    Atlas/background paths (any other width) are left untouched."""
    pat = re.compile(
        r"<path style='[^']*' stroke='#([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})' "
        r"fill='none' stroke-width='([0-9.]+)'\s*d='([^']*)'></path>")

    def line(col, a, w, d):
        return (f"<path style='opacity: 1; ' stroke='#{col}{a}' fill='none' "
                f"stroke-width='{w}' d='{d}'></path>")

    def repl(m):
        body, alpha, width, d = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        try:
            w = float(width)
        except ValueError:
            return m.group(0)
        kept = abs(w - _W_KEPT) < 0.5
        added = abs(w - _W_ADDED) < 0.5
        if not (kept or added):           # atlas / background
            return m.group(0)
        if body.lower() in (_RED_REMOVED.lstrip("#"), _GREY_MISSED.lstrip("#")):
            return line(body, alpha, _RECON_THIN_W, d)
        outline = line(body, alpha, _RECON_OUTLINE_W, d)
        if kept:
            return outline + line("000000", alpha, _RECON_INNER_W, d)
        return outline

    return pat.sub(repl, text)


def run_sysmap(node, no_post):
    """Metabolic maps for an ancestral node, coloured by curated key system: the
    single 'recon' map, plus the 'before'/'after' pair."""
    rows = NODES[node]()
    cog_col = curated_cog_groups(node)
    n_rem = sum(1 for _, p, q, _ in rows if p and not q)
    n_add = sum(1 for _, p, q, _ in rows if q and not p)
    print(f"{node.upper()} sysmap: {len(cog_col)} curated-system COGs; "
          f"removed {n_rem}, added {n_add}")
    jobs = [
        (f"metabolic_{node}_recon", selection_recon_sysmap(rows, cog_col), True),
        (f"metabolic_{node}_before", selection_sysmap(rows, cog_col, "input"), False),
        (f"metabolic_{node}_after", selection_sysmap(rows, cog_col, "output"), False),
    ]
    for name, sel, recon_style in jobs:
        svg = os.path.join(FIGDIR, name + ".svg")
        if not no_post:
            print(f"iPath POST: {name}")
            ipath_post(sel, svg, default_color="#e8eaec", default_opacity="0.30",
                       default_width="1.4", recon_style=recon_style)
        if os.path.exists(svg):
            svg_to_pdf(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", choices=["lbca", "laca", "ecoli", "both"], default="both")
    ap.add_argument("--no-post", action="store_true", help="skip iPath POST, just convert existing SVGs")
    ap.add_argument("--sysmap", action="store_true",
                    help="(lbca/laca) two maps before/after, coloured by curated key system")
    ap.add_argument("--tsv", default="recover_ecoli_bachard.tsv",
                    help="(--node ecoli) recovery TSV: COG_ID/fn/truth/input_present/denoised_prob")
    ap.add_argument("--fn", type=float, default=0.9, help="(--node ecoli) false-negative level to render")
    ap.add_argument("--tag", default="", help="(--node ecoli) filename suffix, e.g. _fn04")
    A = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    if A.node == "ecoli":
        run_ecoli(A.no_post, fn=A.fn, tsv=A.tsv, tag=A.tag)
        return
    nodes = ["lbca", "laca"] if A.node == "both" else [A.node]

    for node in nodes:
        if A.sysmap:
            run_sysmap(node, A.no_post)
            continue
        rows = NODES[node]()
        n_pre = sum(p for _, p, _, _ in rows)
        n_post = sum(q for _, _, q, _ in rows)
        n_kept = sum(1 for _, p, q, _ in rows if p and q)
        n_rec = sum(1 for _, p, q, _ in rows if (not p) and q)
        n_sil = sum(1 for _, p, q, _ in rows if p and (not q))
        print(f"{node.upper()}: input-proposed {n_pre}, denoised-present {n_post} "
              f"(kept {n_kept}, recovered {n_rec}, silenced {n_sil})")
        jobs = [
            (f"metabolic_{node}_overlay", selection_overlay(rows)),
            (f"metabolic_{node}_pre", selection_single(rows, "pre")),
            (f"metabolic_{node}_post", selection_single(rows, "post")),
        ]
        for name, sel in jobs:
            svg = os.path.join(FIGDIR, name + ".svg")
            if not A.no_post:
                print(f"iPath POST: {name}")
                ipath_post(sel, svg)
            if os.path.exists(svg):
                svg_to_pdf(svg)


if __name__ == "__main__":
    main()
