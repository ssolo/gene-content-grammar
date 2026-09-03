#!/usr/bin/env python3
"""Torch-free unit and smoke tests for fp_context_diagnostics.

Covers the split parser and its glob refusal, TSV IO, the metric functions
(ROC-AUC, average precision), and an end-to-end summarize on a synthetic extant
TSV. The forward pass is not exercised: it needs the checkpoints, and the
validation feathers for the extant mode; its commands are in the docstring of
fp_context_diagnostics.py.

Run:  python scripts/test_fp_context_diagnostics.py   (no pytest needed)
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fp_context_diagnostics as D

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name}')


def test_parse_splits():
    check('parse 1-10', D.parse_splits('1-10') == list(range(1, 11)))
    check('parse 1,3,5', D.parse_splits('1,3,5') == [1, 3, 5])
    check('parse 5', D.parse_splits('5') == [5])
    check('parse 1-3,7', D.parse_splits('1-3,7') == [1, 2, 3, 7])
    try:
        D.parse_splits('*'); ok = False
    except SystemExit:
        ok = True
    check('parse * refused', ok)


def test_resolve_models():
    tmpl = 'gsd_results_x_split{split}/model_ho3.pth'
    pairs = D.resolve_models(tmpl, [1, 2], allow_glob=False)
    check('resolve expands {split}', [s for s, _ in pairs] == [1, 2]
          and pairs[0][1].endswith('gsd_results_x_split1/model_ho3.pth'))
    try:
        D.resolve_models('gsd_results_x_split*/model_ho3.pth', [1], allow_glob=False); ok = False
    except SystemExit:
        ok = True
    check('resolve glob refused', ok)
    pairs = D.resolve_models('gsd_results_x_split*_{split}/m.pth', [1], allow_glob=True)
    check('resolve glob allowed with flag', len(pairs) == 1)
    try:
        D.resolve_models('no_placeholder/model.pth', [1], allow_glob=False); ok = False
    except SystemExit:
        ok = True
    check('resolve requires {split}', ok)


def test_load_value_tsv():
    with tempfile.NamedTemporaryFile('w', suffix='.tsv', delete=False) as f:
        f.write('COG_ID\tinput_prob\tmean_actual\n')
        f.write('COG0001\t0.9\t0.95\n')
        f.write('COG0002_1\t0.3\t0.40\n')   # sub-family suffix collapses to COG0002
        f.write('COG0002_2\t0.5\t0.60\n')   # collapse takes the max: 0.5 / 0.60
        f.write('notacog\t1.0\t1.0\n')
        path = f.name
    ip = D.load_value_tsv(path, 'input_prob')
    ma = D.load_value_tsv(path, 'mean_actual')
    os.unlink(path)
    check('load_value_tsv input_prob', abs(ip['COG0001'] - 0.9) < 1e-9)
    check('load_value_tsv suffix-collapse max', abs(ip['COG0002'] - 0.5) < 1e-9)
    check('load_value_tsv second col', abs(ma['COG0002'] - 0.60) < 1e-9)
    check('load_value_tsv drops non-COG', 'notacog' not in ip)


def test_logit():
    check('logit 0.5 ~ 0', abs(D.logit(np.array([0.5]))[0]) < 1e-9)
    check('logit clamps 0/1', np.isfinite(D.logit(np.array([0.0, 1.0]))).all())
    check('logit monotone', D.logit(np.array([0.8]))[0] > D.logit(np.array([0.2]))[0])


def test_metrics():
    y = np.array([0, 0, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    s_reversed = np.array([0.9, 0.8, 0.2, 0.1])
    check('roc_auc perfect=1', abs(D.roc_auc(y, s_perfect) - 1.0) < 1e-9)
    check('roc_auc reversed=0', abs(D.roc_auc(y, s_reversed) - 0.0) < 1e-9)
    check('roc_auc ties=0.5', abs(D.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9)
    check('ap perfect=1', abs(D.average_precision(y, s_perfect) - 1.0) < 1e-9)
    check('roc_auc nan-safe', np.isfinite(D.roc_auc(y, np.array([0.1, np.nan, 0.8, 0.9]))))
    check('roc_auc no-pos=nan', np.isnan(D.roc_auc(np.array([0, 0]), np.array([0.1, 0.2]))))


def test_summarize_smoke():
    # Synthetic extant TSV in which the injected FPs have low context support
    # (q_mask_absent ~ 0.1) and true genes high (~ 0.9), so the cavity score
    # -q_mask_absent_mean separates them almost perfectly.
    rng = np.random.RandomState(0)
    n = 60
    known_fp = np.array([1] * 20 + [0] * 40)
    truth = 1 - known_fp
    qabs = np.where(known_fp == 1, 0.1, 0.9) + rng.normal(0, 0.02, n)
    qneu = qabs + 0.05
    qfull = np.where(known_fp == 1, 0.7, 0.95)   # model keeps both on full input
    df_lines = ['\t'.join(['COG_ID', 'truth', 'known_fp', 'true_input', 'q_full_mean',
                           'q_mask_neutral_mean', 'q_mask_absent_mean', 'self_anchor_absent',
                           'dropout_mean'])]
    for i in range(n):
        df_lines.append('\t'.join([f'COG{i:04d}', str(truth[i]), str(known_fp[i]),
                                   str(truth[i]), f'{qfull[i]:.4f}', f'{qneu[i]:.4f}',
                                   f'{qabs[i]:.4f}', f'{qfull[i]-qabs[i]:.4f}',
                                   f'{qabs[i]:.4f}']))
    with tempfile.NamedTemporaryFile('w', suffix='.tsv', delete=False) as f:
        f.write('\n'.join(df_lines) + '\n')
        path = f.name

    class A:
        tsv = path
    buf = io.StringIO()
    with redirect_stdout(buf):
        D.mode_summarize(A())
    os.unlink(path)
    txt = buf.getvalue()
    check('summarize runs + prints ROC-AUC', 'ROC-AUC' in txt and 'pruning table' in txt)
    auc = D.roc_auc(known_fp, -qabs)
    check('summarize separable AUC>0.95', auc > 0.95)


def main():
    for t in [test_parse_splits, test_resolve_models, test_load_value_tsv,
              test_logit, test_metrics, test_summarize_smoke]:
        print(t.__name__)
        t()
    print(f'\n==== PASS {PASS}  FAIL {FAIL} ====')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
