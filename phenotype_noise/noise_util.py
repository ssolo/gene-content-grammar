#!/usr/bin/env python3
"""Deterministic fp/fn noise on a COG count table, shared by the denoise step
and the prediction step so both arms see the same noisy input.

Corruption is driven by an explicit per-(split, fn, fp) seed, so it is
bit-reproducible across the machines the two steps run on.
"""
import numpy as np


def flip_with_fractional_noise(X, fp_rate, fn_rate, seed, hard_fn=True):
    """Return a noisy copy of the (n, d) count table X.

    fn_rate is a fraction of each genome's present features (set to 0 under
    hard_fn, otherwise decremented by one); fp_rate is a fraction of all d
    features, incremented by one. The result is clipped at 0.
    """
    rng = np.random.default_rng(seed)
    Xn = X.astype(np.float32).copy()
    n, d = Xn.shape
    for i in range(n):
        pos = np.nonzero(X[i] > 0)[0]
        n_fn = int(round(fn_rate * len(pos)))
        if n_fn > 0:
            sel = rng.permutation(len(pos))[:n_fn]
            if hard_fn:
                Xn[i, pos[sel]] = 0
            else:
                Xn[i, pos[sel]] -= 1
        n_fp = int(round(fp_rate * d))
        if n_fp > 0:
            sel = rng.permutation(d)[:n_fp]
            Xn[i, sel] += 1
    return np.clip(Xn, 0, None)


def noise_seed(base, split, fn, fp):
    """Stable integer seed for one (split, fn, fp) cell of the noise sweep.

    fn is quantised to 1/100 and fp to 1/1000: rates closer together than that
    share a seed, and hence the same flips.
    """
    return int(base * 1_000_003 + split * 9176 + int(round(fn * 100)) * 53 + int(round(fp * 1000)))
