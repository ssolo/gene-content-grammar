# -*- coding: utf-8 -*-
"""
Ising Denoiser — ancestral genome reconstruction by iterative mean-field denoising.

Gene tree–species tree reconciliation (ALE, GeneRax, ALeRax) infers which genes
were present at ancestral nodes, with high false-negative rates (FN ≈ 0.5–0.9 at
deep nodes such as LBCA and LUCA).  This package learns pairwise couplings among
N = 4 789 COG gene families from ~90 000 extant prokaryotic genomes and uses them
to restore the missing genes.

The focal model is an iterative mean-field Ising denoiser with hidden state,
Onsager (TAP) correction and module-completeness conditioning:

    x^{t+1} = tanh( h + (1 + (1+γ_field)·φ_t(x₀)) · coupling
                      + (1+γ_skip)·σ_t(x₀) · x₀ )
    z^{t+1} = tanh( x^{t+1} Uᵀ + z^t W̃ )

    coupling  =  x^t J̃  +  z^t Aᵀ  −  x^t (1 − x^t²) @ J²    (TAP/Onsager)

    J ∈ ℝ^{N×N}   symmetric zero-diagonal learned pairwise coupling
    h ∈ ℝ^N       per-gene bias / external field
    z ∈ ℝ^H       hidden state (H = 1000)
    σ_t, φ_t      per-gene skip and field gates (scalar + diagonal + low-rank)
    γ_skip/γ_field module-completeness conditioning, output of an MLP
                  acting on f_obs = Mᵀ((x₀+1)/2) / |M| ∈ [0,1]^M

The per-iteration equations, including the TAP terms elided above, are in
`models.py`.

Models
------
  ModuleConditionedDenoiser    The equations above.

  NoHiddenDenoiser             Hidden state removed (no A, U, W), so the
                               coupling collapses to x^t J̃ (+ TAP).

  HigherOrderDenoiser          ModuleConditionedDenoiser plus an
                               attention-parameterised additive field
                               Δ(x, z, t).

  NoHiddenHigherOrderDenoiser  Δ on top of NoHiddenDenoiser, with the attention
                               reading (x, t) alone.  The production model.

  SetTransformerDenoiser       Attention in place of an explicit J; single-pass,
                               iterative, or hybrid Ising-transformer behaviour
                               selected by constructor flags.

The `onsager` constructor argument selects the TAP terms: `'none'` none,
`'within'` the J² and W² reaction terms only, `'full'` also the cross-block A²
(visible) and U² (hidden) terms with independent A, U, and `'tied'` the same
with U ≡ Aᵀ, under which the bare pairwise iteration is block TAP for a
well-defined energy (the gates, conditioning and temperature are learned
amortisation on top and do not derive from that energy).  Production training
uses `onsager='full'`.

Training entry points
---------------------
  train_denovo.py                 From-scratch 8-stage curriculum tempering the
                                  noise distribution and the trainable parameter
                                  set together; `--no-hidden` trains
                                  NoHiddenDenoiser.
  train_higher_order_nohidden.py  Adds attention-Δ to a NoHiddenDenoiser
                                  checkpoint via the HO1/HO2/HO3 stages.
  train_add_hidden.py             Adds the hidden machinery (A, U, W) to a
                                  trained NoHidden HO3 checkpoint from a
                                  transparent A = 0 start.
"""
__version__ = "0.11.0"
