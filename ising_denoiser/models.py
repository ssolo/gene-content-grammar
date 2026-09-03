"""
Model architectures for the Ising denoiser.

All models reconstruct a clean binary genome x ∈ {±1}^N from a noisy
observation x₀ corrupted by reconciliation errors (high false-negative,
low false-positive rate), through the shared interface

    forward(x₀) → (x_denoised, auxiliary_state)

with x_denoised ∈ (-1, +1)^N continuous (threshold at 0 for hard decisions)
and auxiliary_state the final hidden state z, or None where there is none.

  ModuleConditionedDenoiser    mean-field/TAP Ising denoiser with hidden state
  NoHiddenDenoiser             the same without hidden spins (A, U, W dropped)
  HigherOrderDenoiser          + attention-parameterised additive field Δ
  NoHiddenHigherOrderDenoiser  Δ on top of NoHiddenDenoiser
  SetTransformerDenoiser       self-attention in place of an explicit J

See MODELS.md for parameter counts and provenance.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


def set_coupling_controls(model, freq_tsv=None, shrink_alpha=None,
                          genome_mass_norm=False, device=None):
    """Set a loaded NoHidden* denoiser's coupling controls in place (frozen),
    from the per-COG frequency p_c in ``freq_tsv`` (same COG order as J):
    coupling_shrink = sqrt(p_c / (p_c + alpha)), gmass_kbar = sum_c p_c.  Both
    default off, leaving the forward identical to the trained model.
    """
    if shrink_alpha is None and not genome_mass_norm:
        model.coupling_shrink = None
        model.gmass_kbar = None
        return 'coupling-controls: OFF (baseline)'
    import pandas as pd
    dev = device or next(model.parameters()).device
    p = torch.tensor(pd.read_csv(freq_tsv, sep='\t')['p_c'].to_numpy(),
                     dtype=torch.float32, device=dev)
    assert p.numel() == model.N, \
        f'freq length {p.numel()} != model.N {model.N} (COG-order mismatch)'
    desc = []
    if shrink_alpha is not None:
        model.coupling_shrink = torch.sqrt(p / (p + float(shrink_alpha)))
        desc.append(f'shrink(alpha={shrink_alpha})')
    else:
        model.coupling_shrink = None
    if genome_mass_norm:
        model.gmass_kbar = torch.tensor(float(p.sum()), device=dev)
        desc.append(f'gmass-norm(kbar={float(p.sum()):.1f})')
    else:
        model.gmass_kbar = None
    return 'coupling-controls: ' + ' + '.join(desc)


def lowrank_tail_penalty(J, rank=16, oversample=16):
    """Frobenius norm of J outside its top-`rank` subspace, via a randomized
    range sketch.  The projection is redrawn every call, so the value is
    mildly stochastic."""
    N = J.shape[0]
    Q, _ = torch.linalg.qr(J @ torch.randn(N, min(rank + oversample, N),
                                           device=J.device, dtype=J.dtype))
    tail = J - Q @ (Q.t() @ J)
    return tail.pow(2).sum().sqrt()


def nuclear_norm_lowrank(J, rank=64):
    """Randomized nuclear-norm sketch of J: shrinks the dominant modes, where
    lowrank_tail_penalty shrinks the tail."""
    N = J.shape[0]; r = min(rank, N)
    Q, _ = torch.linalg.qr(J @ torch.randn(N, r, device=J.device, dtype=J.dtype))
    return torch.linalg.svdvals(Q.t() @ J).sum()


def zeromean_coupling_penalty(J):
    """Penalise each gene's total coupling drive away from zero, so h carries
    the marginal load and J only the zero-mean modular deviations."""
    return J.sum(dim=1).pow(2).mean()


class ModuleConditioningMLP(nn.Module):
    """Maps the observable module-completeness profile to per-timestep gate
    modulations.

    f_obs = M^T ((x₀ + 1)/2) / |M| ∈ [0,1]^M is the fraction of member genes
    present in each of M = 419 functional modules (26 COG categories, 66 COG
    pathways, 327 KEGG modules), for the membership matrix M ∈ {0,1}^{N×M}.
    The output (γ^skip_t, γ^field_t) is applied as (1 + γ), so γ = 0 is
    neutral, and the last layer is zero-init.
    """

    def __init__(self, n_modules, T, hidden_dim=128):
        super().__init__()
        self.T = T
        self.mlp = nn.Sequential(
            nn.Linear(n_modules, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * T),
        )
        nn.init.zeros_(self.mlp[4].weight)
        nn.init.zeros_(self.mlp[4].bias)

    def forward(self, f):
        """Map (B, n_modules) completeness fractions to per-timestep skip and
        field modulations, each (B, T)."""
        out = self.mlp(f)
        return out[:, :self.T], out[:, self.T:]


class ModuleConditionedDenoiser(nn.Module):
    r"""Module-completeness conditioned Ising denoiser with hidden state.

    The pairwise model is a joint Ising model over visible spins x ∈ {±1}^N
    and hidden spins z ∈ {±1}^H::

        E(x, z) = -h^T x - (1/2) x^T J x - (1/2) z^T W z - z^T A x

    with J = J^T, W = W^T zero-diagonal and A ∈ ℝ^{N×H} the visible↔hidden
    coupling.  This is the ``'tied'`` case, U ≡ A^T; in ``'full'`` mode the
    visible→hidden coupling U ∈ ℝ^{H×N} is independent, so the model is
    asymmetric and no scalar energy corresponds to it.

    T unrolled mean-field/TAP iterations refine m^t ∈ (-1,+1)^N, n^t ∈
    (-1,+1)^H from m^0 = x₀, n^0 = 0, with x₀ fixed::

        c^x     = J̃ m^t + A n^t                             (1)
                  - m^t ⊙ ((1 - m^t²) J̃²)                   (2a)
                  - m^t ⊙ ((1 - n^t²) (A²)^T)               (2b)
        pre^x   = h + (1 + (1 + γ_field_t) φ_t(x₀)) ⊙ c^x
                    + (1 + γ_skip_t) σ_t(x₀) ⊙ x₀           (3)
        m^{t+1} = tanh(pre^x / τ_t)                         (4)
        c^z     = Ũ m^{t+1} + W̃ n^t                         (5)
                  - n^t ⊙ ((1 - n^t²) W̃²)                   (6a)
                  - n^t ⊙ ((1 - m^{t+1}²) U²)   ['full']    (6b)
                  - n^t ⊙ ((1 - m^{t+1}²) A²)   ['tied']    (6c)
        n^{t+1} = tanh(c^z)                                 (7)

    J̃ and W̃ are the symmetric zero-diagonal projections of _Js and _Ws, Ũ is
    U ('full') or A^T ('tied'), the squares are elementwise, σ_t and φ_t are
    the per-timestep gates of x₀, γ_skip_t and γ_field_t come from
    ModuleConditioningMLP(f_obs), and τ_t = softplus(a_t + b_t ρ) with ρ the
    observed gene density (#present in x₀) / N.

    Tying U = A^T makes the bare pairwise part a TAP fixed-point iteration for
    E; the gates, conditioning and temperature are learned amortisation on top
    and do not derive from E, so the full iteration is not coordinate ascent
    on E.  ``onsager`` selects which terms apply: False/'none' none (naive
    mean field); 'within' (2a) and (6a); 'full' + (2b) and (6b), with A and U
    independent; 'tied' + (2b) and (6c), U ≡ A^T (single-matrix EBM).

    f_obs needs no estimate of the FN rate, and at inference it can be
    recomputed from the output for a multi-pass refinement.  ``module_cond``
    is a registered sub-module, so all its parameters take part in every
    forward pass — DDP-safe.
    """

    def __init__(self, N, H=1000, T=12, skip_rank=32, field_rank=16,
                 n_modules=419, cond_hidden=128, adaptive_temp=False,
                 onsager=False):
        super().__init__()
        # T is the number of unrolled iterations, not a temperature.
        self.N = N; self.H = H; self.T = T
        self.skip_rank = skip_rank; self.field_rank = field_rank
        self.adaptive_temp = adaptive_temp
        # onsager=True is accepted for backward compatibility, as 'full'.
        if onsager is True:
            onsager = 'full'
        elif onsager is False or onsager == 'none':
            onsager = False
        assert onsager in (False, 'within', 'full', 'tied'), \
            f"onsager must be one of 'none', 'within', 'full', 'tied'; got {onsager!r}"
        self.onsager = onsager

        self.register_buffer('_eye_N', torch.eye(N))
        self.register_buffer('_eye_H', torch.eye(H))

        self.h = nn.Parameter(torch.zeros(N))           # external field
        self.J = nn.Parameter(torch.zeros(N, N))        # pairwise coupling
        self.A = nn.Parameter(torch.empty(N, H))        # hidden → visible
        nn.init.xavier_uniform_(self.A, gain=0.5)
        if self.onsager != 'tied':
            self.U = nn.Parameter(torch.empty(H, N))    # visible → hidden
            nn.init.xavier_uniform_(self.U, gain=0.5)
        # In 'tied' mode U is A.t(); no separate parameter.
        self.W = nn.Parameter(torch.zeros(H, H))        # hidden-hidden
        self.symmetric_W = True

        self.skip_alpha = nn.Parameter(
            torch.ones(T) + 0.05 * torch.randn(T))
        self.skip_w = nn.Parameter(torch.zeros(T, N))
        self.skip_V = nn.Parameter(
            torch.randn(T, N, skip_rank) * 0.001)
        self.field_u = nn.Parameter(torch.zeros(T, N))
        self.field_d = nn.Parameter(torch.zeros(T, N))
        self.field_P = nn.Parameter(
            torch.randn(T, N, field_rank) * 0.001)

        # a_t = 0.5413 because softplus(0.5413) ≈ 1, so τ starts at 1.
        if adaptive_temp:
            self.temp_a = nn.Parameter(torch.full((T,), 0.5413))
            self.temp_b = nn.Parameter(torch.zeros(T))

        self.module_cond = ModuleConditioningMLP(n_modules, T, cond_hidden)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Migrate the older ParameterList layout ('skip_alpha.0', ... vs a
        stacked (T, ...) tensor) and bridge 'tied'/'full' checkpoints."""
        # A 'tied' model has no U parameter, so drop the checkpoint's.
        if self.onsager == 'tied' and 'U' in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != 'U'}
        # Tied checkpoint into a 'full' model: synthesise U = A.t() so the
        # full model starts from the tied solution.
        if (self.onsager == 'full' and 'U' not in state_dict
                and 'A' in state_dict):
            state_dict = dict(state_dict)  # don't mutate caller's dict
            state_dict['U'] = state_dict['A'].t().contiguous()

        if 'skip_alpha.0' in state_dict and 'skip_alpha' not in state_dict:
            per_t_params = {
                'skip_alpha': (self.T,),
                'skip_w':     (self.T, self.N),
                'skip_V':     (self.T, self.N, self.skip_rank),
                'field_u':    (self.T, self.N),
                'field_d':    (self.T, self.N),
                'field_P':    (self.T, self.N, self.field_rank),
            }
            if self.adaptive_temp:
                per_t_params['temp_a'] = (self.T,)
                per_t_params['temp_b'] = (self.T,)

            new_sd = {}
            for k, v in state_dict.items():
                parts = k.split('.')
                if len(parts) >= 2 and parts[0] in per_t_params:
                    continue  # restacked below, not dropped
                new_sd[k] = v

            for prefix in per_t_params:
                tensors = []
                for t in range(self.T):
                    key = f'{prefix}.{t}'
                    if key in state_dict:
                        tensors.append(state_dict[key])
                if tensors:
                    new_sd[prefix] = torch.stack(tensors)

            state_dict = new_sd

        # Squeeze trailing dims: old checkpoints store (T, 1) for params
        # that are now (T,) — e.g. skip_alpha, temp_a, temp_b.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if isinstance(v, torch.Tensor) and k in self.state_dict():
                expected = self.state_dict()[k].shape
                if v.shape != expected and v.squeeze().shape == expected:
                    state_dict[k] = v.squeeze()

        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _Js(self):
        r"""J̃ = (J + J^T)/2 ⊙ (1 - I_N), recomputed every forward rather than
        maintained on ``self.J`` through the optimiser."""
        M = 0.5 * (self.J + self.J.t())
        return M * (1 - self._eye_N)

    def _Ws(self):
        r"""W̃ = (W + W^T)/2 ⊙ (1 - I_H), or W ⊙ (1 - I_H) with the
        ``symmetric_W`` opt-out cleared."""
        M = 0.5 * (self.W + self.W.t()) if self.symmetric_W else self.W
        return M * (1 - self._eye_H)

    def _skip(self, t, x0):
        V = self.skip_V[t]
        return self.skip_alpha[t] + self.skip_w[t] * x0 + (x0 @ V) @ V.t()

    def _field(self, t, x0):
        P = self.field_P[t]
        return self.field_u[t] * x0 + self.field_d[t] + (x0 @ P) @ P.t()

    def _compute_pre_tanh(self, x, z, x0, t, gs, gf,
                          all_field, all_skip, coupling):
        """Pre-tanh visible field, eq. (3); subclasses override to add terms.
        ``coupling`` already has the TAP term subtracted, and ``all_field`` /
        ``all_skip`` are the gate outputs precomputed in forward()."""
        return (self.h
                + (1 + gf * all_field[t]) * coupling
                + gs * all_skip[t] * x0)

    def forward(self, xn, M_mod=None, M_sizes=None, return_trajectory=False):
        r"""Run the T iterations of the class docstring.  ``M_mod``/``M_sizes``
        may be None, for an unconditioned run (γ = 0).  Returns m^T and n^T,
        plus a (T, B, N) stack of m^1 .. m^T under ``return_trajectory``.
        """
        Js = self._Js()
        Ws = self._Ws()
        x0 = xn; x = xn.clone()                                          # m^0 = x₀
        B = x.size(0); z = x.new_zeros(B, self.H)                        # n^0 = 0

        if M_mod is not None:
            present = (x0 + 1.0) / 2.0
            f_obs = (present @ M_mod) / M_sizes.unsqueeze(0).clamp(min=1)
            gs_all, gf_all = self.module_cond(f_obs)
        else:
            gs_all = x.new_zeros(B, self.T)
            gf_all = x.new_zeros(B, self.T)

        if self.adaptive_temp:
            rho = (x0 + 1).sum(-1, keepdim=True) / (2 * self.N)

        # Parameter-only, so hoisted out of the T-loop; 'tied' reuses A².
        if self.onsager:
            J2 = Js * Js
            W2 = Ws * Ws
            if self.onsager in ('full', 'tied'):
                A2 = self.A * self.A
                if self.onsager == 'full':
                    U2 = self.U * self.U

        At = self.A.t()
        # In 'tied' mode Ũ = A^T, so Ut is A itself — aliased, not copied.
        if self.onsager == 'tied':
            Ut = self.A
        else:
            Ut = self.U.t()
        T = self.T

        # x₀ is fixed across iterations while the per-t gate parameters are
        # not, so stacking along t gives all T gate outputs in two bmms.
        x0_T = x0.unsqueeze(0).expand(T, -1, -1)                                # no copy

        proj_V  = torch.bmm(x0_T, self.skip_V)
        lr_skip = torch.bmm(proj_V, self.skip_V.transpose(1, 2))
        all_skip = (self.skip_alpha.view(T, 1, 1)
                    + self.skip_w.unsqueeze(1) * x0_T
                    + lr_skip)                                                  # σ_t(x₀)

        proj_P   = torch.bmm(x0_T, self.field_P)
        lr_field = torch.bmm(proj_P, self.field_P.transpose(1, 2))
        all_field = (self.field_u.unsqueeze(1) * x0_T
                     + self.field_d.unsqueeze(1)
                     + lr_field)                                                # φ_t(x₀)

        traj = [] if return_trajectory else None
        for t in range(T):
            coupling = x @ Js + z @ At                                          # eq. (1)

            if self.onsager:
                coupling = coupling - x * ((1 - x * x) @ J2)                    # eq. (2a)
                if self.onsager in ('full', 'tied'):
                    coupling = coupling - x * ((1 - z * z) @ A2.t())            # eq. (2b)

            gs = (1 + gs_all[:, t].unsqueeze(1))
            gf = (1 + gf_all[:, t].unsqueeze(1))

            pre = self._compute_pre_tanh(x, z, x0, t, gs, gf,
                                         all_field, all_skip, coupling)

            if self.adaptive_temp:
                tau = F.softplus(self.temp_a[t] + self.temp_b[t] * rho)
                pre = pre / tau

            x = torch.tanh(pre)                                                 # eq. (4)
            if return_trajectory:
                traj.append(x)

            z_pre = x @ Ut + z @ Ws                                             # eq. (5)

            if self.onsager:
                z_pre = z_pre - z * ((1 - z * z) @ W2)                          # eq. (6a)
                if self.onsager == 'full':
                    z_pre = z_pre - z * ((1 - x * x) @ U2.t())                  # eq. (6b)
                elif self.onsager == 'tied':
                    z_pre = z_pre - z * ((1 - x * x) @ A2)                      # eq. (6c)
            z = torch.tanh(z_pre)                                               # eq. (7)

        self.last_z = z                                                         # external aux heads
        if return_trajectory:
            return x, z, torch.stack(traj, dim=0)
        return x, z

    def ising_params(self):        return [self.h, self.J]
    def hidden_params(self):
        params = [self.W, self.A]
        if self.onsager != 'tied':
            params.append(self.U)
        return params
    def scalar_skip_params(self):  return [self.skip_alpha]
    def diag_gate_params(self):    return [self.skip_w, self.field_u, self.field_d]
    def lr_skip_params(self):      return [self.skip_V]
    def lr_field_params(self):     return [self.field_P]
    def cond_params(self):         return list(self.module_cond.parameters())
    def temp_params(self):
        if not self.adaptive_temp:
            return []
        return [self.temp_a, self.temp_b]

    def set_trainable(self, ising=True, hidden=True, scalar_skip=True,
                      diag=True, lrs=True, lrf=True, cond=True, **_):
        """Freeze/unfreeze parameter groups for the staged curriculum:
        ``ising`` (h, J), ``hidden`` (A, U, W), ``scalar_skip``, ``diag``
        (skip_w, field_u, field_d), ``lrs`` (skip_V), ``lrf`` (field_P),
        ``cond``.  Temperature params stay trainable."""
        for p in self.ising_params():        p.requires_grad = ising
        for p in self.hidden_params():       p.requires_grad = hidden
        for p in self.scalar_skip_params():  p.requires_grad = scalar_skip
        for p in self.diag_gate_params():    p.requires_grad = diag
        for p in self.lr_skip_params():      p.requires_grad = lrs
        for p in self.lr_field_params():     p.requires_grad = lrf
        for p in self.cond_params():         p.requires_grad = cond
        for p in self.temp_params():         p.requires_grad = True

    def trainable_str(self):
        gs = [('ising', self.ising_params()), ('hidden', self.hidden_params()),
              ('sSkip', self.scalar_skip_params()), ('diag', self.diag_gate_params()),
              ('lrS', self.lr_skip_params()), ('lrF', self.lr_field_params()),
              ('cond', self.cond_params())]
        parts = [f"{n}={sum(p.numel() for p in ps):,}"
                 for n, ps in gs if ps and any(p.requires_grad for p in ps)]
        tot = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{tot:,} ({', '.join(parts)})"

    def make_optimizer(self, base_lr, wd, j_lr_frac=0.1, j_wd=None):
        """Optimizer with differential LR and weight decay for J."""
        groups = []
        j_ps = [p for p in [self.J] if p.requires_grad]
        if j_ps:
            groups.append({'params': j_ps, 'lr': base_lr * j_lr_frac,
                           'weight_decay': j_wd if j_wd is not None else wd})
        other = [p for p in self.parameters() if p.requires_grad and p is not self.J]
        if other:
            groups.append({'params': other, 'lr': base_lr})
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

    def masked_forward(self, x_clean, mp=0.15, return_z=False):
        r"""Single-step pseudolikelihood forward, run before the T-step
        curriculum to learn (h, J, A, U)::

            ctx_i = x_clean_i · (1 - mask_i),  mask_i ∼ Bernoulli(mp)
            z     = tanh(ctx Ũ)
            h_i   = h_i + Σ_j J̃_ij ctx_j + Σ_k A_ik z_k    (eq. M.1)

        Masking to 0 rather than ±1 keeps a masked position from contributing
        evidence either way.  τ_0 is the only temperature used.
        """
        Js = self._Js()
        mask = torch.rand_like(x_clean) < mp
        ctx = x_clean * (~mask).float()
        Ut = self.A if self.onsager == 'tied' else self.U.t()
        z = torch.tanh(ctx @ Ut)
        logits = self.h + ctx @ Js + z @ self.A.t()                         # eq. (M.1)
        if self.adaptive_temp:
            rho = (x_clean + 1).sum(-1, keepdim=True) / (2 * self.N)
            tau = F.softplus(self.temp_a[0] + self.temp_b[0] * rho)
            logits = logits / tau
        if return_z:
            return logits[mask], x_clean[mask], mask.sum().item(), z
        return logits[mask], x_clean[mask], mask.sum().item()

    @torch.no_grad()
    def clamp_norms(self, max_norm=25.0):
        """Clamp A and (if present) U Frobenius norms to max_norm."""
        ps = [self.A] if self.onsager == 'tied' else [self.A, self.U]
        for p in ps:
            n = p.data.norm()
            if n > max_norm:
                p.data.mul_(max_norm / n)

    def norm_str(self):
        cw = sum(p.data.norm().item() for p in self.cond_params())
        u_str = (f"U={self.U.data.norm():.1f} " if self.onsager != 'tied'
                 else "")
        s = (f"J={self.J.data.norm():.1f} h={self.h.data.norm():.1f} "
             f"A={self.A.data.norm():.1f} {u_str}W={self.W.data.norm():.1f} "
             f"a={self.skip_alpha.data.mean():.2f} "
             f"V={self.skip_V.data.norm(dim=(1,2)).mean():.3f} "
             f"P={self.field_P.data.norm(dim=(1,2)).mean():.3f} "
             f"cond={cw:.1f}")
        if self.adaptive_temp:
            tau_03 = F.softplus(self.temp_a + self.temp_b * 0.3).mean().item()
            tau_07 = F.softplus(self.temp_a + self.temp_b * 0.7).mean().item()
            s += f" tau(.3)={tau_03:.3f} tau(.7)={tau_07:.3f}"
        return s

    def config_str(self):
        tot = sum(p.numel() for p in self.parameters())
        cn = sum(p.numel() for p in self.cond_params())
        s = (f"ModuleConditionedDenoiser N={self.N} H={self.H} T={self.T} "
             f"sK={self.skip_rank} fK={self.field_rank} "
             f"adaptive_temp={self.adaptive_temp} total={tot:,} (cond={cn:,})")
        if self.onsager:
            s += f" onsager={self.onsager}"
        return s

    def gate_diagnostics(self):
        lines = []
        for t in range(self.T):
            s = (f"    t={t}: a={self.skip_alpha.data[t].item():.3f} "
                 f"|w|={self.skip_w.data[t].norm().item():.3f} "
                 f"|V|={self.skip_V.data[t].norm().item():.3f} | "
                 f"|u|={self.field_u.data[t].norm().item():.3f} "
                 f"d={self.field_d.data[t].mean().item():+.4f} "
                 f"|P|={self.field_P.data[t].norm().item():.3f}")
            if self.adaptive_temp:
                tau_03 = F.softplus(self.temp_a[t] + self.temp_b[t] * 0.3).item()
                tau_07 = F.softplus(self.temp_a[t] + self.temp_b[t] * 0.7).item()
                s += f"  tau(0.3)={tau_03:.3f} tau(0.7)={tau_07:.3f}"
            lines.append(s)
        return '\n'.join(lines)


class NoHiddenDenoiser(nn.Module):
    r"""Module-conditioned mean-field denoiser without hidden spins.

    The only state is the visible magnetisation m = E[x] ∈ (-1, +1)^N, one
    entry per COG family.  From m^0 = x₀::

        c^t     = J̃ m^t  -  m^t ⊙ ((1 - m^t²) J̃²)       (1), (2)
        pre^t   = h + (1 + (1 + γ_field_t) φ_t(x₀)) ⊙ c^t
                    + (1 + γ_skip_t) σ_t(x₀) ⊙ x₀        (3)
        m^{t+1} = tanh(pre^t / τ_t)                      (4)

    Symbols are as in ``ModuleConditionedDenoiser``, with the gates::

        σ_t(x₀)   skip gate:  α_t + w_t ⊙ x₀ + x₀ V_t V_t^T
        φ_t(x₀)   field gate: u_t ⊙ x₀ + d_t + x₀ P_t P_t^T

    Eq. (2) is the second-order (Onsager) correction to the pairwise term; it
    needs soft marginals (1 - m² > 0), which the unrolled tanh keeps alive.
    J̃² is elementwise, so it costs one (B, N)·(N, N) matmul, as does (1).
    ``onsager`` is a plain bool here; the ``'within'/'full'/'tied'`` modes
    belong to ``ModuleConditionedDenoiser``.
    """

    def __init__(self, N, T=12, skip_rank=32, field_rank=16,
                 n_modules=419, cond_hidden=128, adaptive_temp=False,
                 onsager=True):
        super().__init__()
        # T is the number of unrolled iterations, not a temperature.
        self.N = N; self.T = T
        self.skip_rank = skip_rank; self.field_rank = field_rank
        self.adaptive_temp = adaptive_temp
        self.onsager = onsager

        self.register_buffer('_eye_N', torch.eye(N))

        self.h = nn.Parameter(torch.zeros(N))
        self.J = nn.Parameter(torch.zeros(N, N))

        self.skip_alpha = nn.Parameter(
            torch.ones(T) + 0.05 * torch.randn(T))
        self.skip_w = nn.Parameter(torch.zeros(T, N))
        self.skip_V = nn.Parameter(
            torch.randn(T, N, skip_rank) * 0.001)
        self.field_u = nn.Parameter(torch.zeros(T, N))
        self.field_d = nn.Parameter(torch.zeros(T, N))
        self.field_P = nn.Parameter(
            torch.randn(T, N, field_rank) * 0.001)

        if adaptive_temp:
            self.temp_a = nn.Parameter(torch.full((T,), 0.5413))
            self.temp_b = nn.Parameter(torch.zeros(T))

        self.module_cond = ModuleConditioningMLP(n_modules, T, cond_hidden)

        # Inference-time coupling controls, set post-load by
        # set_coupling_controls(): coupling_shrink (N,) multiplies J~ by the
        # outer product s_i s_j, gmass_kbar scales the pairwise drive by
        # sqrt(kbar/k_n) per genome.  With both None the forward matches the
        # trained model exactly.
        self.coupling_shrink = None
        self.gmass_kbar = None

        # Platt affine on the output logit, separately for the gain regime
        # (input absent) and the loss regime (input present):
        # logit' = (1+da_s) * logit + db_s, stored as
        # [da_gain, db_gain, da_loss, db_loss]; init 0 => identity.
        self.gainloss_cal = nn.Parameter(torch.zeros(4))
        self.use_gainloss_cal = False

    def calibrate_p(self, m, x_input, eps=1e-4):
        """Per-state recalibrated presence probability p in (0,1) from the
        magnetisation m and the noisy input.  Gain regime = input absent
        (x<0), loss regime = input present (x>0)."""
        logit = 2.0 * torch.atanh(m.clamp(-1.0 + eps, 1.0 - eps))   # logit of (m+1)/2
        is_loss = (x_input > 0).to(logit.dtype)
        c = self.gainloss_cal
        a = 1.0 + (c[0] * (1.0 - is_loss) + c[2] * is_loss)
        b = c[1] * (1.0 - is_loss) + c[3] * is_loss
        return torch.sigmoid(a * logit + b)

    def _Js(self):
        r"""J̃ = (J + J^T)/2 ⊙ (1 - I_N); the diagonal is absorbed into h.

        The projection is recomputed each forward rather than maintained on
        ``self.J`` through the optimiser.  This same J̃ is scored by the Besag
        pseudo-likelihood term of the ELBO loss in ``train_denovo.py``.  With
        ``coupling_shrink`` set, J̃ is further multiplied by the frozen rank-1
        mask s_i s_j.
        """
        M = (self.J + self.J.t()) * 0.5
        M = M * (1 - self._eye_N)
        if self.coupling_shrink is not None:
            s = self.coupling_shrink
            M = M * (s.unsqueeze(0) * s.unsqueeze(1))
        return M

    def _compute_pre_tanh(self, x, z, x0, t, gs, gf,
                          all_field, all_skip, coupling):
        """Pre-tanh visible field, eq. (3); subclasses override to add terms.
        ``z`` is always None, and stays in the signature to match
        ``HigherOrderDenoiser._compute_pre_tanh``."""
        return (self.h
                + (1 + gf * all_field[t]) * coupling
                + gs * all_skip[t] * x0)

    def forward(self, xn, M_mod=None, M_sizes=None, return_trajectory=False):
        r"""Run the T iterations of the class docstring, which describe the
        default path: with ``gmass_kbar`` set, the coupling of eq. (1)-(2) is
        further scaled by sqrt(kbar / k_n).  Returns m^T and None (the hidden
        slot), plus a (T, B, N) stack of m^1 .. m^T under ``return_trajectory``.
        """
        Js = self._Js()
        x0 = xn; x = xn.clone()                          # m^0 = x₀
        B = x.size(0); T = self.T

        if M_mod is not None:
            present = (x0 + 1.0) * 0.5
            f_obs = (present @ M_mod) / M_sizes.unsqueeze(0).clamp(min=1)
            gs_all, gf_all = self.module_cond(f_obs)
        else:
            gs_all = x.new_zeros(B, T)
            gf_all = x.new_zeros(B, T)

        if self.adaptive_temp:
            rho = (x0 + 1).sum(-1, keepdim=True) / (2 * self.N)

        # Genome-mass normalisation g(k_n) = sqrt(kbar / k_n), k_n = #present
        # genes in x₀ (inference control, default off).
        gmass = None
        if self.gmass_kbar is not None:
            kn = (x0 + 1).sum(-1, keepdim=True) * 0.5
            gmass = torch.sqrt(self.gmass_kbar / kn.clamp(min=1.0))

        # J̃ depends only on parameters, so its square is hoisted out of the loop.
        if self.onsager:
            J2 = Js * Js

        # x₀ is fixed across iterations while the per-t gate parameters are
        # not, so stacking along t gives all T gate outputs in two bmms.
        x0_T = x0.unsqueeze(0).expand(T, -1, -1)                                # no copy

        proj_V  = torch.bmm(x0_T, self.skip_V)
        lr_skip = torch.bmm(proj_V, self.skip_V.transpose(1, 2))
        all_skip = (self.skip_alpha.view(T, 1, 1)
                    + self.skip_w.unsqueeze(1) * x0_T
                    + lr_skip)                                                  # σ_t(x₀)

        proj_P   = torch.bmm(x0_T, self.field_P)
        lr_field = torch.bmm(proj_P, self.field_P.transpose(1, 2))
        all_field = (self.field_u.unsqueeze(1) * x0_T
                     + self.field_d.unsqueeze(1)
                     + lr_field)                                                # φ_t(x₀)

        traj = [] if return_trajectory else None
        for t in range(T):
            coupling = x @ Js                                                   # eq. (1)

            if self.onsager:
                coupling = coupling - x * ((1 - x * x) @ J2)                    # eq. (2)

            # The higher-order Δ of the subclass is left unscaled.
            if gmass is not None:
                coupling = coupling * gmass

            gs = 1 + gs_all[:, t].unsqueeze(1)
            gf = 1 + gf_all[:, t].unsqueeze(1)

            pre = self._compute_pre_tanh(x, None, x0, t, gs, gf,
                                          all_field, all_skip, coupling)

            if self.adaptive_temp:
                tau = F.softplus(self.temp_a[t] + self.temp_b[t] * rho)
                pre = pre / tau

            x = torch.tanh(pre)                                                 # eq. (4)
            if return_trajectory:
                traj.append(x)

        self.last_z = None
        if return_trajectory:
            return x, None, torch.stack(traj, dim=0)
        return x, None

    def ising_params(self):        return [self.h, self.J]
    def hidden_params(self):       return []
    def scalar_skip_params(self):  return [self.skip_alpha]
    def diag_gate_params(self):    return [self.skip_w, self.field_u, self.field_d]
    def lr_skip_params(self):      return [self.skip_V]
    def lr_field_params(self):     return [self.field_P]
    def cond_params(self):         return list(self.module_cond.parameters())
    def temp_params(self):
        if not self.adaptive_temp:
            return []
        return [self.temp_a, self.temp_b]

    def set_trainable(self, ising=True, hidden=True, scalar_skip=True,
                      diag=True, lrs=True, lrf=True, cond=True):
        """Freeze/unfreeze parameter groups; ``hidden`` is ignored here."""
        for p in self.ising_params():        p.requires_grad = ising
        for p in self.scalar_skip_params():  p.requires_grad = scalar_skip
        for p in self.diag_gate_params():    p.requires_grad = diag
        for p in self.lr_skip_params():      p.requires_grad = lrs
        for p in self.lr_field_params():     p.requires_grad = lrf
        for p in self.cond_params():         p.requires_grad = cond
        for p in self.temp_params():         p.requires_grad = True

    def trainable_str(self):
        gs = [('ising', self.ising_params()),
              ('sSkip', self.scalar_skip_params()), ('diag', self.diag_gate_params()),
              ('lrS', self.lr_skip_params()), ('lrF', self.lr_field_params()),
              ('cond', self.cond_params())]
        parts = [f"{n}={sum(p.numel() for p in ps):,}"
                 for n, ps in gs if ps and any(p.requires_grad for p in ps)]
        tot = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{tot:,} ({', '.join(parts)})"

    def make_optimizer(self, base_lr, wd, j_lr_frac=0.1, j_wd=None):
        groups = []
        j_ps = [p for p in [self.J] if p.requires_grad]
        if j_ps:
            groups.append({'params': j_ps, 'lr': base_lr * j_lr_frac,
                           'weight_decay': j_wd if j_wd is not None else wd})
        other = [p for p in self.parameters() if p.requires_grad and p is not self.J]
        if other:
            groups.append({'params': other, 'lr': base_lr})
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

    def masked_forward(self, x_clean, mp=0.15, return_z=False):
        """Single-step pseudolikelihood forward, learning (h, J) before the
        T-step curriculum.  Unlike the hidden-state version, the TAP correction
        is applied here too, to the masked context."""
        Js = self._Js()
        mask = torch.rand_like(x_clean) < mp
        ctx = x_clean * (~mask).float()
        logits = self.h + ctx @ Js
        if self.onsager:
            logits = logits - ctx * ((1 - ctx * ctx) @ (Js * Js))
        if self.adaptive_temp:
            rho = (x_clean + 1).sum(-1, keepdim=True) / (2 * self.N)
            tau = F.softplus(self.temp_a[0] + self.temp_b[0] * rho)
            logits = logits / tau
        if return_z:
            return logits[mask], x_clean[mask], mask.sum().item(), None
        return logits[mask], x_clean[mask], mask.sum().item()

    @torch.no_grad()
    def clamp_norms(self, max_norm=25.0):
        """No-op (no hidden params to clamp)."""
        pass

    def norm_str(self):
        cw = sum(p.data.norm().item() for p in self.cond_params())
        s = (f"J={self.J.data.norm():.1f} h={self.h.data.norm():.1f} "
             f"a={self.skip_alpha.data.mean():.2f} "
             f"V={self.skip_V.data.norm(dim=(1,2)).mean():.3f} "
             f"P={self.field_P.data.norm(dim=(1,2)).mean():.3f} "
             f"cond={cw:.1f}")
        if self.adaptive_temp:
            tau_03 = F.softplus(self.temp_a + self.temp_b * 0.3).mean().item()
            tau_07 = F.softplus(self.temp_a + self.temp_b * 0.7).mean().item()
            s += f" tau(.3)={tau_03:.3f} tau(.7)={tau_07:.3f}"
        return s

    def config_str(self):
        tot = sum(p.numel() for p in self.parameters())
        cn = sum(p.numel() for p in self.cond_params())
        return (f"NoHiddenDenoiser N={self.N} T={self.T} "
                f"sK={self.skip_rank} fK={self.field_rank} "
                f"onsager={self.onsager} adaptive_temp={self.adaptive_temp} "
                f"total={tot:,} (cond={cn:,})")

    def gate_diagnostics(self):
        lines = []
        for t in range(self.T):
            s = (f"    t={t}: a={self.skip_alpha.data[t].item():.3f} "
                 f"|w|={self.skip_w.data[t].norm().item():.3f} "
                 f"|V|={self.skip_V.data[t].norm().item():.3f} | "
                 f"|u|={self.field_u.data[t].norm().item():.3f} "
                 f"d={self.field_d.data[t].mean().item():+.4f} "
                 f"|P|={self.field_P.data[t].norm().item():.3f}")
            if self.adaptive_temp:
                tau_03 = F.softplus(self.temp_a[t] + self.temp_b[t] * 0.3).item()
                tau_07 = F.softplus(self.temp_a[t] + self.temp_b[t] * 0.7).item()
                s += f"  tau(0.3)={tau_03:.3f} tau(0.7)={tau_07:.3f}"
            lines.append(s)
        return '\n'.join(lines)


# nn.TransformerEncoderLayer does not reliably take its "fast path" to SDPA,
# and materialising the full (B, nhead, N, N) attention matrix costs ~367 MB
# per sample per layer in bf16 at the default N = 4789.  Calling
# F.scaled_dot_product_attention directly gets flash attention on Ampere+,
# O(N·d) instead of O(N²): ~5 MB per sample per layer at N = 4789, d = 512.

class MemoryEfficientEncoderLayer(nn.Module):
    """Pre-norm encoder layer with explicit SDPA, interface-compatible with
    nn.TransformerEncoderLayer(batch_first=True, norm_first=True)."""

    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation='gelu'):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout_p = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)     # fused QKV
        self.attn_out_proj = nn.Linear(d_model, d_model)

        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.ff_dropout = nn.Dropout(dropout)
        self.resid_dropout1 = nn.Dropout(dropout)
        self.resid_dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = F.gelu if activation == 'gelu' else F.relu

    def _attn(self, x):
        r"""Multi-head self-attention on a fused QKV projection.  With no mask
        and no attention bias, SDPA takes the flash kernel."""
        B, N, D = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()            # (3, B, nhead, N, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dp = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        out = out.transpose(1, 2).contiguous().view(B, N, D)     # concat heads
        return self.attn_out_proj(out)

    def forward(self, x):
        r"""Pre-norm residual block.  Residual dropout follows each sub-layer;
        attention-probability dropout is SDPA's own ``dropout_p``."""
        x = x + self.resid_dropout1(self._attn(self.norm1(x)))
        x = x + self.resid_dropout2(
            self.ff2(self.ff_dropout(self.activation(self.ff1(self.norm2(x))))))
        return x


class MemoryEfficientEncoder(nn.Module):
    """Stack of MemoryEfficientEncoderLayer.  ``.layers`` is a ModuleList, so
    the checkpointing loops in ``_run_encoder`` can iterate it."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryEfficientEncoderLayer(d_model, nhead, dim_feedforward,
                                         dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HigherOrderModule(nn.Module):
    """Attention-parameterised higher-order field Δ(x, z, t): reads the
    visible state, the hidden state and the iteration index, and returns an
    additive per-gene field Δ ∈ ℝ^N.

    The output head has zero weight and zero bias at init, so Δ ≡ 0 and the
    model starts as its pairwise parent.  Weights are shared across the T
    iterations, with t entering as a learned embedding.
    """

    def __init__(self, N, H, T, d_model=128, nhead=4, n_layers=1,
                 dim_feedforward=512, dropout=0.1, use_checkpoint=True,
                 no_hidden=False):
        super().__init__()
        self.N = N
        self.H = H
        self.T = T
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        self.no_hidden = no_hidden

        # Per-gene identity embedding (no ordering is assumed).
        self.gene_embedding = nn.Parameter(torch.randn(N, d_model) * 0.02)

        self.x_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Skipped entirely in no-hidden mode, so no attn.z_proj.* keys appear
        # in the state_dict.
        if not no_hidden:
            self.z_proj = nn.Sequential(
                nn.Linear(H, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.t_embedding = nn.Embedding(T, d_model)

        self.input_norm = nn.LayerNorm(d_model)

        self.encoder = MemoryEfficientEncoder(
            d_model=d_model, nhead=nhead, num_layers=n_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)

        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # Xavier over every Linear except the Δ head, whose zero init must
        # survive this sweep.
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.output_head:
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _run_encoder(self, h):
        """Optional per-layer checkpointing, to cap activation memory over the
        T shared-weight passes."""
        if not (self.use_checkpoint and self.training and h.requires_grad):
            return self.encoder(h)
        for layer in self.encoder.layers:
            h = ckpt(layer, h, use_reentrant=False)
        return h

    def forward(self, x, z, t_idx):
        r"""Compute the additive higher-order field Δ(x, z, t)::

            h_i^0 = gene_emb_i + x_proj(x_i) + broadcast(z_proj(z)) + t_emb_t
            h^L   = encoder(LayerNorm(h^0))
            Δ_i   = output_head(LayerNorm(h^L)_i)

        ``z`` is None in no-hidden mode; ``t_idx`` is an int in [0, T).
        """
        gene_emb = self.gene_embedding.unsqueeze(0)
        val_emb = self.x_proj(x.unsqueeze(-1))
        t_emb = self.t_embedding.weight[t_idx].view(1, 1, -1)
        if self.no_hidden:
            h = gene_emb + val_emb + t_emb
        else:
            z_emb = self.z_proj(z).unsqueeze(1)
            h = gene_emb + val_emb + z_emb + t_emb

        h = self.input_norm(h)
        h = self._run_encoder(h)
        delta = self.output_head(self.output_norm(h)).squeeze(-1)
        return delta


class HigherOrderDenoiser(ModuleConditionedDenoiser):
    r"""ModuleConditionedDenoiser plus an attention-parameterised higher-order
    field Δ(m, n, t).

    At fixed x₀ the parent's pairwise + TAP update is bilinear in the iterated
    state (m, n); Δ = HigherOrderModule(m, n, t) is not.  It enters eq. (3) as
    an additive residual, outside the field gate; the attention reads z as a
    reduced-rank summary.
    """

    def __init__(self, N, H=1000, T=12, skip_rank=32, field_rank=16,
                 n_modules=419, cond_hidden=128, adaptive_temp=False,
                 onsager=False,
                 attn_d_model=128, attn_nhead=4, attn_n_layers=1,
                 attn_dim_feedforward=512, attn_dropout=0.1,
                 attn_use_checkpoint=True):
        super().__init__(N=N, H=H, T=T, skip_rank=skip_rank,
                         field_rank=field_rank, n_modules=n_modules,
                         cond_hidden=cond_hidden,
                         adaptive_temp=adaptive_temp, onsager=onsager)
        self.attn = HigherOrderModule(
            N=N, H=H, T=T,
            d_model=attn_d_model, nhead=attn_nhead,
            n_layers=attn_n_layers,
            dim_feedforward=attn_dim_feedforward,
            dropout=attn_dropout,
            use_checkpoint=attn_use_checkpoint,
        )
        # Default attention LR fraction; training scripts may overwrite.
        self.attn_lr_frac = 0.3

    @torch._dynamo.disable
    def _attn_delta(self, x, z, t):
        """Compute attention Δ outside any dynamo graph: torch.utils.checkpoint
        does not compose with dynamo's DDP-aware graph splitting, so it and the
        attention stay eager while the pairwise loop stays compiled.

        ``x`` and ``z`` are detached so Δ's backward does not run back through
        the unrolled iteration into (J, h, A, U, W).  Without it an unfrozen J
        takes two autograd-hook fires per backward — one from the bilinear
        coupling, one via attn(x=m^t) — and DDP's reducer raises "marked as
        ready twice".
        """
        x_in = x.detach()
        z_in = z.detach() if z is not None else None
        if self.attn.use_checkpoint and self.training:
            return ckpt(self.attn, x_in, z_in, t, use_reentrant=False)
        return self.attn(x_in, z_in, t)

    def _compute_pre_tanh(self, x, z, x0, t, gs, gf,
                          all_field, all_skip, coupling):
        """Parent pre-tanh plus Δ, added outside the field gate."""
        pre = super()._compute_pre_tanh(x, z, x0, t, gs, gf,
                                         all_field, all_skip, coupling)
        delta = self._attn_delta(x, z, t)
        pre = pre + delta
        return pre

    def attn_params(self):
        return list(self.attn.parameters())

    def set_trainable(self, ising=True, hidden=True, scalar_skip=True,
                      diag=True, lrs=True, lrf=True, cond=True,
                      attn=True, **_):
        super().set_trainable(ising=ising, hidden=hidden,
                              scalar_skip=scalar_skip, diag=diag,
                              lrs=lrs, lrf=lrf, cond=cond)
        for p in self.attn_params():
            p.requires_grad = attn

    def make_optimizer(self, base_lr, wd, j_lr_frac=0.1, j_wd=None):
        """AdamW: J at base_lr * j_lr_frac, attn.* at base_lr * attn_lr_frac,
        everything else at base_lr."""
        attn_lr_frac = getattr(self, 'attn_lr_frac', 0.3)
        groups = []

        j_ps = [p for p in [self.J] if p.requires_grad]
        if j_ps:
            groups.append({
                'params': j_ps,
                'lr': base_lr * j_lr_frac,
                'weight_decay': j_wd if j_wd is not None else wd,
            })

        attn_ids = {id(p) for p in self.attn_params()}
        attn_ps = [p for p in self.attn_params() if p.requires_grad]
        if attn_ps:
            groups.append({
                'params': attn_ps,
                'lr': base_lr * attn_lr_frac,
                'weight_decay': wd,
            })

        other = [p for p in self.parameters()
                 if p.requires_grad
                 and p is not self.J
                 and id(p) not in attn_ids]
        if other:
            groups.append({'params': other, 'lr': base_lr})

        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

    def config_str(self):
        s = super().config_str()
        a = sum(p.numel() for p in self.attn_params())
        s += (f" | attn d={self.attn.d_model}"
              f" L={self.attn.n_layers} params={a:,}")
        return s

    def norm_str(self):
        s = super().norm_str()
        s += f" attn_out={self.attn.output_head.weight.data.norm():.3f}"
        return s

    @torch.no_grad()
    def delta_norm(self, x, z_init=None):
        """Mean per-gene L2 of Δ(x, z_init, 0).  This class has no
        ``delta_sq``, so the explicit λ_Δ E‖Δ‖² penalty is unavailable; Δ is
        held down by the zero-init head, the detached attention input,
        ``attn_lr_frac`` and weight decay."""
        if z_init is None:
            z_init = x.new_zeros(x.size(0), self.H)
        delta = self.attn(x, z_init, 0)
        return (delta.norm(dim=-1) / (self.N ** 0.5)).mean().item()


class NoHiddenHigherOrderDenoiser(NoHiddenDenoiser):
    r"""NoHiddenDenoiser plus an attention-parameterised higher-order field
    Δ(x, t): ``HigherOrderDenoiser`` without hidden spins, so the attention
    reads only x and t.  The inherited ``module_cond`` MLP still uses M_mod
    for the (γ_skip, γ_field) gates; the attention does not.
    """

    def __init__(self, N, T=12, skip_rank=32, field_rank=16,
                 n_modules=419, cond_hidden=128, adaptive_temp=False,
                 onsager=False,
                 attn_d_model=128, attn_nhead=4, attn_n_layers=1,
                 attn_dim_feedforward=512, attn_dropout=0.1,
                 attn_use_checkpoint=True):
        super().__init__(N=N, T=T, skip_rank=skip_rank,
                         field_rank=field_rank, n_modules=n_modules,
                         cond_hidden=cond_hidden,
                         adaptive_temp=adaptive_temp, onsager=onsager)
        self.attn = HigherOrderModule(
            N=N, H=1, T=T,                       # H is unused when no_hidden=True
            d_model=attn_d_model, nhead=attn_nhead,
            n_layers=attn_n_layers,
            dim_feedforward=attn_dim_feedforward,
            dropout=attn_dropout,
            use_checkpoint=attn_use_checkpoint,
            no_hidden=True,
        )
        self.attn_lr_frac = 0.3

    @torch._dynamo.disable
    def _attn_delta(self, x, t):
        """Compute attention Δ outside any dynamo graph.  ``x`` is detached;
        see :meth:`HigherOrderDenoiser._attn_delta`.
        """
        x_in = x.detach()
        if self.attn.use_checkpoint and self.training:
            return ckpt(self.attn, x_in, None, t, use_reentrant=False)
        return self.attn(x_in, None, t)

    def _compute_pre_tanh(self, x, z, x0, t, gs, gf,
                          all_field, all_skip, coupling):
        """Parent pre-tanh plus Δ, added outside the field gate."""
        pre = super()._compute_pre_tanh(x, z, x0, t, gs, gf,
                                         all_field, all_skip, coupling)
        return pre + self._attn_delta(x, t)

    def attn_params(self):
        return list(self.attn.parameters())

    def set_trainable(self, ising=True, hidden=True, scalar_skip=True,
                      diag=True, lrs=True, lrf=True, cond=True,
                      attn=True, **_):
        # The parent accepts and ignores `hidden`.
        super().set_trainable(ising=ising, hidden=hidden,
                              scalar_skip=scalar_skip, diag=diag,
                              lrs=lrs, lrf=lrf, cond=cond)
        for p in self.attn_params():
            p.requires_grad = attn

    def make_optimizer(self, base_lr, wd, j_lr_frac=0.1, j_wd=None):
        groups = []
        j_ps = [p for p in [self.J] if p.requires_grad]
        if j_ps:
            groups.append({'params': j_ps, 'lr': base_lr * j_lr_frac,
                           'weight_decay': j_wd if j_wd is not None else wd})
        attn_ps = [p for p in self.attn_params() if p.requires_grad]
        attn_ids = {id(p) for p in attn_ps}
        if attn_ps:
            groups.append({'params': attn_ps,
                           'lr': base_lr * self.attn_lr_frac})
        other = [p for p in self.parameters()
                 if p.requires_grad
                 and p is not self.J
                 and id(p) not in attn_ids]
        if other:
            groups.append({'params': other, 'lr': base_lr})
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

    def config_str(self):
        s = super().config_str()
        a = sum(p.numel() for p in self.attn_params())
        s += (f" | attn d={self.attn.d_model}"
              f" L={self.attn.n_layers} no_hidden=True params={a:,}")
        return s

    def norm_str(self):
        s = super().norm_str()
        s += f" attn_out={self.attn.output_head.weight.data.norm():.3f}"
        return s

    @torch.no_grad()
    def delta_norm(self, x):
        """Mean per-gene L2 of Δ(x, 0); the penalty itself is ``delta_sq``."""
        delta = self.attn(x, None, 0)
        return (delta.norm(dim=-1) / (self.N ** 0.5)).mean().item()

    def delta_sq(self, x, t=0):
        """Mean Δ(x, t)²: the λ_Δ E‖Δ‖² penalty, evaluated once at step t
        rather than summed over the T steps.  ``x`` is detached; see
        :meth:`HigherOrderDenoiser._attn_delta`."""
        delta = self.attn(x.detach(), None, t)
        return (delta * delta).mean()


# One implementation, three behaviours selected by hyperparameter:
#   T=1, no gates, no adaptive_temp  -> single-pass Set Transformer
#   T>1 + gates                      -> iterative Set Transformer (shared
#                                       encoder weights, per-iteration skip)
#   T>1 + gates + adaptive_temp      -> hybrid Ising-Transformer (the above
#                                       plus a rho-conditioned tau_t)


class SetTransformerDenoiser(nn.Module):
    """Set-transformer genome denoiser: N per-gene tokens in ℝ^d, refined by
    T passes of one shared encoder::

        h_i^0   = gene_embedding[i] + value_proj(x_0_i) + module_cond(f_obs)
        h^{t+1} = encoder(h^t) + α_t · h^0
        y_i     = tanh(head(LayerNorm(h_i^T)) / τ)

    value_proj is a 2-layer MLP over the ±1 value, replaced by mask_token at
    MLM-masked positions; module conditioning enters as an additive per-token
    bias broadcast over N, not a multiplicative gate; α_t ∈ ℝ is learned,
    initialised at 1; τ = softplus(a + b·ρ) if adaptive_temp else 1, for ρ the
    gene density of the input.  forward returns (y, None), like the Ising
    denoisers' (x, z).
    """

    def __init__(self, N=4789, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 T=8, n_modules=419, cond_hidden=128,
                 adaptive_temp=True, use_checkpoint=True,
                 cond_refresh_every=3, cond_refresh_detach=True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.N = N
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.T = T
        self.n_modules = n_modules
        self.adaptive_temp = adaptive_temp
        self.use_checkpoint = use_checkpoint
        # cond_refresh_every = 0 disables the mid-iteration conditioning
        # refresh, leaving f_obs from x^0 only; cond_refresh_detach keeps
        # gradient from flowing back through the soft x^t estimate.
        self.cond_refresh_every = cond_refresh_every
        self.cond_refresh_detach = cond_refresh_detach

        # Per-gene identity embedding (no ordering is assumed).
        self.gene_embedding = nn.Parameter(torch.randn(N, d_model) * 0.02)

        self.value_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Replaces value_proj at MLM-masked positions.
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        self.input_norm = nn.LayerNorm(d_model)

        # Module conditioning: f_obs → a per-genome bias added to every token.
        # The final layer is zero-init, so the model starts unconditioned.
        self.module_cond = nn.Sequential(
            nn.Linear(n_modules, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, d_model),
        )
        with torch.no_grad():
            nn.init.zeros_(self.module_cond[-1].weight)
            nn.init.zeros_(self.module_cond[-1].bias)

        # Encoder weights are shared across the T iterations.
        self.encoder = MemoryEfficientEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)

        self.skip_gates = nn.Parameter(torch.ones(T))            # α_t
        if adaptive_temp:
            self.temp_a = nn.Parameter(torch.full((T,), 0.5413))
            self.temp_b = nn.Parameter(torch.zeros(T))

        # Small Xavier gain rather than a zero init: the output starts near
        # zero, but ∂logit/∂h = W_head stays non-zero so the encoder still
        # gets gradient on the first step.  Only the bias is zero.
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.02)
        nn.init.zeros_(self.output_head.bias)

        # Xavier elsewhere; the two heads above keep their own init.
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.output_head \
               and m is not self.module_cond[-1]:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_h_static(self, x, mask=None):
        """gene_embedding + value_proj(x), with mask_token substituted at
        masked positions.  Left un-normed and unconditioned: invariant across
        the T iterations, so the mid-iteration refresh reuses it."""
        val_emb = self.value_proj(x.unsqueeze(-1))
        if mask is not None:
            val_emb = torch.where(mask.unsqueeze(-1),
                                   self.mask_token.to(val_emb.dtype)
                                       .view(1, 1, -1).expand_as(val_emb),
                                   val_emb)
        return self.gene_embedding.unsqueeze(0) + val_emb

    def _compute_cond_bias(self, x, M_mod, M_sizes):
        """f_obs(x) → module_cond → per-genome bias (B, 1, d)."""
        present = (x + 1.0) / 2.0
        f_obs = (present @ M_mod) / M_sizes.unsqueeze(0).clamp(min=1)
        return self.module_cond(f_obs).unsqueeze(1)

    def _build_input(self, x, mask=None, M_mod=None, M_sizes=None,
                     use_cond=True):
        """Initial token representation h^0: _build_h_static +
        _compute_cond_bias + input_norm.  ``use_cond=False`` skips module
        conditioning."""
        h_static = self._build_h_static(x, mask=mask)
        if use_cond and M_mod is not None:
            h_static = h_static + self._compute_cond_bias(x, M_mod, M_sizes)
        return self.input_norm(h_static)

    def _run_encoder(self, h):
        """Run the encoder, checkpointing each layer whenever training.  This
        is NOT gated on ``h.requires_grad``: under T>1 iteration the encoder
        runs T times and un-checkpointed activations stack T-fold.
        """
        if self.use_checkpoint and self.training:
            for layer in self.encoder.layers:
                h = ckpt(layer, h, use_reentrant=False)
            return h
        return self.encoder(h)

    def forward(self, xn, M_mod=None, M_sizes=None):
        r"""The T shared-weight encoder passes of the class docstring.

        Every ``cond_refresh_every`` iterations, f_obs is recomputed from a
        soft estimate x^t = tanh(head(output_norm(h^t))) of the current genome
        (detached by default) and the skip anchor h^0 is rebuilt from it.  τ is
        applied to the final logit only: ``temp_a``/``temp_b`` are sized (T,),
        but only index [-1] is read.
        """
        x0 = xn
        h_static = self._build_h_static(x0, mask=None)

        if M_mod is not None:
            cond_bias = self._compute_cond_bias(x0, M_mod, M_sizes)
        else:
            cond_bias = None

        if cond_bias is not None:
            h0 = self.input_norm(h_static + cond_bias)
        else:
            h0 = self.input_norm(h_static)
        h = h0

        rho = None
        if self.adaptive_temp:
            rho = (x0 + 1.0).sum(dim=-1, keepdim=True) / (2 * self.N)

        refresh = (self.cond_refresh_every > 0 and M_mod is not None
                   and cond_bias is not None)
        for t in range(self.T):
            h = self._run_encoder(h) + self.skip_gates[t] * h0

            # Skipped on the last iteration, where no pass remains to use it.
            if (refresh and t < self.T - 1
                    and (t + 1) % self.cond_refresh_every == 0):
                logit_t = self.output_head(self.output_norm(h)).squeeze(-1)
                x_t = torch.tanh(logit_t)
                if self.cond_refresh_detach:
                    x_t = x_t.detach()
                cond_bias = self._compute_cond_bias(x_t, M_mod, M_sizes)
                h0 = self.input_norm(h_static + cond_bias)

        logit = self.output_head(self.output_norm(h)).squeeze(-1)

        if self.adaptive_temp:
            tau = F.softplus(self.temp_a[-1] + self.temp_b[-1] * rho)
            logit = logit / tau

        return torch.tanh(logit), None

    def forward_mlm(self, x_clean, mask_prob=0.15):
        """MLM pre-training forward: mask a fraction of gene positions, replace
        them with mask_token, and run one encoder pass — no iteration, no
        gated skip, no module conditioning."""
        B, N = x_clean.shape
        mask = torch.rand(B, N, device=x_clean.device) < mask_prob

        h = self._build_input(x_clean, mask=mask, M_mod=None, M_sizes=None,
                               use_cond=False)
        h = self._run_encoder(h)
        logit = self.output_head(self.output_norm(h)).squeeze(-1)
        return logit[mask], x_clean[mask], int(mask.sum().item())

    def encoder_params(self):
        return list(self.encoder.parameters())

    def embedding_params(self):
        ps = [self.gene_embedding, self.mask_token]
        ps.extend(self.value_proj.parameters())
        ps.extend(self.input_norm.parameters())
        return ps

    def module_cond_params(self):
        return list(self.module_cond.parameters())

    def gate_params(self):
        return [self.skip_gates]

    def temp_params(self):
        if not self.adaptive_temp:
            return []
        return [self.temp_a, self.temp_b]

    def head_params(self):
        ps = list(self.output_norm.parameters())
        ps.extend(self.output_head.parameters())
        return ps

    def set_trainable(self, encoder=True, embedding=True, module_cond=True,
                      gates=True, temp=True, head=True, **_):
        """Freeze / unfreeze parameter groups.  Unknown kwargs are ignored, so
        one STAGE_TRAINABLE dict drives this class and the Ising denoisers
        alike, though their flag names differ."""
        for p in self.encoder_params():      p.requires_grad = encoder
        for p in self.embedding_params():    p.requires_grad = embedding
        for p in self.module_cond_params():  p.requires_grad = module_cond
        for p in self.gate_params():         p.requires_grad = gates
        for p in self.temp_params():         p.requires_grad = temp
        for p in self.head_params():         p.requires_grad = head

    def trainable_str(self):
        gs = [('enc', self.encoder_params()),
              ('emb', self.embedding_params()),
              ('cond', self.module_cond_params()),
              ('gates', self.gate_params()),
              ('temp', self.temp_params()),
              ('head', self.head_params())]
        parts = [f"{n}={sum(p.numel() for p in ps):,}"
                 for n, ps in gs if ps and any(p.requires_grad for p in ps)]
        return ' '.join(parts)

    def gate_str(self):
        """Compact diagnostic of the learned skip gates and τ(ρ)."""
        with torch.no_grad():
            g = self.skip_gates.data.cpu().tolist()
            s = f"skip={[round(v, 3) for v in g]}"
            if self.adaptive_temp:
                for rho in (0.3, 0.7):
                    tau = F.softplus(self.temp_a.data + self.temp_b.data * rho)
                    s += f" τ(ρ={rho})={[round(v, 3) for v in tau.cpu().tolist()]}"
            return s

    def make_optimizer(self, base_lr, wd, encoder_lr_frac=0.3):
        """AdamW with the encoder at base_lr * encoder_lr_frac and everything
        else at base_lr."""
        enc_ids = {id(p) for p in self.encoder_params()}
        enc_ps = [p for p in self.encoder_params() if p.requires_grad]
        other_ps = [p for p in self.parameters()
                    if p.requires_grad and id(p) not in enc_ids]
        groups = []
        if enc_ps:
            groups.append({'params': enc_ps, 'lr': base_lr * encoder_lr_frac})
        if other_ps:
            groups.append({'params': other_ps, 'lr': base_lr})
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

    def config_str(self):
        tot = sum(p.numel() for p in self.parameters())
        enc = sum(p.numel() for p in self.encoder_params())
        cond = sum(p.numel() for p in self.module_cond_params())
        return (f"SetTransformerDenoiser N={self.N} d={self.d_model} "
                f"heads={self.nhead} L={self.num_layers} "
                f"ff={self.dim_feedforward} T={self.T} "
                f"adaptive_temp={self.adaptive_temp} "
                f"total={tot:,} (enc={enc:,}, cond={cond:,})")

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Tolerate a source checkpoint trained at a different T, where
        skip_gates / temp_a / temp_b differ in length: a shorter source is
        padded with the neutral values (skip_gates 1.0, temp_a 0.5413 so τ = 1,
        temp_b 0.0), a longer one truncated."""
        for key, neutral in (('skip_gates', 1.0),
                              ('temp_a', 0.5413),
                              ('temp_b', 0.0)):
            if key in state_dict and key in self.state_dict():
                src = state_dict[key]
                tgt = self.state_dict()[key]
                if src.shape != tgt.shape and src.ndim == 1 and tgt.ndim == 1:
                    if src.numel() < tgt.numel():
                        padded = src.new_full(tgt.shape, neutral)
                        padded[:src.numel()] = src
                        state_dict[key] = padded
                    elif src.numel() > tgt.numel():
                        state_dict[key] = src[:tgt.numel()]
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


