"""
Data loading and noise augmentation for the Ising denoiser.

The training data are binary gene-content profiles (±1 encoding) for
N = 4 789 COG families across ~90 000 extant prokaryotic genomes, split by
phylum into train/val sets and stored as Apache Feather files.  Each row is
a genome; each column is a COG family; values are +1 (gene present) or −1
(gene absent).

Noise augmentation simulates gene tree–species tree reconciliation error:

  False negatives (FN): true +1 flipped to −1 — genes present at an
      ancestral node but missed by the reconciliation.  Dominant error mode
      (FN ≈ 0.5–0.9 at deep nodes).

  False positives (FP): true −1 flipped to +1 — spurious inferred gain.
      Rare (FP ≈ 0.01 empirically).

Each genome yields K replicates per epoch; the FN rate of each corrupted
replicate is drawn from a configurable distribution, which is how the
training curriculum selects a noise regime.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReconciliationNoiseDataset(Dataset):
    """Generates K noise-augmented replicates per clean genome.

    __getitem__ returns (K_reps, clean).  The first *nc* replicates are left
    clean (zero noise), so every batch also carries passthrough examples; the
    remaining K − nc are independently corrupted.

    The false-negative rate of each corrupted copy is drawn from one of
    (curriculum stage in brackets):

      'uniform'    FN ~ Uniform(0, fn_max)                            default
      'beta_low'   FN ~ Beta(1, 5) × fn_max   mean ≈ 0.17 × fn_max    [S3a]
      'beta_mid'   FN ~ Beta(2, 3) × fn_max   mean ≈ 0.40 × fn_max    [S3b]
      'beta_high'  FN ~ Beta(2, 1) × fn_max   mean ≈ 0.67 × fn_max
      'beta_hard'  FN ~ Beta(5, 1) × fn_max   mean ≈ 0.83 × fn_max    [S3d]
      'beta_vhard' FN ~ Beta(8, 1) × fn_max   mean ≈ 0.89 × fn_max

    beta_vhard paired with fn_max = 0.95 concentrates training mass at
    FN 0.85–0.95, the LBCA/LACA regime.
    """

    def __init__(self, clean_tensor, fn_max=0.9, fp=0.01, K=32,
                 clean_frac=0.125, fn_dist='uniform',
                 fp_max=None, fp_dist='uniform', fp_mode='uniform',
                 module_membership=None, module_sizes=None,
                 module_inject_rate=0.0,
                 module_delete_rate=0.0,
                 module_swap_rate=0.0,
                 module_frac_absent=0.2,
                 module_frac_present=0.8,
                 module_swap_max=3,
                 marginal_fp_rate=0.0, marginal_fp_max=0.0,
                 marginal_fp_fn=0.1, marginal_freq=None,
                 return_masks=False):
        """
        Args:
            clean_tensor: (N_train, N_cogs) ±1 tensor of clean genomes.
            fp:           Fixed false-positive rate, used when fp_max is None.
            clean_frac:   Fraction of K that are clean copies (no noise).
            fn_dist:      FN sampling distribution; see the class docstring.
            fp_max:       If set, each corrupted copy draws its own FP from
                          [0, fp_max] via fp_dist, an FP curriculum parallel
                          to the FN one.
            fp_dist:      'uniform', 'beta_low' or 'beta_high'.
            fp_mode:      'uniform' flips absent COGs independently;
                          'marginal' samples them proportional to cross-genome
                          prevalence (common-gene contamination).
            module_membership: (N_cogs, n_modules) binary tensor; enables
                          module-coherent corruption on top of the per-COG
                          noise.  Typically the M_kegg block of
                          data/module_matrix_kegg.pt (327 KEGG modules;
                          coherent FP is defined on pathways, not on the
                          broader functional categories).
            module_sizes: (n_modules,) number of member COGs per module.
            module_inject_rate: per corrupted copy, probability of injecting
                          one absent module's members as +1 (coherent FP).
            module_delete_rate: probability of setting one present module's
                          members to −1 (coherent FN).
            module_swap_rate:   probability of grafting up to module_swap_max
                          modules from another training genome onto this one.
            module_frac_absent / module_frac_present: present-fraction at or
                          below / at or above which a module counts as absent
                          / present in xc.
            module_swap_max: cap on modules per swap event.
            return_masks: if True, also return a boolean mask of positions
                          that are false positives relative to the clean
                          genome.
        """
        self.clean = clean_tensor
        self.fn_max = fn_max
        self.fp = fp
        self.K = K
        self.nc = max(1, int(K * clean_frac))
        self.fn_dist = fn_dist
        self.fp_max = fp_max
        self.fp_dist = fp_dist
        self.fp_mode = fp_mode
        self.M = module_membership
        self.M_sizes = module_sizes
        self.mod_inject_rate = module_inject_rate
        self.mod_delete_rate = module_delete_rate
        self.mod_swap_rate = module_swap_rate
        self.mod_frac_absent = module_frac_absent
        self.mod_frac_present = module_frac_present
        self.mod_swap_max = module_swap_max
        # Marginal-FP corruption: a fraction of the corrupted copies get low FN
        # plus dense FP drawn by per-COG cross-genome frequency.  They share a
        # clean target with the high-FN/low-FP copies, so the model must learn
        # both fill-in and trim.
        self.marg_fp_rate = marginal_fp_rate
        self.marg_fp_max = marginal_fp_max
        self.marg_fp_fn = marginal_fp_fn
        self.return_masks = return_masks
        # Default contamination prior is COG prevalence in the loaded training
        # set.  An explicit marginal_freq must be aligned to the same COG
        # column order -- e.g. archaea-only p_c while training on the mixed set.
        if marginal_fp_rate > 0 or fp_mode == 'marginal':
            self.marginal = (marginal_freq if marginal_freq is not None
                             else (self.clean == 1).float().mean(0))
        else:
            self.marginal = None
        self.module_members = None
        if self.M is not None:
            self.module_members = [
                torch.where(self.M[:, j] > 0)[0]
                for j in range(self.M.shape[1])
            ]

    def __len__(self):
        return self.clean.size(0)

    def _sample_fn(self):
        """Sample a false-negative rate from the configured distribution."""
        if self.fn_dist == 'uniform':
            return torch.rand(1).item() * self.fn_max
        elif self.fn_dist == 'beta_low':
            return min(torch.distributions.Beta(1, 5).sample().item() * self.fn_max,
                       self.fn_max)
        elif self.fn_dist == 'beta_mid':
            return min(torch.distributions.Beta(2, 3).sample().item() * self.fn_max,
                       self.fn_max)
        elif self.fn_dist == 'beta_high':
            return min(torch.distributions.Beta(2, 1).sample().item() * self.fn_max,
                       self.fn_max)
        elif self.fn_dist == 'beta_hard':
            return min(torch.distributions.Beta(5, 1).sample().item() * self.fn_max,
                       self.fn_max)
        elif self.fn_dist == 'beta_vhard':
            return min(torch.distributions.Beta(8, 1).sample().item() * self.fn_max,
                       self.fn_max)
        return torch.rand(1).item() * self.fn_max

    def _sample_fp(self):
        """Sample a false-positive rate for one corrupted copy.

        Returns the fixed scalar `fp` when fp_max is None; otherwise draws
        from [0, fp_max] via fp_dist.
        """
        if self.fp_max is None:
            return self.fp
        if self.fp_dist == 'uniform':
            return torch.rand(1).item() * self.fp_max
        elif self.fp_dist == 'beta_low':
            return min(torch.distributions.Beta(1, 5).sample().item() * self.fp_max,
                       self.fp_max)
        elif self.fp_dist == 'beta_high':
            return min(torch.distributions.Beta(2, 1).sample().item() * self.fp_max,
                       self.fp_max)
        return torch.rand(1).item() * self.fp_max

    def _inject_per_cog_fp(self, rep, is_pos, is_neg, fp):
        """Inject absent COGs either uniformly or by marginal prevalence."""
        if fp <= 0:
            return
        if self.fp_mode == 'marginal' and self.marginal is not None:
            n_abs = int(is_neg.sum().item())
            n_add = int(round(fp * n_abs))
            if n_add <= 0:
                return
            w = self.marginal.clone()
            w[is_pos] = 0.0
            nz = int((w > 0).sum().item())
            if nz > 0:
                pick = torch.multinomial(w, min(n_add, nz), replacement=False)
                rep[pick] = 1
        else:
            rep[is_neg & (torch.rand_like(rep) < fp)] = 1

    # ---- Module-coherent corruption helpers

    def _module_fracs(self, xc):
        """Per-module fraction of members present in xc (xc is +/-1)."""
        if self.M is None:
            return None
        presence = (xc == 1).float()
        sizes = self.M_sizes.clamp(min=1).float()
        return (self.M.t() @ presence) / sizes

    def _inject_module(self, rep, fracs):
        """Flip all members of one absent module to +1 in rep (coherent FP)."""
        cand = torch.where(fracs < self.mod_frac_absent)[0]
        if cand.numel() == 0:
            return
        m = cand[torch.randint(0, cand.numel(), (1,)).item()].item()
        rep[self.module_members[m]] = 1

    def _delete_module(self, rep, fracs):
        """Flip all members of one present module to -1 in rep (coherent FN)."""
        cand = torch.where(fracs > self.mod_frac_present)[0]
        if cand.numel() == 0:
            return
        m = cand[torch.randint(0, cand.numel(), (1,)).item()].item()
        rep[self.module_members[m]] = -1

    def _swap_signature(self, rep, fracs_self, i):
        """Graft modules from a randomly chosen donor training genome.

        Picks modules present in the donor but absent in xc and injects up
        to mod_swap_max of them: coherent FP from a different organism.
        """
        n_genomes = self.clean.size(0)
        if n_genomes < 2:
            return
        j = torch.randint(0, n_genomes, (1,)).item()
        if j == i:
            return
        xc_other = self.clean[j]
        fracs_other = self._module_fracs(xc_other)
        diff = torch.where(
            (fracs_other > self.mod_frac_present)
            & (fracs_self < self.mod_frac_absent)
        )[0]
        if diff.numel() == 0:
            return
        n_inject = min(self.mod_swap_max, diff.numel())
        pick = diff[torch.randperm(diff.numel())[:n_inject]]
        for m in pick.tolist():
            rep[self.module_members[m]] = 1

    def __getitem__(self, i):
        xc = self.clean[i]
        is_pos = (xc == 1)
        is_neg = (xc == -1)
        reps = xc.unsqueeze(0).expand(self.K, -1).clone()
        fp_masks = torch.zeros_like(reps, dtype=torch.bool) if self.return_masks else None
        fracs = self._module_fracs(xc) if self.M is not None else None
        for k in range(self.nc, self.K):
            # Prune example: low FN plus dense marginal-frequency FP.
            if self.marginal is not None and torch.rand(1).item() < self.marg_fp_rate:
                fn = torch.rand(1).item() * self.marg_fp_fn
                reps[k][is_pos & (torch.rand_like(xc) < fn)] = -1
                n_true = int(is_pos.sum().item())
                frac = 0.1 + torch.rand(1).item() * max(0.0, self.marg_fp_max - 0.1)
                n_add = int(frac * n_true)
                if n_add > 0:
                    w = self.marginal.clone()
                    w[is_pos] = 0.0                                  # absent COGs only
                    nz = int((w > 0).sum().item())
                    if nz > 0:
                        pick = torch.multinomial(w, min(n_add, nz), replacement=False)
                        reps[k][pick] = 1
                if self.return_masks:
                    fp_masks[k] = is_neg & (reps[k] == 1)
                continue
            # Rescue example: per-COG FN / FP.
            fn = self._sample_fn()
            fp = self._sample_fp()
            reps[k][is_pos & (torch.rand_like(xc) < fn)] = -1
            self._inject_per_cog_fp(reps[k], is_pos, is_neg, fp)
            if self.M is not None:
                r = torch.rand(1).item()
                t = 0.0
                if r < (t := t + self.mod_inject_rate):
                    self._inject_module(reps[k], fracs)
                elif r < (t := t + self.mod_delete_rate):
                    self._delete_module(reps[k], fracs)
                elif r < (t := t + self.mod_swap_rate):
                    self._swap_signature(reps[k], fracs, i)
            if self.return_masks:
                fp_masks[k] = is_neg & (reps[k] == 1)
        if self.return_masks:
            return reps, xc, fp_masks
        return reps, xc


def collate_rep(batch):
    """Collate K-replicate batches into (noisy, clean) tensors.

    Input batch: list of (K_reps[K, N], clean[N]) from ReconciliationNoiseDataset.
    Output: (noisy[B*K, N], clean_expanded[B*K, N]), K contiguous per genome.
    """
    has_masks = len(batch[0]) == 3
    if has_masks:
        noisy_list, clean_list, fp_mask_list = zip(*batch)
    else:
        noisy_list, clean_list = zip(*batch)
    noisy = torch.cat(noisy_list, 0)
    clean = torch.stack(clean_list, 0)
    K = noisy_list[0].size(0)
    clean_exp = clean.repeat_interleave(K, dim=0)
    if has_masks:
        fp_mask = torch.cat(fp_mask_list, 0)
        return noisy, clean_exp, fp_mask
    return noisy, clean_exp


def gpu_noise_augment(clean_batch, fn_max=0.9, fp=0.01, K=16,
                      clean_frac=0.125, fn_dist='beta_high'):
    """Apply the per-COG noise model on device, for when the CPU dataset path
    is the bottleneck.  Module-coherent and marginal-FP corruption are not
    reproduced here.

    Args:
        clean_batch: (B, N) ±1 tensor already on the device.
        clean_frac:  Fraction of K that are clean copies.

    Returns:
        noisy: (B*K, N) corrupted replicates.
        clean_exp: (B*K, N) expanded clean targets, K contiguous per genome.
    """
    B, N = clean_batch.shape
    nc = max(1, int(K * clean_frac))
    dev = clean_batch.device

    reps = clean_batch.unsqueeze(1).expand(B, K, N).clone()

    n_corrupt = K - nc
    if n_corrupt > 0:
        if fn_dist == 'uniform':
            fn_rates = torch.rand(B, n_corrupt, device=dev) * fn_max
        elif fn_dist == 'beta_low':
            fn_rates = torch.distributions.Beta(1, 5).sample((B, n_corrupt)).to(dev) * fn_max
        elif fn_dist == 'beta_mid':
            fn_rates = torch.distributions.Beta(2, 3).sample((B, n_corrupt)).to(dev) * fn_max
        elif fn_dist == 'beta_high':
            fn_rates = torch.distributions.Beta(2, 1).sample((B, n_corrupt)).to(dev) * fn_max
        elif fn_dist == 'beta_hard':
            fn_rates = torch.distributions.Beta(5, 1).sample((B, n_corrupt)).to(dev) * fn_max
        elif fn_dist == 'beta_vhard':
            fn_rates = torch.distributions.Beta(8, 1).sample((B, n_corrupt)).to(dev) * fn_max
        else:
            fn_rates = torch.rand(B, n_corrupt, device=dev) * fn_max
        fn_rates = fn_rates.clamp(max=fn_max)

        # A view into reps, not a copy: the masked writes below corrupt reps
        # itself, which is what gets reshaped into `noisy`.
        corrupt_slice = reps[:, nc:, :]             # (B, n_corrupt, N)
        is_pos = (clean_batch == 1).unsqueeze(1)
        is_neg = (clean_batch == -1).unsqueeze(1)

        rand_fn = torch.rand(B, n_corrupt, N, device=dev)
        rand_fp = torch.rand(B, n_corrupt, N, device=dev)

        fn_thresh = fn_rates.unsqueeze(-1)

        corrupt_slice[is_pos.expand_as(corrupt_slice) & (rand_fn < fn_thresh)] = -1
        corrupt_slice[is_neg.expand_as(corrupt_slice) & (rand_fp < fp)] = 1

    noisy = reps.reshape(B * K, N)              # (B*K, N), K contiguous
    clean_exp = clean_batch.repeat_interleave(K, dim=0)
    return noisy, clean_exp


def load_feathers(train_path, val_path, frac=1.0):
    """Load train/val COG data from Feather files.

    Columns whose name starts with 'COG' are the gene-presence variables;
    values > 0 map to +1, everything else to −1.  Both tensors are returned
    in the sorted COG order of `vocab`, which is the column order every
    model, module matrix and evaluation script assumes.

    Args:
        frac: Subsampling fraction, for quick runs on a slice of the data.

    Returns:
        train_tensor: (N_train, N_cogs) ±1 float tensor.
        val_tensor:   (N_val, N_cogs) ±1 float tensor.
        vocab:        Sorted list of COG column names.
    """
    # Skipping the taxonomy/metadata columns keeps RAM sane during distributed
    # load, since each rank independently reads the feather.
    def _cog_cols(path):
        import pyarrow.ipc as _ipc
        with _ipc.open_file(path) as reader:
            names = reader.schema.names
        return [n for n in names if n.startswith('COG')]
    tdf = pd.read_feather(train_path, columns=_cog_cols(train_path))
    vdf = pd.read_feather(val_path,   columns=_cog_cols(val_path))
    if frac < 1:
        tdf = tdf.sample(frac=frac, random_state=42)
        vdf = vdf.sample(frac=frac, random_state=42)
    vocab = sorted([c for c in tdf.columns if c.startswith('COG')])

    def to_pm1(df, cols):
        arr = df[cols].to_numpy(np.float32)
        return torch.tensor((arr > 0).astype(np.float32) * 2 - 1, dtype=torch.float32)

    return to_pm1(tdf, vocab), to_pm1(vdf, vocab), vocab
