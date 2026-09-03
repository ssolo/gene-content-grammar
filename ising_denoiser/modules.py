"""
Module completeness: membership matrix, auxiliary head, and conditioning utilities.

The module membership matrix M ∈ {0,1}^{N×n_mod} maps each of the N = 4 789
COG gene families to one or more functional modules.  The KEGG-augmented
matrix concatenates three annotation layers: COG functional categories
(single-letter NCBI codes), COG pathway groupings (e.g. "Ribosome"), and
KEGG metabolic modules (e.g. M00001 "Glycolysis").

M is used twice: as the target of an auxiliary loss (a small MLP predicts
module completeness fractions from the final hidden state z), and, in
ModuleConditionedDenoiser, as conditioning input, where the observed
per-module fractions are fed to the conditioning MLP.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_module_fracs(x, M, sizes):
    """Per-module fraction of member genes present:

        f_m = (1/|M_m|) Σ_{i ∈ M_m} 𝟙[x_i = +1]

    Args:
        x:     (B, N) genome in ±1 encoding (+1 = present, −1 = absent).
        M:     (N, n_mod) binary module membership matrix.
        sizes: (n_mod,) number of COGs per module (column sums of M).

    Returns:
        (B, n_mod) fraction of genes present per module, in [0, 1].
    """
    present = (x + 1.0) / 2.0
    counts = present @ M
    return counts / sizes.unsqueeze(0).clamp(min=1)


class ModuleCompletenessPredictor(nn.Module):
    """Auxiliary head predicting module completeness from the hidden state.

    Must be kept outside the DDP-wrapped model: it is called outside forward(),
    which makes DDP mark its parameters ready twice.  Register its parameters
    with the optimizer via add_param_group() instead.

    The near-zero init puts initial predictions at ≈ 0.5, so the auxiliary loss
    does not swamp the reconstruction loss early in training.
    """

    def __init__(self, H, n_modules, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(H, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_modules), nn.Sigmoid())
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=0.1)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight, gain=0.1)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, z):
        """z: (B, H) → (B, n_modules) predicted fractions."""
        return self.mlp(z)


def module_aux_loss(z_final, x_clean, aux_head, M, sizes):
    """MSE between predicted and true module completeness.

    Args:
        z_final: (B, H) hidden state from the final denoiser iteration.
        x_clean: (B, N) clean (uncorrupted) genome in ±1 encoding.
        M:       (N, n_mod) binary module membership matrix.
        sizes:   (n_mod,) column sums of M.
    """
    pred = aux_head(z_final)
    target = compute_module_fracs(x_clean, M, sizes)
    return F.mse_loss(pred, target)


def load_module_matrix(path, dev, H=1000):
    """Load the COG → module membership matrix and build the auxiliary head.

    Accepts three on-disk key layouts: 'M' (combined), 'M_cat' + 'M_path'
    (categories and COG pathways, concatenated here), or 'M_cat' alone.
    `H` must equal the hidden width of the model the head is trained with.

    Returns:
        (aux_head, M_mod, M_sizes), or (None, None, None) if `path` is missing.
    """
    if not Path(path).exists():
        return None, None, None

    data = torch.load(path, weights_only=False)

    if 'M' in data:
        M = data['M'].to(dev)
        sizes = data['sizes'].to(dev)
    elif 'M_cat' in data and 'M_path' in data:
        M = torch.cat([data['M_cat'], data['M_path']], dim=1).to(dev)
        sizes = torch.cat([data['cat_sizes'], data['path_sizes']]).to(dev)
    elif 'M_cat' in data:
        M = data['M_cat'].to(dev)
        sizes = data['cat_sizes'].to(dev)
    else:
        raise KeyError(f"module_matrix has keys {list(data.keys())}, need M or M_cat")

    head = ModuleCompletenessPredictor(H, M.shape[1]).to(dev)
    return head, M, sizes
