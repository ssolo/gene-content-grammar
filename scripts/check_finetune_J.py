#!/usr/bin/env python3
"""Check whether the coupling-controls fine-tune reshaped the model as intended.

Prints, for a baseline and a fine-tuned checkpoint, the singular spectrum of the
symmetrised coupling J~ (the target of the nuclear-norm penalty), the row-sum
magnitudes (the target of the zero-mean penalty, which keeps the field h carrying
base rates), and the learned per-state calibration head.

Usage: python3 scripts/check_finetune_J.py <baseline_ckpt> <finetuned_ckpt>
"""
import sys

import torch

sys.path.insert(0, 'scripts')
from analyze_ancestral_node import build_model_for_ckpt          # noqa: E402
from ising_denoiser.modules import load_module_matrix            # noqa: E402
from ising_denoiser.training import strip_compile_prefix         # noqa: E402


def load(ck, dev, N, n_mod):
    sd = strip_compile_prefix({k.replace('module.', ''): v
                               for k, v in torch.load(ck, map_location=dev, weights_only=True).items()})
    m, _, _, _ = build_model_for_ckpt(sd, dev, N, n_mod, ck)
    m.load_state_dict(sd, strict=False); m.eval()
    return m, sd


def diag(m, sd, name):
    J = m._Js().detach().float()
    s = torch.linalg.svdvals(J)
    nuc = s.sum().item(); fro = (s.pow(2).sum().sqrt()).item()
    pr = (s.sum().pow(2) / s.pow(2).sum()).item()           # participation ratio = effective rank
    top16 = (s[:16].sum() / nuc).item()
    rowsum = J.sum(dim=1)
    print(f'\n[{name}]')
    print(f'  J~ : nuclear={nuc:.0f}  frob={fro:.1f}  participation-ratio(eff.rank)={pr:.1f}  '
          f'top-16 SV / nuclear = {top16:.3f}')
    print(f'       top5 singular values: {[round(x,1) for x in s[:5].tolist()]}')
    print(f'  zero-mean: mean|row-sum|={rowsum.abs().mean().item():.3f}  max|row-sum|={rowsum.abs().max().item():.2f}')
    gc = sd.get('gainloss_cal')
    if gc is not None:
        gc = gc.tolist()
        print(f'  per-state head gainloss_cal [da_gain,db_gain,da_loss,db_loss] = '
              f'{[round(x,3) for x in gc]}  (0,0,0,0 = identity)')
    else:
        print('  per-state head: (not in checkpoint)')


def main():
    base_ck, ft_ck = sys.argv[1], sys.argv[2]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mm = torch.load('data/module_matrix_kegg.pt', weights_only=False); N = len(mm['cog_names'])
    _, M_mod, _ = load_module_matrix('data/module_matrix_kegg.pt', dev, 1000); n_mod = M_mod.shape[1]
    mb, sb = load(base_ck, dev, N, n_mod); diag(mb, sb, 'baseline ' + base_ck.split('/')[0])
    mf, sf = load(ft_ck, dev, N, n_mod); diag(mf, sf, 'fine-tuned ' + ft_ck.split('/')[0])


if __name__ == '__main__':
    main()
