"""
Reptile hyperparameter ablation study.

Explores three key variables:
    K            — inner gradient steps:   {1, 5, 10, 20}
    inner_lr     — inner loop LR:          {0.001, 0.005, 0.01, 0.05}
    update_freq  — Reptile update timing:  {0=epoch, 1, 10, 50}

Also tests matched_compute mode for update_freq ablation:
    Fixes total inner-loop steps per epoch so that coupling-frequency
    effects can be isolated from total-compute effects.

All ablations use LOSO on held_out=1, seed=1 (single fold, fast).
For a full ablation, run across all subjects and seeds — see run_ablation.sh.

Usage:
    python run_ablation.py --ablation K --held_out 1 --seed 1
    python run_ablation.py --ablation inner_lr --held_out 1 --seed 1
    python run_ablation.py --ablation update_freq --held_out 1 --seed 1
    python run_ablation.py --ablation update_freq_matched --held_out 1 --seed 1
"""

import argparse
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, '../EEGNex')

from data_utils import load_dataset, build_loso_split
from baseline_trainer import build_baseline_model
from reptile_trainer import build_reptile_model, train_reptile
from evaluate import run_full_evaluation
from meta_init import LORA_LAYER_NAMES


# ── Ablation grids ────────────────────────────────────────────────────────────

ABLATION_GRIDS = {
    'K': {
        'vary':    'inner_steps',
        'values':  [1, 5, 10, 20],
        'fixed': {
            'inner_lr':    0.01,
            'update_freq': 0,
            'matched_compute': False,
        },
    },
    'inner_lr': {
        'vary':   'inner_lr',
        'values': [0.001, 0.005, 0.01, 0.05],
        'fixed': {
            'inner_steps': 5,
            'update_freq': 0,
            'matched_compute': False,
        },
    },
    'update_freq': {
        # Varies update frequency WITHOUT fixing total compute.
        # Compares: once per epoch vs tighter coupling.
        'vary':   'update_freq',
        'values': [0, 1, 10, 50],   # 0 = per epoch
        'fixed': {
            'inner_steps': 5,
            'inner_lr':    0.01,
            'matched_compute': False,
        },
    },
    'update_freq_matched': {
        # Varies update frequency WITH matched total compute.
        # Isolates coupling-frequency effect from compute effect.
        'vary':   'update_freq',
        'values': [0, 1, 10, 50],
        'fixed': {
            'inner_steps': 5,
            'inner_lr':    0.01,
            'matched_compute': True,
        },
    },
}

# ── Base config ───────────────────────────────────────────────────────────────

BASE_CONFIG = {
    'freq_min':     8,
    'freq_max':     45,
    'rank':         8,
    'alpha':        24,
    'epochs':       100,
    'batch_size':   64,
    'lr':           1e-3,
    'weight_decay': 0.01,
    'inner_steps':  5,
    'inner_lr':     0.01,
    'inner_batch':  32,
    'meta_lr':      0.1,
    'update_freq':  0,
    'matched_compute': False,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid   = ABLATION_GRIDS[args.ablation]
    values = grid['values']

    # Load data once — reused across all values
    print(f'Loading {args.dataset}...')
    data, labels, subject_ids, sessions, _ = load_dataset(
        args.dataset, freq_min=BASE_CONFIG['freq_min'],
        freq_max=BASE_CONFIG['freq_max'],
    )

    held_out_0idx = args.held_out - 1
    (train_X, train_y, train_sids,
     cal_X,   cal_y,   cal_sids,
     test_X,  test_y,  test_sids) = build_loso_split(
        data, labels, subject_ids, sessions,
        held_out_0idx, dataset=args.dataset,
    )

    n_channels       = train_X.shape[1]
    n_times          = train_X.shape[2]
    n_classes        = len(np.unique(train_y))
    n_train_subjects = len(np.unique(train_sids))
    held_out_slot    = n_train_subjects

    print(f'  Channels={n_channels} | Times={n_times} | '
          f'Classes={n_classes} | Train subjects={n_train_subjects}')

    # Run one experiment per grid value
    for val in values:
        cfg = BASE_CONFIG.copy()
        cfg[grid['vary']] = val
        cfg.update(grid['fixed'])

        # Human-readable label for wandb
        vary_key = grid['vary']
        label    = f'{vary_key}={val}'
        if args.ablation == 'update_freq_matched':
            label += '_matched'

        print(f'\n{"="*60}')
        print(f' Ablation: {args.ablation} | {label}')
        print(f'{"="*60}')

        wandb.init(
            project='reptile_lora_ablation',
            name=f'{args.ablation}_{label}_{args.dataset}'
                 f'_heldout{args.held_out}_seed{args.seed}',
            config={**cfg,
                    'ablation':    args.ablation,
                    'vary_key':    vary_key,
                    'vary_value':  val,
                    'dataset':     args.dataset,
                    'held_out':    args.held_out,
                    'seed':        args.seed},
            reinit=True,
        )

        model = build_reptile_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )

        model, meta_init = train_reptile(
            model, train_X, train_y, train_sids, cfg, device
        )

        # Load meta-init into held-out slot
        meta_init.load_into_slot(model, held_out_slot)

        total_norm = sum(
            meta_init.weights[name]['A'].norm().item() +
            meta_init.weights[name]['B'].norm().item()
            for name in LORA_LAYER_NAMES
        )
        wandb.log({'meta_init_norm': total_norm, 'vary_value': val})

        results = run_full_evaluation(
            model            = model,
            cal_X            = cal_X,
            cal_y            = cal_y,
            test_X           = test_X,
            test_y           = test_y,
            subject_slot     = held_out_slot,
            device           = device,
            condition_name   = f'reptile_{label}',
            held_out_subject = held_out_0idx,
            seed             = args.seed,
        )

        results['vary_value']  = val
        results['ablation']    = args.ablation
        results['dataset']     = args.dataset
        wandb.log(results)

        print(f'  Zero-shot: {results["zero_shot_acc"]:.3f} | '
              f'N=10: {results.get("few_shot_acc_N10", float("nan")):.3f}')

        wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, required=True,
                        choices=list(ABLATION_GRIDS.keys()))
    parser.add_argument('--dataset',  type=str, default='BCI2a',
                        choices=['BCI2a', 'BCI2b'])
    parser.add_argument('--held_out', type=int, default=1,
                        choices=list(range(1, 10)))
    parser.add_argument('--seed',     type=int, default=1)
    args = parser.parse_args()
    main(args)