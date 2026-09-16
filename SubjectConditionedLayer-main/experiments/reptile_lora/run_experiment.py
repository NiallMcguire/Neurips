"""
Main entry point for Reptile LoRA vs Baseline LoRA experiments.

Supports BCI2a (22ch, 4-class) and BCI2b (3ch, 2-class).

Usage:
    python run_experiment.py --condition baseline_lora       --dataset BCI2a --held_out 9 --seed 1
    python run_experiment.py --condition reptile_lora        --dataset BCI2b --held_out 9 --seed 1
    python run_experiment.py --condition reptile_lora_grassmann --dataset BCI2a --held_out 9 --seed 1

Conditions:
    baseline_lora          — zero initialisation for held-out subject (original paper)
    reptile_lora           — Euclidean Reptile meta-initialisation
    reptile_lora_grassmann — Riemannian Reptile on Grassmann manifold G(r,n)

The Grassmann condition is identical to reptile_lora in all hyperparameters,
training loop, and evaluation. Only the meta-update geometry differs — A matrices
are averaged on the Grassmann manifold rather than in Euclidean space.

LOSO evaluation:
    Train on subjects {1..9} \\ {held_out}
    Evaluate on held_out subject:
        zero_shot:    no calibration trials, no fine-tuning
        few_shot_N5:  fine-tune on 5  calibration trials
        few_shot_N10: fine-tune on 10 calibration trials
        few_shot_N20: fine-tune on 20 calibration trials
        few_shot_N50: fine-tune on 50 calibration trials

All results logged to wandb project 'reptile_lora_bci'.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, '../EEGNex')

from data_utils import load_dataset, build_loso_split, euclidean_align
from baseline_trainer import build_baseline_model, train_baseline
from reptile_trainer import (
    build_reptile_model,
    train_reptile,
    train_reptile_grassmann,
)
from evaluate import run_full_evaluation
from meta_init import LORA_LAYER_NAMES
from inits import apply_init, subject_covariances

# Optional heavy arms — imported lazily to avoid hard dependency when unused
try:
    from maml_trainer import build_maml_model, train_maml
    _HAS_MAML = True
except Exception:
    _HAS_MAML = False

try:
    from hypernet import (AdapterHypernetwork, subject_context,
                          train_hypernetwork_regression, context_dim)
    _HAS_HN = True
except Exception:
    _HAS_HN = False


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    'freq_min': 8,
    'freq_max': 45,

    # Model — same rank/alpha for both datasets for fair comparison
    'rank':  8,
    'alpha': 24,

    # Backbone training
    'epochs':       100,
    'batch_size':   64,
    'lr':           1e-3,
    'weight_decay': 0.01,

    # Reptile inner loop
    'inner_steps': 5,
    'inner_lr':    0.01,
    'inner_batch': 32,

    # Reptile outer update
    'meta_lr': 0.1,

    # Backbone freezing (reviewer 8 / brief: frozen EEGNeX trunk)
    'freeze_backbone': False,

    # Meta-learned layer scope (reviewer 5): None = all 4 conv adapters
    'meta_layers': None,
}

# Valid conditions
CONDITIONS = ['baseline_lora', 'reptile_lora', 'reptile_lora_grassmann',
              'average_adapters', 'shared_adapter',
              'donor_random', 'donor_nearest',
              'compute_matched', 'maml', 'hypernet']


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = DEFAULT_CONFIG.copy()
    if args.freeze_backbone:
        cfg['freeze_backbone'] = True

    # meta_lr semantics differ by condition: Reptile uses it as an
    # interpolation coefficient (default 0.1), while MAML uses it as an
    # Adam learning rate on the meta-parameters (needs ~1e-3). The CLI
    # override wins; otherwise pick a per-condition default.
    if args.meta_lr is not None:
        cfg['meta_lr'] = args.meta_lr
    elif args.condition == 'maml':
        cfg['meta_lr'] = 1e-3

    held_out_0idx = args.held_out - 1

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project='reptile_lora_bci',
        name=f'{args.condition}_{args.dataset}_heldout{args.held_out}_seed{args.seed}',
        config={**cfg,
                'condition': args.condition,
                'dataset':   args.dataset,
                'held_out':  args.held_out,
                'seed':      args.seed},
        reinit=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f'\nLoading {args.dataset} | held-out subject: {args.held_out}')
    data, labels, subject_ids, sessions, _ = load_dataset(
        args.dataset,
        freq_min=cfg['freq_min'],
        freq_max=cfg['freq_max'],
    )

    (train_X, train_y, train_sids,
     cal_X,   cal_y,   cal_sids,
     test_X,  test_y,  test_sids) = build_loso_split(
        data, labels, subject_ids, sessions,
        held_out_0idx, dataset=args.dataset,
    )

    # ── Euclidean Alignment (optional, unlabelled) ────────────────────────────
    if args.ea:
        print('Applying Euclidean Alignment (unlabelled, per-subject whitening;'
              ' fit on train sessions only to avoid test leakage)')
        aligned = euclidean_align(data, subject_ids,
                                  sessions=sessions, dataset=args.dataset)
        (train_X, train_y, train_sids,
         cal_X,   cal_y,   cal_sids,
         test_X,  test_y,  test_sids) = build_loso_split(
            aligned, labels, subject_ids, sessions,
            held_out_0idx, dataset=args.dataset,
        )

    n_channels       = train_X.shape[1]
    n_times          = train_X.shape[2]
    n_classes        = len(np.unique(train_y))
    n_train_subjects = len(np.unique(train_sids))

    print(f'  Train:            {len(train_X)} trials from {n_train_subjects} subjects')
    print(f'  Calibration pool: {len(cal_X)} trials from held-out subject')
    print(f'  Test:             {len(test_X)} trials from held-out subject')
    print(f'  Channels: {n_channels} | Times: {n_times} | Classes: {n_classes}')

    wandb.log({
        'n_channels': n_channels,
        'n_times':    n_times,
        'n_classes':  n_classes,
    })

    held_out_slot = n_train_subjects

    # ── Build model and train ──────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    init_info = {}

    if args.condition == 'baseline_lora':
        print(f'\n=== BASELINE LoRA ({args.dataset}) ===')
        print('Held-out subject initialised from ZERO (original paper behaviour)')

        model = build_baseline_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, tracker = train_baseline(
            model, train_X, train_y, train_sids, cfg, device
        )

        print(f'\nHeld-out slot {held_out_slot} remains at zero initialisation')

    elif args.condition == 'reptile_lora':
        print(f'\n=== REPTILE LoRA — Euclidean ({args.dataset}) ===')
        print('Meta-update: Euclidean average in R^(n×r)')
        print('Held-out subject initialised from A_meta, B_meta')

        model = build_reptile_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, meta_init, tracker = train_reptile(
            model, train_X, train_y, train_sids, cfg, device
        )

        print(f'\nLoading A_meta, B_meta into held-out slot {held_out_slot}')
        init_info = apply_init(model, held_out_slot, 'reptile',
                               meta_init=meta_init, device=device)

        total_norm = sum(
            meta_init.weights[name]['A'].norm().item() +
            meta_init.weights[name]['B'].norm().item()
            for name in meta_init.layer_names
        )
        print(f'Meta-init total Euclidean norm: {total_norm:.4f}')
        wandb.log({'meta_init_norm': total_norm})

    elif args.condition == 'reptile_lora_grassmann':
        print(f'\n=== REPTILE LoRA — Grassmann ({args.dataset}) ===')
        print('Meta-update: Riemannian Reptile on G(r, n) for A matrices')
        print('Held-out subject initialised from A_meta (Grassmann), B_meta (Euclidean)')

        model = build_reptile_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, meta_init = train_reptile_grassmann(
            model, train_X, train_y, train_sids, cfg, device
        )

        print(f'\nLoading Grassmann A_meta, B_meta into held-out slot {held_out_slot}')
        meta_init.load_into_slot(model, held_out_slot)

        total_norm = sum(
            meta_init.weights[name]['A'].norm().item() +
            meta_init.weights[name]['B'].norm().item()
            for name in LORA_LAYER_NAMES
        )
        print(f'Meta-init total norm: {total_norm:.4f}')
        wandb.log({'meta_init_norm': total_norm})

    elif args.condition in ('average_adapters', 'shared_adapter',
                            'donor_random', 'donor_nearest'):
        # Cheap non-meta baselines (reviewer request 1): same joint training
        # as baseline_lora, then a non-zero init loaded into the held-out slot.
        print(f'\n=== {args.condition.upper()} ({args.dataset}) ===')

        if args.condition == 'shared_adapter':
            # True shared adapter: all training subjects route to slot 0,
            # held-out slot is 1. Build a model with 2 slots and remap all
            # training sids to 0 so they jointly optimise one adapter.
            shared_cfg = dict(cfg)
            model = build_baseline_model(
                n_channels, n_classes, n_times,
                1,  # n_train_subjects=1 -> num_adapters = 2 (slots 0,1)
                shared_cfg, device,
            )
            held_out_slot = 1
            train_sids_shared = np.zeros_like(train_sids)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Trainable parameters: {n_params:,}')
            wandb.log({'n_params': n_params})
            model, tracker = train_baseline(
                model, train_X, train_y, train_sids_shared, shared_cfg, device
            )
            init_info = apply_init(model, held_out_slot, 'shared',
                                   layer_names=cfg['meta_layers'])
            print(f'Init: {init_info}')
            wandb.log(init_info)
        else:
            model = build_baseline_model(
                n_channels, n_classes, n_times,
                n_train_subjects, cfg, device,
            )

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Trainable parameters: {n_params:,}')
            wandb.log({'n_params': n_params})

            model, tracker = train_baseline(
                model, train_X, train_y, train_sids, cfg, device
            )

            strategy = {
                'average_adapters': 'average',
                'donor_random':     'donor_random',
                'donor_nearest':    'donor_nearest',
            }[args.condition]

            init_kwargs = dict(
                n_train_subjects = n_train_subjects,
                rng              = rng,
                layer_names      = cfg['meta_layers'],
                device           = device,
            )
            if strategy == 'donor_nearest':
                train_covs = {
                    s: subject_covariances(train_X[train_sids == s])
                    for s in np.unique(train_sids)
                }
                # Use calibration pool only (unlabelled train sessions of the
                # held-out subject) — never test data.
                init_kwargs['held_out_cov'] = subject_covariances(cal_X)
                init_kwargs['train_covs'] = train_covs

            init_info = apply_init(model, held_out_slot, strategy, **init_kwargs)
            print(f'Init: {init_info}')
            wandb.log(init_info)

    elif args.condition == 'compute_matched':
        # Reviewer 2 fairness control: baseline joint training, then extra
        # gradient steps so total compute matches the Reptile arm's
        # inner-loop spend. Held-out slot stays at zero-init.
        print(f'\n=== COMPUTE-MATCHED BASELINE ({args.dataset}) ===')
        # Reptile spends ~ n_train_subjects * inner_steps * updates inner steps.
        # With update_freq=0 (per-epoch), updates = epochs, so extra steps
        # ~= n_train_subjects * inner_steps * epochs.
        cfg['extra_grad_steps'] = int(
            n_train_subjects * cfg['inner_steps'] * cfg['epochs']
        )
        print(f'  Extra grad steps: {cfg["extra_grad_steps"]}')

        model = build_baseline_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, tracker = train_baseline(
            model, train_X, train_y, train_sids, cfg, device
        )
        print(f'\nHeld-out slot {held_out_slot} remains at zero initialisation')

    elif args.condition == 'maml':
        if not _HAS_MAML:
            raise RuntimeError('maml_trainer unavailable')
        print(f'\n=== MAML LoRA ({args.dataset}) ===')
        model = build_maml_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, meta_init, tracker = train_maml(
            model, train_X, train_y, train_sids, cfg, device
        )
        init_info = apply_init(model, held_out_slot, 'reptile',
                               meta_init=meta_init, device=device)
        wandb.log({'init': 'maml_init'})

    elif args.condition == 'hypernet':
        if not _HAS_HN:
            raise RuntimeError('hypernet unavailable')
        print(f'\n=== HYPERNETWORK LoRA ({args.dataset}) ===')
        # Train a joint baseline model, then train H to regress onto the
        # trained per-subject adapters, then init held-out from H(c_heldout).
        model = build_baseline_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, tracker = train_baseline(
            model, train_X, train_y, train_sids, cfg, device
        )

        # Build contexts and targets for regression
        layer_names = cfg['meta_layers'] or LORA_LAYER_NAMES
        contexts = {
            int(s): subject_context(train_X[train_sids == s])
            for s in np.unique(train_sids)
        }
        targets = {}
        for s in np.unique(train_sids):
            s = int(s)
            targets[s] = {}
            for name in layer_names:
                layer = getattr(model, name)
                targets[s][name] = {
                    'A': layer.lora_A[s].weight.data.clone(),
                    'B': layer.lora_B[s].weight.data.clone(),
                }

        hypernet = AdapterHypernetwork(
            model, layer_names=layer_names, n_channels=n_channels,
            device=device)
        hypernet = train_hypernetwork_regression(
            hypernet, contexts, targets,
            epochs=cfg.get('hn_epochs', 200),
            lr=cfg.get('hn_lr', 1e-3), device=device)

        # Context from calibration pool only (unlabelled train sessions
        # of the held-out subject) — never test data.
        held_ctx = subject_context(cal_X)
        hypernet.load_into_slot(model, held_out_slot, held_ctx)
        init_info = {'init': 'hypernetwork'}
        wandb.log(init_info)

    else:
        raise ValueError(f'Unknown condition: {args.condition}. '
                         f'Choose from: {CONDITIONS}')

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f'\n=== Evaluation on held-out subject {args.held_out} ({args.dataset}) ===')

    results = run_full_evaluation(
        model            = model,
        cal_X            = cal_X,
        cal_y            = cal_y,
        test_X           = test_X,
        test_y           = test_y,
        subject_slot     = held_out_slot,
        device           = device,
        condition_name   = args.condition,
        held_out_subject = held_out_0idx,
        seed             = args.seed,
    )

    results['dataset'] = args.dataset
    results.update(init_info)

    # ── Save checkpoint for downstream diagnostics ─────────────────────────────
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = (f'checkpoints/{args.condition}_{args.dataset}'
                 f'_heldout{args.held_out}_seed{args.seed}.pt')
    torch.save({
        'state_dict':  model.state_dict(),
        'condition':   args.condition,
        'dataset':     args.dataset,
        'held_out':    args.held_out,
        'seed':        args.seed,
        'n_train_subjects': n_train_subjects,
        'config':      cfg,
    }, ckpt_path)
    print(f'\nCheckpoint saved to {ckpt_path}')

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f'\n=== Final Summary ===')
    print(f'  Condition:     {args.condition}')
    print(f'  Dataset:       {args.dataset}')
    print(f'  Held-out:      subject {args.held_out}')
    print(f'  Seed:          {args.seed}')
    print(f'  Zero-shot acc: {results["zero_shot_acc"]:.3f}')
    for N in [5, 10, 20, 50]:
        key = f'few_shot_acc_N{N}'
        if key in results:
            print(f'  N={N:3d} acc:    {results[key]:.3f}')

    wandb.log(results)
    wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                        choices=CONDITIONS)
    parser.add_argument('--dataset',   type=str, default='BCI2a',
                        choices=['BCI2a', 'BCI2b'])
    parser.add_argument('--held_out',  type=int, required=True,
                        choices=list(range(1, 10)),
                        help='Subject to hold out (1-indexed, 1-9)')
    parser.add_argument('--seed',      type=int, default=1)
    parser.add_argument('--ea',        action='store_true',
                        help='Apply Euclidean Alignment preprocessing')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze EEGNeX backbone, train adapters only')
    parser.add_argument('--meta_lr',   type=float, default=None,
                        help='Outer/meta step size. Reptile: interpolation '
                             'coefficient (default 0.1). MAML: Adam LR on '
                             'meta-params (default 1e-3). Overrides config.')
    args = parser.parse_args()
    main(args)
