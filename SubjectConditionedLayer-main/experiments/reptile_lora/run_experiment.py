"""
Main entry point for Reptile LoRA vs Baseline LoRA experiments.

Usage:
    python run_experiment.py --condition baseline_lora --held_out 9 --seed 1
    python run_experiment.py --condition reptile_lora  --held_out 9 --seed 1

The held_out argument is 1-indexed (1-9) matching BCI2a subject numbering.

LOSO evaluation:
    Train on subjects {1..9} \ {held_out}
    Evaluate on held_out subject:
        zero_shot:      no calibration trials, no fine-tuning
        few_shot_N5:    fine-tune on 5  calibration trials
        few_shot_N10:   fine-tune on 10 calibration trials
        few_shot_N20:   fine-tune on 20 calibration trials
        few_shot_N50:   fine-tune on 50 calibration trials

All results logged to wandb project 'reptile_lora_bci'.
"""

import argparse
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, '../EEGNex')

from data_utils import load_bci2a, build_loso_split
from baseline_trainer import build_baseline_model, train_baseline
from reptile_trainer import build_reptile_model, train_reptile
from evaluate import run_full_evaluation
from meta_init import LORA_LAYER_NAMES


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    'freq_min': 8,
    'freq_max': 45,

    # Model (matching existing EEGNeX experiments)
    'rank':  8,
    'alpha': 24,      # alpha = 3 * rank as per existing EEGNeX.py

    # Backbone training
    'epochs':       100,
    'batch_size':   64,
    'lr':           1e-3,
    'weight_decay': 0.01,

    # Reptile inner loop
    'inner_steps': 5,      # K gradient steps per subject per epoch
    'inner_lr':    0.01,   # SGD learning rate for inner loop
    'inner_batch': 32,     # trials per inner step

    # Reptile outer update
    'meta_lr': 0.1,        # Reptile step size (epsilon)
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = DEFAULT_CONFIG.copy()

    # Convert held_out from 1-indexed (user-facing) to 0-indexed (internal)
    held_out_0idx = args.held_out - 1

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb.init(
        project='reptile_lora_bci',
        name=f'{args.condition}_heldout{args.held_out}_seed{args.seed}',
        config={**cfg,
                'condition':   args.condition,
                'held_out':    args.held_out,
                'seed':        args.seed},
        reinit=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f'\nLoading BCI2a | held-out subject: {args.held_out}')
    data, labels, subject_ids, sessions, _ = load_bci2a(
        freq_min=cfg['freq_min'],
        freq_max=cfg['freq_max'],
    )

    (train_X, train_y, train_sids,
     cal_X,   cal_y,   cal_sids,
     test_X,  test_y,  test_sids) = build_loso_split(
        data, labels, subject_ids, sessions, held_out_0idx
    )

    n_channels       = train_X.shape[1]
    n_times          = train_X.shape[2]
    n_classes        = len(np.unique(train_y))
    n_train_subjects = len(np.unique(train_sids))

    print(f'  Train: {len(train_X)} trials from {n_train_subjects} subjects')
    print(f'  Calibration pool: {len(cal_X)} trials from held-out subject')
    print(f'  Test: {len(test_X)} trials from held-out subject')
    print(f'  Channels: {n_channels} | Times: {n_times} | Classes: {n_classes}')

    # The held-out subject's adapter slot index
    # Training slots: 0 .. n_train_subjects-1
    # Held-out slot:  n_train_subjects
    held_out_slot = n_train_subjects

    # ── Build model and train ─────────────────────────────────────────────────
    if args.condition == 'baseline_lora':
        print(f'\n=== BASELINE LoRA ===')
        print('Held-out subject initialised from ZERO (original paper behaviour)')

        model = build_baseline_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        train_baseline(model, train_X, train_y, train_sids, cfg, device)

        # Held-out slot stays at zero — no further action needed
        print(f'\nHeld-out slot {held_out_slot} remains at zero initialisation')

    elif args.condition == 'reptile_lora':
        print(f'\n=== REPTILE LoRA ===')
        print('Held-out subject will be initialised from A_meta, B_meta')

        model = build_reptile_model(
            n_channels, n_classes, n_times,
            n_train_subjects, cfg, device,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {n_params:,}')
        wandb.log({'n_params': n_params})

        model, meta_init = train_reptile(
            model, train_X, train_y, train_sids, cfg, device
        )

        # Load meta-initialisation into held-out subject's slot
        print(f'\nLoading A_meta, B_meta into held-out slot {held_out_slot}')
        meta_init.load_into_slot(model, held_out_slot)

        # Log the norm of the meta-initialisation as a sanity check
        total_norm = 0.0
        for name in LORA_LAYER_NAMES:
            total_norm += meta_init.weights[name]['A'].norm().item()
            total_norm += meta_init.weights[name]['B'].norm().item()
        print(f'Meta-init total norm: {total_norm:.4f}')
        wandb.log({'meta_init_norm': total_norm})

    else:
        raise ValueError(f'Unknown condition: {args.condition}')

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f'\n=== Evaluation on held-out subject {args.held_out} ===')

    results = run_full_evaluation(
        model     = model,
        cal_X     = cal_X,
        cal_y     = cal_y,
        test_X    = test_X,
        test_y    = test_y,
        subject_slot = held_out_slot,
        device    = device,
        condition_name   = args.condition,
        held_out_subject = held_out_0idx,
        seed      = args.seed,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n=== Final Summary ===')
    print(f'  Condition:     {args.condition}')
    print(f'  Held-out:      subject {args.held_out}')
    print(f'  Seed:          {args.seed}')
    print(f'  Zero-shot acc: {results["zero_shot_acc"]:.3f}')
    for N in [5, 10, 20, 50]:
        key = f'few_shot_acc_N{N}'
        if key in results:
            print(f'  N={N:3d} acc:    {results[key]:.3f}')

    wandb.log(results)
    wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                        choices=['baseline_lora', 'reptile_lora'])
    parser.add_argument('--held_out', type=int, required=True,
                        choices=list(range(1, 10)),
                        help='Subject to hold out (1-indexed, 1-9)')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
