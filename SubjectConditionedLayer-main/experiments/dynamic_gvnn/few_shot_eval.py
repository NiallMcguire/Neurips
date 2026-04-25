"""
Few-Shot Adaptation Experiment.

Tests how quickly each method adapts to a new subject with minimal data.

A model pre-trained on all 9 subjects is fine-tuned using only the
subject-specific parameters (GVNN adapters or LoRA weights) on N
calibration trials from a target subject, where N in [5, 10, 20, 50, 100].

The key question is adaptation efficiency:
  - Which method reaches usable accuracy with the fewest trials?
  - Does the richer dynamic representation in GVNN adapters make
    them easier to fine-tune than static LoRA adapters?

Metrics logged:
  - Accuracy vs N curve (per condition, per subject)
  - Kappa vs N curve
  - Number of parameters updated during fine-tuning
  - Fine-tuning time per N

Comparisons:
  static_lora  vs  gvnn_subject  — same parameter count, different structure
  gvnn_subject vs  full          — GVNN adapters alone vs full combination

Usage:
  python few_shot_eval.py --condition gvnn_subject --target_subject 1 --seed 1
"""

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, '../../')
from EEGNeX import EEGNeX
from utils import get_BNCI2014001
from dynamic_subject_layer import (
    SubjectConditionedGVNNLayer,
    GVNNWrappedEEGNeX,
)
from ablation_study import (
    EEGDataset, build_model, train_one_epoch, evaluate,
    cohen_kappa, load_data,
)
from cross_subject_eval import get_subject_params

FEW_SHOT_NS   = [5, 10, 20, 50, 100]
FINETUNE_EPOCHS = 30
ALL_SUBJECTS  = list(range(1, 10))


def pretrain_model(condition, n_channels, n_classes, n_times, n_subjects, cfg,
                   train_X, train_y, train_sid, device):
    """Train model on all subjects and return it."""
    model = build_model(condition, n_channels, n_classes, n_times, n_subjects, cfg)
    model = model.to(device)

    ds = EEGDataset(train_X, train_y, train_sid)
    loader = DataLoader(ds, batch_size=cfg['batch_size'],
                        shuffle=True, drop_last=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'], weight_decay=0.01)

    print(f'  Pre-training {condition} on {len(train_X)} trials...')
    for epoch in range(cfg['epochs']):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        if epoch % 20 == 0:
            print(f'    Epoch {epoch:3d} | loss {loss:.4f} | acc {acc:.3f}')

    return model


def few_shot_finetune(model, condition, target_sid, cal_X, cal_y, cal_sid,
                      eval_X, eval_y, eval_sid, N, cfg, device):
    """
    Fine-tune subject-specific parameters on N calibration trials.
    Returns (accuracy, kappa, fine-tuning time in seconds).
    """
    # Deep copy model state so different N values are independent
    import copy
    model_n = copy.deepcopy(model)
    model_n = model_n.to(device)

    # Freeze all, then unfreeze subject-specific params only
    for p in model_n.parameters():
        p.requires_grad = False
    subject_params = get_subject_params(model_n, condition)
    n_finetune_params = 0
    for p in subject_params:
        p.requires_grad = True
        n_finetune_params += p.numel()

    # Sample N stratified calibration trials
    n_classes = len(np.unique(cal_y))
    if N >= len(cal_X):
        idx = np.arange(len(cal_X))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                                     train_size=N, random_state=42)
        idx, _ = next(sss.split(cal_X, cal_y))

    cal_X_n   = cal_X[idx]
    cal_y_n   = cal_y[idx]
    cal_sid_n = cal_sid[idx]

    cal_ds = EEGDataset(cal_X_n, cal_y_n, cal_sid_n)
    cal_loader = DataLoader(cal_ds, batch_size=min(N, 16),
                            shuffle=True, num_workers=0)

    eval_ds     = EEGDataset(eval_X, eval_y, eval_sid)
    eval_loader = DataLoader(eval_ds, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_n.parameters()),
        lr=cfg['lr'] * 0.1,
        weight_decay=0.01,
    )

    t0 = time.time()
    for _ in range(FINETUNE_EPOCHS):
        train_one_epoch(model_n, cal_loader, criterion, opt, device)
    ft_time = time.time() - t0

    _, acc, pred, true, _ = evaluate(model_n, eval_loader, criterion, device)
    kappa = cohen_kappa(true, pred, n_classes)

    return acc, kappa, ft_time, n_finetune_params


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    cfg = {
        'freq': [8, 45], 'rank': args.rank, 'alpha': args.alpha,
        'node_fn': args.node_fn, 'epochs': args.pretrain_epochs,
        'batch_size': args.batch_size, 'lr': args.lr, 'weight_decay': 0.01,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project='dynamic_gvnn_fewshot',
        name=f'{args.condition}_sub{args.target_subject}_seed{args.seed}',
        config={**cfg, 'condition': args.condition,
                'target_subject': args.target_subject, 'seed': args.seed,
                'finetune_epochs': FINETUNE_EPOCHS},
        reinit=True,
    )

    # ── load data ───────────────────────────────────────────────────────────
    print(f'Loading BCI2a...')
    data, labels, meta, _ = get_BNCI2014001(
        subject=ALL_SUBJECTS,
        freq_min=cfg['freq'][0], freq_max=cfg['freq'][1],
    )
    data = data[:, :, 244:756]

    subjects     = np.array(meta['subject'].values)
    subject_ids  = subjects - 1       # 0-indexed

    # Pre-train on all subjects (including target — simulates realistic
    # scenario where model was trained on a large population dataset)
    train_mask = meta['session'] == '0train'
    test_mask  = meta['session'] == '1test'

    train_X   = data[train_mask];  train_y   = labels[train_mask]
    train_sid = subject_ids[train_mask]
    test_X    = data[test_mask];   test_y    = labels[test_mask]
    test_sid  = subject_ids[test_mask]

    n_subjects = len(ALL_SUBJECTS)
    n_channels = train_X.shape[1]
    n_times    = train_X.shape[2]
    n_classes  = len(np.unique(train_y))

    # ── pre-train ───────────────────────────────────────────────────────────
    print(f'\nPre-training {args.condition}...')
    model = pretrain_model(
        args.condition, n_channels, n_classes, n_times, n_subjects, cfg,
        train_X, train_y, train_sid, device,
    )

    # Overall pre-train performance (before any per-subject adaptation)
    full_test_ds = EEGDataset(test_X, test_y, test_sid)
    full_loader  = DataLoader(full_test_ds, batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=2)
    criterion    = nn.CrossEntropyLoss()
    _, pretrain_acc, _, _, _ = evaluate(model, full_loader, criterion, device)
    wandb.log({'pretrain_test_acc': pretrain_acc})
    print(f'Pre-train test acc (all subjects): {pretrain_acc:.3f}')

    # ── few-shot on target subject ───────────────────────────────────────────
    target_sid_val = args.target_subject - 1    # 0-indexed

    # Split target subject's test data into calibration pool + evaluation
    tgt_mask_test  = test_sid == target_sid_val
    tgt_X   = test_X[tgt_mask_test]
    tgt_y   = test_y[tgt_mask_test]
    tgt_sid_arr = test_sid[tgt_mask_test]

    # 50% calibration pool / 50% eval — fixed split so all N are comparable
    split = int(0.5 * len(tgt_X))
    cal_X   = tgt_X[:split];   cal_y   = tgt_y[:split]
    cal_sid = tgt_sid_arr[:split]
    eval_X  = tgt_X[split:];   eval_y  = tgt_y[split:]
    eval_sid_arr = tgt_sid_arr[split:]

    print(f'\nFew-shot fine-tuning on subject {args.target_subject}')
    print(f'  Calibration pool: {len(cal_X)} | Eval: {len(eval_X)}')

    # Zero-shot baseline (no fine-tuning)
    eval_ds_base  = EEGDataset(eval_X, eval_y, eval_sid_arr)
    eval_loader_b = DataLoader(eval_ds_base, batch_size=cfg['batch_size'],
                               shuffle=False, num_workers=0)
    _, zs_acc, zs_pred, zs_true, _ = evaluate(
        model, eval_loader_b, criterion, device
    )
    zs_kappa = cohen_kappa(zs_true, zs_pred, n_classes)
    print(f'  Zero-shot (N=0) | acc {zs_acc:.3f} | kappa {zs_kappa:.3f}')
    wandb.log({
        'few_shot_N': 0, 'few_shot_acc': zs_acc, 'few_shot_kappa': zs_kappa,
        'finetune_time': 0.0,
    })

    # Fine-tuning for each N
    for N in FEW_SHOT_NS:
        if N > len(cal_X):
            print(f'  N={N} exceeds calibration pool ({len(cal_X)}), skipping')
            continue

        acc, kappa, ft_time, n_ft_params = few_shot_finetune(
            model, args.condition, target_sid_val,
            cal_X, cal_y, cal_sid,
            eval_X, eval_y, eval_sid_arr,
            N, cfg, device,
        )

        print(f'  N={N:3d} | acc {acc:.3f} | kappa {kappa:.3f} | '
              f'{ft_time:.1f}s | params {n_ft_params:,}')

        wandb.log({
            'few_shot_N':          N,
            'few_shot_acc':        acc,
            'few_shot_kappa':      kappa,
            'finetune_time':       ft_time,
            'n_finetune_params':   n_ft_params,
            'target_subject':      args.target_subject,
            'condition':           args.condition,
        })

    wandb.finish()
    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                        choices=['vanilla', 'static_lora', 'gvnn_pop',
                                 'gvnn_subject', 'gvnn_additive', 'full'])
    parser.add_argument('--target_subject', type=int, required=True,
                        choices=list(range(1, 10)))
    parser.add_argument('--seed',            type=int, default=1)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size',      type=int, default=64)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--rank',            type=int, default=4)
    parser.add_argument('--alpha',           type=float, default=1.0)
    parser.add_argument('--node_fn',         type=str, default='combined',
                        choices=['corr', 'lde', 'combined'])
    args = parser.parse_args()
    main(args)
