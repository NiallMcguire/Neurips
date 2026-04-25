"""
Cross-Subject Evaluation: Leave-One-Subject-Out (LOSO).

Directly addresses the core limitation of the Subject-Conditioned Layer paper:
"for previously unseen subjects, predictions can only be made using the
shared weights W_general" (Klein et al., 2025, Section 6).

For each held-out subject s in 1..9:
  - Train on subjects {1..9} \ {s}
  - Evaluate on subject s in three modes:

    zero_shot   — model uses only W_general (no subject-specific adapters)
                  For GVNN: A_s, B_s initialised to zero (W_s = W_general)
                  For LoRA: subject_id set to -1 (adapter never activates)

    few_shot_N  — model's subject-specific parameters (A_s, B_s or LoRA)
                  fine-tuned on N=5,10,20,50 calibration trials from s,
                  then evaluated on remaining trials.

Research questions answered:
  Q5. Does W_general from gvnn_subject generalise better to new subjects
      than W_general from static_lora?  (zero_shot comparison)
  Q6. Does GVNN's subject-specific adapter fine-tune faster/better than
      LoRA adapters with few calibration trials?  (few_shot comparison)

Usage:
  python cross_subject_eval.py --condition gvnn_subject --held_out 9 --seed 1
  python cross_subject_eval.py --condition static_lora  --held_out 9 --seed 1
"""

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import wandb
from sklearn.model_selection import train_test_split

sys.path.insert(0, '../../')
from EEGNeX import EEGNeX
from utils import get_BNCI2014001
from dynamic_subject_layer import (
    GVNNPreprocessLayer,
    SubjectConditionedGVNNLayer,
    AdditiveSubjectDynamicLayer,
    GVNNWrappedEEGNeX,
)
from ablation_study import (
    EEGDataset, build_model, train_one_epoch, evaluate,
    cohen_kappa, per_subject_accuracy,
)


FEW_SHOT_NS = [5, 10, 20, 50]
ALL_SUBJECTS = list(range(1, 10))


# ── helpers ──────────────────────────────────────────────────────────────────

def get_subject_params(model, condition):
    """
    Return only the subject-specific parameters of the model.
    These are the parameters fine-tuned during few-shot adaptation.
    """
    if condition in ('gvnn_subject', 'gvnn_additive'):
        # lora_A and lora_B inside the GVNN layer
        return [model.gvnn.lora_A, model.gvnn.lora_B]
    elif condition == 'full':
        # GVNN adapters + LoRA adapters inside backbone
        gvnn_params = [model.gvnn.lora_A, model.gvnn.lora_B]
        lora_params = [p for n, p in model.backbone.named_parameters()
                       if 'lora_A' in n or 'lora_B' in n]
        return gvnn_params + lora_params
    elif condition == 'static_lora':
        return [p for n, p in model.named_parameters()
                if 'lora_A' in n or 'lora_B' in n]
    else:
        # vanilla / gvnn_pop — no subject-specific parameters, fine-tune all
        return list(model.parameters())


def zero_shot_subject_id(condition, device):
    """
    Return a subject_id value that effectively disables subject-specific
    adaptation, forcing the model to use only the shared W_general.

    For GVNN: return an index whose A, B are initialised to zero (the
              held-out subject gets a fresh slot initialised to zero).
    For LoRA: return -1 so no adapter mask activates (matches EEGNeX.py
              convention from the original codebase).
    """
    if condition in ('gvnn_subject', 'gvnn_additive', 'full'):
        # Use subject slot 8 (held-out subject) which was NOT trained
        # The slot was initialised with B=0 so W_s = W_general
        return None   # handled per-sample below
    else:
        return -1


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    cfg = {
        'freq': [8, 45], 'rank': args.rank, 'alpha': args.alpha,
        'node_fn': args.node_fn, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr, 'weight_decay': 0.01,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project='dynamic_gvnn_loso',
        name=f'{args.condition}_heldout{args.held_out}_seed{args.seed}',
        config={**cfg, 'condition': args.condition,
                'held_out': args.held_out, 'seed': args.seed},
        reinit=True,
    )

    # ── load all data ───────────────────────────────────────────────────────
    print(f'Loading BCI2a, held-out subject: {args.held_out}')
    data, labels, meta, _ = get_BNCI2014001(
        subject=ALL_SUBJECTS,
        freq_min=cfg['freq'][0], freq_max=cfg['freq'][1],
    )
    data = data[:, :, 244:756]

    # subject IDs remapped to 0-indexed; held-out gets index = held_out - 1
    subjects = np.array(meta['subject'].values)
    subject_ids_0 = subjects - 1          # 1-9 -> 0-8

    train_mask = subjects != args.held_out
    test_mask  = subjects == args.held_out

    train_X   = data[train_mask];   train_y   = labels[train_mask]
    train_sid = subject_ids_0[train_mask]
    test_X    = data[test_mask];    test_y    = labels[test_mask]
    test_sid  = subject_ids_0[test_mask]   # index = held_out - 1

    n_subjects  = len(ALL_SUBJECTS)
    n_channels  = train_X.shape[1]
    n_times     = train_X.shape[2]
    n_classes   = len(np.unique(train_y))

    train_ds = EEGDataset(train_X, train_y, train_sid)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, drop_last=True, num_workers=2)

    # ── build and train on seen subjects ────────────────────────────────────
    print(f'Training {args.condition} on subjects {ALL_SUBJECTS} \\ {args.held_out}')
    model = build_model(args.condition, n_channels, n_classes,
                        n_times, n_subjects, cfg)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )

    for epoch in range(cfg['epochs']):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | train acc {t_acc:.3f}')
        wandb.log({'train_epoch': epoch, 'train_acc_loso': t_acc,
                   'train_loss_loso': t_loss})

    # ── zero-shot evaluation on held-out subject ────────────────────────────
    # For GVNN: use held-out subject's adapter slot (initialised with B=0
    # so W_s = W_general — no task-specific information has been provided).
    # For LoRA: set subject_id to -1 so no adapter activates.
    print(f'\nZero-shot evaluation on subject {args.held_out}')

    if args.condition in ('vanilla', 'static_lora', 'gvnn_pop'):
        zero_shot_sid = np.full(len(test_X), -1, dtype=int)
    else:
        # Use the held-out subject's adapter slot (never updated during training)
        zero_shot_sid = test_sid.copy()

    zs_ds     = EEGDataset(test_X, test_y, zero_shot_sid)
    zs_loader = DataLoader(zs_ds, batch_size=cfg['batch_size'],
                           shuffle=False, num_workers=2)

    _, zs_acc, zs_pred, zs_true, _ = evaluate(model, zs_loader, criterion, device)
    zs_kappa = cohen_kappa(zs_true, zs_pred, n_classes)
    print(f'  Zero-shot acc: {zs_acc:.3f} | kappa: {zs_kappa:.3f}')

    wandb.log({
        'zero_shot_acc':   zs_acc,
        'zero_shot_kappa': zs_kappa,
        'held_out_subject': args.held_out,
    })

    # ── few-shot fine-tuning ────────────────────────────────────────────────
    # Freeze everything except subject-specific parameters, then fine-tune
    # on N calibration trials from the held-out subject.
    print(f'\nFew-shot fine-tuning (N = {FEW_SHOT_NS})')

    # Identify subject-specific parameters
    subject_params = get_subject_params(model, args.condition)
    subject_param_ids = set(id(p) for p in subject_params)

    # Freeze all params then unfreeze subject-specific ones
    for p in model.parameters():
        p.requires_grad = False
    for p in subject_params:
        p.requires_grad = True

    # Split held-out data: calibration pool + evaluation set
    # Stratified split so class balance is maintained
    cal_pool_X, eval_X, cal_pool_y, eval_y, cal_pool_sid, eval_sid_arr = (
        train_test_split(test_X, test_y, test_sid,
                         test_size=0.5, stratify=test_y, random_state=args.seed)
    )

    eval_ds     = EEGDataset(eval_X, eval_y, eval_sid_arr)
    eval_loader = DataLoader(eval_ds, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=2)

    for N in FEW_SHOT_NS:
        # Re-load model from the trained state each time so N-shot comparisons
        # are independent (don't benefit from previous N-shot fine-tuning)
        model_n = build_model(args.condition, n_channels, n_classes,
                              n_times, n_subjects, cfg)
        model_n.load_state_dict(model.state_dict())
        model_n = model_n.to(device)

        # Freeze again
        for p in model_n.parameters():
            p.requires_grad = False
        for p in get_subject_params(model_n, args.condition):
            p.requires_grad = True

        # Sample N calibration trials (stratified by class)
        if N >= len(cal_pool_X):
            cal_idx = np.arange(len(cal_pool_X))
        else:
            cal_idx = []
            for cls in np.unique(cal_pool_y):
                cls_idx = np.where(cal_pool_y == cls)[0]
                n_cls = max(1, N // n_classes)
                chosen = np.random.choice(cls_idx, size=n_cls, replace=False)
                cal_idx.extend(chosen.tolist())
            cal_idx = np.array(cal_idx)

        cal_X_n   = cal_pool_X[cal_idx]
        cal_y_n   = cal_pool_y[cal_idx]
        cal_sid_n = cal_pool_sid[cal_idx]

        cal_ds_n  = EEGDataset(cal_X_n, cal_y_n, cal_sid_n)
        cal_loader_n = DataLoader(cal_ds_n, batch_size=min(N, 16),
                                  shuffle=True, num_workers=0)

        opt_n = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_n.parameters()),
            lr=cfg['lr'] * 0.1,     # lower LR for fine-tuning
            weight_decay=0.01,
        )

        # Fine-tune for 20 epochs
        for _ in range(20):
            train_one_epoch(model_n, cal_loader_n, criterion, opt_n, device)

        _, fs_acc, fs_pred, fs_true, _ = evaluate(
            model_n, eval_loader, criterion, device
        )
        fs_kappa = cohen_kappa(fs_true, fs_pred, n_classes)
        print(f'  N={N:3d} | acc {fs_acc:.3f} | kappa {fs_kappa:.3f}')

        wandb.log({
            f'few_shot_acc_N{N}':   fs_acc,
            f'few_shot_kappa_N{N}': fs_kappa,
            'held_out_subject': args.held_out,
            'N_shots': N,
        })

    wandb.finish()
    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                        choices=['vanilla', 'static_lora', 'gvnn_pop',
                                 'gvnn_subject', 'gvnn_additive', 'full'])
    parser.add_argument('--held_out', type=int, required=True,
                        choices=list(range(1, 10)))
    parser.add_argument('--seed',       type=int, default=1)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--rank',       type=int, default=4)
    parser.add_argument('--alpha',      type=float, default=1.0)
    parser.add_argument('--node_fn',    type=str, default='combined',
                        choices=['corr', 'lde', 'combined'])
    args = parser.parse_args()
    main(args)
