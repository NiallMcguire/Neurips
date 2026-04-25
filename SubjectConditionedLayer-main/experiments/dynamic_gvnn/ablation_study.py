"""
Ablation Study: Dynamic Subject-Conditioned GVNN layers vs baselines.

Six conditions tested on BCI Competition IV 2a (22ch, 4-class) and
optionally 2b (3ch, 2-class):

  vanilla          — EEGNeX, no subject conditioning, no GVNN
  static_lora      — EEGNeX + LoRA adapters in conv layers (original paper)
  gvnn_pop         — GVNN preprocessing (shared W_general) + EEGNeX vanilla
  gvnn_subject     — SubjectConditionedGVNNLayer + EEGNeX vanilla  [our method]
  gvnn_additive    — AdditiveSubjectDynamicLayer + EEGNeX vanilla  [ablation]
  full             — SubjectConditionedGVNNLayer + EEGNeX LoRA     [full combo]

Research questions answered:
  Q1. Does dynamic connectivity help over static LoRA?  (gvnn_pop vs static_lora)
  Q2. Does subject-specific dynamic graph help over shared dynamic graph?
      (gvnn_subject vs gvnn_pop)
  Q3. Is multiplicative Hadamard fusion necessary?      (gvnn_subject vs gvnn_additive)
  Q4. Do GVNN and LoRA combine beneficially?            (full vs each alone)

Usage:
  python ablation_study.py --condition gvnn_subject --dataset BCI2a --seed 1
  python ablation_study.py --condition static_lora   --dataset BCI2a --seed 2
"""

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, '../../')                 # repo root
from EEGNeX import EEGNeX
from utils import get_BNCI2014001, get_BNCI2014004
from dynamic_subject_layer import (
    GVNNPreprocessLayer,
    SubjectConditionedGVNNLayer,
    AdditiveSubjectDynamicLayer,
    GVNNWrappedEEGNeX,
)


# ── dataset ──────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.subject_id = torch.LongTensor(subject_ids)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_id[idx]


# ── model factory ────────────────────────────────────────────────────────────

def build_model(condition, n_channels, n_classes, n_times, n_subjects, cfg):
    """
    Returns the model for a given condition string.
    All subject IDs are expected in [0, n_subjects).
    """
    # LoRA EEGNeX backbone (for static_lora and full conditions)
    def make_lora_eegnex():
        return EEGNeX(
            n_chans=n_channels, n_outputs=n_classes, n_times=n_times,
            mode='LoRA', rank=cfg['rank'], alpha=cfg['alpha'],
            num_adapters=n_subjects,
        )

    # Vanilla EEGNeX backbone (for conditions where GVNN does the subject work)
    def make_vanilla_eegnex():
        return EEGNeX(
            n_chans=n_channels, n_outputs=n_classes, n_times=n_times,
            mode='vanilla',
        )

    if condition == 'vanilla':
        # Baseline: no subject conditioning at all
        backbone = make_vanilla_eegnex()
        # Wrap so forward(X, subject_id) works uniformly — subject_id ignored
        class VanillaWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, X, subject_id):
                return self.model(X, subject_id)
        return VanillaWrapper(backbone)

    elif condition == 'static_lora':
        # Original Subject-Conditioned Layer paper baseline
        return make_lora_eegnex()

    elif condition == 'gvnn_pop':
        # GVNN preprocessing (shared W_general) + vanilla EEGNeX
        gvnn = GVNNPreprocessLayer(
            n_channels=n_channels,
            node_fn=cfg['node_fn'],
            learnable_support=True,
        )
        backbone = make_vanilla_eegnex()
        return GVNNWrappedEEGNeX(gvnn, backbone)

    elif condition == 'gvnn_subject':
        # Our proposed method: subject-conditioned dynamic graph
        gvnn = SubjectConditionedGVNNLayer(
            n_channels=n_channels,
            n_subjects=n_subjects,
            rank=cfg['rank'],
            node_fn=cfg['node_fn'],
            learnable_general=True,
        )
        backbone = make_vanilla_eegnex()
        return GVNNWrappedEEGNeX(gvnn, backbone)

    elif condition == 'gvnn_additive':
        # Ablation: additive combination (tests whether Hadamard fusion needed)
        gvnn = AdditiveSubjectDynamicLayer(
            n_channels=n_channels,
            n_subjects=n_subjects,
            rank=cfg['rank'],
            node_fn=cfg['node_fn'],
        )
        backbone = make_vanilla_eegnex()
        return GVNNWrappedEEGNeX(gvnn, backbone)

    elif condition == 'full':
        # Full combination: subject-conditioned GVNN + LoRA in conv layers
        gvnn = SubjectConditionedGVNNLayer(
            n_channels=n_channels,
            n_subjects=n_subjects,
            rank=cfg['rank'],
            node_fn=cfg['node_fn'],
            learnable_general=True,
        )
        backbone = make_lora_eegnex()
        return GVNNWrappedEEGNeX(gvnn, backbone)

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ── data loading ─────────────────────────────────────────────────────────────

def load_data(dataset, cfg):
    if dataset == 'BCI2a':
        data, labels, meta, channels = get_BNCI2014001(
            subject=list(range(1, 10)),
            freq_min=cfg['freq'][0],
            freq_max=cfg['freq'][1],
        )
        train_mask = meta['session'] == '0train'
        test_mask  = meta['session'] == '1test'
        crop = slice(244, 756)       # 512 samples as per existing code

    elif dataset == 'BCI2b':
        data, labels, meta, channels = get_BNCI2014004(
            subject=list(range(1, 10)),
            freq_min=cfg['freq'][0],
            freq_max=cfg['freq'][1],
        )
        train_mask = (
            (meta['session'] == 'session_0') |
            (meta['session'] == 'session_1') |
            (meta['session'] == 'session_2')
        )
        test_mask = (
            (meta['session'] == 'session_3') |
            (meta['session'] == 'session_4')
        )
        crop = slice(None)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_data   = data[train_mask][:, :, crop]
    train_labels = labels[train_mask]
    train_meta   = meta[train_mask]

    test_data   = data[test_mask][:, :, crop]
    test_labels = labels[test_mask]
    test_meta   = meta[test_mask]

    # Map subject IDs (1-9) to zero-indexed (0-8) for adapter indexing
    def remap(m):
        subjects = np.array(m['subject'].values)
        return subjects - subjects.min()

    train_sids = remap(train_meta)
    test_sids  = remap(test_meta)

    return (train_data, train_labels, train_sids,
            test_data,  test_labels,  test_sids)


# ── metrics ──────────────────────────────────────────────────────────────────

def cohen_kappa(y_true, y_pred, n_classes):
    """Cohen's kappa from numpy arrays."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred)


def per_subject_accuracy(y_true, y_pred, subject_ids):
    """Dict of accuracy per subject."""
    accs = {}
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        accs[int(sid)] = (y_pred[mask] == y_true[mask]).mean()
    return accs


# ── training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y, sid in loader:
        X, y, sid = X.to(device), y.to(device), sid.to(device)
        optimizer.zero_grad()
        logits = model(X, sid)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_pred, all_true, all_sid = 0.0, [], [], []
    for X, y, sid in loader:
        X, y, sid = X.to(device), y.to(device), sid.to(device)
        logits = model(X, sid)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        all_pred.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(y.cpu().numpy())
        all_sid.extend(sid.cpu().numpy())
    n = len(all_true)
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    all_sid  = np.array(all_sid)
    acc = (all_pred == all_true).mean()
    return total_loss / n, acc, all_pred, all_true, all_sid


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    cfg = {
        'freq':     [8, 45],
        'rank':     args.rank,
        'alpha':    args.alpha,
        'node_fn':  args.node_fn,
        'epochs':   args.epochs,
        'batch_size': args.batch_size,
        'lr':       args.lr,
        'weight_decay': 0.01,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── wandb ──────────────────────────────────────────────────────────────
    wandb.init(
        project='dynamic_gvnn_ablation',
        name=f'{args.condition}_{args.dataset}_seed{args.seed}',
        config={**cfg, 'condition': args.condition, 'dataset': args.dataset,
                'seed': args.seed},
        reinit=True,
    )

    # ── data ───────────────────────────────────────────────────────────────
    print(f'Loading {args.dataset}...')
    (train_X, train_y, train_sids,
     test_X,  test_y,  test_sids) = load_data(args.dataset, cfg)

    n_subjects = int(train_sids.max()) + 1
    n_channels = train_X.shape[1]
    n_times    = train_X.shape[2]
    n_classes  = len(np.unique(train_y))

    print(f'  Subjects: {n_subjects} | Channels: {n_channels} | '
          f'Times: {n_times} | Classes: {n_classes}')
    print(f'  Train: {len(train_X)} | Test: {len(test_X)}')

    train_ds = EEGDataset(train_X, train_y, train_sids)
    test_ds  = EEGDataset(test_X,  test_y,  test_sids)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  drop_last=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'],
                              shuffle=False, drop_last=False, num_workers=2)

    # ── model ──────────────────────────────────────────────────────────────
    print(f'Building model: {args.condition}')
    model = build_model(args.condition, n_channels, n_classes,
                        n_times, n_subjects, cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable parameters: {n_params:,}')
    wandb.log({'n_params': n_params})

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )

    # Cosine LR schedule with linear warmup (matches PBT paper)
    warmup_steps = 10
    total_steps  = cfg['epochs']
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── training ───────────────────────────────────────────────────────────
    best_test_acc = 0.0
    epoch_times   = []

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        test_loss, test_acc, pred, true, sids = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        kappa = cohen_kappa(true, pred, n_classes)
        best_test_acc = max(best_test_acc, test_acc)

        log_dict = {
            'epoch':      epoch,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'test_loss':  test_loss,
            'test_acc':   test_acc,
            'kappa':      kappa,
            'epoch_time': epoch_time,
            'lr':         scheduler.get_last_lr()[0],
        }

        # Per-subject accuracy every 10 epochs
        if epoch % 10 == 0:
            per_sub = per_subject_accuracy(true, pred, sids)
            for sid, acc in per_sub.items():
                log_dict[f'acc_subject_{sid+1}'] = acc

        wandb.log(log_dict)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | '
                  f'train {train_acc:.3f} | test {test_acc:.3f} | '
                  f'kappa {kappa:.3f} | {epoch_time:.1f}s')

    # ── final summary ───────────────────────────────────────────────────────
    _, final_acc, pred, true, sids = evaluate(model, test_loader, criterion, device)
    final_kappa   = cohen_kappa(true, pred, n_classes)
    per_sub_final = per_subject_accuracy(true, pred, sids)
    mean_epoch_t  = np.mean(epoch_times)

    summary = {
        'final_test_acc':   final_acc,
        'best_test_acc':    best_test_acc,
        'final_kappa':      final_kappa,
        'mean_epoch_time':  mean_epoch_t,
        'n_params':         n_params,
        'condition':        args.condition,
        'dataset':          args.dataset,
        'seed':             args.seed,
    }
    for sid, acc in per_sub_final.items():
        summary[f'final_acc_subject_{sid+1}'] = acc

    wandb.log(summary)
    print('\n=== Final Results ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                        choices=['vanilla', 'static_lora', 'gvnn_pop',
                                 'gvnn_subject', 'gvnn_additive', 'full'])
    parser.add_argument('--dataset',   type=str, default='BCI2a',
                        choices=['BCI2a', 'BCI2b'])
    parser.add_argument('--seed',      type=int, default=1)
    parser.add_argument('--epochs',    type=int, default=100)
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--rank',      type=int, default=4)
    parser.add_argument('--alpha',     type=float, default=1.0)
    parser.add_argument('--node_fn',   type=str, default='combined',
                        choices=['corr', 'lde', 'combined'])
    args = parser.parse_args()
    main(args)
