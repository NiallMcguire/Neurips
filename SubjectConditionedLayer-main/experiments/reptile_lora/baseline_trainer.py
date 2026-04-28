"""
Baseline trainer: original Subject-Conditioned Layer paper approach.

Trains EEGNeX with LoRAConvPerSubject adapters on n_training_subjects.
Adapters are jointly trained with the backbone.
The held-out subject's adapter slot is left at zero initialisation
(exactly as in the original paper for unseen subjects).

At evaluation time, the held-out subject's slot is optionally fine-tuned
on N calibration trials from the evaluate.py few-shot function.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.insert(0, '../EEGNex')
from EEGNeX import EEGNeX

from data_utils import EEGDataset


def build_baseline_model(n_channels, n_classes, n_times,
                         n_train_subjects, config, device):
    """
    Build EEGNeX in LoRA mode with n_train_subjects + 1 adapter slots.
    The extra slot (index n_train_subjects) is the held-out subject slot.
    It is never updated during training, so it stays at zero.
    """
    model = EEGNeX(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        mode='LoRA',
        rank=config['rank'],
        alpha=config['alpha'],
        num_adapters=n_train_subjects + 1,   # +1 for held-out subject
    ).to(device)
    return model


def train_baseline(model, train_X, train_y, train_sids, config, device):
    """
    Standard joint training on all training subjects.
    Identical to the original EEGNeX.py training loop.

    train_sids are 0-indexed and already remapped to 0..n_train-1.
    The held-out subject slot (n_train_subjects) is never seen during training.
    """
    dataset = EEGDataset(train_X, train_y, train_sids)
    loader  = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    print(f'Baseline training | epochs={config["epochs"]} | '
          f'batch={config["batch_size"]}')

    for epoch in range(config['epochs']):
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for X_batch, y_batch, sid_batch in loader:
            X_batch   = X_batch.to(device)
            y_batch   = y_batch.to(device)
            sid_batch = sid_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch, sid_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            n          += len(y_batch)

        acc = correct / n

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | loss {total_loss/n:.4f} | acc {acc:.3f}')

        wandb.log({
            'epoch':      epoch,
            'train_loss': total_loss / n,
            'train_acc':  acc,
        })

    return model
