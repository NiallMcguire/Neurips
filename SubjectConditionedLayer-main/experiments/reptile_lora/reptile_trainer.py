"""
Reptile trainer: meta-trains A_meta, B_meta alongside standard backbone training.

Architecture: identical EEGNeX LoRA to baseline.
Difference:   after each training epoch, a Reptile meta-update moves
              A_meta, B_meta toward each training subject's adapted parameters.

At evaluation time, the held-out subject's adapter slot is initialised from
A_meta, B_meta rather than zeros. This is the only difference from baseline.

Reptile loop per iteration:
    FOR each training subject s:
        1. Load A_meta, B_meta into subject s's adapter slot
        2. Run K inner gradient steps on subject s's data
        3. Collect phi_s = updated adapter weights
    Reptile update: A_meta += meta_lr * mean(phi_s - A_meta)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.insert(0, '../EEGNex')
from EEGNeX import EEGNeX

from data_utils import EEGDataset, build_per_subject_dict
from meta_init import MetaInit, get_adapter_params, LORA_LAYER_NAMES


def build_reptile_model(n_channels, n_classes, n_times,
                        n_train_subjects, config, device):
    """
    Identical model to baseline. Extra slot for held-out subject.
    """
    model = EEGNeX(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        mode='LoRA',
        rank=config['rank'],
        alpha=config['alpha'],
        num_adapters=n_train_subjects + 1,
    ).to(device)
    return model


# ── Inner loop ────────────────────────────────────────────────────────────────

def inner_loop(model, meta_init, subject_slot, X_s, y_s,
               K, inner_lr, inner_batch, device):
    """
    Run K gradient steps on subject s's data starting from A_meta, B_meta.

    1. Load meta weights into subject_slot
    2. Build optimiser over subject_slot's adapter params only
    3. Run K SGD steps
    4. Return updated weights (phi_K)

    The backbone and all other slots are untouched.
    """
    # Step 1: initialise this slot from meta
    meta_init.load_into_slot(model, subject_slot)

    # Step 2: optimiser over this slot's adapter params only
    adapter_params = get_adapter_params(model, subject_slot)
    inner_opt = torch.optim.SGD(adapter_params, lr=inner_lr, momentum=0.9)
    criterion  = nn.CrossEntropyLoss()

    n_trials = len(X_s)

    # Step 3: K gradient steps
    model.train()
    for k in range(K):
        idx     = torch.randperm(n_trials)[:inner_batch]
        X_batch = X_s[idx].to(device)
        y_batch = y_s[idx].to(device)
        sid     = torch.full((len(idx),), subject_slot,
                             dtype=torch.long, device=device)

        inner_opt.zero_grad()
        logits = model(X_batch, sid)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        inner_opt.step()

    # Step 4: collect phi_K
    phi = meta_init.extract_from_slot(model, subject_slot)
    return phi


# ── Full Reptile training ─────────────────────────────────────────────────────

def train_reptile(model, train_X, train_y, train_sids, config, device):
    """
    Joint backbone training + Reptile meta-update.

    Phase 1 (outer step): standard AdamW on the full model for one epoch.
    Phase 2 (Reptile step): for each subject, inner loop K steps,
                            then Reptile update of meta-init.

    The two phases alternate each epoch. This keeps the backbone and
    meta-initialisation co-evolving, which is important because A_meta
    should match the backbone's current feature space.
    """
    # Build per-subject dict for inner loop
    subject_data = build_per_subject_dict(train_X, train_y, train_sids)
    n_subjects   = len(subject_data)

    # Meta-initialisation — starts from model's initial adapter weights
    meta_init = MetaInit(model, device)

    # Full dataset for outer (backbone) training step
    dataset = EEGDataset(train_X, train_y, train_sids)
    loader  = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    criterion  = nn.CrossEntropyLoss()
    outer_opt  = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    K          = config['inner_steps']
    inner_lr   = config['inner_lr']
    inner_batch = config['inner_batch']
    meta_lr    = config['meta_lr']

    print(f'Reptile training | epochs={config["epochs"]} | K={K} | '
          f'inner_lr={inner_lr} | meta_lr={meta_lr}')

    for epoch in range(config['epochs']):

        # ── Phase 1: outer backbone step ───────────────────────────────────
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for X_batch, y_batch, sid_batch in loader:
            X_batch   = X_batch.to(device)
            y_batch   = y_batch.to(device)
            sid_batch = sid_batch.to(device)

            outer_opt.zero_grad()
            logits = model(X_batch, sid_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            outer_opt.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            n          += len(y_batch)

        train_acc  = correct / n
        train_loss = total_loss / n

        # ── Phase 2: Reptile meta-update ───────────────────────────────────
        # For each training subject, run inner loop from meta-init
        # and collect the resulting phi_K
        phi_list = []
        for sid, (X_s, y_s) in subject_data.items():
            phi = inner_loop(
                model, meta_init, sid,
                X_s, y_s, K, inner_lr, inner_batch, device,
            )
            phi_list.append(phi)

        # Reptile update: move meta toward average of all phi_K
        delta_norm = meta_init.delta_norm(phi_list)
        meta_init.reptile_update(phi_list, meta_lr)

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | loss {train_loss:.4f} | '
                  f'acc {train_acc:.3f} | delta_norm {delta_norm:.4f}')

        wandb.log({
            'epoch':        epoch,
            'train_loss':   train_loss,
            'train_acc':    train_acc,
            'delta_norm':   delta_norm,
        })

    return model, meta_init
