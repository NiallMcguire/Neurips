"""
Reptile trainer with full hyperparameter ablation support.

Three key variables explored:

1. K (inner_steps): number of inner gradient steps per subject.
   Controls how far phi_K moves from the meta-initialisation.
   Tested: {1, 5, 10, 20}

2. inner_lr: learning rate for the inner loop SGD.
   Should ideally match the adaptation LR used at test time.
   Tested: {0.001, 0.005, 0.01, 0.05}

3. update_freq: how often the Reptile meta-update fires.
   0  = once per epoch (original behaviour)
   M  = every M backbone gradient steps (tighter coupling)
   Tested: {1, 10, 50, 0 (epoch)}

Additional mode: matched_compute
   Fixes total inner-loop gradient steps per epoch and distributes
   them differently across update frequency settings.

Grassmann variant (new):
   train_reptile_grassmann() — identical training loop but the meta-update
   uses Riemannian Reptile on the Grassmann manifold G(r, n) for A matrices.
   B matrices remain Euclidean. See meta_init.py and grassmann_utils.py.
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


# ── Inner loop (shared by both Euclidean and Grassmann trainers) ──────────────

def inner_loop(model, meta_init, subject_slot, X_s, y_s,
               K, inner_lr, inner_batch, device):
    """
    Run K SGD steps on subject s starting from meta-init.
    Returns phi_K — the adapted adapter weights.
    Backbone and all other adapter slots are untouched.
    """
    meta_init.load_into_slot(model, subject_slot)

    adapter_params = get_adapter_params(model, subject_slot)
    inner_opt  = torch.optim.SGD(adapter_params, lr=inner_lr, momentum=0.9)
    criterion  = nn.CrossEntropyLoss()
    n_trials   = len(X_s)

    model.train()
    for _ in range(K):
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

    return meta_init.extract_from_slot(model, subject_slot)


# ── Euclidean Reptile step and trainer (original, unchanged) ──────────────────

def reptile_step(model, meta_init, subject_data,
                 K, inner_lr, inner_batch, meta_lr, device):
    """
    Run inner loop for all training subjects and apply Euclidean Reptile update.
    Returns delta_norm for logging.
    """
    phi_list = []
    for sid, (X_s, y_s) in subject_data.items():
        phi = inner_loop(model, meta_init, sid,
                         X_s, y_s, K, inner_lr, inner_batch, device)
        phi_list.append(phi)

    delta_norm = meta_init.delta_norm(phi_list)
    meta_init.reptile_update(phi_list, meta_lr)
    return delta_norm


def train_reptile(model, train_X, train_y, train_sids, config, device):
    """
    Joint backbone training + Euclidean Reptile meta-update.

    update_freq controls when the Reptile step fires:
        0         → once per epoch after all backbone steps (original)
        M > 0     → every M backbone gradient steps (tighter coupling)

    When update_freq > 0 and matched_compute is True, K is scaled
    down so total inner-loop steps per epoch stays constant.
    """
    subject_data = build_per_subject_dict(train_X, train_y, train_sids)
    meta_init    = MetaInit(model, device)

    dataset = EEGDataset(train_X, train_y, train_sids)
    loader  = DataLoader(dataset, batch_size=config['batch_size'],
                         shuffle=True, drop_last=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    outer_opt = torch.optim.AdamW(model.parameters(),
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'])

    K            = config['inner_steps']
    inner_lr     = config['inner_lr']
    inner_batch  = config['inner_batch']
    meta_lr      = config['meta_lr']
    update_freq  = config.get('update_freq', 0)
    matched      = config.get('matched_compute', False)

    steps_per_epoch = len(loader)

    if update_freq > 0 and matched:
        updates_per_epoch = max(1, steps_per_epoch // update_freq)
        K_actual = max(1, K // updates_per_epoch)
    else:
        K_actual = K

    print(f'Reptile (Euclidean) | epochs={config["epochs"]} K={K_actual} '
          f'inner_lr={inner_lr} meta_lr={meta_lr} '
          f'update_freq={"epoch" if update_freq==0 else update_freq}')

    global_step = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        epoch_delta_norm = 0.0
        reptile_count    = 0

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
            global_step += 1

            if update_freq > 0 and global_step % update_freq == 0:
                dn = reptile_step(model, meta_init, subject_data,
                                  K_actual, inner_lr, inner_batch,
                                  meta_lr, device)
                epoch_delta_norm += dn
                reptile_count    += 1

        train_acc  = correct / n
        train_loss = total_loss / n

        if update_freq == 0:
            dn = reptile_step(model, meta_init, subject_data,
                              K_actual, inner_lr, inner_batch,
                              meta_lr, device)
            epoch_delta_norm = dn
            reptile_count    = 1

        avg_delta = epoch_delta_norm / max(reptile_count, 1)

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | loss {train_loss:.4f} | '
                  f'acc {train_acc:.3f} | delta_norm {avg_delta:.4f}')

        wandb.log({
            'epoch':           epoch,
            'train_loss':      train_loss,
            'train_acc':       train_acc,
            'delta_norm':      avg_delta,
            'reptile_updates': reptile_count,
            'K_actual':        K_actual,
        })

    return model, meta_init


# ── Grassmann Reptile step and trainer (new) ──────────────────────────────────

def reptile_step_grassmann(model, meta_init, subject_data,
                           K, inner_lr, inner_batch, meta_lr, device):
    """
    Run inner loop for all training subjects and apply Grassmann Reptile update.

    The inner loop is identical to the Euclidean version — SGD on adapter
    parameters in the standard Euclidean sense. Only the meta-update differs:
    A matrices are updated via Riemannian Reptile on G(r, n) rather than
    Euclidean averaging.

    Returns geodesic delta_norm for logging.
    """
    phi_list = []
    for sid, (X_s, y_s) in subject_data.items():
        phi = inner_loop(model, meta_init, sid,
                         X_s, y_s, K, inner_lr, inner_batch, device)
        phi_list.append(phi)

    # Use geodesic health-check metric for logging
    delta_norm = meta_init.delta_norm_grassmann(phi_list)
    # Grassmann update: A via manifold, B via Euclidean
    meta_init.reptile_update_grassmann(phi_list, meta_lr)
    return delta_norm


def train_reptile_grassmann(model, train_X, train_y, train_sids, config, device):
    """
    Joint backbone training + Riemannian Reptile meta-update on G(r, n).

    Identical to train_reptile() in every respect except the meta-update:
        - reptile_step_grassmann() is called instead of reptile_step()
        - reptile_update_grassmann() moves Q_meta along a geodesic rather
          than averaging matrices in Euclidean space
        - delta_norm logged is geodesic distance (radians), not Euclidean norm

    This means the backbone, inner loop, batch sampling, and all
    hyperparameters are held constant — only the geometry of the
    meta-update changes. A clean ablation of Euclidean vs Riemannian.

    Note: update_freq and matched_compute are supported identically to
    train_reptile() for consistency, though the default quick experiment
    uses update_freq=0 (once per epoch).
    """
    subject_data = build_per_subject_dict(train_X, train_y, train_sids)
    meta_init    = MetaInit(model, device)

    dataset = EEGDataset(train_X, train_y, train_sids)
    loader  = DataLoader(dataset, batch_size=config['batch_size'],
                         shuffle=True, drop_last=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    outer_opt = torch.optim.AdamW(model.parameters(),
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'])

    K            = config['inner_steps']
    inner_lr     = config['inner_lr']
    inner_batch  = config['inner_batch']
    meta_lr      = config['meta_lr']
    update_freq  = config.get('update_freq', 0)
    matched      = config.get('matched_compute', False)

    steps_per_epoch = len(loader)

    if update_freq > 0 and matched:
        updates_per_epoch = max(1, steps_per_epoch // update_freq)
        K_actual = max(1, K // updates_per_epoch)
    else:
        K_actual = K

    print(f'Reptile (Grassmann) | epochs={config["epochs"]} K={K_actual} '
          f'inner_lr={inner_lr} meta_lr={meta_lr} '
          f'update_freq={"epoch" if update_freq==0 else update_freq}')

    global_step = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        epoch_delta_norm = 0.0
        reptile_count    = 0

        for X_batch, y_batch, sid_batch in loader:
            X_batch   = X_batch.to(device)
            y_batch   = y_batch.to(device)
            sid_batch = sid_batch.to(device)

            # ── Backbone step (identical to Euclidean trainer) ─────────────────
            outer_opt.zero_grad()
            logits = model(X_batch, sid_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            outer_opt.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            n          += len(y_batch)
            global_step += 1

            # ── Intra-epoch Grassmann Reptile update ───────────────────────────
            if update_freq > 0 and global_step % update_freq == 0:
                dn = reptile_step_grassmann(
                    model, meta_init, subject_data,
                    K_actual, inner_lr, inner_batch, meta_lr, device,
                )
                epoch_delta_norm += dn
                reptile_count    += 1

        train_acc  = correct / n
        train_loss = total_loss / n

        # ── Per-epoch Grassmann Reptile update ────────────────────────────────
        if update_freq == 0:
            dn = reptile_step_grassmann(
                model, meta_init, subject_data,
                K_actual, inner_lr, inner_batch, meta_lr, device,
            )
            epoch_delta_norm = dn
            reptile_count    = 1

        avg_delta = epoch_delta_norm / max(reptile_count, 1)

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | loss {train_loss:.4f} | '
                  f'acc {train_acc:.3f} | geo_delta {avg_delta:.4f}')

        wandb.log({
            'epoch':           epoch,
            'train_loss':      train_loss,
            'train_acc':       train_acc,
            'delta_norm':      avg_delta,   # geodesic distance, not Euclidean
            'reptile_updates': reptile_count,
            'K_actual':        K_actual,
        })

    return model, meta_init
