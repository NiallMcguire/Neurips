"""
MAML baseline for per-subject LoRA adapters (reviewer request 1, "if feasible").

Model-agnostic meta-learning over the adapter parameters. Unlike Reptile
(first-order, moves the init toward adapted weights), MAML backpropagates
through the inner adaptation loop (second-order), so the init is shaped
by the *gradient* of the post-adaptation loss.

Implemented with a functional inner loop over the adapter slot weights.
The backbone is treated as a fixed feature extractor during the inner
loop (its parameters are not adapted), consistent with the frozen-trunk
framing. A standard joint backbone step runs alongside, as in the
Reptile trainer.

Because the LoRAConvPerSubject forward selects adapters by index with a
Python loop and boolean mask, a fully functional rewrite would be
invasive. Instead we use `torch.func` (functional_call) for the inner
loop on a per-subject basis, treating the chosen slot's (A, B) as the
adapted parameters and the rest of the model as fixed buffers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb

from data_utils import EEGDataset, build_per_subject_dict
from meta_init import MetaInit, get_adapter_params, LORA_LAYER_NAMES, freeze_backbone
from exposure import ExposureTracker


def build_maml_model(n_channels, n_classes, n_times,
                     n_train_subjects, config, device):
    """Same architecture as the Reptile arm."""
    import sys
    sys.path.insert(0, '../EEGNex')
    from EEGNeX import EEGNeX
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


def _slot_param_names(slot, layer_names):
    names = []
    for lname in layer_names:
        names.append(f'{lname}.lora_A.{slot}.weight')
        names.append(f'{lname}.lora_B.{slot}.weight')
    return names


def maml_inner_and_meta(model, meta_params, X_s, y_s, slot,
                        K, inner_lr, inner_batch, device):
    """
    Second-order MAML step for one subject.

    meta_params: dict of {param_name: tensor} for the slot's adapter
                 weights (the meta-learned init).
    Runs K inner steps (differentiably) then evaluates a meta loss on a
    held-out mini-batch, returning meta gradients w.r.t. meta_params.
    """
    layer_names = [n.split('.lora_')[0] for n in meta_params.keys()][::2]
    # Unique layer names preserving order
    seen = []
    for n in meta_params.keys():
        ln = n.split('.lora_')[0]
        if ln not in seen:
            seen.append(ln)
    layer_names = seen

    criterion = nn.CrossEntropyLoss()
    n_trials = len(X_s)

    # Start inner loop from the meta init
    fast = {k: v for k, v in meta_params.items()}

    for _ in range(K):
        idx = torch.randperm(n_trials)[:inner_batch]
        X_b = X_s[idx].to(device)
        y_b = y_s[idx].to(device)
        sid = torch.full((len(idx),), slot, dtype=torch.long, device=device)

        logits = _functional_forward(model, fast, X_b, sid, slot, layer_names)
        loss = criterion(logits, y_b)
        grads = torch.autograd.grad(loss, list(fast.values()),
                                    create_graph=True,
                                    allow_unused=True)
        # Unused params (e.g. a slot not hit by this batch) get zero grad.
        fast = {k: v - inner_lr * (g if g is not None else torch.zeros_like(v))
                for (k, v), g in zip(fast.items(), grads)}

    # Meta loss on a fresh batch
    idx = torch.randperm(n_trials)[:inner_batch]
    X_b = X_s[idx].to(device)
    y_b = y_s[idx].to(device)
    sid = torch.full((len(idx),), slot, dtype=torch.long, device=device)
    meta_logits = _functional_forward(model, fast, X_b, sid, slot, layer_names)
    meta_loss = criterion(meta_logits, y_b)
    meta_grads = torch.autograd.grad(meta_loss, list(meta_params.values()),
                                     allow_unused=True)
    meta_grads = [g if g is not None else torch.zeros_like(p)
                  for g, p in zip(meta_grads, meta_params.values())]
    return meta_loss.item(), dict(zip(meta_params.keys(), meta_grads))


def _functional_forward(model, fast_params, X, sid, slot, layer_names):
    """
    Run the model with the slot's adapter weights overridden by
    fast_params, differentiably, without mutating the modules.

    Uses torch.func.functional_call: the provided fast_params (which
    carry the autograd graph through the inner loop) replace the named
    parameters for this single call only. All other parameters use their
    module values. This is the standard second-order MAML pattern.
    """
    from torch.func import functional_call

    # functional_call expects {param_name: tensor}; our fast_params keys
    # already match model.named_parameters() naming.
    logits = functional_call(model, fast_params, (X, sid))
    return logits


def train_maml(model, train_X, train_y, train_sids, config, device):
    """
    MAML over per-subject adapters + joint backbone training.

    Returns (model, meta_init, tracker) where meta_init is a MetaInit
    holding the MAML-learned init for the held-out slot.
    """
    subject_data = build_per_subject_dict(train_X, train_y, train_sids)
    layer_names = config.get('meta_layers', None) or LORA_LAYER_NAMES
    meta_init = MetaInit(model, device, layer_names=layer_names)

    dataset = EEGDataset(train_X, train_y, train_sids)
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=True, drop_last=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    if config.get('freeze_backbone', False):
        freeze_backbone(model)
    outer_opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['lr'], weight_decay=config['weight_decay'])

    # Meta parameters: the init for each slot, optimised with meta_lr.
    # We meta-learn a single shared init (slot 0's structure) used for all.
    meta_slot = 0
    meta_param_names = _slot_param_names(meta_slot, layer_names)
    meta_params = {n: p.clone().detach().requires_grad_(True)
                   for n, p in model.named_parameters()
                   if n in meta_param_names}
    meta_opt = torch.optim.Adam(list(meta_params.values()),
                                lr=config['meta_lr'])

    tracker = ExposureTracker()
    tracker.start()

    K = config['inner_steps']
    inner_lr = config['inner_lr']
    inner_batch = config['inner_batch']

    print(f'MAML | epochs={config["epochs"]} K={K} inner_lr={inner_lr} '
          f'meta_lr={config["meta_lr"]}')

    for epoch in range(config['epochs']):
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for X_batch, y_batch, sid_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            sid_batch = sid_batch.to(device)

            outer_opt.zero_grad()
            logits = model(X_batch, sid_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            outer_opt.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            n += len(y_batch)
            tracker.record_backbone_step(len(y_batch), sid_batch.cpu().numpy())

        # MAML meta-step once per epoch over all subjects
        meta_opt.zero_grad()
        epoch_meta_loss = 0.0
        for sid, (X_s, y_s) in subject_data.items():
            ml, grads = maml_inner_and_meta(
                model, meta_params, X_s, y_s, sid,
                K, inner_lr, inner_batch, device)
            epoch_meta_loss += ml
            for k, g in grads.items():
                if meta_params[k].grad is None:
                    meta_params[k].grad = g.clone()
                else:
                    meta_params[k].grad += g
            tracker.record_inner_step(inner_batch * K, sid)
        meta_opt.step()
        tracker.record_reptile_update()

        # Sync meta_params back into the MetaInit store (slot 0 structure)
        for name in layer_names:
            meta_init.weights[name]['A'] = \
                meta_params[f'{name}.lora_A.{meta_slot}.weight'].detach().clone()
            meta_init.weights[name]['B'] = \
                meta_params[f'{name}.lora_B.{meta_slot}.weight'].detach().clone()

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | loss {total_loss/n:.4f} | '
                  f'acc {correct/n:.3f} | meta_loss {epoch_meta_loss:.4f}')

        wandb.log({'epoch': epoch, 'train_loss': total_loss/n,
                   'train_acc': correct/n,
                   'meta_loss': epoch_meta_loss})

    wandb.log(tracker.summary())
    return model, meta_init, tracker
