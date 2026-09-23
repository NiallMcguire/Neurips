"""
hypernet_objective: compares hypernetwork training objectives under
identical conditions (frozen shared backbone, same descriptor, same
architecture, same training budget/early stopping).

Does not modify any existing module. Reuses, unmodified:
    data_utils.{load_dataset, build_loso_split, euclidean_align}
    baseline_trainer.{build_baseline_model, train_baseline}
    reptile_trainer.{build_reptile_model, train_reptile}
    meta_init.{get_adapter_params, freeze_backbone, unfreeze_all,
               LORA_LAYER_NAMES}
    inits.{delta_w, init_average, apply_init, subject_covariances}
    hypernet.{subject_context, AdapterHypernetwork, context_dim}
    evaluate.{N_ADAPT_STEPS, ADAPT_LR, ADAPT_BATCH}
    run_experiment.DEFAULT_CONFIG

Design notes (read before changing anything):

Shared frozen backbone
    A single EEGNeX+LoRA model is jointly trained once per
    (dataset, held_out, seed, ea) via the existing train_baseline().
    Its state dict is then copied into every other model instance built
    for this experiment (any slot count), and the backbone/classifier
    parameters are frozen via meta_init.freeze_backbone(). Every
    condition below (H1-H4, B1-B4, controls) therefore shares byte-
    identical, frozen backbone weights. Batch-norm layers are kept in
    eval() mode throughout all adapter/hypernetwork fitting in this
    script (deliberately, unlike the rest of the codebase) so the
    shared backbone's running statistics never drift between
    conditions.

Per-subject "fitted adapters" (used as H1/H2 regression targets, and
as B3's average source)
    For each of the 8 training-subject slots, an adapter is fit from
    scratch (AdamW, hard-label cross-entropy) on that subject's own
    trials only, backbone frozen. This is what "fitted adapter" means
    throughout this module.

Leave-one-source-out validation for H1-H4
    Among the 8 training-subject slots, the subject at index
    (seed - 1) % n_train_subjects is excluded from the hypernetwork's
    training loss entirely, and used only to monitor early-stopping
    validation loss. The final hypernetwork is trained on the
    remaining 7 subjects. This is stated explicitly per the task's
    "state which" requirement.

Few-shot protocol here differs from the rest of the codebase in ORDERING
    only, not in the meaning of N. evaluate.py's FEW_SHOT_NS = [5, 10, 20, 50]
    are TOTAL trial counts (not per-class), drawn as a class-stratified
    RANDOM subset of size N. This experiment uses the same N values (plus
    N=0 for zero-shot, evaluated separately) and the same TOTAL-count
    semantics, but draws the FIRST N trials in original, unshuffled order
    from the held-out subject's calibration pool (their '0train' session —
    i.e. cal_X/cal_y from the existing LOSO split) for fine-tuning, per the
    task's explicit request, then evaluates on the REMAINING calibration
    trials (cal_X[N:], cal_y[N:]). Because trials are taken in order rather
    than stratified, class balance across the N trials is not guaranteed.
    The fine-tuning optimiser settings (steps/lr/batch) are imported
    unmodified from evaluate.py so they match the rest of the pipeline.

Descriptor is always computed from PRE-EA (raw) data
    subject_context() is EA-blind: computing it from EA-whitened trials
    collapses its across-subject SD by ~4 orders of magnitude (measured on
    BCI2a fold 1/seed 1: median SD 0.581 raw vs 0.000046 EA-transformed),
    making it uninformative about subject identity. Both the EA and non-EA
    arms of this experiment therefore compute every subject_context() call
    (training-subject contexts, held-out context, and the shuffled/mean
    descriptor controls) from the raw, un-aligned split, even when `--ea`
    is set and EA is applied to the data used for backbone/adapter
    training and evaluation.
"""

import argparse
import copy
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, '../EEGNex')

from data_utils import load_dataset, build_loso_split, euclidean_align
from baseline_trainer import build_baseline_model, train_baseline
from reptile_trainer import build_reptile_model, train_reptile
from meta_init import (get_adapter_params, freeze_backbone, unfreeze_all,
                       LORA_LAYER_NAMES)
from inits import delta_w, init_average, apply_init, subject_covariances
from hypernet import subject_context, AdapterHypernetwork, context_dim
from evaluate import N_ADAPT_STEPS, ADAPT_LR, ADAPT_BATCH
from run_experiment import DEFAULT_CONFIG


# ── Budget constants (identical across H1-H4; documented, not tunable per condition) ──

FIT_STEPS   = 300     # per-subject / oracle / B2-pooled adapter fitting
FIT_LR      = 1e-3
FIT_BATCH   = 32

HN_MAX_EPOCHS = 300
HN_PATIENCE   = 20
HN_LR         = 1e-3
HN_WD         = 1e-4   # weight decay — needed since only 7 training subjects

FEWSHOT_NS  = [5, 10, 20, 50]   # matches evaluate.py's FEW_SHOT_NS (total trial counts)
SEEDS       = [1, 2, 3]

CONDITIONS_H = ['recon_factors', 'recon_product', 'end_to_end', 'recon_then_e2e']
CONDITIONS_B = ['backbone_only', 'joint_shared_adapter',
                'average_fitted_adapters', 'reptile_zero_shot']
CONTROLS     = ['none', 'shuffled', 'mean']   # 'none' = real descriptor


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ── Backbone sharing ───────────────────────────────────────────────────────────

def train_shared_backbone(n_channels, n_classes, n_times, n_train_subjects,
                          cfg, train_X, train_y, train_sids, device):
    """Joint train once; return a frozen snapshot of its state dict."""
    model = build_baseline_model(n_channels, n_classes, n_times,
                                 n_train_subjects, cfg, device)
    model, _ = train_baseline(model, train_X, train_y, train_sids, cfg, device)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def copy_frozen_backbone(model, shared_sd, layer_names):
    """Copy backbone/classifier weights (skip lora_A/lora_B) and freeze them."""
    own_sd = model.state_dict()
    for k, v in shared_sd.items():
        if '.lora_A.' in k or '.lora_B.' in k:
            continue
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k].copy_(v)
    model.load_state_dict(own_sd)
    freeze_backbone(model, layer_names)
    return model


# ── Hard-label adapter fitting (per subject / oracle / pooled) ────────────────

def fit_adapter_hardlabel(model, X, y, slot, layer_names, device,
                          steps=FIT_STEPS, lr=FIT_LR, batch=FIT_BATCH):
    """Fit one adapter slot from scratch on (X, y), backbone frozen+eval()."""
    model.eval()   # keep shared frozen backbone's BN stats untouched
    params = get_adapter_params(model, slot, layer_names)
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(X)
    for _ in range(steps):
        idx = np.random.choice(n, size=min(batch, n), replace=n < batch)
        Xb  = torch.from_numpy(X[idx]).float().to(device)
        yb  = torch.from_numpy(y[idx]).long().to(device)
        sid = torch.full((len(idx),), slot, dtype=torch.long, device=device)
        opt.zero_grad()
        loss = crit(model(Xb, sid), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


# ── Functional forward for end-to-end hypernetwork training ───────────────────

def generated_fast_params(hypernet, context_tensor, slot, layer_names):
    """Fast-weight dict for torch.func.functional_call, graph-connected to H."""
    preds = hypernet(context_tensor)
    fast = {}
    for name in layer_names:
        fast[f'{name}.lora_A.{slot}.weight'] = preds[name]['A'][0]
        fast[f'{name}.lora_B.{slot}.weight'] = preds[name]['B'][0]
    return fast


def functional_forward(model, fast_params, X, sid):
    from torch.func import functional_call
    return functional_call(model, fast_params, (X, sid))


def compose_dw(A, B):
    """(alpha/r) omitted deliberately: cancels in the requested relative losses."""
    return torch.einsum('oj,jikl->oikl', B.squeeze(-1).squeeze(-1), A)


# ── Generic hypernetwork trainer with early stopping (shared by H1-H4) ────────

def train_hypernet(objective, model, layer_names, n_channels, device,
                   pool_ctx, pool_targets, pool_data,
                   val_ctx, val_target, val_data,
                   init_state=None):
    """
    objective: 'recon_factors' | 'recon_product' | 'end_to_end'
    pool_ctx/pool_targets/pool_data: dicts keyed by the 7 pooled subject slots
    val_ctx/val_target/val_data: the single leave-one-source-out subject
    Returns (hypernet, best_val_loss).
    """
    hypernet = AdapterHypernetwork(model, layer_names=layer_names,
                                   n_channels=n_channels, device=device)
    if init_state is not None:
        hypernet.load_state_dict(init_state)
    opt = torch.optim.AdamW(hypernet.parameters(), lr=HN_LR, weight_decay=HN_WD)

    model.eval()
    subjects = sorted(pool_ctx.keys())
    best_val = float('inf')
    best_state = copy.deepcopy(hypernet.state_dict())
    bad = 0

    def step_loss(ctx_np, target, data, slot, train_mode):
        ctx = torch.from_numpy(ctx_np).float().to(device)
        if objective == 'recon_factors':
            pred = hypernet(ctx)
            loss = 0.0
            for name in layer_names:
                loss = loss + F.mse_loss(pred[name]['A'][0], target[name]['A'].to(device))
                loss = loss + F.mse_loss(pred[name]['B'][0], target[name]['B'].to(device))
            return loss
        if objective == 'recon_product':
            pred = hypernet(ctx)
            loss = 0.0
            for name in layer_names:
                dw_pred = compose_dw(pred[name]['A'][0], pred[name]['B'][0])
                loss = loss + torch.linalg.norm((dw_pred - target[name]['dW'].to(device)).flatten())
            return loss
        if objective == 'end_to_end':
            X_s, y_s = data
            if train_mode:
                idx = np.random.choice(len(X_s), size=min(FIT_BATCH, len(X_s)),
                                       replace=len(X_s) < FIT_BATCH)
            else:
                idx = np.arange(len(X_s))
            Xb = torch.from_numpy(X_s[idx]).float().to(device)
            yb = torch.from_numpy(y_s[idx]).long().to(device)
            sid = torch.full((len(idx),), slot, dtype=torch.long, device=device)
            fast = generated_fast_params(hypernet, ctx, slot, layer_names)
            logits = functional_forward(model, fast, Xb, sid)
            return F.cross_entropy(logits, yb)
        raise ValueError(objective)

    for epoch in range(HN_MAX_EPOCHS):
        hypernet.train()
        for s in subjects:
            opt.zero_grad()
            loss = step_loss(pool_ctx[s], pool_targets.get(s), pool_data.get(s),
                             s, train_mode=True)
            loss.backward()
            opt.step()

        hypernet.eval()
        val_loss = step_loss(val_ctx, val_target, val_data,
                             list(pool_ctx.keys())[0], train_mode=False)
        # ^ slot id is irrelevant for the loss value itself (only shapes matter)
        val_loss = float(val_loss.detach().cpu().item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(hypernet.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= HN_PATIENCE:
                break

    hypernet.load_state_dict(best_state)
    return hypernet, best_val


# ── Descriptor controls ────────────────────────────────────────────────────────

def apply_descriptor_control(control, real_ctx, pool_ctx, rng):
    if control == 'none':
        return real_ctx
    if control == 'shuffled':
        donor = rng.choice(list(pool_ctx.keys()))
        return pool_ctx[donor]
    if control == 'mean':
        return np.mean(np.stack(list(pool_ctx.values())), axis=0)
    raise ValueError(control)


# ── Evaluation: zero-shot / few-shot / dW diagnostic ───────────────────────────

@torch.no_grad()
def balanced_acc(model, X, y, slot, device, batch_size=64):
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        Xb = torch.from_numpy(X[start:end]).float().to(device)
        sid = torch.full((end - start,), slot, dtype=torch.long, device=device)
        preds.extend(model(Xb, sid).argmax(1).cpu().numpy().tolist())
    return balanced_accuracy_score(y, np.array(preds))


def finetune_first_k(model, cal_X, cal_y, slot, k, layer_names, device):
    """Fine-tune on the FIRST k calibration trials; return the model copy and
    the (X, y) of the remaining trials for evaluation."""
    model_copy = copy.deepcopy(model)
    freeze_backbone(model_copy, layer_names)
    X_k, y_k = cal_X[:k], cal_y[:k]
    X_rest, y_rest = cal_X[k:], cal_y[k:]

    params = get_adapter_params(model_copy, slot, layer_names)
    opt = torch.optim.SGD(params, lr=ADAPT_LR, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    model_copy.train()
    n = len(X_k)
    for _ in range(N_ADAPT_STEPS):
        replace = n < ADAPT_BATCH
        idx = np.random.choice(n, size=min(ADAPT_BATCH, n), replace=replace)
        Xb = torch.from_numpy(X_k[idx]).float().to(device)
        yb = torch.from_numpy(y_k[idx]).long().to(device)
        sid = torch.full((len(idx),), slot, dtype=torch.long, device=device)
        opt.zero_grad()
        loss = crit(model_copy(Xb, sid), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    unfreeze_all(model_copy)
    return model_copy, X_rest, y_rest


def dw_diagnostic(model, slot, oracle_dw, layer_names):
    out = {}
    for name in layer_names:
        layer = getattr(model, name)
        gen = delta_w(layer, slot)
        denom = oracle_dw[name].norm().item()
        out[name] = (gen - oracle_dw[name]).norm().item() / denom if denom > 0 else float('nan')
    return out


# ── Row helper ──────────────────────────────────────────────────────────────

def make_row(condition, control, ea, seed, dataset, held_out, k, bal_acc, dw_err):
    row = {
        'condition': condition, 'descriptor_control': control, 'ea': ea,
        'seed': seed, 'dataset': dataset, 'held_out_subject': held_out,
        'k': k, 'balanced_accuracy': bal_acc,
    }
    for name in LORA_LAYER_NAMES:
        row[f'dw_err_{name}'] = dw_err.get(name, float('nan'))
    return row


CSV_FIELDS = (['condition', 'descriptor_control', 'ea', 'seed', 'dataset',
              'held_out_subject', 'k', 'balanced_accuracy']
             + [f'dw_err_{n}' for n in LORA_LAYER_NAMES])


# ── Fold driver ────────────────────────────────────────────────────────────────

def run_fold(dataset, held_out_1idx, seed, ea, device, quick=False):
    set_all_seeds(seed)
    # train_baseline()/train_reptile() (reused unmodified) call wandb.log();
    # keep a disabled run active so those calls are no-ops here.
    wandb.init(mode='disabled', reinit=True)
    held_out_0idx = held_out_1idx - 1

    cfg = dict(DEFAULT_CONFIG)
    if quick:
        cfg['epochs'] = 5

    data, labels, subject_ids, sessions, _ = load_dataset(dataset)
    (train_X, train_y, train_sids,
     cal_X, cal_y, cal_sids,
     test_X, test_y, test_sids) = build_loso_split(
        data, labels, subject_ids, sessions, held_out_0idx, dataset=dataset)

    # Kept for descriptor computation regardless of `ea`: subject_context()
    # must never see EA-whitened trials (see module docstring — EA collapses
    # its across-subject SD by ~4 orders of magnitude on BCI2a fold 1).
    train_X_raw, train_sids_raw, cal_X_raw = train_X, train_sids, cal_X

    if ea:
        aligned = euclidean_align(data, subject_ids, sessions=sessions, dataset=dataset)
        (train_X, train_y, train_sids,
         cal_X, cal_y, cal_sids,
         test_X, test_y, test_sids) = build_loso_split(
            aligned, labels, subject_ids, sessions, held_out_0idx, dataset=dataset)

    n_channels = train_X.shape[1]
    n_times    = train_X.shape[2]
    n_classes  = len(np.unique(train_y))
    n_train    = len(np.unique(train_sids))
    layer_names = LORA_LAYER_NAMES
    held_out_slot = n_train

    fit_steps = 20 if quick else FIT_STEPS
    hn_max_epochs = 10 if quick else HN_MAX_EPOCHS
    hn_patience = 3 if quick else HN_PATIENCE

    print(f'[{dataset} held_out={held_out_1idx} seed={seed} ea={ea}] '
         f'training shared backbone ({n_train} subjects)...')
    shared_sd = train_shared_backbone(n_channels, n_classes, n_times, n_train,
                                      cfg, train_X, train_y, train_sids, device)

    # Main model: n_train+1 slots, frozen shared backbone, then fit every
    # training-subject adapter independently (hard-label).
    model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
    model = copy_frozen_backbone(model, shared_sd, layer_names)

    print('  fitting per-subject adapters...')
    for s in range(n_train):
        Xs = train_X[train_sids == s]
        ys = train_y[train_sids == s]
        fit_adapter_hardlabel(model, Xs, ys, s, layer_names, device, steps=fit_steps)

    print('  fitting oracle adapter on held-out subject (diagnostic only)...')
    fit_adapter_hardlabel(model, cal_X, cal_y, held_out_slot, layer_names,
                          device, steps=fit_steps)
    oracle_dw = {name: delta_w(getattr(model, name), held_out_slot).clone()
                for name in layer_names}

    contexts = {s: subject_context(train_X_raw[train_sids_raw == s]) for s in range(n_train)}
    targets_factors = {}
    targets_product = {}
    train_data = {}
    for s in range(n_train):
        targets_factors[s] = {
            name: {'A': getattr(model, name).lora_A[s].weight.data.clone(),
                  'B': getattr(model, name).lora_B[s].weight.data.clone()}
            for name in layer_names
        }
        targets_product[s] = {
            name: {'dW': delta_w(getattr(model, name), s).clone()}
            for name in layer_names
        }
        train_data[s] = (train_X[train_sids == s], train_y[train_sids == s])

    held_ctx = subject_context(cal_X_raw)

    val_subject = (seed - 1) % n_train
    pool_subjects = [s for s in range(n_train) if s != val_subject]
    print(f'  leave-one-source-out validation subject (slot): {val_subject}')

    rows = []
    rng = np.random.default_rng(seed * 1000 + held_out_1idx)

    def eval_all(condition, control, gen_model, gen_slot):
        zs_acc = balanced_acc(gen_model, test_X, test_y, gen_slot, device)
        dw_err = dw_diagnostic(gen_model, gen_slot, oracle_dw, layer_names)
        rows.append(make_row(condition, control, ea, seed, dataset, held_out_1idx,
                             0, zs_acc, dw_err))
        for k in FEWSHOT_NS:
            if k >= len(cal_X):
                continue
            ft_model, X_rest, y_rest = finetune_first_k(
                gen_model, cal_X, cal_y, gen_slot, k, layer_names, device)
            if len(X_rest) == 0:
                continue
            fk_acc = balanced_acc(ft_model, X_rest, y_rest, gen_slot, device)
            rows.append(make_row(condition, control, ea, seed, dataset, held_out_1idx,
                                 k, fk_acc, dw_err))

    # ── B1: backbone only (zero adapter) ────────────────────────────────────
    print('  B1 backbone_only ...')
    b1_model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
    b1_model = copy_frozen_backbone(b1_model, shared_sd, layer_names)
    eval_all('backbone_only', 'none', b1_model, held_out_slot)

    # ── B2: joint shared adapter (pooled, frozen backbone) ──────────────────
    print('  B2 joint_shared_adapter ...')
    b2_model = build_baseline_model(n_channels, n_classes, n_times, 1, cfg, device)
    b2_model = copy_frozen_backbone(b2_model, shared_sd, layer_names)
    fit_adapter_hardlabel(b2_model, train_X, train_y, 0, layer_names, device,
                          steps=fit_steps)
    with torch.no_grad():
        for name in layer_names:
            layer = getattr(b2_model, name)
            layer.lora_A[1].weight.data.copy_(layer.lora_A[0].weight.data)
            layer.lora_B[1].weight.data.copy_(layer.lora_B[0].weight.data)
    eval_all('joint_shared_adapter', 'none', b2_model, 1)

    # ── B3: average of fitted adapters, dW space ────────────────────────────
    print('  B3 average_fitted_adapters ...')
    b3_model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
    b3_model = copy_frozen_backbone(b3_model, shared_sd, layer_names)
    for s in range(n_train):
        Xs = train_X[train_sids == s]
        ys = train_y[train_sids == s]
        fit_adapter_hardlabel(b3_model, Xs, ys, s, layer_names, device, steps=fit_steps)
    init_average(b3_model, held_out_slot, n_train, layer_names, device)
    eval_all('average_fitted_adapters', 'none', b3_model, held_out_slot)

    # ── B4: Reptile meta-init zero-shot (frozen shared backbone) ────────────
    print('  B4 reptile_zero_shot ...')
    b4_cfg = dict(cfg)
    b4_cfg['freeze_backbone'] = True
    if quick:
        b4_cfg['epochs'] = 5
    b4_model = build_reptile_model(n_channels, n_classes, n_times, n_train, b4_cfg, device)
    b4_model = copy_frozen_backbone(b4_model, shared_sd, layer_names)
    b4_model, meta_init, _ = train_reptile(b4_model, train_X, train_y, train_sids,
                                           b4_cfg, device)
    apply_init(b4_model, held_out_slot, 'reptile', meta_init=meta_init, device=device)
    eval_all('reptile_zero_shot', 'none', b4_model, held_out_slot)

    # ── H1: recon_factors (current method) ──────────────────────────────────
    print('  H1 recon_factors ...')
    pool_ctx = {s: contexts[s] for s in pool_subjects}
    pool_tgt_f = {s: targets_factors[s] for s in pool_subjects}
    hn1, _ = train_hypernet('recon_factors', model, layer_names, n_channels, device,
                            pool_ctx, pool_tgt_f, {},
                            contexts[val_subject], targets_factors[val_subject], None)
    h1_model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
    h1_model = copy_frozen_backbone(h1_model, shared_sd, layer_names)
    for control in CONTROLS if False else ['none']:  # H1 has no controls
        ctx = apply_descriptor_control(control, held_ctx, pool_ctx, rng)
        hn1.load_into_slot(h1_model, held_out_slot, ctx)
        eval_all('recon_factors', control, h1_model, held_out_slot)

    # ── H2: recon_product ─────────────────────────────────────────────────
    print('  H2 recon_product ...')
    pool_tgt_p = {s: targets_product[s] for s in pool_subjects}
    hn2, _ = train_hypernet('recon_product', model, layer_names, n_channels, device,
                            pool_ctx, pool_tgt_p, {},
                            contexts[val_subject], targets_product[val_subject], None)
    for control in CONTROLS:
        h2_model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
        h2_model = copy_frozen_backbone(h2_model, shared_sd, layer_names)
        ctx = apply_descriptor_control(control, held_ctx, pool_ctx, rng)
        hn2.load_into_slot(h2_model, held_out_slot, ctx)
        eval_all('recon_product', control, h2_model, held_out_slot)

    # ── H3: end_to_end ───────────────────────────────────────────────────────
    print('  H3 end_to_end ...')
    pool_data = {s: train_data[s] for s in pool_subjects}
    hn3, _ = train_hypernet('end_to_end', model, layer_names, n_channels, device,
                            pool_ctx, {}, pool_data,
                            contexts[val_subject], None, train_data[val_subject])
    for control in CONTROLS:
        h3_model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
        h3_model = copy_frozen_backbone(h3_model, shared_sd, layer_names)
        ctx = apply_descriptor_control(control, held_ctx, pool_ctx, rng)
        hn3.load_into_slot(h3_model, held_out_slot, ctx)
        eval_all('end_to_end', control, h3_model, held_out_slot)

    # ── H4: recon_then_e2e (init from H2, fine-tune with H3 objective) ──────
    print('  H4 recon_then_e2e ...')
    hn4, _ = train_hypernet('end_to_end', model, layer_names, n_channels, device,
                            pool_ctx, {}, pool_data,
                            contexts[val_subject], None, train_data[val_subject],
                            init_state=copy.deepcopy(hn2.state_dict()))
    for control in CONTROLS:
        h4_model = build_baseline_model(n_channels, n_classes, n_times, n_train, cfg, device)
        h4_model = copy_frozen_backbone(h4_model, shared_sd, layer_names)
        ctx = apply_descriptor_control(control, held_ctx, pool_ctx, rng)
        hn4.load_into_slot(h4_model, held_out_slot, ctx)
        eval_all('recon_then_e2e', control, h4_model, held_out_slot)

    return rows


def write_csv(rows, path):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary_table(rows):
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r['condition'], r['descriptor_control'], r['k'])].append(r['balanced_accuracy'])
    print(f"\n{'condition':22s} {'control':9s} {'k':>3s} {'n':>3s} {'mean_bal_acc':>12s}")
    for (cond, ctrl, k), vals in sorted(agg.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        print(f'{cond:22s} {ctrl:9s} {k:3d} {len(vals):3d} {np.mean(vals):12.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BCI2a', choices=['BCI2a', 'BCI2b'])
    parser.add_argument('--held_out', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ea', action='store_true')
    parser.add_argument('--out', default='hypernet_objective_results.csv')
    parser.add_argument('--sanity', action='store_true',
                        help='Fold 1, seed 1, both EA settings, reduced budget.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.sanity:
        out_path = 'hypernet_objective_sanity.csv'
        if os.path.exists(out_path):
            os.remove(out_path)
        all_rows = []
        for ea in (False, True):
            t0 = time.perf_counter()
            rows = run_fold(args.dataset, 1, 1, ea, device, quick=True)
            print(f'  ea={ea} done in {time.perf_counter() - t0:.1f}s, '
                 f'{len(rows)} rows')
            write_csv(rows, out_path)
            all_rows.extend(rows)
        print_summary_table(all_rows)
        return

    if args.held_out is None or args.seed is None:
        parser.error('--held_out and --seed are required unless --sanity is set')

    rows = run_fold(args.dataset, args.held_out, args.seed, args.ea, device)
    write_csv(rows, args.out)
    print_summary_table(rows)


if __name__ == '__main__':
    main()
