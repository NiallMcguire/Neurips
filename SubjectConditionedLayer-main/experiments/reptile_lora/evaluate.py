"""
Shared evaluation for baseline and Reptile conditions.

evaluate_zero_shot:
    New subject uses whatever initialisation is provided (zeros or meta).
    No gradient updates. Pure forward pass.

evaluate_few_shot:
    New subject starts from provided initialisation.
    Fine-tunes adapter on N calibration trials for n_adapt_steps steps.
    Evaluates on held-out test set.

Both functions are condition-agnostic. The only difference between
baseline and Reptile is what initialisation is loaded before calling them.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from meta_init import get_adapter_params, freeze_backbone, unfreeze_all, LORA_LAYER_NAMES


FEW_SHOT_NS   = [5, 10, 20, 50]
N_ADAPT_STEPS = 10      # gradient steps during few-shot adaptation
ADAPT_LR      = 5e-4    # lower than training LR — careful fine-tuning
ADAPT_BATCH   = 16      # small batch for few-shot


# ── Shared forward helpers ────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, X, y, subject_slot, device, batch_size=64):
    """
    Evaluate model on (X, y) using adapter slot `subject_slot`.
    Returns accuracy, kappa, predictions, true labels.
    """
    model.eval()
    all_pred, all_true = [], []

    n = len(X)
    for start in range(0, n, batch_size):
        end      = min(start + batch_size, n)
        X_batch  = torch.from_numpy(X[start:end]).float().to(device)
        y_batch  = y[start:end]
        sid      = torch.full((end - start,), subject_slot,
                              dtype=torch.long, device=device)
        logits   = model(X_batch, sid)
        pred     = logits.argmax(1).cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(y_batch.tolist())

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    acc   = (all_pred == all_true).mean()
    kappa = cohen_kappa_score(all_true, all_pred)
    return acc, kappa, all_pred, all_true


# ── Zero-shot evaluation ──────────────────────────────────────────────────────

def evaluate_zero_shot(model, test_X, test_y, subject_slot, device):
    """
    Evaluate with no adaptation. Whatever weights are currently in
    `subject_slot` are used directly.

    For baseline: slot was left at zero initialisation.
    For Reptile:  slot was loaded with A_meta, B_meta.
    """
    acc, kappa, pred, true = run_eval(
        model, test_X, test_y, subject_slot, device
    )
    return {
        'zero_shot_acc':   acc,
        'zero_shot_kappa': kappa,
        'pred':            pred,
        'true':            true,
    }


# ── Few-shot evaluation ───────────────────────────────────────────────────────

def evaluate_few_shot(model, cal_X, cal_y, test_X, test_y,
                      subject_slot, device, N, n_adapt_steps=N_ADAPT_STEPS,
                      adapt_lr=ADAPT_LR):
    """
    Fine-tune adapter on N calibration trials, then evaluate on test set.

    The model's backbone and all other adapter slots are frozen.
    Only `subject_slot`'s lora_A and lora_B are updated.

    For baseline: starts from zeros in subject_slot.
    For Reptile:  starts from A_meta, B_meta already loaded into subject_slot.
    """
    import copy

    # Work on a copy so the original model is unchanged for subsequent N values
    model_copy = copy.deepcopy(model)
    model_copy.to(device)

    # Freeze everything except this subject's adapter
    freeze_backbone(model_copy)

    # Subsample N calibration trials — stratified by class
    n_classes = len(np.unique(cal_y))
    if N >= len(cal_X):
        idx = np.arange(len(cal_X))
    else:
        idx = []
        for cls in np.unique(cal_y):
            cls_idx = np.where(cal_y == cls)[0]
            n_cls   = max(1, N // n_classes)
            chosen  = np.random.choice(cls_idx,
                                       size=min(n_cls, len(cls_idx)),
                                       replace=False)
            idx.extend(chosen.tolist())
        idx = np.array(idx)

    X_cal_n = cal_X[idx]
    y_cal_n = cal_y[idx]

    # Optimise only this subject's adapter weights
    adapter_params = get_adapter_params(model_copy, subject_slot)
    opt = torch.optim.SGD(adapter_params, lr=adapt_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_copy.train()
    for step in range(n_adapt_steps):
        # Sample a mini-batch (with replacement when N is tiny)
        replace  = len(X_cal_n) < ADAPT_BATCH
        bidx     = np.random.choice(len(X_cal_n), size=min(ADAPT_BATCH, len(X_cal_n)),
                                    replace=replace)
        X_batch  = torch.from_numpy(X_cal_n[bidx]).float().to(device)
        y_batch  = torch.from_numpy(y_cal_n[bidx]).long().to(device)
        sid      = torch.full((len(bidx),), subject_slot,
                              dtype=torch.long, device=device)

        logits = model_copy(X_batch, sid)
        loss   = criterion(logits, y_batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        opt.step()

    # Evaluate adapted model
    acc, kappa, pred, true = run_eval(
        model_copy, test_X, test_y, subject_slot, device
    )

    # Restore gradient state
    unfreeze_all(model_copy)

    return {
        'acc':   acc,
        'kappa': kappa,
        'pred':  pred,
        'true':  true,
        'N':     N,
    }


# ── Full evaluation suite ─────────────────────────────────────────────────────

def run_full_evaluation(model, cal_X, cal_y, test_X, test_y,
                        subject_slot, device, condition_name,
                        held_out_subject, seed):
    """
    Run zero-shot and all few-shot evaluations.
    Returns a flat dict suitable for wandb logging.
    """
    import wandb

    results = {
        'condition':       condition_name,
        'held_out_subject': held_out_subject + 1,   # back to 1-indexed for logging
        'seed':            seed,
    }

    # Zero-shot
    zs = evaluate_zero_shot(model, test_X, test_y, subject_slot, device)
    results['zero_shot_acc']   = zs['zero_shot_acc']
    results['zero_shot_kappa'] = zs['zero_shot_kappa']

    print(f'  Zero-shot | acc {zs["zero_shot_acc"]:.3f} | '
          f'kappa {zs["zero_shot_kappa"]:.3f}')

    wandb.log({
        **results,
        'few_shot_N': 0,
        'few_shot_acc': zs['zero_shot_acc'],
        'few_shot_kappa': zs['zero_shot_kappa'],
    })

    # Few-shot
    for N in FEW_SHOT_NS:
        if N > len(cal_X):
            print(f'  N={N} exceeds calibration pool ({len(cal_X)}), skipping')
            continue

        fs = evaluate_few_shot(
            model, cal_X, cal_y, test_X, test_y,
            subject_slot, device, N,
        )

        results[f'few_shot_acc_N{N}']   = fs['acc']
        results[f'few_shot_kappa_N{N}'] = fs['kappa']

        print(f'  N={N:3d}    | acc {fs["acc"]:.3f} | kappa {fs["kappa"]:.3f}')

        wandb.log({
            **results,
            'few_shot_N':     N,
            'few_shot_acc':   fs['acc'],
            'few_shot_kappa': fs['kappa'],
        })

    return results
