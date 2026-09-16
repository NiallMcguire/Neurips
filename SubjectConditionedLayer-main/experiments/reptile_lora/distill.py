"""
Distillation of per-subject teachers into rank-8 adapter targets (Phase 3, gated).

When the hypernetwork is trained by regression onto pre-computed
per-subject target adapters, those targets can be improved by first
distilling a full-capacity per-subject teacher into the rank-8 adapter.

Gate: the teacher must beat the plain hard-label rank-8 adapter on that
subject's held-out trials. If it does not, fall back to hard-label
targets. This branch exists only for target-regression hypernetwork
training, not end-to-end.

Teacher: a full-capacity per-subject model = frozen shared backbone +
         a FULL-rank (non-low-rank) per-subject conv correction, or
         equivalently a per-subject fine-tuned copy of the backbone's
         adapted layers. Here we implement the simplest full-capacity
         teacher: a per-subject fine-tune of the backbone with a
         higher-rank adapter, then distill its soft outputs into the
         rank-8 adapter.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_init import get_adapter_params, freeze_backbone, unfreeze_all


def distill_teacher_into_adapter(model, teacher_logits_fn, X_s, slot,
                                 layer_names=None, device='cpu',
                                 steps=200, lr=1e-3, batch=32, temp=4.0):
    """
    Distill a teacher's soft outputs into subject slot's rank-8 adapter.

    model:             the LoRA model (backbone shared/frozen).
    teacher_logits_fn: callable(X_batch) -> logits from the full-capacity
                       teacher for this subject.
    X_s:               (N, C, T) this subject's trials (numpy).
    slot:              adapter slot to write into.
    temp:              distillation temperature.

    Returns the adapted slot's (A, B) via the model (in place).
    """
    freeze_backbone(model, layer_names)
    adapter_params = get_adapter_params(model, slot, layer_names)
    opt = torch.optim.SGD(adapter_params, lr=lr, momentum=0.9)

    X = torch.from_numpy(X_s).float()
    n = len(X)
    model.train()

    for step in range(steps):
        idx = np.random.choice(n, size=min(batch, n), replace=n < batch)
        X_b = X[idx].to(device)
        sid = torch.full((len(idx),), slot, dtype=torch.long, device=device)

        with torch.no_grad():
            t_logits = teacher_logits_fn(X_b)

        s_logits = model(X_b, sid)
        loss = F.kl_div(
            F.log_softmax(s_logits / temp, dim=1),
            F.softmax(t_logits / temp, dim=1),
            reduction='batchmean',
        ) * (temp ** 2)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        opt.step()

    unfreeze_all(model)
    return model


def teacher_beats_hardlabel(teacher_acc, hardlabel_acc):
    """
    Gate: only use distilled targets if the full-capacity teacher
    outperforms the plain hard-label rank-8 adapter on held-out trials.
    """
    return teacher_acc > hardlabel_acc
