"""
Cheap non-meta initialisation baselines (reviewer request 1 — decisive item).

Every baseline is just "what gets loaded into the held-out subject's
adapter slot before evaluation", reusing the init-agnostic plumbing in
evaluate.py unchanged.

Baselines:
    zero          — original paper behaviour (A ~ N(0,0.02), B = 0)
    average       — average of trained adapters, computed in ΔW space
                    and re-factorised to rank r via SVD. Averaging raw
                    (A, B) factors is invalid: the factorisation is only
                    defined up to an invertible r×r transform.
    shared        — single adapter slot shared by all training subjects
    donor_random  — copy one random training subject's adapter
    donor_nearest — copy the training subject whose trials are nearest
                    to the held-out subject in mean-covariance
                    (log-Euclidean) space
    reptile       — Reptile centroid (MetaInit), the meta arm

For conv adapters, ΔW for a layer is the composition of the rank
down-projection conv A (kernel k, dilation d) with the 1×1 up-projection
conv B. As a linear operator on the rank channel axis, the effective
per-(spatial-frequency) weight is W_eff = (α/r) · W_B @ W_A_spatial,
where W_A_spatial is the (rank × in_ch × k) A weight viewed per kernel
position and W_B is (out_ch × rank). We materialise the operator norm
via the singular values of the composed linear map. For 1×1 B and a
(k×k)-spatial A, the composed operator applied to input X is
B(A(X)); its effective weight tensor (as a single conv) is obtained by
convolving the B 1×1 kernel with A's kernel along the channel axis:

    W_eff[o, i, kh, kw] = (α/r) * Σ_j B[o, j, 0, 0] · A[j, i, kh, kw]

This gives a well-defined (out_ch, in_ch, k, k) tensor per subject that
IS comparable across subjects (unlike the raw factors), which is what
we average.
"""

import numpy as np
import torch
import torch.nn.functional as F

from meta_init import LORA_LAYER_NAMES, _resolve_layer_names


# ── ΔW utilities ──────────────────────────────────────────────────────────────

def delta_w(layer, slot, alpha=None, rank=None):
    """
    Effective per-subject weight delta for one LoRAConvPerSubject layer.

    Returns tensor of shape (out_channels, in_channels, kH, kW):
        W_eff[o, i, kh, kw] = (α/r) * Σ_j B[o, j, 0, 0] · A[j, i, kh, kw]

    This is comparable across subjects; the raw (A, B) factors are not.
    """
    alpha = layer.alpha if alpha is None else alpha
    rank  = layer.rank  if rank  is None else rank

    A = layer.lora_A[slot].weight.data    # (rank, in_ch, kH, kW)
    B = layer.lora_B[slot].weight.data    # (out_ch, rank, 1, 1)

    # einsum: out_ch, in_ch, kH, kW
    W = torch.einsum('oj,jikl->oikl', B.squeeze(-1).squeeze(-1), A)
    return (alpha / rank) * W


def factorize_delta_w(W, rank, kernel_size, dilation=1, device='cpu'):
    """
    Re-factorise a (out_ch, in_ch, kH, kW) effective delta into rank-r
    (A, B) conv factors via SVD of the flattened (out_ch × in_ch*kH*kW)
    matrix. Returns (A_weight, B_weight) matching LoRAConvPerSubject shapes:
        A: (rank, in_ch, kH, kW), B: (out_ch, rank, 1, 1)
    """
    out_ch, in_ch, kH, kW = W.shape
    M = W.reshape(out_ch, -1).cpu().numpy()          # (out_ch, in_ch*kH*kW)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    r = min(rank, U.shape[1])
    sqrt_S = np.sqrt(S[:r])

    B_flat = U[:, :r] * sqrt_S                       # (out_ch, r)
    A_flat = (Vt[:r, :].T * sqrt_S).T                # (r, in_ch*kH*kW)

    B_w = torch.from_numpy(B_flat).float().reshape(out_ch, r, 1, 1)
    A_w = torch.from_numpy(A_flat).float().reshape(r, in_ch, kH, kW)
    return A_w.to(device), B_w.to(device)


def load_delta_w_into_slot(layer, W, slot, device):
    """
    Write an effective delta W (out_ch, in_ch, kH, kW) into adapter slot
    `slot` of a LoRAConvPerSubject layer by re-factorising to rank r.
    The loaded adapter reproduces W up to the (α/r) scaling, which is
    folded into the factors (B scaled by 1, A absorbs 1 — we normalise
    so that (α/r)·A·B = W exactly).
    """
    rank = layer.rank
    alpha = layer.alpha
    A_w, B_w = factorize_delta_w(W, rank, layer.lora_A[slot].kernel_size,
                                 device=device)
    # Undo the α/r scaling applied in forward: store A' = A·(r/α)^{1/2},
    # B' = B·(r/α)^{1/2} — but simpler: scale so (α/r)·A'·B' = W.
    # A_w, B_w satisfy A_w·B_w ≈ W (unscaled), so multiply both by
    # sqrt(r/α) to keep norms balanced.
    scale = np.sqrt(rank / alpha)
    with torch.no_grad():
        layer.lora_A[slot].weight.data.copy_(A_w * scale)
        layer.lora_B[slot].weight.data.copy_(B_w * scale)


# ── Baseline initialisers ─────────────────────────────────────────────────────

def init_zero(model, slot, layer_names=None):
    """Held-out slot left at its (zero-B) initialisation. No-op if unused."""
    return {}


def init_average(model, slot, n_train_subjects, layer_names=None, device='cpu'):
    """
    Average the trained adapters' effective deltas and load into `slot`.
    Averaging in ΔW space, not factor space (gauge-correct).
    """
    names = _resolve_layer_names(layer_names)
    for name in names:
        layer = getattr(model, name)
        Ws = torch.stack(
            [delta_w(layer, s) for s in range(n_train_subjects)]
        )
        W_mean = Ws.mean(dim=0)
        load_delta_w_into_slot(layer, W_mean, slot, device)
    return {'init': 'average_of_adapters'}


def init_shared(model, slot, layer_names=None):
    """
    Shared-adapter baseline: copy slot 0 (which all subjects trained
    through, if the model was built with a single shared adapter) into
    the held-out slot.
    """
    names = _resolve_layer_names(layer_names)
    with torch.no_grad():
        for name in names:
            layer = getattr(model, name)
            layer.lora_A[slot].weight.data.copy_(layer.lora_A[0].weight.data)
            layer.lora_B[slot].weight.data.copy_(layer.lora_B[0].weight.data)
    return {'init': 'shared_adapter'}


def init_donor_random(model, slot, n_train_subjects, rng, layer_names=None):
    """Copy a uniformly random training subject's adapter."""
    donor = int(rng.integers(0, n_train_subjects))
    names = _resolve_layer_names(layer_names)
    with torch.no_grad():
        for name in names:
            layer = getattr(model, name)
            layer.lora_A[slot].weight.data.copy_(layer.lora_A[donor].weight.data)
            layer.lora_B[slot].weight.data.copy_(layer.lora_B[donor].weight.data)
    return {'init': 'donor_random', 'donor': donor}


def subject_covariances(X_s, eps=1e-6):
    """
    Mean trial covariance for one subject, regularised.
    X_s: (N, C, T) numpy array for one subject.
    Returns (C, C) mean covariance.
    """
    covs = []
    for trial in X_s:
        C = np.cov(trial)
        covs.append(C)
    M = np.mean(covs, axis=0)
    M += eps * np.trace(M) / M.shape[0] * np.eye(M.shape[0])
    return M


def log_euclidean_distance(C1, C2):
    """Log-Euclidean distance between two SPD matrices."""
    from scipy.linalg import logm
    d = logm(C1) - logm(C2)
    return float(np.linalg.norm(d, ord='fro'))


def init_donor_nearest(model, slot, n_train_subjects,
                       held_out_cov, train_covs, layer_names=None):
    """
    Copy the training subject whose mean covariance is nearest to the
    held-out subject's (log-Euclidean). Uses unlabelled data only.
    train_covs: dict {train_slot: (C, C) covariance}.
    """
    dists = {
        s: log_euclidean_distance(held_out_cov, C)
        for s, C in train_covs.items()
    }
    donor = min(dists, key=dists.get)
    names = _resolve_layer_names(layer_names)
    with torch.no_grad():
        for name in names:
            layer = getattr(model, name)
            layer.lora_A[slot].weight.data.copy_(layer.lora_A[donor].weight.data)
            layer.lora_B[slot].weight.data.copy_(layer.lora_B[donor].weight.data)
    return {'init': 'donor_nearest', 'donor': donor,
            'donor_dist': dists[donor]}


def apply_init(model, slot, strategy, *, n_train_subjects=None,
               meta_init=None, rng=None, held_out_cov=None,
               train_covs=None, layer_names=None, device='cpu'):
    """
    Dispatch an initialisation strategy into the held-out slot.
    Returns an info dict for logging.
    """
    if strategy == 'zero':
        return init_zero(model, slot, layer_names)
    if strategy == 'average':
        return init_average(model, slot, n_train_subjects, layer_names, device)
    if strategy == 'shared':
        return init_shared(model, slot, layer_names)
    if strategy == 'donor_random':
        return init_donor_random(model, slot, n_train_subjects, rng, layer_names)
    if strategy == 'donor_nearest':
        return init_donor_nearest(model, slot, n_train_subjects,
                                  held_out_cov, train_covs, layer_names)
    if strategy == 'reptile':
        meta_init.load_into_slot(model, slot)
        return {'init': 'reptile_centroid'}
    raise ValueError(f'Unknown init strategy: {strategy}')
