"""
Conditional / amortised adapter initialisation via hypernetwork (Phase 3).

Replaces the single Reptile centroid with a per-subject predicted init:

    theta_s = H(c_s)

where c_s is a cheap unlabelled context descriptor for subject s
(per-channel log-variance + flattened upper-triangular mean covariance,
the same statistics Euclidean Alignment uses), and H is a hypernetwork
emitting the (A, B) LoRA factor weights for every adapted layer.

Two training modes:
    end_to_end — task loss backpropagated through the emitted adapters
                 into H (requires a differentiable functional forward).
    regression — H is trained to match pre-computed per-subject target
                 adapters by MSE on the factor tensors. Targets may be
                 distilled (see distill.py) or plain hard-label adapters.

The amortised claim: at test time, creating a new subject's init is a
single forward pass H(c_new) — no inner optimisation loop.
"""

import numpy as np
import torch
import torch.nn as nn

from meta_init import _resolve_layer_names


# ── Context descriptor ────────────────────────────────────────────────────────

def subject_context(X_s, eps=1e-6):
    """
    Unlabelled context vector c_s for one subject.

    X_s: (N, C, T) numpy array of one subject's trials.
    Returns a 1-D float32 numpy vector:
        [ per-channel log-variance (C,) ,
          log-eigenvalues of the mean trial covariance (C,) ]
    Log-transform keeps the descriptor scale-stable across subjects.

    IMPORTANT: X_s must be PRE-Euclidean-Alignment (raw) trials. EA whitens
    every subject to a common covariance frame, which collapses this
    descriptor's across-subject variance to near zero (measured ~1e-4x the
    raw-data spread on BCI2a fold 1) and makes it uninformative about
    subject identity. Callers that also apply EA to the data used for
    model training/evaluation must compute this descriptor from the
    un-aligned data separately.
    """
    trials = X_s
    N, C, _ = trials.shape

    # per-channel variance, averaged over trials
    chan_var = trials.var(axis=2).mean(axis=0)          # (C,)
    log_var = np.log(chan_var + eps)

    # mean covariance
    covs = np.stack([np.cov(t) for t in trials])
    M = covs.mean(axis=0)
    M += eps * np.trace(M) / C * np.eye(C)
    # log-eigenvalues as a stable covariance summary
    eigvals = np.linalg.eigvalsh(M)
    log_eig = np.log(np.maximum(eigvals, eps))

    return np.concatenate([log_var, log_eig]).astype(np.float32)


def context_dim(n_channels):
    return 2 * n_channels


# ── Hypernetwork ──────────────────────────────────────────────────────────────

class AdapterHypernetwork(nn.Module):
    """
    Emits LoRA (A, B) weights for every adapted layer from a context vector.

    For each adapted layer l with A shape (r, in, kh, kw) and B shape
    (out, r, 1, 1), a layer-specific head maps the shared embedding to
    the flattened concatenation of both factor tensors.

    The output is scaled to match the empirical norm of trained adapters
    (via a learnable per-layer gain initialised small), so H starts near
    the zero-init and learns deviations — mirroring the B=0 trick.
    """

    def __init__(self, model, layer_names=None, n_channels=22,
                 context_extra=0, hidden=256, device='cpu'):
        super().__init__()
        self.layer_names = _resolve_layer_names(layer_names)
        self.device = device

        ctx_dim = context_dim(n_channels) + context_extra

        # Record per-layer factor shapes from the model
        self.shapes = {}
        total_out = 0
        for name in self.layer_names:
            layer = getattr(model, name)
            a_shape = tuple(layer.lora_A[0].weight.shape)
            b_shape = tuple(layer.lora_B[0].weight.shape)
            n_a = int(np.prod(a_shape))
            n_b = int(np.prod(b_shape))
            self.shapes[name] = {'A': a_shape, 'B': b_shape,
                                 'n_a': n_a, 'n_b': n_b}
            total_out += n_a + n_b

        self.encoder = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden, spec['n_a'] + spec['n_b'])
            for name, spec in self.shapes.items()
        })

        # Small initial gain so initial predictions are near-zero
        # (adapters start as identity, like B=0 init).
        self.gain = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in self.layer_names
        })

        self.to(device)

    def forward(self, context):
        """
        context: (ctx_dim,) or (batch, ctx_dim) tensor.
        Returns {layer_name: {'A': tensor, 'B': tensor}} with shapes
        matching the model's adapter factors.
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        h = self.encoder(context)
        out = {}
        for name in self.layer_names:
            raw = self.heads[name](h) * torch.sigmoid(self.gain[name])
            spec = self.shapes[name]
            n_a = spec['n_a']
            A = raw[:, :n_a].reshape(-1, *spec['A'])
            B = raw[:, n_a:].reshape(-1, *spec['B'])
            out[name] = {'A': A, 'B': B}
        return out

    @torch.no_grad()
    def load_into_slot(self, model, slot, context):
        """
        Predict adapters for `context` and write them into `slot`
        of the model (inference-time amortised init).
        """
        if not isinstance(context, torch.Tensor):
            context = torch.from_numpy(context).float()
        context = context.to(self.device)
        preds = self.forward(context)
        for name in self.layer_names:
            layer = getattr(model, name)
            # preds are batched; take first (single subject)
            layer.lora_A[slot].weight.data.copy_(preds[name]['A'][0])
            layer.lora_B[slot].weight.data.copy_(preds[name]['B'][0])


# ── Regression-mode training ──────────────────────────────────────────────────

def train_hypernetwork_regression(hypernet, contexts, targets,
                                  epochs=200, lr=1e-3, device='cpu'):
    """
    Train H to regress onto pre-computed per-subject target adapters.

    contexts: {subject_slot: context vector (numpy)}
    targets:  {subject_slot: {layer_name: {'A': tensor, 'B': tensor}}}
    Returns the trained hypernetwork.
    """
    opt = torch.optim.AdamW(hypernet.parameters(), lr=lr)
    slots = sorted(contexts.keys())

    for epoch in range(epochs):
        total = 0.0
        for slot in slots:
            ctx = torch.from_numpy(contexts[slot]).float().to(device)
            pred = hypernet(ctx)
            loss = 0.0
            for name in hypernet.layer_names:
                loss = loss + nn.functional.mse_loss(
                    pred[name]['A'][0], targets[slot][name]['A'].to(device))
                loss = loss + nn.functional.mse_loss(
                    pred[name]['B'][0], targets[slot][name]['B'].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 50 == 0:
            print(f'  HN epoch {epoch:4d} | mse {total/len(slots):.6f}')

    return hypernet
