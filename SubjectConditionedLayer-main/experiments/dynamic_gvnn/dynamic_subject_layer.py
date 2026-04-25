"""
Dynamic Subject-Conditioned GVNN Layers.

Three layer variants building on the Subject-Conditioned Layer paper:

1. GVNNPreprocessLayer
   Population-level dynamic graph. No subject specificity.
   Tests whether dynamic connectivity alone helps over static LoRA.

2. SubjectConditionedGVNNLayer  (our proposed method)
   W_s = W_general + scale * A_s @ B_s.T   (full-rank subject support)
   Omega_t = W_s * J_t                      (subject-conditioned dynamic graph)
   Fixes all three problems identified in the analysis:
     - Full rank preserved (W_general full rank + low-rank correction)
     - Clean separation maintained (addition before Hadamard)
     - Dynamics are genuinely subject-specific (J_t filtered through W_s)

3. AdditiveSubjectDynamicLayer
   Additive variant where the three components are explicitly separated:
   out(t) = w_pop * W_general @ x_t         (population, static)
          + w_sub * (A_s @ B_s.T) @ x_t    (subject-specific, static)
          + w_dyn * (W_general * J_t) @ x_t (dynamic, shared not subject-specific)
   Used as an ablation to test whether the multiplicative Hadamard fusion
   is necessary or whether additive combination is sufficient.

All layers use fully vectorised operations — no Python loops over T.
Input/output: (batch, n_channels, n_times).
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Shared utility: compute J for all timesteps simultaneously
# ──────────────────────────────────────────────────────────────

def _compute_J_vectorised(X, node_fn='combined'):
    """
    Compute instantaneous connectivity tensor for all timesteps at once.

    X:        (B, C, T)
    Returns:  J (B, C, C, T)

    corr  — rank-1 instantaneous correlation  (xi - mean_i)(xj - mean_j)
    lde   — rank-3 Local Dirichlet Energy     (xi - xj)^2
    combined — sum of both
    """
    B, C, T = X.shape

    if node_fn in ('corr', 'combined'):
        mu = X.mean(dim=2, keepdim=True)              # (B, C, 1)
        X_c = X - mu                                  # (B, C, T)
        # J_corr[b,i,j,t] = X_c[b,i,t] * X_c[b,j,t]
        J_corr = torch.einsum('bit,bjt->bijt', X_c, X_c)

    if node_fn in ('lde', 'combined'):
        X2 = X ** 2                                   # (B, C, T)
        # (xi - xj)^2 = xi^2 + xj^2 - 2*xi*xj
        J_lde = (X2.unsqueeze(2) + X2.unsqueeze(1)
                 - 2 * torch.einsum('bit,bjt->bijt', X, X))

    if node_fn == 'corr':
        return J_corr
    elif node_fn == 'lde':
        return J_lde
    else:
        return J_corr + J_lde


# ──────────────────────────────────────────────────────────────
# Condition 3: Population-level GVNN (no subject specificity)
# ──────────────────────────────────────────────────────────────

class GVNNPreprocessLayer(nn.Module):
    """
    GVNN preprocessing with shared W_general across all subjects.
    Tests whether dynamic connectivity helps at all, independent of
    subject-specific adaptation.

    forward(X) -> (B, C, T)   — no subject_id needed
    """

    def __init__(self, n_channels, node_fn='combined', learnable_support=True):
        super().__init__()
        self.node_fn = node_fn

        W_init = torch.eye(n_channels)
        if learnable_support:
            self.W_general = nn.Parameter(W_init)
        else:
            self.register_buffer('W_general', W_init)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, subject_id=None):
        # subject_id ignored — included for API compatibility
        J = _compute_J_vectorised(X, self.node_fn)          # (B, C, C, T)
        W = self.W_general.unsqueeze(0).unsqueeze(-1)        # (1, C, C, 1)
        Omega = W * J                                         # (B, C, C, T)
        z = torch.einsum('bijt,bjt->bit', Omega, X)         # (B, C, T)
        return self.alpha * X + self.beta * z


# ──────────────────────────────────────────────────────────────
# Condition 4: Subject-Conditioned GVNN  (proposed method)
# ──────────────────────────────────────────────────────────────

class SubjectConditionedGVNNLayer(nn.Module):
    """
    Our proposed method: GVNN with subject-conditioned full-rank support.

    W_s      = W_general + scale * A_s @ B_s.T
    Omega_t  = W_s * J_t

    W_s is full rank because W_general is full rank and the LoRA correction
    is small (B initialised to zero at start). The GVNN rank-lifting theorem
    therefore applies, giving full-rank Omega_t.

    forward(X, subject_id) -> (B, C, T)
    """

    def __init__(self, n_channels, n_subjects, rank=4,
                 node_fn='combined', learnable_general=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_subjects = n_subjects
        self.rank = rank
        self.node_fn = node_fn
        self.scale = 1.0 / rank

        W_init = torch.eye(n_channels)
        if learnable_general:
            self.W_general = nn.Parameter(W_init)
        else:
            self.register_buffer('W_general', W_init)

        # B init to zero so W_s = W_general for all subjects at start.
        # Model begins as the pure population model and learns per-subject
        # deviations during training — same init strategy as original paper.
        self.lora_A = nn.Parameter(
            torch.randn(n_subjects, n_channels, rank) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(n_subjects, n_channels, rank)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

    def get_subject_support(self, subject_id):
        """
        Build subject-specific full-rank support matrix.
        subject_id: (B,) int tensor with values in [0, n_subjects)
        Returns W_s: (B, C, C)
        """
        A_s = self.lora_A[subject_id]                            # (B, C, rank)
        B_s = self.lora_B[subject_id]                            # (B, C, rank)
        correction = self.scale * torch.bmm(A_s, B_s.transpose(1, 2))  # (B, C, C)
        W_s = self.W_general.unsqueeze(0) + correction           # (B, C, C)
        return W_s

    def forward(self, X, subject_id):
        B, C, T = X.shape
        W_s = self.get_subject_support(subject_id)               # (B, C, C)
        J   = _compute_J_vectorised(X, self.node_fn)             # (B, C, C, T)
        Omega = W_s.unsqueeze(-1) * J                            # (B, C, C, T)
        z = torch.einsum('bijt,bjt->bit', Omega, X)             # (B, C, T)
        return self.alpha * X + self.beta * z


# ──────────────────────────────────────────────────────────────
# Condition 5: Additive variant (ablation of Hadamard fusion)
# ──────────────────────────────────────────────────────────────

class AdditiveSubjectDynamicLayer(nn.Module):
    """
    Additive combination — three explicitly separated components.

    out(t) = w_pop * W_general @ x_t
           + w_sub * (A_s @ B_s.T) @ x_t
           + w_dyn * (W_general * J_t) @ x_t

    Unlike SubjectConditionedGVNNLayer, the dynamic term is shared across
    subjects (filtered only by W_general, not W_s). This is the ablation
    that tests whether the multiplicative Hadamard fusion of subject-specific
    support with J_t is necessary.

    forward(X, subject_id) -> (B, C, T)
    """

    def __init__(self, n_channels, n_subjects, rank=4, node_fn='combined'):
        super().__init__()
        self.rank = rank
        self.node_fn = node_fn
        self.scale = 1.0 / rank

        self.W_general = nn.Parameter(torch.eye(n_channels))
        self.lora_A = nn.Parameter(
            torch.randn(n_subjects, n_channels, rank) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(n_subjects, n_channels, rank)
        )

        # Learnable weights for each additive component — initialised so
        # model starts near a standard linear layer
        self.w_pop  = nn.Parameter(torch.tensor(1.0))
        self.w_sub  = nn.Parameter(torch.tensor(0.5))
        self.w_dyn  = nn.Parameter(torch.tensor(0.5))
        self.alpha_skip = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, subject_id):
        B, C, T = X.shape

        A_s = self.lora_A[subject_id]                                # (B, C, rank)
        B_s = self.lora_B[subject_id]                                # (B, C, rank)
        W_sub = self.scale * torch.bmm(A_s, B_s.transpose(1, 2))    # (B, C, C)

        J = _compute_J_vectorised(X, self.node_fn)                   # (B, C, C, T)

        # Population: W_general @ x for all t
        pop = torch.einsum('ij,bjt->bit', self.W_general, X)        # (B, C, T)

        # Subject-specific static
        sub = torch.einsum('bij,bjt->bit', W_sub, X)                # (B, C, T)

        # Dynamic: shared W_general filtered by J_t
        Omega_dyn = self.W_general.unsqueeze(0).unsqueeze(-1) * J   # (B, C, C, T)
        dyn = torch.einsum('bijt,bjt->bit', Omega_dyn, X)           # (B, C, T)

        z = self.w_pop * pop + self.w_sub * sub + self.w_dyn * dyn
        return self.alpha_skip * X + (1 - self.alpha_skip) * z


# ──────────────────────────────────────────────────────────────
# Wrapper: prepend GVNN layer to any EEGNeX-style backbone
# ──────────────────────────────────────────────────────────────

class GVNNWrappedEEGNeX(nn.Module):
    """
    Wraps a GVNN preprocessing layer around an EEGNeX backbone.

    forward(X, subject_id) -> logits
    """

    def __init__(self, gvnn_layer, backbone):
        super().__init__()
        self.gvnn = gvnn_layer
        self.backbone = backbone

    def forward(self, X, subject_id):
        X = self.gvnn(X, subject_id)
        return self.backbone(X, subject_id)
