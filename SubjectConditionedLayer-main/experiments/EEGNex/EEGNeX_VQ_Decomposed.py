# EEGNeX with Decomposed VQ Latent Space  — v6
# ============================================================
# z_shared  -- VQ-regularised continuous representation, shared across subjects
# z_subject -- continuous subject residual
#
# Changes vs v5:
#   1. codebook_size     64   -> 24  (8 per class, eliminates structural dead codes)
#   2. commitment_beta   0.25 -> 0.5 (stronger pull, reduces continuous-discrete gap)
# ============================================================

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split as sk_split

from braindecode.models import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint


# =============================================================================
# 1. Vector Quantizer  (Gumbel-Softmax, temperature annealed)
# =============================================================================

class VectorQuantizer(nn.Module):
    """
    Codebook lookup with Gumbel-Softmax for differentiable assignment.

    Training:
        Soft weighted sum over codebook entries via Gumbel-Softmax.
        Temperature annealed from temp_start -> temp_end over training.
        As temperature -> 0 this recovers hard argmin.

    Inference / retrieval:
        Hard argmin, returns nearest codebook entry.

    The VQ loss acts as a regulariser that forces z_e_shared to organise
    into discrete clusters. Classification acts on continuous z_e_shared.

    EMA codebook updates + dead code restart with threshold 0.1.
    """

    def __init__(
        self,
        codebook_size: int = 24,
        embedding_dim: int = 128,
        commitment_beta: float = 0.5,
        ema_decay: float = 0.95,
        temp_start: float = 1.0,
        temp_end: float = 0.1,
    ):
        super().__init__()
        self.K          = codebook_size
        self.d          = embedding_dim
        self.beta       = commitment_beta
        self.ema_decay  = ema_decay
        self.temp_start = temp_start
        self.temp_end   = temp_end
        self.temperature = temp_start

        self.codebook = nn.Embedding(self.K, self.d)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.K, 1.0 / self.K)

        self.register_buffer("ema_cluster_size", torch.zeros(self.K))
        self.register_buffer("ema_embedding_sum", self.codebook.weight.data.clone())

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e : [B, d]  continuous encoder output (shared half)

        Returns:
            z_q_soft : [B, d]  soft weighted codebook vector (training)
                               hard nearest codebook vector (eval)
            z_q_hard : [B, d]  hard nearest codebook vector (always)
            indices  : [B]     hard assignment indices
            weights  : [B, K]  soft assignment weights (for coherence loss)
            loss_vq  : scalar  codebook + commitment loss
        """
        # Pairwise squared L2 distances  [B, K]
        dists = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * (z_e @ self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        # Hard assignment (always computed for EMA and inference)
        indices  = dists.argmin(dim=1)
        z_q_hard = self.codebook(indices)

        if self.training:
            # Gumbel-Softmax soft assignment
            logits_vq = -dists / self.temperature              # [B, K]
            weights   = F.softmax(logits_vq, dim=1)           # [B, K]
            z_q_soft  = weights @ self.codebook.weight         # [B, d]

            # EMA codebook update using hard assignments
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.K).float()   # [B, K]
                new_cluster_size  = one_hot.sum(0)             # [K]
                new_embedding_sum = one_hot.t() @ z_e          # [K, d]

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    new_cluster_size * (1 - self.ema_decay)
                )
                self.ema_embedding_sum.mul_(self.ema_decay).add_(
                    new_embedding_sum * (1 - self.ema_decay)
                )

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.K * 1e-5)
                    * n
                )
                self.codebook.weight.data.copy_(
                    self.ema_embedding_sum / smoothed.unsqueeze(1)
                )

                # Dead code restart — threshold 0.1
                dead_mask = self.ema_cluster_size < 0.1
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    rand_idx = torch.randint(
                        0, z_e.shape[0], (n_dead,), device=z_e.device
                    )
                    self.codebook.weight.data[dead_mask] = z_e[rand_idx].detach()
                    self.ema_cluster_size[dead_mask]     = 1.0
                    self.ema_embedding_sum[dead_mask]    = z_e[rand_idx].detach()

        else:
            # Inference: hard assignment
            z_q_soft = z_q_hard
            weights  = F.one_hot(indices, self.K).float()

        # VQ losses computed against hard assignment
        loss_codebook = F.mse_loss(z_q_hard, z_e.detach())
        loss_commit   = F.mse_loss(z_e, z_q_hard.detach())
        loss_vq       = loss_codebook + self.beta * loss_commit

        return z_q_soft, z_q_hard, indices, weights, loss_vq

    def set_temperature(self, epoch: int, total_epochs: int):
        """Anneal temperature linearly from temp_start to temp_end."""
        frac = epoch / max(total_epochs - 1, 1)
        self.temperature = self.temp_start + frac * (self.temp_end - self.temp_start)

    def dead_code_fraction(self) -> float:
        return (self.ema_cluster_size < 1.0).float().mean().item()


# =============================================================================
# 2. Subject Residual Head
# =============================================================================

class SubjectResidualHead(nn.Module):
    """
    Per-subject embedding bias injected before a shared MLP.
    Encodes individual variation that the shared VQ-regularised
    representation cannot and should not represent.
    """

    def __init__(self, input_dim: int, num_subjects: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.subject_embedding = nn.Embedding(num_subjects, input_dim)
        nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.01)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z_e_subject: torch.Tensor, subject_id: torch.LongTensor):
        bias = self.subject_embedding(subject_id)
        return self.mlp(z_e_subject + bias)


# =============================================================================
# 3. Reconstruction Decoder
# =============================================================================

class ReconstructionDecoder(nn.Module):
    """
    Reconstructs z_e from (z_q_soft, z_subject).
    Forces z_subject to encode whatever individual variation VQ discarded.
    Target is z_e.detach() so reconstruction gradients do not perturb encoder.
    """

    def __init__(self, d_shared: int, d_subject: int):
        super().__init__()
        in_dim = d_shared + d_subject

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, in_dim),
        )

    def forward(self, z_q_soft: torch.Tensor, z_subject: torch.Tensor):
        return self.decoder(torch.cat([z_q_soft, z_subject], dim=-1))


# =============================================================================
# 4. EEGNeX Backbone  (fully shared, no subject-specific weights)
# =============================================================================

class EEGNeXBackbone(EEGModuleMixin, nn.Module):
    """
    Standard EEGNeX convolutional encoder shared across all subjects.
    filter_1=16 gives out_features=256, d_shared=128.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        activation=nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 16,
        filter_2: int = 32,
        drop_prob: float = 0.5,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        max_norm_conv: float = 1.0,
    ):
        super().__init__(
            n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info,
            n_times=n_times, input_window_seconds=input_window_seconds, sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.filter_1 = filter_1
        f3 = filter_2 * depth_multiplier

        kb = (1, kernel_block_1_2)
        k4 = (1, kernel_block_4);  d4 = (1, dilation_block_4)
        p4 = (1, avg_pool_block4)
        k5 = (1, kernel_block_5);  d5 = (1, dilation_block_5)
        p5 = (1, avg_pool_block5)

        self._p4 = avg_pool_block4
        self._p5 = avg_pool_block5
        self.out_features = self._calc_out(avg_pool_block4, avg_pool_block5)

        self.block_1 = nn.Sequential(
            Rearrange("b c t -> b 1 c t"),
            nn.Conv2d(1, filter_1, kernel_size=kb, padding="same", bias=False),
            nn.BatchNorm2d(filter_1),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kernel_size=kb, padding="same", bias=False),
            nn.BatchNorm2d(filter_2),
        )
        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(
                filter_2, f3, max_norm=max_norm_conv,
                kernel_size=(self.n_chans, 1), groups=filter_2, bias=False,
            ),
            nn.BatchNorm2d(f3),
            activation(),
            nn.AvgPool2d(kernel_size=p4, padding=(0, 1)),
            nn.Dropout(p=drop_prob),
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(f3, filter_2, kernel_size=k4, dilation=d4, padding="same", bias=False),
            nn.BatchNorm2d(filter_2),
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(filter_2, filter_1, kernel_size=k5, dilation=d5, padding="same", bias=False),
            nn.BatchNorm2d(filter_1),
            activation(),
            nn.AvgPool2d(kernel_size=p5, padding=(0, 1)),
            nn.Dropout(p=drop_prob),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x

    def _calc_out(self, p4: int, p5: int) -> int:
        T3 = math.floor((self.n_times + 2 - p4) / p4) + 1
        T5 = math.floor((T3          + 2 - p5) / p5) + 1
        return self.filter_1 * T5


# =============================================================================
# 5. VQ Decomposed Head
# =============================================================================

class VQDecomposedHead(nn.Module):
    """
    Splits z_e into shared and subject components.
    Classifier acts on continuous z_e_shared — clean gradient path.
    VQ provides structural regularisation on z_e_shared.
    """

    def __init__(
        self,
        in_features: int,
        n_outputs: int,
        num_subjects: int,
        codebook_size: int = 24,
        commitment_beta: float = 0.5,
        ema_decay: float = 0.95,
        temp_start: float = 1.0,
        temp_end: float = 0.1,
        max_norm_clf: float = 0.25,
    ):
        super().__init__()
        self.d_shared  = in_features // 2
        self.d_subject = in_features - self.d_shared

        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=self.d_shared,
            commitment_beta=commitment_beta,
            ema_decay=ema_decay,
            temp_start=temp_start,
            temp_end=temp_end,
        )
        self.subject_head = SubjectResidualHead(
            input_dim=self.d_subject,
            num_subjects=num_subjects,
        )
        self.decoder = ReconstructionDecoder(
            d_shared=self.d_shared,
            d_subject=self.d_subject,
        )
        # Classifier acts on continuous z_e_shared — not z_q
        self.classifier = LinearWithConstraint(
            in_features=self.d_shared,
            out_features=n_outputs,
            max_norm=max_norm_clf,
        )

    def forward(self, z_e: torch.Tensor, subject_id: torch.LongTensor):
        z_e_shared  = z_e[:, :self.d_shared]
        z_e_subject = z_e[:, self.d_shared:]

        z_q_soft, z_q_hard, indices, weights, loss_vq = self.vq(z_e_shared)
        z_subject = self.subject_head(z_e_subject, subject_id)
        z_e_recon = self.decoder(z_q_soft, z_subject)

        # Clean gradient path: continuous z_e_shared -> classifier
        logits = self.classifier(z_e_shared)

        return logits, z_e_shared, z_q_soft, z_subject, z_e_recon, weights, loss_vq


# =============================================================================
# 6. Full Model
# =============================================================================

class EEGNeXVQDecomposed(nn.Module):

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        num_subjects: int,
        filter_1: int = 16,
        filter_2: int = 32,
        depth_multiplier: int = 2,
        drop_prob: float = 0.5,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        codebook_size: int = 24,
        commitment_beta: float = 0.5,
        ema_decay: float = 0.95,
        temp_start: float = 1.0,
        temp_end: float = 0.1,
    ):
        super().__init__()

        self.backbone = EEGNeXBackbone(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            filter_1=filter_1, filter_2=filter_2, depth_multiplier=depth_multiplier,
            drop_prob=drop_prob, kernel_block_1_2=kernel_block_1_2,
            kernel_block_4=kernel_block_4, dilation_block_4=dilation_block_4,
            avg_pool_block4=avg_pool_block4, kernel_block_5=kernel_block_5,
            dilation_block_5=dilation_block_5, avg_pool_block5=avg_pool_block5,
        )
        self.head = VQDecomposedHead(
            in_features=self.backbone.out_features,
            n_outputs=n_outputs,
            num_subjects=num_subjects,
            codebook_size=codebook_size,
            commitment_beta=commitment_beta,
            ema_decay=ema_decay,
            temp_start=temp_start,
            temp_end=temp_end,
        )

    def forward(self, x: torch.Tensor, subject_id: torch.LongTensor):
        z_e = self.backbone(x)
        logits, z_e_shared, z_q_soft, z_subject, z_e_recon, weights, loss_vq = \
            self.head(z_e, subject_id)
        return logits, z_e_shared, z_q_soft, z_subject, z_e_recon, z_e, weights, loss_vq

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cross-subject retrieval pathway.
        Hard discrete z_q — no subject_id, z_subject discarded.
        """
        z_e = self.backbone(x)
        z_e_shared = z_e[:, :self.head.d_shared]
        _, z_q_hard, _, _, _ = self.head.vq(z_e_shared)
        return z_q_hard

    def set_temperature(self, epoch: int, total_epochs: int):
        self.head.vq.set_temperature(epoch, total_epochs)


# =============================================================================
# 7. Loss Function
# =============================================================================

def compute_loss(
    logits, labels,
    z_e_shared, z_subject,
    z_e_recon, z_e,
    vq_weights, loss_vq,
    lambda_vq: float        = 0.5,
    lambda_recon: float     = 0.5,
    lambda_ortho: float     = 1.0,
    lambda_coherence: float = 0.5,
):
    """
    L = L_cls + lambda_vq*L_vq + lambda_recon*L_recon
      + lambda_ortho*L_ortho + lambda_coherence*L_coherence

    L_cls       : CrossEntropy on continuous z_e_shared
    L_vq        : codebook+commitment regulariser
    L_recon     : MSE(recon, z_e) — z_subject encodes what VQ discarded
    L_ortho     : squared cosine similarity — disentanglement constraint
    L_coherence : within-class VQ weight variance — class-coherent codebook
    """
    L_cls   = F.cross_entropy(logits, labels)
    L_recon = F.mse_loss(z_e_recon, z_e.detach())

    d = min(z_e_shared.shape[1], z_subject.shape[1])
    z_s = F.normalize(z_e_shared[:, :d], dim=-1)
    z_u = F.normalize(z_subject[:, :d],  dim=-1)
    L_ortho = (z_s * z_u).sum(dim=-1).pow(2).mean()

    n_classes   = logits.shape[1]
    L_coherence = torch.tensor(0.0, device=logits.device)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 1:
            mean_w      = vq_weights[mask].mean(0, keepdim=True)
            L_coherence += (vq_weights[mask] - mean_w).pow(2).mean()
    L_coherence = L_coherence / n_classes

    total = (
        L_cls
        + lambda_vq        * loss_vq
        + lambda_recon     * L_recon
        + lambda_ortho     * L_ortho
        + lambda_coherence * L_coherence
    )
    return {
        "total":     total,
        "cls":       L_cls,
        "vq":        loss_vq,
        "recon":     L_recon,
        "ortho":     L_ortho,
        "coherence": L_coherence,
    }


# =============================================================================
# 8. Codebook Warm-Start
# =============================================================================

def init_codebook_from_data(model, dataloader, device, n_batches: int = 10):
    """
    Initialise VQ codebook from real encoder outputs before training.
    Prevents cold-start collapse from random uniform initialisation.
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for i, (X, y, s) in enumerate(dataloader):
            if i >= n_batches:
                break
            z_e = model.backbone(X.to(device))
            samples.append(z_e[:, :model.head.d_shared].cpu())

    samples = torch.cat(samples, dim=0)
    K = model.head.vq.K

    idx = (
        torch.randperm(len(samples))[:K]
        if len(samples) >= K
        else torch.randint(0, len(samples), (K,))
    )

    init_weights = samples[idx].to(device)
    model.head.vq.codebook.weight.data.copy_(init_weights)
    model.head.vq.ema_embedding_sum.copy_(init_weights)
    model.head.vq.ema_cluster_size.fill_(1.0)

    print(f"Codebook warm-started from {len(samples)} real encoder outputs "
          f"(K={K}, d={model.head.d_shared})")
    model.train()


# =============================================================================
# 9. Training Script  (PhysionetMI, 109 subjects)
# =============================================================================

def main(config):

    from utils import get_PhysionetMI
    import os

    # Verify data is present
    DATA_DIR = os.path.expanduser("~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0")
    missing = [
        (s, r) for s in config["subject"] for r in [4, 6, 8, 10, 12, 14]
        if not os.path.exists(
            os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf")
        )
    ]
    if missing:
        print(f"ERROR: {len(set(s for s,_ in missing))} subjects not downloaded.")
        print("Run: python download_physionet.py")
        raise SystemExit(1)

    # Load data
    print(f"Loading PhysionetMI for {len(config['subject'])} subjects...")
    data, labels, meta, channels = get_PhysionetMI(
        subject=config["subject"],
        freq_min=config["freq"][0],
        freq_max=config["freq"][1],
    )

    # Keep left_hand=0, right_hand=1, feet=2
    mask   = np.isin(labels, [0, 1, 2])
    data   = data[mask]
    labels = labels[mask]
    meta   = meta.iloc[mask].reset_index(drop=True)
    labels = np.array([{0: 0, 1: 1, 2: 2}[l] for l in labels])

    # 0-index subject IDs for embedding table
    unique_subjects = sorted(meta["subject"].unique())
    subject_to_idx  = {s: i for i, s in enumerate(unique_subjects)}
    subject_ids     = np.array([subject_to_idx[s] for s in meta["subject"]])
    num_subjects    = len(unique_subjects)

    # Crop to 512 samples
    n_t = data.shape[2]
    if n_t >= 512:
        start = (n_t - 512) // 2
        data  = data[:, :, start:start + 512]
    else:
        print(f"Warning: time dim is {n_t}, expected >= 512")

    print(f"Data: {data.shape} | Labels: {np.bincount(labels)} | Subjects: {num_subjects}")

    # Train / test split
    train_idx, test_idx = sk_split(
        np.arange(len(data)),
        test_size=0.2,
        stratify=labels,
        random_state=config["seed"],
    )

    class TensorsDataset(Dataset):
        def __init__(self, X, y, sids):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
            self.s = torch.LongTensor(sids)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return self.X[i], self.y[i], self.s[i]

    train_loader = DataLoader(
        TensorsDataset(data[train_idx], labels[train_idx], subject_ids[train_idx]),
        batch_size=config["batch_size"], shuffle=True,
    )
    valid_loader = DataLoader(
        TensorsDataset(data[test_idx], labels[test_idx], subject_ids[test_idx]),
        batch_size=config["batch_size"], shuffle=False,
    )

    # Model
    cuda   = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"Using device: {device}")

    model = EEGNeXVQDecomposed(
        n_chans=data.shape[1],
        n_outputs=len(np.unique(labels)),
        n_times=data.shape[2],
        num_subjects=num_subjects,
        filter_1=config["filter_1"],
        codebook_size=config["codebook_size"],
        commitment_beta=config["commitment_beta"],
        ema_decay=config["ema_decay"],
        temp_start=config["temp_start"],
        temp_end=config["temp_end"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}")
    print(f"  out_features (z_e) : {model.backbone.out_features}")
    print(f"  d_shared (VQ dim)  : {model.head.d_shared}")
    print(f"  d_subject (resid.) : {model.head.d_subject}")
    print(f"  Codebook size K    : {config['codebook_size']}")

    # Codebook warm-start
    init_codebook_from_data(model, train_loader, device, config["warmstart_batches"])

    # Optimiser and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_valid_acc   = 0.0
    patience_counter = 0

    # Training loop
    for epoch in range(config["epochs"]):

        # Anneal Gumbel temperature
        model.set_temperature(epoch, config["epochs"])
        current_temp = model.head.vq.temperature

        model.train()
        running = {k: 0.0 for k in ["total", "cls", "vq", "recon", "ortho", "coherence"]}
        train_correct = 0

        for inputs, targets, sids in train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            sids    = sids.to(device)

            optimizer.zero_grad()

            logits, z_e_shared, z_q_soft, z_subject, z_e_recon, z_e, weights, loss_vq = \
                model(inputs, sids)

            losses = compute_loss(
                logits, targets,
                z_e_shared, z_subject,
                z_e_recon, z_e,
                weights, loss_vq,
                lambda_vq=config["lambda_vq"],
                lambda_recon=config["lambda_recon"],
                lambda_ortho=config["lambda_ortho"],
                lambda_coherence=config["lambda_coherence"],
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            train_correct += (logits.argmax(1) == targets).sum().item()

        scheduler.step()
        for k in running:
            running[k] /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss    = 0.0
        valid_correct = 0

        with torch.no_grad():
            for inputs, targets, sids in valid_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                sids    = sids.to(device)

                logits, z_e_shared, z_q_soft, z_subject, z_e_recon, z_e, weights, loss_vq = \
                    model(inputs, sids)

                losses = compute_loss(
                    logits, targets,
                    z_e_shared, z_subject,
                    z_e_recon, z_e,
                    weights, loss_vq,
                    lambda_vq=config["lambda_vq"],
                    lambda_recon=config["lambda_recon"],
                    lambda_ortho=config["lambda_ortho"],
                    lambda_coherence=config["lambda_coherence"],
                )

                valid_loss    += losses["total"].item()
                valid_correct += (logits.argmax(1) == targets).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc   = valid_correct / len(valid_loader.dataset)
        dead_frac   = model.head.vq.dead_code_fraction()

        print(
            f"Ep {epoch+1:3d}/{config['epochs']} | "
            f"train={train_acc:.4f} valid={valid_acc:.4f} | "
            f"cls={running['cls']:.3f} vq={running['vq']:.3f} "
            f"recon={running['recon']:.3f} ortho={running['ortho']:.3f} "
            f"coh={running['coherence']:.3f} | "
            f"dead={dead_frac:.1%} temp={current_temp:.3f}"
        )

        wandb.log({
            "train_acc":            train_acc,
            "valid_acc":            valid_acc,
            "train_loss_total":     running["total"],
            "train_loss_cls":       running["cls"],
            "train_loss_vq":        running["vq"],
            "train_loss_recon":     running["recon"],
            "train_loss_ortho":     running["ortho"],
            "train_loss_coherence": running["coherence"],
            "valid_loss":           valid_loss,
            "dead_code_frac":       dead_frac,
            "vq_temperature":       current_temp,
        })

        if valid_acc > best_valid_acc:
            best_valid_acc   = valid_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_vq_decomposed.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}. Best: {best_valid_acc:.4f}")
                break

    print(f"\nBest valid accuracy: {best_valid_acc:.4f}")

    # Cross-subject evaluation
    print("\n--- Cross-subject eval ---")
    model.load_state_dict(torch.load("best_vq_decomposed.pt"))
    model.eval()

    cs_correct_hard = 0
    cs_correct_cont = 0

    with torch.no_grad():
        for inputs, targets, sids in valid_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            # Hard discrete code — true cross-subject retrieval setting
            z_q_hard = model.encode_shared(inputs)
            logits_hard = model.head.classifier(z_q_hard)
            cs_correct_hard += (logits_hard.argmax(1) == targets).sum().item()

            # Continuous z_e_shared — upper bound for shared-only pathway
            z_e = model.backbone(inputs)
            z_e_shared = z_e[:, :model.head.d_shared]
            logits_cont = model.head.classifier(z_e_shared)
            cs_correct_cont += (logits_cont.argmax(1) == targets).sum().item()

    cs_acc_hard = cs_correct_hard / len(valid_loader.dataset)
    cs_acc_cont = cs_correct_cont / len(valid_loader.dataset)

    print(f"z_q hard (discrete, cross-subject)  : {cs_acc_hard:.4f}")
    print(f"z_e_shared continuous (upper bound)  : {cs_acc_cont:.4f}")
    print(f"Discretisation cost                  : {cs_acc_cont - cs_acc_hard:.4f}")

    wandb.log({
        "cross_subject_acc_discrete":   cs_acc_hard,
        "cross_subject_acc_continuous": cs_acc_cont,
        "discretisation_cost":          cs_acc_cont - cs_acc_hard,
    })


# =============================================================================
# 10. Entry Point
# =============================================================================

if __name__ == "__main__":

    config = {
        # Data
        "freq":    [8, 45],
        "subject": list(range(1, 110)),
        "seed":    1,

        # Training
        "epochs":       50,
        "batch_size":   64,
        "lr":           1e-3,
        "weight_decay": 0.01,
        "patience":     10,

        # Backbone
        "filter_1": 16,

        # VQ
        "codebook_size":     24,    # was 64 -- 8 per class, eliminates structural dead codes
        "commitment_beta":   0.5,   # was 0.25 -- stronger pull, reduces continuous-discrete gap
        "ema_decay":         0.95,
        "warmstart_batches": 10,
        "temp_start":        1.0,
        "temp_end":          0.1,

        # Loss weights
        "lambda_vq":        0.5,
        "lambda_recon":     0.5,
        "lambda_ortho":     1.0,
        "lambda_coherence": 0.5,
    }

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    import wandb
    wandb.init(
        project="VQ_Decomposed_EEGNeX_PhysionetMI",
        config=config,
        reinit=False,
    )

    main(config)