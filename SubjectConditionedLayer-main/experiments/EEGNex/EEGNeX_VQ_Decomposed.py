# EEGNeX with Decomposed VQ Latent Space
# ============================================================
# z_shared  -- discrete VQ code, shared across subjects (classify from this)
# z_subject -- continuous subject residual (reconstruct from this, discard at test)
# ============================================================

import math, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split as sk_split

from braindecode.models import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint


# ── 1. Vector Quantizer ───────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=512, embedding_dim=64,
                 commitment_beta=0.25, ema_decay=0.99):
        super().__init__()
        self.K, self.d, self.beta, self.ema_decay = codebook_size, embedding_dim, commitment_beta, ema_decay
        self.codebook = nn.Embedding(self.K, self.d)
        nn.init.uniform_(self.codebook.weight, -1.0/self.K, 1.0/self.K)
        if ema_decay > 0:
            self.register_buffer("ema_cluster_size", torch.zeros(self.K))
            self.register_buffer("ema_embedding_sum", self.codebook.weight.data.clone())

    def forward(self, z_e):
        dists = (z_e.pow(2).sum(1, keepdim=True)
                 - 2 * (z_e @ self.codebook.weight.t())
                 + self.codebook.weight.pow(2).sum(1))
        indices = dists.argmin(1)
        z_q = self.codebook(indices)

        if self.training and self.ema_decay > 0:
            with torch.no_grad():
                oh = F.one_hot(indices, self.K).float()
                self.ema_cluster_size.mul_(self.ema_decay).add_(oh.sum(0) * (1 - self.ema_decay))
                self.ema_embedding_sum.mul_(self.ema_decay).add_((oh.t() @ z_e) * (1 - self.ema_decay))
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
                self.codebook.weight.data.copy_(self.ema_embedding_sum / smoothed.unsqueeze(1))

        loss_vq = F.mse_loss(z_q, z_e.detach()) + self.beta * F.mse_loss(z_e, z_q.detach())
        z_q_st  = z_e + (z_q - z_e).detach()   # straight-through estimator
        return z_q_st, z_q, indices, loss_vq

    def dead_code_fraction(self):
        return (self.ema_cluster_size < 1.0).float().mean().item() if self.ema_decay > 0 else float("nan")


# ── 2. Subject Residual Head ──────────────────────────────────────────────────

class SubjectResidualHead(nn.Module):
    """Per-subject embedding bias injected into a shared MLP."""
    def __init__(self, input_dim, num_subjects, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.subject_embedding = nn.Embedding(num_subjects, input_dim)
        nn.init.normal_(self.subject_embedding.weight, 0.0, 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, z_e_subject, subject_id):
        return self.mlp(z_e_subject + self.subject_embedding(subject_id))


# ── 3. Reconstruction Decoder ─────────────────────────────────────────────────

class ReconstructionDecoder(nn.Module):
    """Reconstructs z_e from (z_shared, z_subject). Forces z_subject to be useful."""
    def __init__(self, d_shared, d_subject):
        super().__init__()
        d = d_shared + d_subject
        self.decoder = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
    def forward(self, z_shared_st, z_subject):
        return self.decoder(torch.cat([z_shared_st, z_subject], dim=-1))


# ── 4. EEGNeX Backbone (shared, no subject-specific weights) ─────────────────

class EEGNeXBackbone(EEGModuleMixin, nn.Module):
    def __init__(self, n_chans=None, n_outputs=None, n_times=None,
                 chs_info=None, input_window_seconds=None, sfreq=None,
                 activation=nn.ELU, depth_multiplier=2, filter_1=8, filter_2=32,
                 drop_prob=0.5, kernel_block_1_2=64, kernel_block_4=16,
                 dilation_block_4=2, avg_pool_block4=4, kernel_block_5=16,
                 dilation_block_5=4, avg_pool_block5=8, max_norm_conv=1.0):
        super().__init__(n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info,
                         n_times=n_times, input_window_seconds=input_window_seconds, sfreq=sfreq)
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.filter_1 = filter_1
        f3 = filter_2 * depth_multiplier
        self._p4 = avg_pool_block4
        self._p5 = avg_pool_block5
        self.out_features = self._calc_out(avg_pool_block4, avg_pool_block5)

        kb=(1,kernel_block_1_2); k4=(1,kernel_block_4); d4=(1,dilation_block_4)
        p4=(1,avg_pool_block4);  k5=(1,kernel_block_5); d5=(1,dilation_block_5)
        p5=(1,avg_pool_block5)

        self.block_1 = nn.Sequential(
            Rearrange("b c t -> b 1 c t"),
            nn.Conv2d(1, filter_1, kb, padding="same", bias=False),
            nn.BatchNorm2d(filter_1))
        self.block_2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kb, padding="same", bias=False),
            nn.BatchNorm2d(filter_2))
        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(filter_2, f3, max_norm=max_norm_conv,
                                 kernel_size=(self.n_chans,1), groups=filter_2, bias=False),
            nn.BatchNorm2d(f3), activation(),
            nn.AvgPool2d(p4, padding=(0,1)), nn.Dropout(drop_prob))
        self.block_4 = nn.Sequential(
            nn.Conv2d(f3, filter_2, k4, dilation=d4, padding="same", bias=False),
            nn.BatchNorm2d(filter_2))
        self.block_5 = nn.Sequential(
            nn.Conv2d(filter_2, filter_1, k5, dilation=d5, padding="same", bias=False),
            nn.BatchNorm2d(filter_1), activation(),
            nn.AvgPool2d(p5, padding=(0,1)), nn.Dropout(drop_prob), nn.Flatten())

    def forward(self, x):
        return self.block_5(self.block_4(self.block_3(self.block_2(self.block_1(x)))))

    def _calc_out(self, p4, p5):
        T3 = math.floor((self.n_times + 2 - p4) / p4) + 1
        T5 = math.floor((T3          + 2 - p5) / p5) + 1
        return self.filter_1 * T5


# ── 5. VQ Decomposed Head ─────────────────────────────────────────────────────

class VQDecomposedHead(nn.Module):
    def __init__(self, in_features, n_outputs, num_subjects,
                 codebook_size=512, commitment_beta=0.25, ema_decay=0.99, max_norm_clf=0.25):
        super().__init__()
        self.d_shared  = in_features // 2
        self.d_subject = in_features - self.d_shared

        self.vq           = VectorQuantizer(codebook_size, self.d_shared, commitment_beta, ema_decay)
        self.subject_head = SubjectResidualHead(self.d_subject, num_subjects)
        self.decoder      = ReconstructionDecoder(self.d_shared, self.d_subject)
        self.classifier   = LinearWithConstraint(self.d_shared, n_outputs, max_norm=max_norm_clf)

    def forward(self, z_e, subject_id):
        z_e_shared, z_e_subject = z_e[:, :self.d_shared], z_e[:, self.d_shared:]
        z_q_st, z_q, indices, loss_vq = self.vq(z_e_shared)
        z_subject = self.subject_head(z_e_subject, subject_id)
        z_e_recon = self.decoder(z_q_st, z_subject)
        logits    = self.classifier(z_q_st)
        return logits, z_q_st, z_subject, z_e_recon, loss_vq


# ── 6. Full Model ─────────────────────────────────────────────────────────────

class EEGNeXVQDecomposed(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, num_subjects,
                 filter_1=8, filter_2=32, depth_multiplier=2, drop_prob=0.5,
                 kernel_block_1_2=64, kernel_block_4=16, dilation_block_4=2,
                 avg_pool_block4=4, kernel_block_5=16, dilation_block_5=4,
                 avg_pool_block5=8, codebook_size=512, commitment_beta=0.25, ema_decay=0.99):
        super().__init__()
        self.backbone = EEGNeXBackbone(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            filter_1=filter_1, filter_2=filter_2, depth_multiplier=depth_multiplier,
            drop_prob=drop_prob, kernel_block_1_2=kernel_block_1_2,
            kernel_block_4=kernel_block_4, dilation_block_4=dilation_block_4,
            avg_pool_block4=avg_pool_block4, kernel_block_5=kernel_block_5,
            dilation_block_5=dilation_block_5, avg_pool_block5=avg_pool_block5)
        self.head = VQDecomposedHead(
            self.backbone.out_features, n_outputs, num_subjects,
            codebook_size, commitment_beta, ema_decay)

    def forward(self, x, subject_id):
        z_e = self.backbone(x)
        logits, z_q_st, z_subject, z_e_recon, loss_vq = self.head(z_e, subject_id)
        return logits, z_q_st, z_subject, z_e_recon, z_e, loss_vq

    def encode_shared(self, x):
        """Retrieval/inference: returns discrete z_q only. Subject discarded."""
        z_e = self.backbone(x)
        _, z_q, _, _ = self.head.vq(z_e[:, :self.head.d_shared])
        return z_q


# ── 7. Loss ───────────────────────────────────────────────────────────────────

def compute_loss(logits, labels, z_q_st, z_subject, z_e_recon, z_e, loss_vq,
                 lambda_vq=1.0, lambda_recon=0.5, lambda_ortho=0.1):
    L_cls   = F.cross_entropy(logits, labels)
    L_recon = F.mse_loss(z_e_recon, z_e.detach())
    d = min(z_q_st.shape[1], z_subject.shape[1])
    L_ortho = (F.normalize(z_q_st[:,:d], dim=-1) * F.normalize(z_subject[:,:d], dim=-1)).sum(-1).pow(2).mean()
    total   = L_cls + lambda_vq*loss_vq + lambda_recon*L_recon + lambda_ortho*L_ortho
    return {"total": total, "cls": L_cls, "vq": loss_vq, "recon": L_recon, "ortho": L_ortho}


# ── 8. Training Script ────────────────────────────────────────────────────────

def main(config):
    from utils import get_PhysionetMI
    import os

    DATA_DIR = os.path.expanduser("~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0")
    missing = [(s,r) for s in config["subject"] for r in [4,6,8,10,12,14]
               if not os.path.exists(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf"))]
    if missing:
        print(f"Missing data. Run: python download_physionet.py"); raise SystemExit(1)

    data, labels, meta, _ = get_PhysionetMI(
        subject=config["subject"], freq_min=config["freq"][0], freq_max=config["freq"][1])

    mask   = np.isin(labels, [0,1,2])
    data   = data[mask]; labels = labels[mask]; meta = meta.iloc[mask].reset_index(drop=True)
    labels = np.array([{0:0,1:1,2:2}[l] for l in labels])

    unique_subjects = sorted(meta["subject"].unique())
    subject_to_idx  = {s:i for i,s in enumerate(unique_subjects)}
    subject_ids     = np.array([subject_to_idx[s] for s in meta["subject"]])
    num_subjects    = len(unique_subjects)

    n_t = data.shape[2]
    if n_t >= 512:
        s = (n_t-512)//2; data = data[:,:,s:s+512]

    train_idx, test_idx = sk_split(np.arange(len(data)), test_size=0.2,
                                   stratify=labels, random_state=config["seed"])

    class DS(Dataset):
        def __init__(self,X,y,s):
            self.X=torch.from_numpy(X).float(); self.y=torch.from_numpy(y).long(); self.s=torch.LongTensor(s)
        def __len__(self): return len(self.X)
        def __getitem__(self,i): return self.X[i],self.y[i],self.s[i]

    train_loader = DataLoader(DS(data[train_idx],labels[train_idx],subject_ids[train_idx]),
                              batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(DS(data[test_idx], labels[test_idx], subject_ids[test_idx]),
                              batch_size=config["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = EEGNeXVQDecomposed(
        n_chans=data.shape[1], n_outputs=len(np.unique(labels)),
        n_times=data.shape[2], num_subjects=num_subjects,
        codebook_size=config["codebook_size"],
        commitment_beta=config["commitment_beta"],
        ema_decay=config["ema_decay"],
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"d_shared={model.head.d_shared}  d_subject={model.head.d_subject}  K={config['codebook_size']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])

    best_acc, patience_counter = 0.0, 0

    for epoch in range(config["epochs"]):
        model.train()
        running = {k:0.0 for k in ["total","cls","vq","recon","ortho"]}
        correct = 0

        for X,y,s in train_loader:
            X,y,s = X.to(device), y.to(device), s.to(device)
            optimizer.zero_grad()
            logits,z_q_st,z_sub,z_recon,z_e,loss_vq = model(X,s)
            losses = compute_loss(logits,y,z_q_st,z_sub,z_recon,z_e,loss_vq,
                                  config["lambda_vq"],config["lambda_recon"],config["lambda_ortho"])
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            for k in running: running[k] += losses[k].item()
            correct += (logits.argmax(1)==y).sum().item()

        scheduler.step()
        for k in running: running[k] /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        model.eval(); v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for X,y,s in valid_loader:
                X,y,s = X.to(device),y.to(device),s.to(device)
                logits,z_q_st,z_sub,z_recon,z_e,loss_vq = model(X,s)
                losses = compute_loss(logits,y,z_q_st,z_sub,z_recon,z_e,loss_vq,
                                      config["lambda_vq"],config["lambda_recon"],config["lambda_ortho"])
                v_loss += losses["total"].item()
                v_correct += (logits.argmax(1)==y).sum().item()

        v_acc = v_correct / len(valid_loader.dataset)
        dead  = model.head.vq.dead_code_fraction()

        print(f"Ep {epoch+1:3d}/{config['epochs']} | train={train_acc:.4f} valid={v_acc:.4f} | "
              f"cls={running['cls']:.3f} vq={running['vq']:.3f} "
              f"recon={running['recon']:.3f} ortho={running['ortho']:.3f} | dead={dead:.1%}")

        wandb.log({"train_acc":train_acc,"valid_acc":v_acc,"train_loss_total":running["total"],
                   "train_loss_cls":running["cls"],"train_loss_vq":running["vq"],
                   "train_loss_recon":running["recon"],"train_loss_ortho":running["ortho"],
                   "valid_loss":v_loss/len(valid_loader),"dead_code_frac":dead})

        if v_acc > best_acc:
            best_acc = v_acc; patience_counter = 0
            torch.save(model.state_dict(), "best_vq_decomposed.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stop. Best: {best_acc:.4f}"); break

    print(f"\nBest valid accuracy: {best_acc:.4f}")

    # Cross-subject eval: z_shared ONLY — no subject adapter used
    print("\n--- Cross-subject eval (z_shared only) ---")
    model.load_state_dict(torch.load("best_vq_decomposed.pt"))
    model.eval(); cs_correct = 0
    with torch.no_grad():
        for X,y,s in valid_loader:
            z_q = model.encode_shared(X.to(device))
            cs_correct += (model.head.classifier(z_q).argmax(1)==y.to(device)).sum().item()
    cs_acc = cs_correct / len(valid_loader.dataset)
    print(f"z_shared-only accuracy: {cs_acc:.4f}")
    wandb.log({"cross_subject_acc_shared_only": cs_acc})


if __name__ == "__main__":
    config = {
        "freq": [8,45], "subject": list(range(1,110)), "seed": 1,
        "epochs": 50, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.01, "patience": 10,
        "codebook_size": 512, "commitment_beta": 0.25, "ema_decay": 0.99,
        "lambda_vq": 1.0, "lambda_recon": 0.5, "lambda_ortho": 0.1,
    }
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"]); random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    import wandb
    wandb.init(project="VQ_Decomposed_EEGNeX_PhysionetMI", config=config, reinit=False)
    main(config)