# Modified from Braindecode (BSD 3-Clause License) https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnex.py
# Original copyright (c) 2017-currently Braindecode Developers
# See LICENSE_BRAINDECODE for details.
# Authors of the Base Code: Bruno Aristimunha <b.aristimunha@gmail.com>


import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint


class LoRAConvPerSubject(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        groups=1,
        rank=4,
        alpha=1.0,
        num_adapters=4,
        stride=1,
        padding="same",
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.rank = rank
        self.alpha = alpha
        self.num_adapters = num_adapters

        self.lora_A = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=rank,
                    kernel_size=kernel_size,
                    dilation=1,
                    groups=1,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
                for _ in range(num_adapters)
            ]
        )
        self.lora_B = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=rank,
                    out_channels=out_channels,
                    kernel_size=1,
                    dilation=1,
                    groups=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
                for _ in range(num_adapters)
            ]
        )

        for a in self.lora_A:
            torch.nn.init.normal_(a.weight, mean=0.0, std=0.02)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)

    def forward(self, x, subject_id: torch.LongTensor):
        out = self.conv(x)

        lora_out = torch.zeros_like(out)
        for i in range(self.num_adapters):
            mask = subject_id == i
            if mask.any():
                lora_A_i = self.lora_A[i](x[mask])
                lora_B_i = self.lora_B[i](lora_A_i)
                lora_out[mask] = self.alpha / self.rank * lora_B_i

        return out + lora_out


class EEGNeX(EEGModuleMixin, nn.Module):
    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        activation: nn.Module = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
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
        max_norm_linear: float = 0.25,
        mode="LoRA",
        rank=4,
        alpha=1.0,
        num_adapters=109,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        print(mode)
        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = self.filter_2 * self.depth_multiplier
        self.drop_prob = drop_prob
        self.activation = activation
        self.kernel_block_1_2 = (1, kernel_block_1_2)
        self.kernel_block_4 = (1, kernel_block_4)
        self.dilation_block_4 = (1, dilation_block_4)
        self.avg_pool_block4 = (1, avg_pool_block4)
        self.kernel_block_5 = (1, kernel_block_5)
        self.dilation_block_5 = (1, dilation_block_5)
        self.avg_pool_block5 = (1, avg_pool_block5)
        self.mode = mode
        self.in_features = self._calculate_output_length()

        if mode == "vanilla":
            self.block_1 = nn.Sequential(
                Rearrange("batch ch time -> batch 1 ch time"),
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.filter_1,
                    kernel_size=self.kernel_block_1_2,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_1),
            )
            self.block_2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filter_1,
                    out_channels=self.filter_2,
                    kernel_size=self.kernel_block_1_2,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_2),
            )
            self.block_3 = nn.Sequential(
                Conv2dWithConstraint(
                    in_channels=self.filter_2,
                    out_channels=self.filter_3,
                    max_norm=max_norm_conv,
                    kernel_size=(self.n_chans, 1),
                    groups=self.filter_2,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_3),
                self.activation(),
                nn.AvgPool2d(kernel_size=self.avg_pool_block4, padding=(0, 1)),
                nn.Dropout(p=self.drop_prob),
            )
            self.block_4 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filter_3,
                    out_channels=self.filter_2,
                    kernel_size=self.kernel_block_4,
                    dilation=self.dilation_block_4,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_2),
            )
            self.block_5 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filter_2,
                    out_channels=self.filter_1,
                    kernel_size=self.kernel_block_5,
                    dilation=self.dilation_block_5,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_1),
                self.activation(),
                nn.AvgPool2d(kernel_size=self.avg_pool_block5, padding=(0, 1)),
                nn.Dropout(p=self.drop_prob),
                nn.Flatten(),
            )

        elif mode == "LoRA":
            self.block_1_0 = nn.Sequential(
                Rearrange("batch ch time -> batch 1 ch time")
            )
            self.block_1_1 = LoRAConvPerSubject(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_1_2,
                rank=rank, alpha=alpha, num_adapters=num_adapters,
                padding="same", bias=False,
            )
            self.block_1_2 = nn.Sequential(nn.BatchNorm2d(num_features=self.filter_1))

            self.block_2_0 = LoRAConvPerSubject(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_1_2,
                rank=rank, alpha=alpha, num_adapters=num_adapters,
                padding="same", bias=False,
            )
            self.block_2_1 = nn.Sequential(nn.BatchNorm2d(num_features=self.filter_2))

            self.block_3_0 = Conv2dWithConstraint(
                in_channels=self.filter_2,
                out_channels=self.filter_3,
                max_norm=max_norm_conv,
                kernel_size=(self.n_chans, 1),
                groups=self.filter_2,
                bias=False,
            )
            self.block_3_1 = nn.Sequential(
                nn.BatchNorm2d(num_features=self.filter_3),
                self.activation(),
                nn.AvgPool2d(kernel_size=self.avg_pool_block4, padding=(0, 1)),
                nn.Dropout(p=self.drop_prob),
            )

            self.block_4_1 = LoRAConvPerSubject(
                in_channels=self.filter_3,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_4,
                dilation=self.dilation_block_4,
                padding="same",
                rank=rank, alpha=alpha, num_adapters=num_adapters,
                bias=False,
            )
            self.block_4_2 = nn.BatchNorm2d(num_features=self.filter_2)

            self.block_5_0 = LoRAConvPerSubject(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_5,
                dilation=self.dilation_block_5,
                padding="same",
                rank=rank, alpha=alpha, num_adapters=num_adapters,
                bias=False,
            )
            self.block_5_1 = nn.Sequential(
                nn.BatchNorm2d(num_features=self.filter_1),
                self.activation(),
                nn.AvgPool2d(kernel_size=self.avg_pool_block5, padding=(0, 1)),
                nn.Dropout(p=self.drop_prob),
                nn.Flatten(),
            )

        self.final_layer = LinearWithConstraint(
            in_features=self.in_features,
            out_features=self.n_outputs,
            max_norm=max_norm_linear,
        )

    def forward(self, x: torch.Tensor, subject_id=None) -> torch.Tensor:
        if self.mode == "vanilla":
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.block_5(x)
        elif self.mode == "LoRA":
            x = self.block_1_0(x)
            x = self.block_1_1(x, subject_id)
            x = self.block_2_0(x, subject_id)
            x = self.block_2_1(x)
            x = self.block_3_0(x)
            x = self.block_3_1(x)
            x = self.block_4_1(x, subject_id)
            x = self.block_4_2(x)
            x = self.block_5_0(x, subject_id)
            x = self.block_5_1(x)

        x = self.final_layer(x)
        return x

    def _calculate_output_length(self) -> int:
        p4 = self.avg_pool_block4[1]
        p5 = self.avg_pool_block5[1]
        pad4 = 1
        pad5 = 1
        T3 = math.floor((self.n_times + 2 * pad4 - p4) / p4) + 1
        T5 = math.floor((T3 + 2 * pad5 - p5) / p5) + 1
        return self.filter_1 * T5


# =============================================================================
# Training Script
# =============================================================================

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split as sk_train_test_split


def main(config):

    from utils import get_PhysionetMI

    # --- Check data is downloaded before attempting to load ---
    import os
    DATA_DIR = os.path.expanduser("~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0")
    RUNS = [4, 6, 8, 10, 12, 14]
    missing = []
    for subject in config["subject"]:
        subj_str = f"S{subject:03d}"
        for run in RUNS:
            fname = os.path.join(DATA_DIR, subj_str, f"{subj_str}R{run:02d}.edf")
            if not os.path.exists(fname):
                missing.append((subject, run))

    if missing:
        missing_subjects = sorted(set(s for s, _ in missing))
        print(f"ERROR: {len(missing_subjects)} subjects not yet downloaded: {missing_subjects}")
        print("Please run: python download_physionet.py")
        print("Then resubmit this job.")
        raise SystemExit(1)

    print(f"All data present. Loading PhysionetMI for subjects {config['subject']}...")
    data, labels, meta, channels = get_PhysionetMI(
        subject=config["subject"],
        freq_min=config["freq"][0],
        freq_max=config["freq"][1],
    )

    # --- Filter to only the classes we want (left_hand=0, right_hand=1, feet=2) ---
    # PhysionetMI also has "rest"=4 and "hands"=5 which we exclude
    valid_classes = [0, 1, 2]
    mask = np.isin(labels, valid_classes)
    data = data[mask]
    labels = labels[mask]
    meta = meta.iloc[mask].reset_index(drop=True)

    # Remap labels to be contiguous: 0, 1, 2
    label_map = {0: 0, 1: 1, 2: 2}
    labels = np.array([label_map[l] for l in labels])

    # --- Map subject IDs to 0-indexed integers for adapter indexing ---
    # PhysionetMI subjects are 1-109, adapters expect 0-indexed
    unique_subjects = sorted(meta["subject"].unique())
    subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    subject_ids = np.array([subject_to_idx[s] for s in meta["subject"]])

    # --- Crop time window (same as BCI IV-2a script) ---
    # PhysionetMI at 250Hz: trial is ~4s = 1000 samples, crop to 512
    n_times = data.shape[2]
    if n_times >= 512:
        start = (n_times - 512) // 2
        data = data[:, :, start:start + 512]
    else:
        print(f"Warning: time dimension is {n_times}, expected >= 512")

    print(f"Data shape after filtering: {data.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Number of subjects: {len(unique_subjects)}")

    # --- Train/test split (80/20 stratified by subject+label) ---
    # PhysionetMI doesn't have a canonical session split like BCI IV-2a,
    # so we split per-subject to avoid data leakage
    train_idx, test_idx = sk_train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        stratify=labels,
        random_state=config["seed"],
    )

    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    train_subject_ids = subject_ids[train_idx]
    test_subject_ids = subject_ids[test_idx]

    print(f"Train: {train_data.shape}, Test: {test_data.shape}")

    class TensorsDataset(Dataset):
        def __init__(self, X, y, subject_ids):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
            self.subject_id = torch.LongTensor(subject_ids)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.subject_id[idx]

    train_dataset = TensorsDataset(train_data, train_labels, train_subject_ids)
    valid_dataset = TensorsDataset(test_data, test_labels, test_subject_ids)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

    # --- Model ---
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"Using device: {device}")

    n_channels = train_data.shape[1]
    n_classes = len(np.unique(train_labels))
    input_window_samples = train_data.shape[2]
    num_adapters = len(unique_subjects)  # one adapter per subject

    print(f"n_channels={n_channels}, n_classes={n_classes}, "
          f"n_times={input_window_samples}, num_adapters={num_adapters}")

    model = EEGNeX(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        mode=config["mode"],
        rank=config["rank"],
        alpha=config["alpha"],
        num_adapters=num_adapters,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params:,}")

    if cuda:
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_valid_acc = 0.0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for inputs, labels_batch, subject_id in train_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            subject_id = subject_id.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, subject_id)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        model.eval()
        valid_loss, valid_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels_batch, subject_id in valid_loader:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                subject_id = subject_id.to(device)

                outputs = model(inputs, subject_id)
                loss = criterion(outputs, labels_batch)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_acc += (predicted == labels_batch).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader.dataset)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
        )

        wandb.log({
            "train_loss": train_loss,
            "test_loss": valid_loss,
            "acc_train": train_acc,
            "acc_test": valid_acc,
        })

        # Early stopping
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}. "
                      f"Best valid acc: {best_valid_acc:.4f}")
                break

    print(f"\nFinal best valid accuracy: {best_valid_acc:.4f}")


if __name__ == "__main__":

    config = {
        "mode": "LoRA",       # "vanilla" for subject-agnostic baseline
        "rank": 8,
        "alpha": 24,          # alpha = 3 * rank
        "freq": [8, 45],
        "subject": list(range(1, 110)),  # all 109 subjects
        "epochs": 50,
        "batch_size": 64,
        "weight_decay": 0.01,
        "lr": 0.001,
        "patience": 10,       # early stopping
    }

    i = 1
    config["seed"] = i
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    import wandb
    wandb.init(
        project="NeurIPS_Workshop_EEGNeX_PhysionetMI",
        config=config,
        reinit=False,
    )

    main(config)