"""
Data utilities for Reptile LoRA experiments.

Loads BCI2a (BNCI2014001) and returns per-subject train/test splits
consistent with the LOSO evaluation structure.

All subject IDs returned are 0-indexed (original 1-9 -> 0-8).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, '../EEGNex')
from utils import get_BNCI2014001


# ── Dataset ───────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        """
        X:           (N, C, T) numpy array
        y:           (N,)      numpy array of int labels
        subject_ids: (N,)      numpy array of 0-indexed subject IDs
        """
        self.X          = torch.from_numpy(X).float()
        self.y          = torch.from_numpy(y).long()
        self.subject_id = torch.from_numpy(subject_ids).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_id[idx]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bci2a(freq_min=8, freq_max=45):
    """
    Load all 9 subjects from BCI Competition IV 2a.
    Returns raw arrays with metadata.
    """
    ALL_SUBJECTS = list(range(1, 10))

    data, labels, meta, channels = get_BNCI2014001(
        subject=ALL_SUBJECTS,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    # Crop to 512 samples matching existing EEGNeX experiments
    data = data[:, :, 244:756]

    # Remap subject IDs 1-9 -> 0-8
    subjects_raw = np.array(meta['subject'].values)
    subject_ids  = subjects_raw - 1

    sessions = np.array(meta['session'].values)

    return data, labels, subject_ids, sessions, channels


def build_loso_split(data, labels, subject_ids, sessions, held_out_subject_0idx):
    """
    Build LOSO train/test split.

    held_out_subject_0idx: 0-indexed subject to hold out (0-8)

    Returns:
        train_X, train_y, train_sids  — training subjects, session T data
        cal_X,   cal_y,   cal_sids    — held-out subject, session T (calibration pool)
        test_X,  test_y,  test_sids   — held-out subject, session E (evaluation)
    """
    train_mask = (
        (subject_ids != held_out_subject_0idx) &
        (sessions == '0train')
    )
    cal_mask = (
        (subject_ids == held_out_subject_0idx) &
        (sessions == '0train')
    )
    test_mask = (
        (subject_ids == held_out_subject_0idx) &
        (sessions == '1test')
    )

    train_X   = data[train_mask]
    train_y   = labels[train_mask]
    train_sids = subject_ids[train_mask]

    # Remap training subject IDs to be contiguous 0..n_train-1
    # e.g. if held out is 2 (0-idx), training subjects are 0,1,3,4,5,6,7,8
    # remapped to                                           0,1,2,3,4,5,6,7
    unique_train = np.sort(np.unique(train_sids))
    remap        = {old: new for new, old in enumerate(unique_train)}
    train_sids_remapped = np.array([remap[s] for s in train_sids])

    cal_X   = data[cal_mask]
    cal_y   = labels[cal_mask]
    cal_sids = np.zeros(len(cal_X), dtype=int)   # new subject always slot 0

    test_X   = data[test_mask]
    test_y   = labels[test_mask]
    test_sids = np.zeros(len(test_X), dtype=int)

    return (train_X, train_y, train_sids_remapped,
            cal_X,  cal_y,  cal_sids,
            test_X, test_y, test_sids)


def build_per_subject_dict(train_X, train_y, train_sids):
    """
    Build a dictionary {subject_id: (X_tensor, y_tensor)} for Reptile inner loop.
    """
    subject_data = {}
    for sid in np.unique(train_sids):
        mask = train_sids == sid
        subject_data[int(sid)] = (
            torch.from_numpy(train_X[mask]).float(),
            torch.from_numpy(train_y[mask]).long(),
        )
    return subject_data
