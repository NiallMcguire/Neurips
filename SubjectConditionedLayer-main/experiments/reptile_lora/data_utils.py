"""
Data utilities for Reptile LoRA experiments.

Supports:
    BCI2a — BNCI2014001, 22 channels, 4-class motor imagery, 9 subjects
    BCI2b — BNCI2014004,  3 channels, 2-class motor imagery, 9 subjects

All subject IDs returned are 0-indexed (original 1-9 -> 0-8).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, '../EEGNex')
from utils import get_BNCI2014001, get_BNCI2014004


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


# ── BCI2a loading ─────────────────────────────────────────────────────────────

def load_bci2a(freq_min=8, freq_max=45):
    """
    Load all 9 subjects from BCI Competition IV 2a (BNCI2014001).
    22 channels, 4-class motor imagery.
    Cropped to 512 samples matching existing EEGNeX experiments.
    """
    ALL_SUBJECTS = list(range(1, 10))

    data, labels, meta, channels = get_BNCI2014001(
        subject=ALL_SUBJECTS,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    # Crop to 512 samples matching existing EEGNeX experiments
    data = data[:, :, 244:756]

    subjects_raw = np.array(meta['subject'].values)
    subject_ids  = subjects_raw - 1    # 1-9 -> 0-8
    sessions     = np.array(meta['session'].values)

    return data, labels, subject_ids, sessions, channels


# ── BCI2b loading ─────────────────────────────────────────────────────────────

def load_bci2b(freq_min=8, freq_max=45):
    """
    Load all 9 subjects from BCI Competition IV 2b (BNCI2014004).
    3 channels (C3, Cz, C4), 2-class motor imagery (left/right hand).
    5 sessions per subject: 0,1,2 for training; 3,4 for test.
    No fixed crop — use full trial from the MOABB paradigm.
    """
    ALL_SUBJECTS = list(range(1, 10))

    data, labels, meta, channels = get_BNCI2014004(
        subject=ALL_SUBJECTS,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    subjects_raw = np.array(meta['subject'].values)
    subject_ids  = subjects_raw - 1    # 1-9 -> 0-8
    sessions     = np.array(meta['session'].values)

    return data, labels, subject_ids, sessions, channels


# ── Dataset config ────────────────────────────────────────────────────────────

# Maps dataset name to (train_sessions, test_sessions)
# BCI2a uses MOABB session labels '0train' and '1test'
# BCI2b uses MOABB session labels 'session_0' ... 'session_4'
DATASET_SESSIONS = {
    'BCI2a': {
        'train': ('0train',),
        'test':  ('1test',),
    },
    'BCI2b': {
        # Verified from MOABB output: sessions are labelled
        # '0train', '1train', '2train', '3test', '4test'
        'train': ('0train', '1train', '2train'),
        'test':  ('3test', '4test'),
    },
}


# ── LOSO split ────────────────────────────────────────────────────────────────

def build_loso_split(data, labels, subject_ids, sessions,
                     held_out_subject_0idx, dataset='BCI2a'):
    """
    Build LOSO train/test split for either dataset.

    held_out_subject_0idx: 0-indexed subject to hold out (0-8)
    dataset: 'BCI2a' or 'BCI2b' — determines which session labels to use

    Returns:
        train_X, train_y, train_sids  — training subjects, train session data
        cal_X,   cal_y,   cal_sids    — held-out subject, train sessions (calibration pool)
        test_X,  test_y,  test_sids   — held-out subject, test sessions (evaluation)
    """
    train_sessions = DATASET_SESSIONS[dataset]['train']
    test_sessions  = DATASET_SESSIONS[dataset]['test']

    in_train_session = np.isin(sessions, train_sessions)
    in_test_session  = np.isin(sessions, test_sessions)

    train_mask = (subject_ids != held_out_subject_0idx) & in_train_session
    cal_mask   = (subject_ids == held_out_subject_0idx) & in_train_session
    test_mask  = (subject_ids == held_out_subject_0idx) & in_test_session

    train_X    = data[train_mask]
    train_y    = labels[train_mask]
    train_sids = subject_ids[train_mask]

    # Remap training subject IDs to contiguous 0..n_train-1
    unique_train = np.sort(np.unique(train_sids))
    remap        = {old: new for new, old in enumerate(unique_train)}
    train_sids_remapped = np.array([remap[s] for s in train_sids])

    cal_X    = data[cal_mask]
    cal_y    = labels[cal_mask]
    cal_sids = np.zeros(len(cal_X), dtype=int)

    test_X    = data[test_mask]
    test_y    = labels[test_mask]
    test_sids = np.zeros(len(test_X), dtype=int)

    return (train_X, train_y, train_sids_remapped,
            cal_X,  cal_y,  cal_sids,
            test_X, test_y, test_sids)


# ── Per-subject dict for Reptile inner loop ───────────────────────────────────

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


# ── Convenience loader ────────────────────────────────────────────────────────

def load_dataset(dataset, freq_min=8, freq_max=45):
    """
    Dispatch to the correct loader by dataset name.
    Returns (data, labels, subject_ids, sessions, channels).
    """
    if dataset == 'BCI2a':
        return load_bci2a(freq_min=freq_min, freq_max=freq_max)
    elif dataset == 'BCI2b':
        return load_bci2b(freq_min=freq_min, freq_max=freq_max)
    else:
        raise ValueError(f'Unknown dataset: {dataset}. Choose BCI2a or BCI2b.')