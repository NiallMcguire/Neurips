#!/usr/bin/env python3
"""
Simplified Dataloader for EEG-EEG vs EEG-Text Alignment
Version: 2.2
Focus: Support within-subject and cross-subject pairing
New (v2.1): dynamic_resample flag for cross-subject EEG-EEG — re-samples the
            document subject on every __getitem__ call rather than locking it
            in at preprocessing time.  Only active when:
              document_type='eeg' AND subject_mode='cross-subject' AND dynamic_resample=True
New (v2.2): Per-channel EEG normalisation — _compute_subject_statistics now stores
            per-channel mean/std vectors instead of a single global scalar per subject.
            This removes the spatial fingerprint (channel amplitude ratios) that the
            model could exploit as a subject-identity shortcut, whilst preserving the
            temporal dynamics that carry semantic content.
            SubjectStratifiedSampler — custom BatchSampler that enforces subject
            diversity within every batch by cycling round-robin across subjects.
            Prevents batches from being dominated by a single subject's data, forcing
            the contrastive loss to discriminate on sentence content rather than on
            easily-separable subject fingerprints.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import warnings


def detect_dataset_format(data_path):
    """Detect whether dataset is original or Nieuwland format"""
    try:
        dataset = np.load(data_path, allow_pickle=True).item()
        ict_pairs = dataset['ict_pairs']

        if len(ict_pairs) == 0:
            return 'original'

        first_pair = ict_pairs[0]

        # Nieuwland format has ICTPair objects with attributes
        if hasattr(first_pair, 'query_eeg') and hasattr(first_pair, 'doc_eeg'):
            return 'nieuwland'
        elif 'query_eeg' in first_pair:
            return 'original'
        else:
            return 'original'
    except:
        return 'original'


def convert_nieuwland_to_original_format(ict_pairs):
    """Convert Nieuwland ICTPair objects to original dictionary format"""
    converted_pairs = []

    for pair in ict_pairs:
        if hasattr(pair, 'query_text'):  # ICTPair object
            converted_pair = {
                'query_text': pair.query_text,
                'query_eeg': pair.query_eeg,
                'query_words': getattr(pair, 'query_words', []),
                'doc_text': pair.doc_text,
                'doc_eeg': pair.doc_eeg,  # This is key for EEG-EEG alignment
                'doc_words': getattr(pair, 'doc_words', []),
                'participant_id': pair.participant_id,
                'sentence_id': pair.sentence_id,
                'fs': getattr(pair, 'fs', 500.0)
            }
        else:  # Already dictionary format
            converted_pair = pair

        converted_pairs.append(converted_pair)

    return converted_pairs


def compute_global_eeg_dimensions(data_path, max_eeg_len=50, dataset_type='auto'):
    """Compute global EEG dimensions for both query and document EEGs"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Detect dataset format
    detected_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
    print(f"Computing EEG dimensions for {detected_format} format...")

    dataset = np.load(data_path, allow_pickle=True).item()
    ict_pairs = dataset['ict_pairs']

    # Convert Nieuwland format if needed
    if detected_format == 'nieuwland':
        ict_pairs = convert_nieuwland_to_original_format(ict_pairs)

    global_max_words = 0
    global_max_time = 0
    global_max_channels = 0

    print(f"Computing global EEG dimensions across {len(ict_pairs)} samples...")

    for pair in ict_pairs[:1000]:  # Sample for efficiency
        # Check both query_eeg and doc_eeg
        for eeg_key in ['query_eeg', 'doc_eeg']:
            eeg_data = pair.get(eeg_key, None)
            if eeg_data is None:
                continue

            try:
                eeg_array = np.array(eeg_data, dtype=np.float32)

                if len(eeg_array.shape) == 3:
                    num_words, time_samples, channels = eeg_array.shape
                    if num_words > max_eeg_len:
                        num_words = max_eeg_len
                    global_max_words = max(global_max_words, num_words)
                    global_max_time = max(global_max_time, time_samples)
                    global_max_channels = max(global_max_channels, channels)
                elif len(eeg_array.shape) == 2:
                    time_samples, channels = eeg_array.shape
                    global_max_words = max(global_max_words, 1)
                    global_max_time = max(global_max_time, time_samples)
                    global_max_channels = max(global_max_channels, channels)
            except Exception:
                continue

    print(f"Global EEG dimensions: {global_max_words}x{global_max_time}x{global_max_channels}")
    return global_max_words, global_max_time, global_max_channels


class SimplifiedEEGDataloader(Dataset):
    """
    Simplified Dataset for EEG-EEG vs EEG-Text alignment experiments
    Version: 2.1

    Key features:
    - Always uses query_eeg as query
    - Toggles between doc_eeg (EEG-EEG) or doc_text (EEG-Text) as document
    - Supports within-subject and cross-subject pairing
    - No masking complexity
    - Simple train/val/test splits

    v2.1 additions:
    - dynamic_resample: When True (and document_type='eeg', subject_mode='cross-subject'),
      re-samples the document subject on every __getitem__ call instead of fixing it at
      preprocessing time.  This exposes the model to the full positive distribution across
      training and prevents memorisation of a single query→doc subject pairing per sentence.
      Has no effect for within-subject or EEG-Text experiments.
    """

    def __init__(self, data_path: str, tokenizer=None, max_text_len: int = 256,
                 max_eeg_len: int = 50, train_ratio: float = 0.8,
                 split: str = 'train', normalize_eeg: bool = True,
                 debug: bool = False, global_eeg_dims: tuple = None,
                 dataset_type: str = 'auto', holdout_subjects: bool = False,
                 document_type: str = 'text', subject_mode: str = 'within-subject',
                 dynamic_resample: bool = False,
                 query_subject: str = None, doc_subject: str = None):
        """
        Args:
            document_type:    'text' for EEG-Text alignment, 'eeg' for EEG-EEG alignment
            subject_mode:     'within-subject' (same person) or 'cross-subject' (different people)
            dynamic_resample: Re-sample the doc subject on every __getitem__ call.
                              Only active when document_type='eeg' AND subject_mode='cross-subject'.
                              When False (default), the doc subject is fixed at preprocessing time.
            query_subject:    If set, only pairs where participant_id == query_subject are used
                              as queries.  Enables fixed two-subject experiments.
            doc_subject:      If set (requires query_subject), the document for every EEG-EEG
                              pair is always taken from this subject.  Ignored for EEG-Text
                              (doc is always text).  Must differ from query_subject.
        """

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_eeg_len = max_eeg_len
        self.normalize_eeg = normalize_eeg
        self.debug = debug
        self.holdout_subjects = holdout_subjects
        self.document_type = document_type
        self.subject_mode = subject_mode
        self.dynamic_resample = dynamic_resample and (document_type == 'eeg') and (subject_mode == 'cross-subject')
        self.query_subject = query_subject
        self.doc_subject   = doc_subject

        if dynamic_resample and not self.dynamic_resample:
            print("⚠️  dynamic_resample=True has no effect: only active for EEG-EEG cross-subject mode.")

        if self.dynamic_resample:
            print("✅ Dynamic doc-subject resampling ENABLED (cross-subject EEG-EEG)")

        # Set global EEG dimensions
        if global_eeg_dims is not None:
            self.global_max_words, self.global_max_time, self.global_max_channels = global_eeg_dims
        else:
            self.global_max_words, self.global_max_time, self.global_max_channels = compute_global_eeg_dimensions(
                data_path, max_eeg_len, dataset_type)

        # Load data
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        dataset = np.load(data_path, allow_pickle=True).item()
        self.ict_pairs = dataset['ict_pairs']
        self.metadata = dataset.get('metadata', {})

        # Handle dataset format conversion
        self.dataset_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
        if self.dataset_format == 'nieuwland':
            if self.debug:
                print(f"Converting Nieuwland format to original format")
            self.ict_pairs = convert_nieuwland_to_original_format(self.ict_pairs)

        if self.debug:
            print(f"Loaded {len(self.ict_pairs)} ICT pairs")
            print(f"Document type: {document_type} ({'EEG-EEG' if document_type == 'eeg' else 'EEG-Text'} alignment)")
            print(f"Subject mode: {subject_mode}")

        # ── Fixed two-subject filtering ────────────────────────────────────────
        # When query_subject is specified, restrict the dataset to pairs where
        # the query comes from that subject only.  This enables clean two-subject
        # experiments: Subject A is always the query, Subject B always the doc.
        if self.query_subject is not None:
            all_subjects = set(p.get('participant_id', 'unknown') for p in self.ict_pairs)
            if self.query_subject not in all_subjects:
                raise ValueError(
                    f"query_subject '{self.query_subject}' not found in dataset. "
                    f"Available: {sorted(all_subjects)}"
                )
            if self.doc_subject is not None and self.doc_subject not in all_subjects:
                raise ValueError(
                    f"doc_subject '{self.doc_subject}' not found in dataset. "
                    f"Available: {sorted(all_subjects)}"
                )
            if self.doc_subject is not None and self.doc_subject == self.query_subject:
                raise ValueError("query_subject and doc_subject must be different subjects.")

            before = len(self.ict_pairs)
            self.ict_pairs = [p for p in self.ict_pairs
                              if p.get('participant_id', 'unknown') == self.query_subject]
            print(f"Filtered to query_subject='{self.query_subject}': "
                  f"{before} → {len(self.ict_pairs)} pairs")
            if self.doc_subject:
                print(f"Doc subject fixed to: '{self.doc_subject}'")
        # ──────────────────────────────────────────────────────────────────────

        # Validate dataset compatibility
        if document_type == 'eeg':
            sample_pair = self.ict_pairs[0]
            if 'doc_eeg' not in sample_pair or sample_pair['doc_eeg'] is None:
                raise ValueError("Dataset does not support EEG-EEG alignment (missing doc_eeg)")
            if self.debug:
                print("✅ Dataset supports EEG-EEG alignment")

        # For cross-subject mode, we need multiple subjects
        if subject_mode == 'cross-subject' and document_type == 'eeg':
            unique_subjects = set(pair['participant_id'] for pair in self.ict_pairs)
            if len(unique_subjects) < 2:
                raise ValueError(f"Cross-subject mode requires at least 2 subjects, found {len(unique_subjects)}")
            if self.debug:
                print(f"✅ Dataset has {len(unique_subjects)} subjects for cross-subject pairing")

        # Create train/val/test split
        random.seed(42)

        if holdout_subjects:
            if self.debug:
                print("Using holdout subjects split")
            self.indices, self.unique_subjects = self._create_subject_split(split, train_ratio, debug)
        else:
            if self.debug:
                print("Using random sample split")
            self.indices, self.unique_subjects = self._create_random_split(split, train_ratio)

        self.pairs = [self.ict_pairs[i] for i in self.indices]

        # Compute per-subject statistics for normalization
        print(f"Computing per-subject EEG statistics for normalization...")
        self.subject_stats = self._compute_subject_statistics()
        print(f"  Computed stats for {len(self.subject_stats)} subjects")

        # For cross-subject mode, build sentence-to-subject mapping.
        # Required for both static and dynamic resampling.
        if subject_mode == 'cross-subject' and document_type == 'eeg':
            self._build_sentence_subject_mapping()

        # Process pairs once
        self.processed_pairs = []
        self._process_pairs()

        if self.debug:
            print(
                f"{split} split: {len(self.processed_pairs)} samples from {len(self.unique_subjects)} unique subjects")
            if len(self.processed_pairs) > 0:
                self._debug_print_sample(0)

    def _compute_subject_statistics(self):
        """
        Compute per-subject, per-channel mean and std for EEG normalisation.

        v2.2: Statistics are now computed independently for every electrode
        channel rather than as a single global scalar across all channels and
        timepoints.  This removes the spatial amplitude fingerprint that
        distinguishes subjects at the channel level (e.g. subject A's frontal
        channels are 3× stronger than subject B's), whilst preserving the
        temporal dynamics within each channel that carry semantic content.

        Stats are stored as 1-D numpy arrays of shape [n_channels] so that
        _normalize_eeg can broadcast them across the [words, time, channels]
        EEG tensor in a single vectorised operation.
        """
        subject_eegs = {}  # participant_id -> list of EEG arrays

        # Collect all EEG data per subject
        for pair in self.pairs:
            participant_id = pair.get('participant_id', 'unknown')

            # Collect query EEG
            query_eeg = pair.get('query_eeg', None)
            if query_eeg is not None:
                subject_eegs.setdefault(participant_id, []).append(query_eeg)

            # Collect doc EEG if available (for EEG-EEG alignment)
            if self.document_type == 'eeg':
                doc_eeg = pair.get('doc_eeg', None)
                if doc_eeg is not None:
                    subject_eegs.setdefault(participant_id, []).append(doc_eeg)

        # Compute per-channel statistics per subject
        subject_stats = {}
        for participant_id, eeg_list in subject_eegs.items():
            if not eeg_list:
                continue

            try:
                # Flatten each EEG to [N_samples, n_channels] by collapsing
                # words and time dimensions, then stack across all recordings.
                channel_rows = []
                for eeg in eeg_list:
                    eeg_array = np.array(eeg, dtype=np.float32)
                    if eeg_array.ndim == 3:
                        # [words, time, channels] → [words*time, channels]
                        n_channels = eeg_array.shape[-1]
                        channel_rows.append(eeg_array.reshape(-1, n_channels))
                    elif eeg_array.ndim == 2:
                        # [time, channels]
                        channel_rows.append(eeg_array)
                    # 1-D or unexpected shapes are skipped

                if not channel_rows:
                    continue

                # Stack → [total_samples, n_channels]
                stacked = np.concatenate(channel_rows, axis=0)

                # Exclude all-zero rows (padding artefacts) from statistics
                non_zero_mask = (stacked != 0).any(axis=1)
                stacked = stacked[non_zero_mask]

                if stacked.shape[0] == 0:
                    continue

                chan_mean = stacked.mean(axis=0)           # [n_channels]
                chan_std  = stacked.std(axis=0)            # [n_channels]
                # Guard against dead channels with zero variance
                chan_std  = np.where(chan_std < 1e-6, 1.0, chan_std)

                subject_stats[participant_id] = {
                    'mean': chan_mean,   # ndarray [n_channels]
                    'std':  chan_std,    # ndarray [n_channels]
                }

                if self.debug and len(subject_stats) <= 3:
                    print(f"    Subject {participant_id}: "
                          f"chan_mean μ={chan_mean.mean():.4f} σ={chan_mean.std():.4f}, "
                          f"chan_std  μ={chan_std.mean():.4f} σ={chan_std.std():.4f} "
                          f"({stacked.shape[1]} channels)")

            except Exception as e:
                if self.debug:
                    print(f"    Warning: Could not compute stats for subject {participant_id}: {e}")
                continue

        return subject_stats

    def _build_sentence_subject_mapping(self):
        """Build mapping of sentences to subjects for cross-subject pairing"""
        self.sentence_to_subjects = {}  # sentence_id -> {participant_id: [pair_indices]}

        for idx, pair in enumerate(self.pairs):
            sentence_id = pair.get('sentence_id', 0)
            participant_id = pair.get('participant_id', 'unknown')

            if sentence_id not in self.sentence_to_subjects:
                self.sentence_to_subjects[sentence_id] = {}

            if participant_id not in self.sentence_to_subjects[sentence_id]:
                self.sentence_to_subjects[sentence_id][participant_id] = []

            self.sentence_to_subjects[sentence_id][participant_id].append(idx)

        if self.debug:
            print(f"Built sentence-subject mapping: {len(self.sentence_to_subjects)} unique sentences")
            multi_subject_sentences = sum(1 for s in self.sentence_to_subjects.values() if len(s) > 1)
            print(f"  Sentences with multiple subjects: {multi_subject_sentences}")

    def _create_subject_split(self, split, train_ratio, debug):
        """Create subject-based train/val/test split"""
        participant_to_indices = {}
        for idx, pair in enumerate(self.ict_pairs):
            participant_id = pair.get('participant_id', 'unknown')
            if participant_id not in participant_to_indices:
                participant_to_indices[participant_id] = []
            participant_to_indices[participant_id].append(idx)

        unique_participants = list(participant_to_indices.keys())
        random.shuffle(unique_participants)

        if debug:
            print(f"Found {len(unique_participants)} unique participants")

        n_participants = len(unique_participants)
        train_end = int(n_participants * 0.8)
        val_end = int(n_participants * 0.9)

        if split == 'train':
            selected_participants = unique_participants[:train_end]
        elif split == 'val':
            selected_participants = unique_participants[train_end:val_end]
        elif split == 'test':
            selected_participants = unique_participants[val_end:]
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

        selected_indices = []
        for participant in selected_participants:
            selected_indices.extend(participant_to_indices[participant])

        if debug:
            print(
                f"Selected {len(selected_indices)} samples for {split} split from {len(selected_participants)} participants")

        return selected_indices, set(selected_participants)

    def _create_random_split(self, split, train_ratio):
        """Create random train/val/test split"""
        shuffled_indices = list(range(len(self.ict_pairs)))
        random.shuffle(shuffled_indices)

        if split in ['train', 'val']:
            split_idx = int(len(self.ict_pairs) * train_ratio)
            if split == 'train':
                selected_indices = shuffled_indices[:split_idx]
            else:  # val
                selected_indices = shuffled_indices[split_idx:]
        elif split == 'test':
            val_start_idx = int(len(self.ict_pairs) * train_ratio)
            val_indices = shuffled_indices[val_start_idx:]
            val_split_point = len(val_indices) // 2
            selected_indices = val_indices[val_split_point:]
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

        unique_subjects = set()
        for idx in selected_indices:
            participant_id = self.ict_pairs[idx].get('participant_id', 'unknown')
            unique_subjects.add(participant_id)

        return selected_indices, unique_subjects

    def _process_pairs(self):
        """Process raw ICT pairs"""
        successful_pairs = 0
        failed_pairs = 0

        for idx, pair in enumerate(self.pairs):
            try:
                processed = self._process_single_pair(pair, idx)
                if processed is not None:
                    self.processed_pairs.append(processed)
                    successful_pairs += 1
                else:
                    failed_pairs += 1
            except Exception as e:
                failed_pairs += 1
                if self.debug:
                    print(f"Failed to process pair {idx}: {e}")
                continue

        if self.debug or failed_pairs > 0:
            print(f"Processed {successful_pairs} pairs successfully, {failed_pairs} failed")

    def _process_single_pair(self, pair, idx):
        """Process single ICT pair"""
        # Extract query EEG (always used)
        query_eeg = pair.get('query_eeg', None)
        if query_eeg is None:
            return None

        query_eeg_processed = self._process_eeg(query_eeg)
        if query_eeg_processed is None:
            return None

        # Extract document based on document_type and subject_mode
        if self.document_type == 'eeg':
            # EEG-EEG alignment
            if self.subject_mode == 'within-subject':
                doc_eeg = pair.get('doc_eeg', None)
                if doc_eeg is None:
                    return None

                doc_eeg_processed = self._process_eeg(doc_eeg)
                if doc_eeg_processed is None:
                    return None

                doc_data = doc_eeg_processed
                doc_participant_id = pair.get('participant_id', 'unknown')

            else:  # cross-subject
                sentence_id = pair.get('sentence_id', 0)
                query_participant_id = pair.get('participant_id', 'unknown')

                if sentence_id in self.sentence_to_subjects:
                    # If doc_subject is fixed, use it; otherwise random.choice
                    if self.doc_subject is not None:
                        if self.doc_subject in self.sentence_to_subjects[sentence_id]:
                            other_participants = [self.doc_subject]
                        else:
                            # doc_subject has no recording for this sentence — skip
                            return None
                    else:
                        other_participants = [p for p in self.sentence_to_subjects[sentence_id].keys()
                                              if p != query_participant_id]

                    if other_participants:
                        doc_participant_id = random.choice(other_participants)
                        doc_pair_indices = self.sentence_to_subjects[sentence_id][doc_participant_id]
                        doc_pair_idx = random.choice(doc_pair_indices)
                        doc_pair = self.pairs[doc_pair_idx]

                        doc_eeg = doc_pair.get('doc_eeg', None)
                        if doc_eeg is None:
                            return None

                        doc_eeg_processed = self._process_eeg(doc_eeg)
                        if doc_eeg_processed is None:
                            return None

                        doc_data = doc_eeg_processed
                    else:
                        # No other participants for this sentence — skip
                        return None
                else:
                    return None

            doc_text = None

        else:  # document_type == 'text'
            doc_text = pair.get('doc_text', '').strip()
            if not doc_text:
                return None

            doc_data = None
            doc_participant_id = pair.get('participant_id', 'unknown')

        query_text = pair.get('query_text', '').strip()

        return {
            'query_text': query_text,
            'query_eeg': query_eeg_processed,
            'doc_text': doc_text,
            'doc_eeg': doc_data,
            'query_participant_id': pair.get('participant_id', 'unknown'),
            'doc_participant_id': doc_participant_id,
            'sentence_id': pair.get('sentence_id', 0),
            'document_type': self.document_type,
            'subject_mode': self.subject_mode,
            'original_idx': idx
        }

    def _process_eeg(self, eeg_data):
        """Process EEG data to consistent format with global padding"""
        try:
            eeg_array = np.array(eeg_data, dtype=np.float32)

            if len(eeg_array.shape) == 3:
                num_words, time_samples, channels = eeg_array.shape
                if num_words > self.max_eeg_len:
                    eeg_array = eeg_array[:self.max_eeg_len]
                    num_words = self.max_eeg_len

                padded_eeg = np.zeros((self.global_max_words, self.global_max_time, self.global_max_channels),
                                      dtype=np.float32)
                padded_eeg[:num_words, :time_samples, :channels] = eeg_array
                return padded_eeg

            elif len(eeg_array.shape) == 2:
                time_samples, channels = eeg_array.shape
                padded_eeg = np.zeros((self.global_max_words, self.global_max_time, self.global_max_channels),
                                      dtype=np.float32)
                padded_eeg[0, :time_samples, :channels] = eeg_array
                return padded_eeg

            else:
                flattened = eeg_array.flatten()
                for channels in [32, 63, 64, 128, 256]:
                    if len(flattened) % channels == 0:
                        time_samples = len(flattened) // channels
                        if time_samples >= 10:
                            reshaped = flattened.reshape(time_samples, channels)
                            padded_eeg = np.zeros(
                                (self.global_max_words, self.global_max_time, self.global_max_channels),
                                dtype=np.float32)
                            padded_eeg[0, :time_samples, :channels] = reshaped
                            return padded_eeg

                if self.debug:
                    print(f"Could not reshape EEG with shape {eeg_array.shape}")
                return None

        except Exception as e:
            if self.debug:
                print(f"EEG processing failed: {e}")
            return None

    def _normalize_eeg(self, eeg_tensor, participant_id='unknown'):
        """
        Normalise EEG tensor using per-subject, per-channel statistics (v2.2).

        v2.1 used a single scalar mean/std per subject, which equalised global
        amplitude but left the relative channel-amplitude pattern intact.  That
        spatial pattern is a strong subject fingerprint the model can exploit.

        v2.2 computes mean and std independently for every electrode channel,
        then subtracts and scales each channel to N(0,1).  This removes the
        spatial fingerprint while keeping the temporal structure within channels.

        Args:
            eeg_tensor:     torch.Tensor of shape [words, time, channels]
            participant_id: subject ID used to look up stored statistics

        Returns:
            Normalised tensor of the same shape.
        """
        if not self.normalize_eeg:
            return eeg_tensor

        if participant_id in self.subject_stats:
            stats      = self.subject_stats[participant_id]
            chan_mean   = torch.tensor(stats['mean'], dtype=torch.float32)  # [C_stored]
            chan_std    = torch.tensor(stats['std'],  dtype=torch.float32)  # [C_stored]

            n_stored = chan_mean.shape[0]
            n_actual = eeg_tensor.shape[-1]          # channels in this tensor

            if n_stored >= n_actual:
                # Trim to the actual channel count (handles padding channels)
                chan_mean = chan_mean[:n_actual]
                chan_std  = chan_std[:n_actual]
            else:
                # Stored stats cover fewer channels than the tensor — pad with
                # identity stats (mean=0, std=1) so padding channels pass through.
                pad = n_actual - n_stored
                chan_mean = torch.cat([chan_mean, torch.zeros(pad)])
                chan_std  = torch.cat([chan_std,  torch.ones(pad)])

            # Broadcast: eeg_tensor is [words, time, channels]
            # chan_mean / chan_std are [channels] → broadcast over [words, time]
            normalized = (eeg_tensor - chan_mean) / chan_std
            return normalized

        else:
            # Fallback: compute per-channel stats on-the-fly from this tensor.
            # Used when subject_stats lookup fails (e.g. unseen subject at test time).
            if eeg_tensor.ndim == 3:
                words, time, channels = eeg_tensor.shape
                # Reshape to [words*time, channels] for per-channel computation
                flat = eeg_tensor.reshape(-1, channels)
                non_zero_rows = flat.abs().sum(dim=1) > 0
                if non_zero_rows.sum() > 0:
                    active = flat[non_zero_rows]
                    chan_mean = active.mean(dim=0)                          # [channels]
                    chan_std  = active.std(dim=0)                           # [channels]
                    chan_std  = torch.where(
                        chan_std < 1e-6,
                        torch.ones_like(chan_std),
                        chan_std
                    )
                    return (eeg_tensor - chan_mean) / chan_std

            return eeg_tensor

    def _tokenize_text(self, text):
        """Tokenize text for document encoding"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text document processing")

        encoded = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_text_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

    def _debug_print_sample(self, idx):
        """Print detailed debug information for a sample"""
        sample = self.processed_pairs[idx]
        print(f"\nSample {idx}:")
        print(f"  Query text: '{sample['query_text'][:100]}...'")
        print(f"  Query EEG shape: {sample['query_eeg'].shape}")
        print(f"  Query participant: {sample['query_participant_id']}")
        print(f"  Document type: {sample['document_type']}")
        print(f"  Subject mode: {sample['subject_mode']}")
        if sample['document_type'] == 'text':
            print(f"  Doc text: '{sample['doc_text'][:100]}...'")
            print(f"  Doc participant: {sample['doc_participant_id']}")
        else:
            print(f"  Doc EEG shape: {sample['doc_eeg'].shape}")
            print(f"  Doc participant: {sample['doc_participant_id']}")
            if sample['subject_mode'] == 'cross-subject':
                print(f"  ✅ Cross-subject pairing: {sample['query_participant_id']} → {sample['doc_participant_id']}")
            else:
                print(f"  Within-subject pairing: {sample['query_participant_id']} → {sample['doc_participant_id']}")

    def __len__(self):
        return len(self.processed_pairs)

    def __getitem__(self, idx):
        """
        Get sample for training with subject-wise normalization.

        When dynamic_resample=True (cross-subject EEG-EEG only), a new document
        subject is sampled on every call.  This prevents the model from memorising
        a fixed query→doc subject pairing per sentence and forces it to learn
        subject-invariant semantic representations.
        """
        sample = self.processed_pairs[idx]

        # ── Query EEG ──────────────────────────────────────────────────────────
        query_eeg_tensor = torch.tensor(sample['query_eeg'], dtype=torch.float32)
        query_eeg_tensor = self._normalize_eeg(query_eeg_tensor, sample['query_participant_id'])

        # ── Document ───────────────────────────────────────────────────────────
        if self.document_type == 'eeg':

            # ── Dynamic resampling path ────────────────────────────────────────
            if self.dynamic_resample:
                sentence_id = sample['sentence_id']
                query_pid = sample['query_participant_id']

                # If doc_subject is fixed, only ever sample from that subject
                if self.doc_subject is not None:
                    other_pids = ([self.doc_subject]
                                  if self.doc_subject in self.sentence_to_subjects.get(sentence_id, {})
                                  else [])
                else:
                    other_pids = [p for p in self.sentence_to_subjects[sentence_id].keys()
                                  if p != query_pid]

                doc_eeg_tensor = None
                doc_pid = sample['doc_participant_id']  # fallback

                if other_pids:
                    # Pick a fresh random subject and a random pair from that subject
                    sampled_pid = random.choice(other_pids)
                    pair_indices = self.sentence_to_subjects[sentence_id][sampled_pid]
                    raw_pair = self.pairs[random.choice(pair_indices)]
                    raw_doc_eeg = raw_pair.get('doc_eeg', None)

                    if raw_doc_eeg is not None:
                        processed = self._process_eeg(raw_doc_eeg)
                        if processed is not None:
                            doc_eeg_tensor = torch.tensor(processed, dtype=torch.float32)
                            doc_eeg_tensor = self._normalize_eeg(doc_eeg_tensor, sampled_pid)
                            doc_pid = sampled_pid

                # Fallback: use the statically-processed doc from preprocessing
                if doc_eeg_tensor is None:
                    doc_eeg_tensor = torch.tensor(sample['doc_eeg'], dtype=torch.float32)
                    doc_eeg_tensor = self._normalize_eeg(doc_eeg_tensor, sample['doc_participant_id'])
                    doc_pid = sample['doc_participant_id']

            # ── Static path (original behaviour) ──────────────────────────────
            else:
                doc_eeg_tensor = torch.tensor(sample['doc_eeg'], dtype=torch.float32)
                doc_eeg_tensor = self._normalize_eeg(doc_eeg_tensor, sample['doc_participant_id'])
                doc_pid = sample['doc_participant_id']

            return {
                'query_eeg': query_eeg_tensor,
                'doc_eeg': doc_eeg_tensor,
                'doc_text_tokens': None,
                'metadata': {
                    'query_participant_id': sample['query_participant_id'],
                    'doc_participant_id': doc_pid,
                    'sentence_id': sample['sentence_id'],
                    'document_type': sample['document_type'],
                    'subject_mode': sample['subject_mode'],
                    'query_text': sample['query_text'],
                    'original_idx': sample['original_idx'],
                    'dynamic_resampled': self.dynamic_resample
                }
            }

        else:  # document_type == 'text'
            doc_tokens = self._tokenize_text(sample['doc_text'])

            return {
                'query_eeg': query_eeg_tensor,
                'doc_eeg': None,
                'doc_text_tokens': doc_tokens,
                'metadata': {
                    'query_participant_id': sample['query_participant_id'],
                    'doc_participant_id': sample['doc_participant_id'],
                    'sentence_id': sample['sentence_id'],
                    'document_type': sample['document_type'],
                    'subject_mode': sample['subject_mode'],
                    'query_text': sample['query_text'],
                    'doc_text': sample['doc_text'],
                    'original_idx': sample['original_idx'],
                    'dynamic_resampled': False
                }
            }


def simple_collate_fn(batch):
    """Simple collate function for batching"""

    document_type = batch[0]['metadata']['document_type']

    query_eegs = torch.stack([item['query_eeg'] for item in batch])

    if document_type == 'eeg':
        doc_eegs = torch.stack([item['doc_eeg'] for item in batch])
        doc_text_tokens = None

    else:  # document_type == 'text'
        doc_eegs = None
        doc_text_tokens = {
            'input_ids': torch.stack([item['doc_text_tokens']['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['doc_text_tokens']['attention_mask'] for item in batch])
        }

    metadata = [item['metadata'] for item in batch]

    return {
        'query_eegs': query_eegs,
        'doc_eegs': doc_eegs,
        'doc_text_tokens': doc_text_tokens,
        'metadata': metadata,
        'document_type': document_type
    }


def debug_batch(batch, print_details=True):
    """Debug utility to inspect a batch"""
    batch_size = len(batch['metadata'])
    document_type = batch['document_type']
    subject_mode = batch['metadata'][0]['subject_mode']

    print(f"\nBatch Debug:")
    print(f"  Batch size: {batch_size}")
    print(f"  Document type: {document_type} ({'EEG-EEG' if document_type == 'eeg' else 'EEG-Text'} alignment)")
    print(f"  Subject mode: {subject_mode}")
    print(f"  Query EEGs: {batch['query_eegs'].shape}")

    if document_type == 'eeg':
        print(f"  Doc EEGs: {batch['doc_eegs'].shape}")
        if subject_mode == 'cross-subject':
            query_subjects = [m['query_participant_id'] for m in batch['metadata']]
            doc_subjects = [m['doc_participant_id'] for m in batch['metadata']]
            cross_pairs = sum(1 for q, d in zip(query_subjects, doc_subjects) if q != d)
            dynamic_flag = batch['metadata'][0].get('dynamic_resampled', False)
            print(f"  Cross-subject pairs: {cross_pairs}/{batch_size}")
            print(f"  Dynamic resampling: {'✅ ON' if dynamic_flag else '❌ OFF'}")
    else:
        print(f"  Doc text input_ids: {batch['doc_text_tokens']['input_ids'].shape}")
        print(f"  Doc text attention_mask: {batch['doc_text_tokens']['attention_mask'].shape}")

    if print_details and batch_size > 0:
        meta = batch['metadata'][0]
        print(f"  Sample query participant: {meta['query_participant_id']}")
        print(f"  Sample doc participant: {meta['doc_participant_id']}")
        print(f"  Sample query: '{meta['query_text'][:50]}...'")
        if document_type == 'text':
            print(f"  Sample doc: '{meta['doc_text'][:50]}...'")



# ──────────────────────────────────────────────────────────────────────────────
# Subject-Stratified Batch Sampler (v2.2)
# ──────────────────────────────────────────────────────────────────────────────

class SubjectStratifiedSampler(Sampler):
    """
    Constructs batches that contain samples from many distinct subjects.

    Without stratification, PyTorch's default random sampler produces batches
    whose composition is proportional to the dataset's subject distribution.
    If one subject contributed 400 of 1 000 total samples, ~40 % of each batch
    will be that subject's data.  In-batch negatives are then dominated by
    same-subject pairs, which the model can trivially discriminate by subject
    identity rather than sentence content — causing the validation loss to
    diverge while training loss keeps falling (the EEG-EEG divergence pattern).

    Algorithm
    ---------
    1. Group sample indices by subject (query_participant_id).
    2. Each epoch, shuffle the within-subject index lists independently.
    3. Build batches by cycling through subjects in round-robin order,
       drawing one sample per subject per cycle until the batch is full.
    4. Batches that cannot be completed (end-of-data) are discarded so every
       batch has exactly `batch_size` samples at uniform subject diversity.
    5. The final list of batches is shuffled before yielding so the model
       sees subjects in different sentence orderings each epoch.

    Args:
        dataset:    A SimplifiedEEGDataloader instance (must have processed_pairs).
        batch_size: Number of samples per batch.
        shuffle:    Shuffle within-subject lists and batch order each epoch.
        seed:       Base random seed; incremented each call to set_epoch().
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0

        # Build subject → [sample indices] mapping from processed_pairs
        subject_to_indices: Dict[str, List[int]] = {}
        for idx, pair in enumerate(dataset.processed_pairs):
            pid = pair.get('query_participant_id', 'unknown')
            subject_to_indices.setdefault(pid, []).append(idx)

        self.subjects        = sorted(subject_to_indices.keys())
        self.subject_indices = subject_to_indices
        self.n_subjects      = len(self.subjects)

        total_samples = sum(len(v) for v in subject_to_indices.values())
        n_batches     = total_samples // batch_size

        print(f"SubjectStratifiedSampler initialised:")
        print(f"  Subjects       : {self.n_subjects}")
        print(f"  Total samples  : {total_samples}")
        print(f"  Batch size     : {batch_size}")
        print(f"  Batches/epoch  : {n_batches}  (drop_last=True)")
        samples_per_subj = {pid: len(v) for pid, v in subject_to_indices.items()}
        min_s = min(samples_per_subj.values())
        max_s = max(samples_per_subj.values())
        print(f"  Samples/subject: min={min_s}, max={max_s}")

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to advance the RNG seed."""
        self._epoch = epoch

    # ------------------------------------------------------------------
    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        # Shuffle within-subject lists independently each epoch
        per_subject: Dict[str, List[int]] = {}
        for pid in self.subjects:
            indices = self.subject_indices[pid].copy()
            if self.shuffle:
                rng.shuffle(indices)
            per_subject[pid] = indices

        pointers   = {pid: 0 for pid in self.subjects}
        exhausted  = set()
        batches: List[List[int]] = []
        current_batch: List[int] = []

        # Round-robin: one sample per subject per pass
        while len(exhausted) < self.n_subjects:
            made_progress = False
            for pid in self.subjects:
                if pid in exhausted:
                    continue
                ptr = pointers[pid]
                if ptr >= len(per_subject[pid]):
                    exhausted.add(pid)
                    continue

                current_batch.append(per_subject[pid][ptr])
                pointers[pid] += 1
                made_progress = True

                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

            if not made_progress:
                break
        # Incomplete final batch is dropped (drop_last behaviour)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield from batch

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        total = sum(len(v) for v in self.subject_indices.values())
        return (total // self.batch_size) * self.batch_size


# Convenience aliases
EEGAlignmentDataloader = SimplifiedEEGDataloader