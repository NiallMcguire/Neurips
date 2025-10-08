#!/usr/bin/env python3
"""
Simplified Dataloader for EEG-EEG vs EEG-Text Alignment
Version: 2.0
Focus: Support within-subject and cross-subject pairing
"""

import numpy as np
import torch
from torch.utils.data import Dataset
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
    Version: 2.0

    Key features:
    - Always uses query_eeg as query
    - Toggles between doc_eeg (EEG-EEG) or doc_text (EEG-Text) as document
    - Supports within-subject and cross-subject pairing
    - No masking complexity
    - Simple train/val/test splits
    """

    def __init__(self, data_path: str, tokenizer=None, max_text_len: int = 256,
                 max_eeg_len: int = 50, train_ratio: float = 0.8,
                 split: str = 'train', normalize_eeg: bool = True,
                 debug: bool = False, global_eeg_dims: tuple = None,
                 dataset_type: str = 'auto', holdout_subjects: bool = False,
                 document_type: str = 'text', subject_mode: str = 'within-subject'):
        """
        Args:
            document_type: 'text' for EEG-Text alignment, 'eeg' for EEG-EEG alignment
            subject_mode: 'within-subject' (same person) or 'cross-subject' (different people)
        """

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_eeg_len = max_eeg_len
        self.normalize_eeg = normalize_eeg
        self.debug = debug
        self.holdout_subjects = holdout_subjects
        self.document_type = document_type
        self.subject_mode = subject_mode

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

        # 🔥 NEW: Compute per-subject statistics for normalization
        print(f"Computing per-subject EEG statistics for normalization...")
        self.subject_stats = self._compute_subject_statistics()
        print(f"  Computed stats for {len(self.subject_stats)} subjects")

        # For cross-subject mode, build sentence-to-subject mapping
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
        Compute per-subject mean and std for EEG normalization
        This is CRITICAL for cross-subject generalization
        """
        subject_eegs = {}  # participant_id -> list of EEG arrays

        # Collect all EEG data per subject
        for pair in self.pairs:
            participant_id = pair.get('participant_id', 'unknown')

            # Collect query EEG
            query_eeg = pair.get('query_eeg', None)
            if query_eeg is not None:
                if participant_id not in subject_eegs:
                    subject_eegs[participant_id] = []
                subject_eegs[participant_id].append(np.array(query_eeg, dtype=np.float32))

            # Collect doc EEG if available (for EEG-EEG alignment)
            if self.document_type == 'eeg':
                doc_eeg = pair.get('doc_eeg', None)
                if doc_eeg is not None:
                    subject_eegs[participant_id].append(np.array(doc_eeg, dtype=np.float32))

        # Compute statistics per subject
        subject_stats = {}
        for participant_id, eeg_list in subject_eegs.items():
            if len(eeg_list) == 0:
                continue

            # Concatenate all EEG data for this subject
            try:
                # Flatten all dimensions except the last (channels)
                all_values = []
                for eeg in eeg_list:
                    eeg_array = np.array(eeg, dtype=np.float32)
                    if len(eeg_array.shape) >= 2:
                        # Flatten to get all channel values
                        all_values.append(eeg_array.reshape(-1))

                if all_values:
                    combined = np.concatenate(all_values)
                    # Remove zeros (padding)
                    non_zero = combined[combined != 0]

                    if len(non_zero) > 0:
                        subject_mean = np.mean(non_zero)
                        subject_std = np.std(non_zero)

                        # Avoid division by zero
                        if subject_std < 1e-6:
                            subject_std = 1.0

                        subject_stats[participant_id] = {
                            'mean': float(subject_mean),
                            'std': float(subject_std)
                        }

                        if self.debug and len(subject_stats) <= 3:
                            print(f"    Subject {participant_id}: mean={subject_mean:.4f}, std={subject_std:.4f}")
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
        # Extract all unique participants
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

        # Split participants: 80% train, 10% val, 10% test
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

        # Collect all indices for selected participants
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
            # Standard 2-way split during training
            split_idx = int(len(self.ict_pairs) * train_ratio)
            if split == 'train':
                selected_indices = shuffled_indices[:split_idx]
            else:  # val
                selected_indices = shuffled_indices[split_idx:]
        elif split == 'test':
            # 3-way split: use validation portion and split it further
            val_start_idx = int(len(self.ict_pairs) * train_ratio)
            val_indices = shuffled_indices[val_start_idx:]

            # Split remaining data: 50% val, 50% test
            val_split_point = len(val_indices) // 2
            selected_indices = val_indices[val_split_point:]  # Second half for test
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

        # Count unique subjects in this split
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

        # Process query EEG
        query_eeg_processed = self._process_eeg(query_eeg)
        if query_eeg_processed is None:
            return None

        # Extract document based on document_type and subject_mode
        if self.document_type == 'eeg':
            # EEG-EEG alignment
            if self.subject_mode == 'within-subject':
                # Within-subject: use doc_eeg from same participant
                doc_eeg = pair.get('doc_eeg', None)
                if doc_eeg is None:
                    return None

                doc_eeg_processed = self._process_eeg(doc_eeg)
                if doc_eeg_processed is None:
                    return None

                doc_data = doc_eeg_processed
                doc_participant_id = pair.get('participant_id', 'unknown')

            else:  # cross-subject
                # Cross-subject: find doc_eeg from different participant, same sentence
                sentence_id = pair.get('sentence_id', 0)
                query_participant_id = pair.get('participant_id', 'unknown')

                # Find other participants who read the same sentence
                if sentence_id in self.sentence_to_subjects:
                    other_participants = [p for p in self.sentence_to_subjects[sentence_id].keys()
                                          if p != query_participant_id]

                    if other_participants:
                        # Randomly select a different participant
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
                        # No other participants for this sentence, skip
                        return None
                else:
                    # Sentence not in mapping, skip
                    return None

            doc_text = None

        else:  # document_type == 'text'
            # EEG-Text alignment: use doc_text (subject_mode doesn't apply)
            doc_text = pair.get('doc_text', '').strip()
            if not doc_text:
                return None

            doc_data = None
            doc_participant_id = pair.get('participant_id', 'unknown')

        # Extract query text for reference
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
                # 3D format: [num_words, time_samples, channels]
                num_words, time_samples, channels = eeg_array.shape
                if num_words > self.max_eeg_len:
                    eeg_array = eeg_array[:self.max_eeg_len]
                    num_words = self.max_eeg_len

                padded_eeg = np.zeros((self.global_max_words, self.global_max_time, self.global_max_channels),
                                      dtype=np.float32)
                padded_eeg[:num_words, :time_samples, :channels] = eeg_array
                return padded_eeg

            elif len(eeg_array.shape) == 2:
                # 2D format: [time_samples, channels]
                time_samples, channels = eeg_array.shape
                padded_eeg = np.zeros((self.global_max_words, self.global_max_time, self.global_max_channels),
                                      dtype=np.float32)
                padded_eeg[0, :time_samples, :channels] = eeg_array
                return padded_eeg

            else:
                # Try to reshape flattened data
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
        Normalize EEG tensor using SUBJECT-SPECIFIC statistics
        This is critical for cross-subject generalization
        """
        if not self.normalize_eeg:
            return eeg_tensor

        # Use subject-specific normalization if available
        if participant_id in self.subject_stats:
            stats = self.subject_stats[participant_id]
            subject_mean = stats['mean']
            subject_std = stats['std']

            # Apply subject-wise z-score normalization
            normalized = (eeg_tensor - subject_mean) / subject_std
            return normalized
        else:
            # Fallback to global normalization if subject stats not available
            if len(eeg_tensor.shape) == 3:
                # Create mask for non-zero values
                non_zero_mask = (eeg_tensor.abs().sum(dim=(1, 2)) > 0).unsqueeze(1).unsqueeze(2)

                if non_zero_mask.sum() > 0:
                    mean = torch.mean(eeg_tensor[non_zero_mask])
                    std = torch.std(eeg_tensor[non_zero_mask])
                    std = torch.where(std == 0, torch.tensor(1e-6), std)
                    return (eeg_tensor - mean) / std

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
        """Get sample for training with subject-wise normalization"""
        sample = self.processed_pairs[idx]

        # Process query EEG (always present) with subject-specific normalization
        query_eeg_tensor = torch.tensor(sample['query_eeg'], dtype=torch.float32)
        query_eeg_tensor = self._normalize_eeg(query_eeg_tensor, sample['query_participant_id'])

        # Process document based on type
        if self.document_type == 'eeg':
            # EEG-EEG alignment with subject-specific normalization
            doc_eeg_tensor = torch.tensor(sample['doc_eeg'], dtype=torch.float32)
            doc_eeg_tensor = self._normalize_eeg(doc_eeg_tensor, sample['doc_participant_id'])

            return {
                'query_eeg': query_eeg_tensor,
                'doc_eeg': doc_eeg_tensor,
                'doc_text_tokens': None,
                'metadata': {
                    'query_participant_id': sample['query_participant_id'],
                    'doc_participant_id': sample['doc_participant_id'],
                    'sentence_id': sample['sentence_id'],
                    'document_type': sample['document_type'],
                    'subject_mode': sample['subject_mode'],
                    'query_text': sample['query_text'],
                    'original_idx': sample['original_idx']
                }
            }

        else:  # document_type == 'text'
            # EEG-Text alignment
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
                    'original_idx': sample['original_idx']
                }
            }


def simple_collate_fn(batch):
    """Simple collate function for batching"""

    # Determine document type from first sample
    document_type = batch[0]['metadata']['document_type']

    # Stack query EEGs (always present)
    query_eegs = torch.stack([item['query_eeg'] for item in batch])

    if document_type == 'eeg':
        # EEG-EEG alignment
        doc_eegs = torch.stack([item['doc_eeg'] for item in batch])
        doc_text_tokens = None

    else:  # document_type == 'text'
        # EEG-Text alignment
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
            print(f"  Cross-subject pairs: {cross_pairs}/{batch_size}")
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


# Convenience aliases
EEGAlignmentDataloader = SimplifiedEEGDataloader