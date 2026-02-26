#!/usr/bin/env python3
"""
Simplified Controller for EEG-EEG vs EEG-Text Alignment Experiments
Version: 2.3
Focus: Compare within-subject vs cross-subject alignment
New (v2.1): text_loss_mode argument for EEG-Text loss formulation control
New (v2.2): dynamic_resample and weighted_multi_positive flags for cross-subject EEG-EEG
New (v2.3): stratified_sampling flag — uses SubjectStratifiedSampler for the train
            DataLoader so every batch contains samples from many distinct subjects.
            per_channel_norm is now always active (v2.2 dataloader default).
"""

import torch
import numpy as np
import random
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from datetime import datetime

from dataloader import (SimplifiedEEGDataloader, simple_collate_fn,
                        compute_global_eeg_dimensions, SubjectStratifiedSampler)
from models import create_alignment_model
from training import train_simplified_model, finish_wandb


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set random seed to {seed}")


def create_output_directory(base_name="eeg_alignment"):
    """Create timestamped output directory for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_name}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def detect_dataset_name(data_path):
    """Detect dataset name from file path"""
    data_path_str = str(data_path).lower()

    if 'nieuwland' in data_path_str:
        return 'Nieuwland'
    elif 'zuco' in data_path_str:
        return 'ZuCo'
    elif 'k408' in data_path_str or 'k-408' in data_path_str:
        return 'K408'
    else:
        return 'Unknown'


def create_simplified_dataloaders(data_path, tokenizer, batch_size=8, max_text_len=256,
                                  max_eeg_len=50, train_ratio=0.8, debug=False,
                                  dataset_type='auto', holdout_subjects=False,
                                  document_type='text', global_eeg_dims=None,
                                  subject_mode='within-subject', multi_positive_eval=False,
                                  dynamic_resample=False, stratified_sampling=False,
                                  query_subject=None, doc_subject=None):
    """
    Create simplified dataloaders for EEG alignment experiments

    Args:
        document_type:        'text' for EEG-Text alignment, 'eeg' for EEG-EEG alignment
        subject_mode:         'within-subject' or 'cross-subject'
        multi_positive_eval:  If True, treat all recordings of same document as relevant during evaluation
        dynamic_resample:     Re-sample the doc subject per __getitem__ call (cross-subject EEG-EEG only).
                              Applied to the train split only — val/test use static sampling for
                              reproducible evaluation.
        stratified_sampling:  If True (and document_type='eeg', subject_mode='cross-subject'), use
                              SubjectStratifiedSampler for the train DataLoader so that every batch
                              contains samples drawn round-robin from many distinct subjects.
                              Has no effect for EEG-Text or within-subject experiments.
    """
    print(f"Loading data from: {data_path}")
    print(f"Document type: {document_type.upper()}")
    print(f"Subject mode: {subject_mode}")
    print(f"Split strategy: {'holdout subjects' if holdout_subjects else 'random sample'}")
    print(f"Multi-positive evaluation: {'ENABLED' if multi_positive_eval else 'DISABLED'}")
    print(f"Dynamic doc-subject resampling (train): {'ENABLED ✅' if dynamic_resample else 'DISABLED ❌'}")

    _stratified_active = (
        stratified_sampling
        and document_type == 'eeg'
        and subject_mode == 'cross-subject'
    )
    print(f"Subject-stratified batching (train):    {'ENABLED ✅' if _stratified_active else 'DISABLED ❌'}")

    if global_eeg_dims is None:
        global_eeg_dims = compute_global_eeg_dimensions(data_path, max_eeg_len, dataset_type)
        print(f"Computed global EEG dimensions: {global_eeg_dims}")

    # dynamic_resample is only applied to the training split.
    # Val and test always use static sampling for reproducible evaluation.
    train_dataset = SimplifiedEEGDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='train', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims,
        dataset_type=dataset_type, holdout_subjects=holdout_subjects,
        document_type=document_type, subject_mode=subject_mode,
        dynamic_resample=dynamic_resample,
        query_subject=query_subject, doc_subject=doc_subject
    )

    val_dataset = SimplifiedEEGDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='val', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims,
        dataset_type=dataset_type, holdout_subjects=holdout_subjects,
        document_type=document_type, subject_mode=subject_mode,
        dynamic_resample=False,
        query_subject=query_subject, doc_subject=doc_subject
    )

    test_dataset = SimplifiedEEGDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='test', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims,
        dataset_type=dataset_type, holdout_subjects=holdout_subjects,
        document_type=document_type, subject_mode=subject_mode,
        dynamic_resample=False,
        query_subject=query_subject, doc_subject=doc_subject
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=not _stratified_active,   # mutually exclusive with sampler
                                  sampler=(SubjectStratifiedSampler(
                                               train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               seed=42)
                                           if _stratified_active else None),
                                  batch_sampler=None,
                                  collate_fn=simple_collate_fn, num_workers=0,
                                  pin_memory=torch.cuda.is_available())

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=simple_collate_fn, num_workers=0,
                                pin_memory=torch.cuda.is_available())

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=simple_collate_fn, num_workers=0,
                                 pin_memory=torch.cuda.is_available())

    print(f"Train: {len(train_dataset)} samples from {len(train_dataset.unique_subjects)} subjects")
    print(f"Val: {len(val_dataset)} samples from {len(val_dataset.unique_subjects)} subjects")
    print(f"Test: {len(test_dataset)} samples from {len(test_dataset.unique_subjects)} subjects")

    return train_dataloader, val_dataloader, test_dataloader, global_eeg_dims


def inspect_dataset(data_path, dataset_type='auto'):
    """Inspect dataset structure and content"""
    print(f"\n=== DATASET INSPECTION ===")
    print(f"Inspecting: {data_path}")

    try:
        from dataloader import detect_dataset_format

        detected_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
        print(f"Dataset format: {detected_format}")

        dataset_name = detect_dataset_name(data_path)
        print(f"Dataset name: {dataset_name}")

        dataset = np.load(data_path, allow_pickle=True).item()
        ict_pairs = dataset.get('ict_pairs', [])
        metadata = dataset.get('metadata', {})

        print(f"Total ICT pairs: {len(ict_pairs)}")
        print(f"Dataset metadata keys: {list(metadata.keys())}")

        if len(ict_pairs) > 0:
            sample_pair = ict_pairs[0]

            if hasattr(sample_pair, 'query_eeg') and hasattr(sample_pair, 'doc_eeg'):
                print("✅ Dataset supports EEG-EEG alignment (has both query_eeg and doc_eeg)")
                print(f"Sample query_eeg shape: {np.array(sample_pair.query_eeg).shape}")
                print(f"Sample doc_eeg shape: {np.array(sample_pair.doc_eeg).shape}")
            elif 'query_eeg' in sample_pair and 'doc_eeg' in sample_pair:
                print("✅ Dataset supports EEG-EEG alignment (has both query_eeg and doc_eeg)")
                print(f"Sample query_eeg shape: {np.array(sample_pair['query_eeg']).shape}")
                print(f"Sample doc_eeg shape: {np.array(sample_pair['doc_eeg']).shape}")
            else:
                print("⚠️ Dataset may not support EEG-EEG alignment (missing doc_eeg)")

        participants = set()
        for pair in ict_pairs[:100]:
            if hasattr(pair, 'participant_id'):
                participants.add(pair.participant_id)
            elif 'participant_id' in pair:
                participants.add(pair['participant_id'])

        print(f"Unique participants (sample): {len(participants)}")

    except Exception as e:
        print(f"Error inspecting dataset: {e}")


def save_experiment_config(config, output_dir):
    """Save experiment configuration to JSON file"""
    config_path = output_dir / "experiment_config.json"

    serializable_config = {
        k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
        for k, v in config.items()
    }

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

    print(f"Saved experiment config to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Simplified EEG Alignment: Compare within-subject vs cross-subject alignment')

    # Data arguments
    parser.add_argument('--data_path', required=True, help='Path to ICT pairs .npy file')
    parser.add_argument('--dataset_type', default='auto', choices=['auto', 'original', 'nieuwland'],
                        help='Dataset type: auto-detect, original format, or nieuwland format')
    parser.add_argument('--inspect_only', action='store_true', help='Only inspect dataset, don\'t train')

    # Core experiment arguments
    parser.add_argument('--document_type', default='text', choices=['text', 'eeg'],
                        help='Document representation: text (EEG-Text) or eeg (EEG-EEG)')
    parser.add_argument('--subject_mode', default='within-subject',
                        choices=['within-subject', 'cross-subject'],
                        help='Subject matching mode: within-subject (same person) or cross-subject (different people)')
    parser.add_argument('--pooling_strategy', default='multi', choices=['multi', 'cls', 'max', 'mean'],
                        help='Pooling strategy for representations')

    # Evaluation arguments
    parser.add_argument('--multi_positive_eval', action='store_true',
                        help='Enable multi-positive evaluation: treat all recordings of same document as relevant '
                             '(recommended for cross-subject EEG-EEG; no effect on EEG-Text eval)')
    parser.add_argument('--multi_positive_train', action='store_true',
                        help='Enable multi-positive training for EEG-EEG: treat all in-batch recordings of same '
                             'sentence as positives (recommended for cross-subject EEG-EEG)')

    # ── v2.2 NEW FLAGS ──────────────────────────────────────────────────────
    parser.add_argument('--dynamic_resample', action='store_true',
                        help=(
                            'Re-sample the document subject on every training step instead of fixing it '
                            'at preprocessing time.  Only active for --document_type eeg and '
                            '--subject_mode cross-subject.  Exposes the model to the full positive '
                            'distribution across training, preventing memorisation of a single '
                            'query→doc subject pairing per sentence. '
                            'Val and test splits always use static sampling for reproducibility.'
                        ))

    parser.add_argument('--weighted_multi_positive', action='store_true',
                        help=(
                            'Weight each co-positive in the multi-positive contrastive loss by its '
                            'cosine similarity to the query in raw (pre-encoder) EEG space. '
                            'Reliable cross-subject pairs (neurologically more similar) contribute '
                            'more gradient; noisy pairs are down-weighted automatically. '
                            'Only active when --multi_positive_train is also set and '
                            '--document_type eeg. Has no effect otherwise.'
                        ))

    parser.add_argument('--stratified_sampling', action='store_true',
                        help=(
                            'Use SubjectStratifiedSampler for the training DataLoader. '
                            'Batches are constructed round-robin across subjects so that '
                            'every batch contains samples from many distinct subjects. '
                            'This prevents the contrastive loss from discriminating on '
                            'subject identity rather than sentence content. '
                            'Only active when --document_type eeg and '
                            '--subject_mode cross-subject. Has no effect otherwise.'
                        ))
    # ────────────────────────────────────────────────────────────────────────

    # Text loss mode argument
    parser.add_argument('--text_loss_mode', default='standard',
                        choices=['standard', 'masked', 'multi_positive'],
                        help=(
                            'Loss formulation for EEG-Text condition (ignored for EEG-EEG). '
                            '"standard": CE where same-sentence batch duplicates are treated as hard negatives. '
                            '"masked": CE where same-sentence off-diagonal positions are excluded from the '
                            'softmax denominator (neither positive nor negative). '
                            '"multi_positive": same multi-positive CE used for EEG-EEG, same-sentence batch '
                            'entries treated as co-positives. '
                            'Applies consistently to training, validation, and test evaluation.'
                        ))

    # Document encoder arguments
    parser.add_argument('--doc_encoder_type', default='bert', choices=['bert', 'eeg'],
                        help=(
                            'Architecture for the document-side encoder. '
                            '"bert": existing TextEncoder (BERT/ColBERT + LoRA) for text documents. '
                            '"eeg": EEGEncoder architecture for text documents — token IDs are converted '
                            'to vectors via frozen BERT token embeddings then passed through EEGEncoder. '
                            'Only relevant when --document_type text. '
                            'For --document_type eeg this flag has no effect.'
                        ))
    parser.add_argument('--freeze_doc_encoder', action='store_true',
                        help=(
                            'Freeze the document-side encoder weights after initialisation. '
                            'For EEG-Text/bert: freezes TextEncoder (no LoRA gradients). '
                            'For EEG-Text/eeg: freezes text_doc_eeg_encoder. '
                            'For EEG-EEG: creates a separate frozen doc EEG encoder '
                            '(query encoder remains trainable).'
                        ))

    # Model arguments
    parser.add_argument('--colbert_model_name', default='colbert-ir/colbertv2.0',
                        help='ColBERT model name (for text encoding)')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--eeg_arch', default='simple', choices=['simple', 'complex', 'transformer'],
                        help='EEG encoder architecture')
    parser.add_argument('--eeg_num_layers', type=int, default=2,
                        help=(
                            'Number of TransformerEncoderLayers in the EEG encoder '
                            '(only applies when --eeg_arch transformer). '
                            'Default 2. Set to 1 to halve transformer capacity — '
                            'recommended for EEG-EEG conditions to reduce overfitting.'
                        ))
    parser.add_argument('--eeg_nhead', type=int, default=8,
                        help=(
                            'Number of attention heads in each TransformerEncoderLayer '
                            '(only applies when --eeg_arch transformer). '
                            'Must divide --hidden_dim evenly. Default 8. '
                            'Use 4 if --hidden_dim is reduced to 256.'
                        ))
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for L2 regularization')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_text_len', type=int, default=256, help='Max text sequence length')
    parser.add_argument('--max_eeg_len', type=int, default=50, help='Max EEG sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    # Experiment arguments
    parser.add_argument('--output_dir', default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--holdout_subjects', action='store_true',
                        help='Use holdout subjects for validation instead of random split')

    # ── Fixed two-subject experiment args ────────────────────────────────────
    parser.add_argument('--query_subject', default=None,
                        help=(
                            'Restrict queries to a single subject ID. '
                            'When set, only pairs where participant_id matches this value '
                            'are used. Enables controlled two-subject experiments. '
                            'If not set, all subjects are used (existing behaviour).'
                        ))
    parser.add_argument('--doc_subject', default=None,
                        help=(
                            'Fix the document subject for EEG-EEG cross-subject experiments. '
                            'Requires --query_subject. Every EEG-EEG pair will use this '
                            'subject as the document. Has no effect for EEG-Text. '
                            'If --query_subject is set but --doc_subject is not, '
                            'the doc subject is chosen randomly per pair (existing behaviour).'
                        ))
    # ────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    # ── Argument validation ──────────────────────────────────────────────────

    if args.document_type == 'eeg' and args.text_loss_mode != 'standard':
        print(f"⚠️  Note: --text_loss_mode={args.text_loss_mode} has no effect when --document_type=eeg. "
              f"EEG-EEG loss is controlled by --multi_positive_train.")

    if args.dynamic_resample and args.document_type != 'eeg':
        print(f"⚠️  Note: --dynamic_resample has no effect when --document_type=text.")

    if args.dynamic_resample and args.subject_mode != 'cross-subject':
        print(f"⚠️  Note: --dynamic_resample has no effect when --subject_mode=within-subject.")

    if args.weighted_multi_positive and not args.multi_positive_train:
        print(f"⚠️  Note: --weighted_multi_positive has no effect without --multi_positive_train.")

    if args.weighted_multi_positive and args.document_type != 'eeg':
        print(f"⚠️  Note: --weighted_multi_positive only applies to EEG-EEG training. "
              f"It has no effect when --document_type=text.")

    if args.stratified_sampling and args.document_type != 'eeg':
        print(f"⚠️  Note: --stratified_sampling has no effect when --document_type=text.")

    if args.stratified_sampling and args.subject_mode != 'cross-subject':
        print(f"⚠️  Note: --stratified_sampling has no effect when --subject_mode=within-subject.")

    # ── Auto-select query/doc subjects if not provided ───────────────────────
    # When --query_subject is not given, behaviour is unchanged (all subjects).
    # When given without --doc_subject, doc subject is random per pair.
    # For two-subject experiments the script should supply both, but if only
    # --query_subject is provided we warn.
    if args.query_subject is None and args.doc_subject is not None:
        raise ValueError("--doc_subject requires --query_subject to also be set.")

    if args.query_subject is not None:
        print(f"\n── Two-subject experiment mode ─────────────────────────────────")
        print(f"   Query subject : {args.query_subject}")
        if args.doc_subject:
            print(f"   Doc subject   : {args.doc_subject}")
        else:
            print(f"   Doc subject   : random per pair (not fixed)")
        print(f"────────────────────────────────────────────────────────────────")
    # ────────────────────────────────────────────────────────────────────────

    if args.eeg_arch == 'transformer' and args.hidden_dim % args.eeg_nhead != 0:
        raise ValueError(
            f"--hidden_dim ({args.hidden_dim}) must be divisible by --eeg_nhead ({args.eeg_nhead}). "
            f"Try --eeg_nhead 4 when --hidden_dim 256.")

    # ────────────────────────────────────────────────────────────────────────

    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name = detect_dataset_name(args.data_path)
    print(f"\n=== DATASET INFO ===")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {args.data_path}")

    if args.inspect_only:
        inspect_dataset(args.data_path, args.dataset_type)
        return

    inspect_dataset(args.data_path, args.dataset_type)

    output_dir = Path(args.output_dir) if args.output_dir else create_output_directory(
        "simplified_eeg_alignment")
    output_dir.mkdir(exist_ok=True)

    # Load tokenizer (only needed if using text documents)
    tokenizer = None
    if args.document_type == 'text':
        print(f"\nLoading tokenizer: {args.colbert_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.colbert_model_name)
        except:
            print(f"ColBERT tokenizer not found, falling back to bert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_dataloader, val_dataloader, test_dataloader, global_eeg_dims = create_simplified_dataloaders(
        data_path=args.data_path, tokenizer=tokenizer, batch_size=args.batch_size,
        max_text_len=args.max_text_len, max_eeg_len=args.max_eeg_len,
        train_ratio=args.train_ratio, debug=args.debug,
        dataset_type=args.dataset_type, holdout_subjects=args.holdout_subjects,
        document_type=args.document_type, subject_mode=args.subject_mode,
        multi_positive_eval=args.multi_positive_eval,
        dynamic_resample=args.dynamic_resample,
        stratified_sampling=args.stratified_sampling,
        query_subject=args.query_subject,
        doc_subject=args.doc_subject
    )

    train_subjects = len(train_dataloader.dataset.unique_subjects)
    val_subjects = len(val_dataloader.dataset.unique_subjects)
    test_subjects = len(test_dataloader.dataset.unique_subjects)

    # Create experiment configuration
    config = {
        'version': 'v2.2',
        'experiment_type': f'eeg_{args.document_type}_alignment',
        'timestamp': datetime.now().isoformat(),

        # Dataset info
        'dataset_name': dataset_name,
        'dataset_path': str(args.data_path),
        'dataset_type': args.dataset_type,

        # Experiment config
        'document_type': args.document_type,
        'subject_mode': args.subject_mode,
        'alignment_task': f'EEG-{args.document_type.upper()}',
        'alignment_method': args.subject_mode,
        'multi_positive_eval': args.multi_positive_eval,
        'multi_positive_train': args.multi_positive_train,
        'dynamic_resample': args.dynamic_resample,
        'weighted_multi_positive': args.weighted_multi_positive,
        'stratified_sampling': args.stratified_sampling,
        'text_loss_mode': args.text_loss_mode,
        'doc_encoder_type': args.doc_encoder_type,
        'freeze_doc_encoder': args.freeze_doc_encoder,

        # Model config
        'colbert_model_name': args.colbert_model_name,
        'hidden_dim': args.hidden_dim,
        'pooling_strategy': args.pooling_strategy,
        'eeg_arch': args.eeg_arch,
        'eeg_num_layers': args.eeg_num_layers,
        'eeg_nhead': args.eeg_nhead,

        # Training config
        'batch_size': args.batch_size,
        'max_text_len': args.max_text_len,
        'max_eeg_len': args.max_eeg_len,
        'train_ratio': args.train_ratio,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,

        # Data config
        'holdout_subjects': args.holdout_subjects,
        'query_subject': args.query_subject,
        'doc_subject': args.doc_subject,
        'seed': args.seed,
        'device': str(device),
        'train_samples': len(train_dataloader.dataset),
        'val_samples': len(val_dataloader.dataset),
        'test_samples': len(test_dataloader.dataset),
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'global_eeg_dims': global_eeg_dims,

        # Experiment flags
        'simplified': True,
        'no_masking': True,
        'dual_encoder_only': True
    }

    print(f"\n=== EXPERIMENT SETUP ===")
    print(f"Dataset: {dataset_name}")
    print(f"Alignment Task: EEG → {args.document_type.upper()}")
    print(f"Subject Mode: {args.subject_mode}")
    print(f"Pooling Strategy: {args.pooling_strategy}")
    print(f"EEG Architecture: {args.eeg_arch}")
    if args.eeg_arch == 'transformer':
        print(f"Transformer Layers: {args.eeg_num_layers}  |  Attention Heads: {args.eeg_nhead}")
    print(f"Multi-Positive Eval:       {'✅ ENABLED' if args.multi_positive_eval else '❌ DISABLED'}")
    print(f"Multi-Positive Train:      {'✅ ENABLED' if args.multi_positive_train else '❌ DISABLED'}")
    print(f"Dynamic Doc-Subj Resample: {'✅ ENABLED (train only)' if args.dynamic_resample else '❌ DISABLED'}")
    print(f"Weighted Multi-Positive:   {'✅ ENABLED' if args.weighted_multi_positive else '❌ DISABLED'}")
    print(f"Stratified Batching:       {'✅ ENABLED (train only)' if args.stratified_sampling else '❌ DISABLED'}")

    if args.document_type == 'text':
        mode_labels = {
            'standard': 'Standard CE — same-sentence duplicates treated as hard negatives',
            'masked': 'Masked CE — same-sentence duplicates excluded from softmax denominator',
            'multi_positive': 'Multi-Positive CE — same-sentence duplicates treated as co-positives'
        }
        print(f"Text Loss Mode: {args.text_loss_mode.upper()} → {mode_labels.get(args.text_loss_mode)}")
        print(f"Doc Encoder Type: {args.doc_encoder_type.upper()}"
              + (" (frozen)" if args.freeze_doc_encoder else " (trainable)"))
    if args.document_type == 'eeg' and args.freeze_doc_encoder:
        print(f"EEG Doc Encoder: FROZEN (separate from query encoder)")

    if args.multi_positive_eval and args.subject_mode == 'cross-subject':
        print(f"  → All recordings of same document will be treated as relevant in evaluation")
    if args.multi_positive_train and args.subject_mode == 'cross-subject':
        print(f"  → All in-batch recordings of same sentence will be treated as positives in training")
    if args.dynamic_resample and args.subject_mode == 'cross-subject' and args.document_type == 'eeg':
        print(f"  → Doc subject is re-sampled per training step (full positive distribution exposed)")
    if args.weighted_multi_positive and args.multi_positive_train and args.document_type == 'eeg':
        print(f"  → Co-positive weights are computed from raw EEG cosine similarity")

    print(f"Training samples: {config['train_samples']} ({train_subjects} subjects)")
    print(f"Validation samples: {config['val_samples']} ({val_subjects} subjects)")
    print(f"Test samples: {config['test_samples']} ({test_subjects} subjects)")
    print(f"Split method: {'Holdout subjects' if args.holdout_subjects else 'Random samples'}")

    # Create model
    print(f"\n=== MODEL CREATION ===")
    model = create_alignment_model(
        document_type=args.document_type,
        colbert_model_name=args.colbert_model_name if args.document_type == 'text' else None,
        hidden_dim=args.hidden_dim,
        eeg_arch=args.eeg_arch,
        pooling_strategy=args.pooling_strategy,
        global_eeg_dims=global_eeg_dims,
        device=device,
        dropout=args.dropout,
        doc_encoder_type=args.doc_encoder_type,
        freeze_doc_encoder=args.freeze_doc_encoder,
        eeg_num_layers=args.eeg_num_layers,
        eeg_nhead=args.eeg_nhead
    )

    if tokenizer and args.document_type == 'text':
        model.set_tokenizer_vocab_size(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created! Total: {total_params:,}, Trainable: {trainable_params:,}")

    config.update({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'tokenizer_vocab_size': len(tokenizer) if tokenizer else None
    })

    save_experiment_config(config, output_dir)

    # Create optimizer and train
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"\n=== TRAINING START ===")
    print(f"Task: EEG queries → {args.document_type.upper()} documents")
    print(f"Method: {args.subject_mode}")
    if args.multi_positive_eval:
        print(f"Evaluation: Multi-positive labels enabled ✅")
    if args.multi_positive_train:
        print(f"Training: Multi-positive labels enabled ✅")
    if args.dynamic_resample:
        print(f"Training: Dynamic doc-subject resampling enabled ✅")
    if args.weighted_multi_positive:
        print(f"Training: Weighted multi-positive (raw EEG cosine weights) enabled ✅")
    if args.document_type == 'text' and args.text_loss_mode != 'standard':
        print(f"Text Loss Mode: {args.text_loss_mode.upper()} ✅")

    trained_model = train_simplified_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        debug=args.debug,
        config=config,
        multi_positive_eval=args.multi_positive_eval,
        multi_positive_train=args.multi_positive_train,
        text_loss_mode=args.text_loss_mode,
        weighted_multi_positive=args.weighted_multi_positive
    )

    # Save trained model
    model_save_path = output_dir / (
        f"simplified_model_{args.document_type}_{args.subject_mode}_"
        f"{args.pooling_strategy}_{args.eeg_arch}"
        f"{'_' + args.text_loss_mode if args.document_type == 'text' and args.text_loss_mode != 'standard' else ''}"
        f"_docenc{args.doc_encoder_type}"
        f"{'_frozen' if args.freeze_doc_encoder else ''}"
        f"{'_dynResample' if args.dynamic_resample else ''}"
        f"{'_wtdMP' if args.weighted_multi_positive else ''}"
        f"{'_stratBatch' if args.stratified_sampling else ''}.pt"
    )
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'tokenizer_vocab_size': len(tokenizer) if tokenizer else None
    }, model_save_path)
    print(f"Saved trained model to: {model_save_path}")

    finish_wandb()
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Dataset: {dataset_name}")
    print(f"Alignment Task: EEG → {args.document_type.upper()}")
    print(f"Subject Mode: {args.subject_mode}")
    print(f"Multi-Positive Eval:       {'ENABLED ✅' if args.multi_positive_eval else 'DISABLED ❌'}")
    print(f"Multi-Positive Train:      {'ENABLED ✅' if args.multi_positive_train else 'DISABLED ❌'}")
    print(f"Dynamic Doc-Subj Resample: {'ENABLED ✅' if args.dynamic_resample else 'DISABLED ❌'}")
    print(f"Weighted Multi-Positive:   {'ENABLED ✅' if args.weighted_multi_positive else 'DISABLED ❌'}")
    print(f"Stratified Batching:       {'ENABLED ✅' if args.stratified_sampling else 'DISABLED ❌'}")
    if args.document_type == 'text':
        print(f"Text Loss Mode: {args.text_loss_mode.upper()}")
        print(f"Doc Encoder Type: {args.doc_encoder_type.upper()}"
              + (" (frozen)" if args.freeze_doc_encoder else " (trainable)"))
    if args.document_type == 'eeg' and args.freeze_doc_encoder:
        print(f"EEG Doc Encoder: FROZEN")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()