#!/usr/bin/env python3
"""
Simplified Controller for EEG-EEG vs EEG-Text Alignment Experiments
Version: 2.0
Focus: Compare within-subject vs cross-subject alignment
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

# Import simplified modules
from dataloader import SimplifiedEEGDataloader, simple_collate_fn, compute_global_eeg_dimensions
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
                                  subject_mode='within-subject', multi_positive_eval=False):
    """
    Create simplified dataloaders for EEG alignment experiments

    Args:
        document_type: 'text' for EEG-Text alignment, 'eeg' for EEG-EEG alignment
        subject_mode: 'within-subject' or 'cross-subject'
        multi_positive_eval: If True, treat all recordings of same document as relevant during evaluation
    """
    print(f"Loading data from: {data_path}")
    print(f"Document type: {document_type.upper()}")
    print(f"Subject mode: {subject_mode}")
    print(f"Split strategy: {'holdout subjects' if holdout_subjects else 'random sample'}")
    print(f"Multi-positive evaluation: {'ENABLED' if multi_positive_eval else 'DISABLED'}")

    # Compute global EEG dimensions if not provided
    if global_eeg_dims is None:
        global_eeg_dims = compute_global_eeg_dimensions(data_path, max_eeg_len, dataset_type)
        print(f"Computed global EEG dimensions: {global_eeg_dims}")

    # Create datasets
    train_dataset = SimplifiedEEGDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='train', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims,
        dataset_type=dataset_type, holdout_subjects=holdout_subjects,
        document_type=document_type, subject_mode=subject_mode
    )

    val_dataset = SimplifiedEEGDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='val', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims,
        dataset_type=dataset_type, holdout_subjects=holdout_subjects,
        document_type=document_type, subject_mode=subject_mode
    )

    test_dataset = SimplifiedEEGDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='test', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims,
        dataset_type=dataset_type, holdout_subjects=holdout_subjects,
        document_type=document_type, subject_mode=subject_mode
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
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

        # Detect format
        detected_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
        print(f"Dataset format: {detected_format}")

        # Detect dataset name
        dataset_name = detect_dataset_name(data_path)
        print(f"Dataset name: {dataset_name}")

        # Load dataset
        dataset = np.load(data_path, allow_pickle=True).item()
        ict_pairs = dataset.get('ict_pairs', [])
        metadata = dataset.get('metadata', {})

        print(f"Total ICT pairs: {len(ict_pairs)}")
        print(f"Dataset metadata keys: {list(metadata.keys())}")

        if len(ict_pairs) > 0:
            sample_pair = ict_pairs[0]

            # Check if we have both query_eeg and doc_eeg
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

        # Basic statistics
        participants = set()
        for pair in ict_pairs[:100]:  # Sample for speed
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

    # Convert non-serializable values
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
                        help='Enable multi-positive evaluation: treat all recordings of same document as relevant (recommended for cross-subject)')

    # Model arguments
    parser.add_argument('--colbert_model_name', default='colbert-ir/colbertv2.0',
                        help='ColBERT model name (for text encoding)')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--eeg_arch', default='simple', choices=['simple', 'complex', 'transformer'],
                        help='EEG encoder architecture')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_text_len', type=int, default=256, help='Max text sequence length')
    parser.add_argument('--max_eeg_len', type=int, default=50, help='Max EEG sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # Experiment arguments
    parser.add_argument('--output_dir', default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--holdout_subjects', action='store_true',
                        help='Use holdout subjects for validation instead of random split')

    args = parser.parse_args()

    # Set random seeds
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Detect dataset name
    dataset_name = detect_dataset_name(args.data_path)
    print(f"\n=== DATASET INFO ===")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {args.data_path}")

    # Inspect dataset
    if args.inspect_only:
        inspect_dataset(args.data_path, args.dataset_type)
        return

    inspect_dataset(args.data_path, args.dataset_type)

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else create_output_directory("simplified_eeg_alignment")
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
        multi_positive_eval=args.multi_positive_eval
    )

    train_subjects = len(train_dataloader.dataset.unique_subjects)
    val_subjects = len(val_dataloader.dataset.unique_subjects)
    test_subjects = len(test_dataloader.dataset.unique_subjects)

    # Create experiment configuration
    config = {
        'version': 'v1',
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
        'multi_positive_eval': args.multi_positive_eval,  # NEW

        # Model config
        'colbert_model_name': args.colbert_model_name,
        'hidden_dim': args.hidden_dim,
        'pooling_strategy': args.pooling_strategy,
        'eeg_arch': args.eeg_arch,

        # Training config
        'batch_size': args.batch_size,
        'max_text_len': args.max_text_len,
        'max_eeg_len': args.max_eeg_len,
        'train_ratio': args.train_ratio,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'patience': args.patience,

        # Data config
        'holdout_subjects': args.holdout_subjects,
        'seed': args.seed,
        'device': str(device),
        'train_samples': len(train_dataloader.dataset),
        'val_samples': len(val_dataloader.dataset),
        'test_samples': len(test_dataloader.dataset),
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'global_eeg_dims': global_eeg_dims,

        # Simplified experiment
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
    print(f"Multi-Positive Eval: {'✅ ENABLED' if args.multi_positive_eval else '❌ DISABLED'}")  # NEW
    if args.multi_positive_eval and args.subject_mode == 'cross-subject':
        print(f"  → All recordings of same document will be treated as relevant")
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
        device=device
    )

    # Update tokenizer vocab size if needed
    if tokenizer and args.document_type == 'text':
        model.set_tokenizer_vocab_size(len(tokenizer))

    # Count parameters
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n=== TRAINING START ===")
    print(f"Task: EEG queries → {args.document_type.upper()} documents")
    print(f"Method: {args.subject_mode}")
    if args.multi_positive_eval:
        print(f"Evaluation: Multi-positive labels enabled ✅")

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
        multi_positive_eval=args.multi_positive_eval  # NEW - Pass to training function
    )

    # Save trained model
    model_save_path = output_dir / f"simplified_model_{args.document_type}_{args.subject_mode}_{args.pooling_strategy}_{args.eeg_arch}.pt"
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
    print(f"Multi-Positive Eval: {'ENABLED ✅' if args.multi_positive_eval else 'DISABLED ❌'}")
    print(f"Results saved in: {output_dir}")




if __name__ == "__main__":
    main()