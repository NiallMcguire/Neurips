#!/usr/bin/env python3
"""
Memory-Efficient Multi-Vector Retrieval Controller for Brain Passage Retrieval
Uses dynamic masking with single dataloader for memory efficiency
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

# Import our custom modules
from mv_dataloader import DynamicMaskingDataloader, simple_collate_fn, compute_global_eeg_dimensions
from mv_models import create_model
from mv_training import train_model, finish_wandb


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set random seed to {seed}")


def create_output_directory(base_name="simple_experiment"):
    """Create timestamped output directory for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_name}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def create_dynamic_dataloaders(data_path, tokenizer, batch_size=8, max_text_len=256,
                               max_eeg_len=50, train_ratio=0.8, debug=False,
                               num_vectors=32, dataset_type='auto', training_masking_level=90,
                               global_eeg_dims=None):
    """Create training and validation dataloaders with DYNAMIC masking support"""
    print(f"Loading data from: {data_path}")

    # Compute global EEG dimensions if not provided
    if global_eeg_dims is None:
        global_eeg_dims = compute_global_eeg_dimensions(data_path, max_eeg_len, dataset_type)
        print(f"Computed global EEG dimensions: {global_eeg_dims[0]}x{global_eeg_dims[1]}x{global_eeg_dims[2]}")

    # Convert training masking level to probability
    training_masking_prob = training_masking_level / 100.0
    print(f"Training with {training_masking_level}% masking probability")

    # Create training dataset with DYNAMIC masking
    train_dataset = DynamicMaskingDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='train', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims, num_vectors=num_vectors,
        dataset_type=dataset_type, initial_masking_probability=training_masking_prob
    )

    # Create validation dataset with DYNAMIC masking (initially same as training)
    val_dataset = DynamicMaskingDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='val', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims, num_vectors=num_vectors,
        dataset_type=dataset_type, initial_masking_probability=training_masking_prob
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=simple_collate_fn, num_workers=0,
                                  pin_memory=torch.cuda.is_available())

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=simple_collate_fn, num_workers=0,
                                pin_memory=torch.cuda.is_available())

    print(f"Created training dataloader with {len(train_dataset)} samples (dynamic masking)")
    print(f"Created validation dataloader with {len(val_dataset)} samples (dynamic masking)")
    print(f"Memory efficient: Single validation dataloader with dynamic masking capability")

    return train_dataloader, val_dataloader, global_eeg_dims


def inspect_dataset(data_path, dataset_type='auto'):
    """Inspect dataset structure and content"""
    print(f"\n=== DATASET INSPECTION ===")
    print(f"Inspecting: {data_path}")

    try:
        from mv_dataloader import detect_dataset_format, convert_nieuwland_to_original_format

        # Detect format
        detected_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
        print(f"Dataset format: {detected_format}")

        # Load and convert if needed
        dataset = np.load(data_path, allow_pickle=True).item()
        ict_pairs = dataset.get('ict_pairs', [])
        metadata = dataset.get('metadata', {})

        if detected_format == 'nieuwland' and len(ict_pairs) > 0:
            ict_pairs = convert_nieuwland_to_original_format(ict_pairs)
            print("Converted Nieuwland format for inspection")

        print(f"Total ICT pairs: {len(ict_pairs)}")
        print(f"Dataset metadata keys: {list(metadata.keys())}")

        if 'creation_date' in metadata:
            print(f"Dataset created: {metadata['creation_date']}")

        # Check runtime masking support
        supports_runtime_masking = metadata.get('supports_runtime_masking', False)
        print(f"Supports runtime masking: {supports_runtime_masking}")

        # Basic statistics
        query_lengths = []
        doc_lengths = []
        participants = set()

        for pair in ict_pairs[:1000]:  # Sample for speed
            if pair.get('query_text'):
                query_lengths.append(len(pair['query_text'].split()))
            if pair.get('doc_text'):
                doc_lengths.append(len(pair['doc_text'].split()))
            if pair.get('participant_id'):
                participants.add(pair['participant_id'])

        if query_lengths:
            print(f"Query length: mean={np.mean(query_lengths):.1f}, std={np.std(query_lengths):.1f}")
        if doc_lengths:
            print(f"Doc length: mean={np.mean(doc_lengths):.1f}, std={np.std(doc_lengths):.1f}")
        print(f"Unique participants: {len(participants)}")

    except Exception as e:
        print(f"Error inspecting dataset: {e}")


def save_experiment_config(config, output_dir):
    """Save experiment configuration to JSON file"""
    config_path = output_dir / "experiment_config.json"

    # Convert non-serializable values
    serializable_config = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
                           for k, v in config.items()}

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"Saved experiment config to: {config_path}")


def handle_multiple_datasets(data_paths, dataset_types, max_eeg_len):
    """Handle loading and combining multiple datasets"""
    from mv_dataloader import load_combined_datasets, compute_combined_eeg_dimensions

    print(f"Loading {len(data_paths)} datasets for combination...")
    all_ict_pairs, combined_metadata = load_combined_datasets(data_paths, dataset_types)
    global_eeg_dims = compute_combined_eeg_dimensions(all_ict_pairs, max_eeg_len)

    # Save combined dataset temporarily - USE FULL PATH
    import tempfile
    import os
    temp_dir = tempfile.gettempdir()
    temp_combined_path = os.path.join(temp_dir, f"temp_combined_dataset_{os.getpid()}.npy")

    print(f"Saving temporary combined dataset to: {temp_combined_path}")
    combined_dataset = {'ict_pairs': all_ict_pairs, 'metadata': combined_metadata}

    try:
        np.save(temp_combined_path, combined_dataset)
        print(f"Successfully saved {len(all_ict_pairs)} combined pairs")
    except Exception as e:
        print(f"Error saving temporary dataset: {e}")
        raise

    return temp_combined_path, 'original', global_eeg_dims


def main():
    parser = argparse.ArgumentParser(
        description='Memory-Efficient Multi-Vector Brain Passage Retrieval with Dynamic Multi-Masking Validation')

    # Data arguments
    parser.add_argument('--data_path', help='Path to ICT pairs .npy file')
    parser.add_argument('--data_paths', nargs='+', help='Paths to multiple ICT pairs .npy files')
    parser.add_argument('--dataset_type', default='auto', choices=['auto', 'original', 'nieuwland'],
                        help='Dataset type: auto-detect, original format, or nieuwland format')
    parser.add_argument('--dataset_types', nargs='*', default=None,
                        help='Dataset types for each path (auto, original, nieuwland)')
    parser.add_argument('--inspect_only', action='store_true', help='Only inspect dataset, don\'t train')

    # Model arguments
    parser.add_argument('--colbert_model_name', default='colbert-ir/colbertv2.0', help='ColBERT model name')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--pooling_strategy', default='max', choices=['multi', 'cls', 'max', 'mean'],
                        help='Pooling strategy: multi-vector, CLS token, max pooling, or mean pooling')
    parser.add_argument('--encoder_type', default='dual', choices=['dual', 'cross'],
                        help='Encoder type: dual (bi-encoder) or cross (cross-encoder)')
    parser.add_argument('--query_type', default='eeg', choices=['eeg', 'text'],
                        help='Query representation type: eeg or text')

    # LoRA arguments
    parser.add_argument('--no_lora', action='store_true', help='Disable LoRA adaptation')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling factor')
    parser.add_argument('--use_pretrained_text', action='store_true',
                        help='Use pretrained ColBERT for text encoding')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_text_len', type=int, default=256, help='Max text sequence length')
    parser.add_argument('--max_eeg_len', type=int, default=50, help='Max EEG sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--eeg_arch', default='simple', choices=['simple', 'complex', 'transformer'],
                        help='EEG encoder architecture')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_vectors', type=int, default=32,
                        help='Number of vectors per sequence (for multi pooling)')

    # Experiment arguments
    parser.add_argument('--output_dir', default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--use_temporal_spatial_decomp', action='store_true',
                        help='Enable temporal-spatial decomposition for EEG')
    parser.add_argument('--decomp_level', default='word', choices=['word', 'sequence'],
                        help='Level of decomposition: word-level or sequence-level')
    parser.add_argument('--use_dual_loss', action='store_true',
                        help='Use dual loss for temporal and spatial components')
    parser.add_argument('--lambda_temporal', type=float, default=1.0,
                        help='Weight for temporal loss component')
    parser.add_argument('--lambda_spatial', type=float, default=1.0,
                        help='Weight for spatial loss component')

    parser.add_argument('--use_sequence_concat', action='store_true',
                        help='Concatenate EEG words into sequences before encoding (baseline for decomposition)')

    # UPDATED Multi-masking validation arguments for DYNAMIC approach
    parser.add_argument('--enable_multi_masking_validation', action='store_true',
                        help='Enable DYNAMIC validation across multiple masking levels during training')
    parser.add_argument('--validation_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100],
                        help='Masking percentages to evaluate during validation (default: 0 25 50 75 90 100)')
    parser.add_argument('--multi_masking_frequency', type=int, default=3,
                        help='Run multi-masking validation every N epochs (default: 3)')
    parser.add_argument('--primary_masking_level', type=int, default=90,
                        help='Primary masking level for early stopping (default: 90)')
    parser.add_argument('--training_masking_level', type=int, default=90,
                        help='Masking level used during training (default: 90)')
    # Add to parser arguments section
    parser.add_argument('--test_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100],
                        help='Masking percentages to evaluate during testing (default: 0 25 50 75 90 100)')
    parser.add_argument('--enable_test_evaluation', action='store_true',
                        help='Enable comprehensive test evaluation after training')

    parser.add_argument('--use_cnn_preprocessing', action='store_true',
                        help='Use CNN preprocessing before transformer (LaBram-style)')

    args = parser.parse_args()

    # Validate inputs
    if not args.data_path and not args.data_paths:
        raise ValueError("Must specify either --data_path or --data_paths")

    # Set random seeds
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Handle single vs multiple datasets
    if args.data_paths and len(args.data_paths) > 1:
        data_path_to_use, dataset_type_to_use, global_eeg_dims = handle_multiple_datasets(
            args.data_paths, args.dataset_types, args.max_eeg_len)
    else:
        data_path_to_use = args.data_path or args.data_paths[0]
        dataset_type_to_use = args.dataset_type
        global_eeg_dims = None

    # Inspect dataset
    inspect_dataset(data_path_to_use, dataset_type_to_use)

    if args.inspect_only:
        print("\nInspection complete. Exiting.")
        return

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else create_output_directory("dynamic_brain_retrieval")
    output_dir.mkdir(exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.colbert_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.colbert_model_name)
    except:
        print(f"ColBERT tokenizer not found, falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Add special tokens
    special_tokens = ['[Q]', '[D]', '[MASK]'] if '[MASK]' not in tokenizer.get_vocab() else ['[Q]', '[D]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create DYNAMIC dataloaders
    print(f"\nCreating DYNAMIC dataloaders...")
    if global_eeg_dims is None:  # Single dataset - compute dimensions
        train_dataloader, val_dataloader, global_eeg_dims = create_dynamic_dataloaders(
            data_path=data_path_to_use, tokenizer=tokenizer, batch_size=args.batch_size,
            max_text_len=args.max_text_len, max_eeg_len=args.max_eeg_len,
            train_ratio=args.train_ratio, debug=args.debug, num_vectors=args.num_vectors,
            dataset_type=dataset_type_to_use, training_masking_level=args.training_masking_level
        )
    else:  # Multi dataset - use pre-computed dimensions
        train_dataloader, val_dataloader, _ = create_dynamic_dataloaders(
            data_path=data_path_to_use, tokenizer=tokenizer, batch_size=args.batch_size,
            max_text_len=args.max_text_len, max_eeg_len=args.max_eeg_len,
            train_ratio=args.train_ratio, debug=args.debug, num_vectors=args.num_vectors,
            dataset_type='original', training_masking_level=args.training_masking_level,
            global_eeg_dims=global_eeg_dims
        )

    # Create experiment configuration
    config = {
        'experiment_type': 'dynamic_brain_retrieval', 'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path_to_use), 'colbert_model_name': args.colbert_model_name,
        'hidden_dim': args.hidden_dim, 'num_vectors': args.num_vectors,
        'batch_size': args.batch_size, 'max_text_len': args.max_text_len,
        'max_eeg_len': args.max_eeg_len, 'train_ratio': args.train_ratio,
        'seed': args.seed, 'device': str(device), 'tokenizer_vocab_size': len(tokenizer),
        'train_samples': len(train_dataloader.dataset), 'val_samples': len(val_dataloader.dataset),
        'eeg_arch': args.eeg_arch, 'learning_rate': args.lr, 'epochs': args.epochs,
        'patience': args.patience, 'use_lora': not args.no_lora, 'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha, 'pooling_strategy': args.pooling_strategy,
        'encoder_type': args.encoder_type, 'query_type': args.query_type,
        'use_pretrained_text': args.use_pretrained_text,
        # DYNAMIC Multi-masking validation configuration
        'enable_multi_masking_validation': args.enable_multi_masking_validation,
        'validation_masking_levels': args.validation_masking_levels,
        'multi_masking_frequency': args.multi_masking_frequency,
        'primary_masking_level': args.primary_masking_level,
        'training_masking_level': args.training_masking_level,
        'dynamic_masking': True,  # NEW: Flag indicating dynamic masking approach
        'memory_efficient': True  # NEW: Flag indicating memory-efficient approach
    }

    print(f"\n=== EXPERIMENT SETUP COMPLETE ===")
    print(f"Training samples: {config['train_samples']}")
    print(f"Validation samples: {config['val_samples']}")
    print(f"Pooling strategy: {args.pooling_strategy}")
    print(f"Training masking level: {args.training_masking_level}%")
    print(f"Memory approach: DYNAMIC masking with single dataloader")
    if args.enable_multi_masking_validation:
        print(f"Multi-masking validation: ENABLED (DYNAMIC)")
        print(f"  Validation masking levels: {args.validation_masking_levels}%")
        print(f"  Multi-masking frequency: every {args.multi_masking_frequency} epochs")
        print(f"  Primary masking level: {args.primary_masking_level}% (for early stopping)")
        print(f"  Memory efficient: Single dataloader with dynamic masking")
    else:
        print(f"Multi-masking validation: DISABLED")

    # CREATE MODEL
    print(f"\n=== MODEL CREATION ===")
    model = create_model(
        colbert_model_name=args.colbert_model_name, hidden_dim=args.hidden_dim,
        eeg_arch=args.eeg_arch, device=device, use_lora=not args.no_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        pooling_strategy=args.pooling_strategy, encoder_type=args.encoder_type,
        global_eeg_dims=global_eeg_dims, query_type=args.query_type,
        use_pretrained_text=args.use_pretrained_text,
        use_temporal_spatial_decomp=args.use_temporal_spatial_decomp,
        decomp_level=args.decomp_level,
        use_dual_loss=args.use_dual_loss,
        lambda_temporal=args.lambda_temporal,
        lambda_spatial=args.lambda_spatial,
        use_sequence_concat=args.use_sequence_concat,
        use_cnn_preprocessing=args.use_cnn_preprocessing  # ADD THIS LINE
    )

    config.update({
        'use_temporal_spatial_decomp': args.use_temporal_spatial_decomp,
        'decomp_level': args.decomp_level,
        'use_dual_loss': args.use_dual_loss,
        'lambda_temporal': args.lambda_temporal,
        'lambda_spatial': args.lambda_spatial,
        'use_sequence_concat': args.use_sequence_concat,
        'use_cnn_preprocessing': args.use_cnn_preprocessing
    })

    if not args.use_pretrained_text:
        model.set_tokenizer_vocab_size(len(tokenizer))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created! Total: {total_params:,}, Trainable: {trainable_params:,}")

    config.update({'total_params': total_params, 'trainable_params': trainable_params})
    save_experiment_config(config, output_dir)

    # CREATE OPTIMIZER AND TRAIN
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n=== TRAINING START (DYNAMIC MASKING) ===")

    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        debug=args.debug,
        config=config,
        # DYNAMIC Multi-masking validation parameters
        enable_multi_masking_validation=args.enable_multi_masking_validation,
        multi_masking_frequency=args.multi_masking_frequency,
        validation_masking_levels=args.validation_masking_levels,
        primary_masking_level=args.primary_masking_level
    )

    # SAVE TRAINED MODEL
    model_save_path = output_dir / f"dynamic_model_{args.pooling_strategy}_{args.eeg_arch}.pt"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'tokenizer_vocab_size': len(tokenizer)
    }, model_save_path)
    print(f"Saved trained model to: {model_save_path}")

    # ADD TEST EVALUATION (after model saving, before finish_wandb())
    if args.enable_test_evaluation:
        print(f"\n=== TEST SET EVALUATION ===")

        # Create test dataloader using same validation split approach
        # For simplicity, we'll use validation data as test data
        # In practice, you'd want a separate test split

        # Import the test function
        from mv_training import test_model

        # Run comprehensive test evaluation
        test_results = test_model(
            model=trained_model,
            test_dataloader=val_dataloader,  # Using val as test for now
            device=device,
            debug=args.debug,
            test_masking_levels=args.test_masking_levels,
            primary_masking_level=args.primary_masking_level
        )

        print(f"Test evaluation completed. Results logged to wandb under 'test_ranking/' section")
    else:
        print("Test evaluation skipped. Use --enable_test_evaluation to run comprehensive testing.")

    finish_wandb()
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Results saved in: {output_dir}")
    print(f"Memory approach: DYNAMIC masking with single dataloader")


if __name__ == "__main__":
    main()