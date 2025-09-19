#!/usr/bin/env python3
"""
Simplified Multi-Vector Models for Brain Passage Retrieval
Supports only CLS and Multi-Vector pooling strategies
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType


class LaBramCNNPreprocessor(nn.Module):
    """LaBram-style CNN preprocessing for EEG patches"""

    def __init__(self, input_dim, hidden_dim, num_layers=3):  # ✅ Correct parameter name
        super().__init__()

        # CNN layers based on LaBram architecture
        if num_layers == 3:
            self.cnn_blocks = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=15, stride=8, padding=7),
                nn.GroupNorm(4, hidden_dim // 4),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, hidden_dim // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, hidden_dim),
                nn.GELU()
            )
        else:
            self.cnn_blocks = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=15, stride=8, padding=7),  # ✅ Fix this
                nn.GroupNorm(8, hidden_dim),
                nn.GELU()
            )

    def forward(self, x):
        """
        Args:
            x: [batch, sequence_len, channels] or [batch, words, time, channels]
        Returns:
            [batch, sequence_len, hidden_dim] or [batch, words, new_time, hidden_dim]
        """
        if len(x.shape) == 4:
            # Word-level: [batch, words, time, channels]
            batch_size, num_words, time_samples, channels = x.shape

            # Process each word separately
            word_features = []
            for word_idx in range(num_words):
                word_data = x[:, word_idx, :, :].transpose(1, 2)  # [batch, channels, time]
                word_feat = self.cnn_blocks(word_data).transpose(1, 2)  # [batch, new_time, hidden_dim]
                word_features.append(word_feat)

            # Stack back: [batch, words, new_time, hidden_dim]
            return torch.stack(word_features, dim=1)

        else:
            # Sequence-level: [batch, sequence_len, channels]
            x = x.transpose(1, 2)  # [batch, channels, sequence_len]
            x = self.cnn_blocks(x)  # [batch, hidden_dim, new_sequence_len]
            return x.transpose(1, 2)  # [batch, new_sequence_len, hidden_dim]


class EEGPositionalEmbeddings(nn.Module):
    """Spatial and temporal positional embeddings for EEG"""

    def __init__(self, hidden_dim, max_words=50, max_time=500, max_channels=64):
        super().__init__()

        # Learnable embeddings
        self.temporal_embeddings = nn.Embedding(max_time, hidden_dim)
        self.spatial_embeddings = nn.Embedding(max_channels, hidden_dim)
        self.word_embeddings = nn.Embedding(max_words, hidden_dim)

        self.max_words = max_words
        self.max_time = max_time
        self.max_channels = max_channels

    def forward(self, x, word_level=True):
        """
        Add positional embeddings to CNN features

        Args:
            x: [batch, words, time, hidden_dim] or [batch, time, hidden_dim]
            word_level: Whether input is word-level or sequence-level
        """
        if word_level and len(x.shape) == 4:
            batch_size, num_words, time_len, hidden_dim = x.shape

            # Create position indices
            word_pos = torch.arange(num_words, device=x.device).unsqueeze(0).repeat(batch_size, 1)  # [batch, words]
            time_pos = torch.arange(time_len, device=x.device)  # [time]

            # Get embeddings
            word_emb = self.word_embeddings(word_pos)  # [batch, words, hidden_dim]
            temp_emb = self.temporal_embeddings(time_pos)  # [time, hidden_dim]

            # Broadcast and add
            word_emb = word_emb.unsqueeze(2)  # [batch, words, 1, hidden_dim]
            temp_emb = temp_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, time, hidden_dim]

            return x + word_emb + temp_emb

        else:
            # Sequence-level: [batch, time, hidden_dim]
            batch_size, time_len, hidden_dim = x.shape

            # Only temporal embeddings for sequence-level
            time_pos = torch.arange(time_len, device=x.device)
            temp_emb = self.temporal_embeddings(time_pos)  # [time, hidden_dim]
            temp_emb = temp_emb.unsqueeze(0)  # [1, time, hidden_dim]

            return x + temp_emb

class SimpleTextEncoder(nn.Module):
    """Simple text encoder trained from scratch (similar complexity to EEG encoder)"""

    def __init__(self, vocab_size, hidden_dim=768, arch='simple'):
        super().__init__()
        self.arch = arch
        self.hidden_dim = hidden_dim

        # Embedding layer (trained from scratch)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        if arch == 'simple':
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif arch == 'complex':
            # Use LayerNorm instead of BatchNorm1d to avoid batch size issues
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        elif arch == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input_ids, attention_mask):
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, hidden_dim]

        if self.arch == 'transformer':
            # Use attention mask for transformer
            padding_mask = ~attention_mask.bool()
            encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        else:
            # For simple/complex: mean pooling over sequence, then apply MLP
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (embedded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            encoded = self.encoder(pooled).unsqueeze(1)  # [batch, 1, hidden_dim]

        return encoded


class SimplifiedBrainRetrieval(nn.Module):
    """
    Simplified Brain Retrieval Model with ColBERT document encoder and LoRA adaptation
    Supports only CLS and Multi-Vector pooling strategies
    """

    def __init__(self, colbert_model_name='colbert-ir/colbertv2.0',
                 hidden_dim=768, eeg_arch='simple', dropout=0.1,
                 use_lora=True, lora_r=16, lora_alpha=32,
                 pooling_strategy='multi', query_type='eeg',
                 use_pretrained_text=True, use_temporal_spatial_decomp=False,
                 decomp_level='word', use_sequence_concat=False, use_cnn_preprocessing=False):
        super().__init__()

        self.query_type = query_type
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        self.eeg_arch = eeg_arch
        self.use_lora = use_lora
        self.use_pretrained_text = use_pretrained_text
        self.use_temporal_spatial_decomp = use_temporal_spatial_decomp
        self.decomp_level = decomp_level
        self.use_sequence_concat = use_sequence_concat  # ADD THIS LINE
        self.use_cnn_preprocessing = use_cnn_preprocessing

        # Create separate encoders for temporal/spatial if needed
        if use_temporal_spatial_decomp:
            self.temporal_eeg_encoder = None  # Will be created dynamically
            self.spatial_eeg_encoder = None  # Will be created dynamically
            self.temporal_projection = nn.Linear(hidden_dim, hidden_dim)
            self.spatial_projection = nn.Linear(hidden_dim, hidden_dim)
            # ADD THIS: projection for concatenated features back to original dimension
            self.combined_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        if use_cnn_preprocessing:
            self.cnn_preprocessor = None  # Will be created dynamically
            self.positional_embeddings = None  # Will be created dynamically
            print(f"CNN preprocessing enabled (LaBram-style)")

        # Validate decomposition settings
        if use_temporal_spatial_decomp and query_type != 'eeg':
            raise ValueError("Temporal-spatial decomposition requires query_type='eeg'")

        # Validate sequence concatenation settings
        if use_sequence_concat and query_type != 'eeg':
            raise ValueError("Sequence concatenation requires query_type='eeg'")

        # Validate pooling strategy
        if pooling_strategy not in ['multi', 'cls', 'max', 'mean']:
            raise ValueError(
                f"Only 'multi', 'cls', 'max', and 'mean' pooling strategies supported, got: {pooling_strategy}")

        # Text encoder - either pretrained or simple from-scratch
        if use_pretrained_text:
            # Original pretrained approach
            print(f"Loading pretrained ColBERT model: {colbert_model_name}")
            try:
                self.text_encoder = AutoModel.from_pretrained(colbert_model_name)
            except:
                print(f"ColBERT model not found, falling back to bert-base-uncased")
                self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
                colbert_model_name = 'bert-base-uncased'

            encoder_dim = self.text_encoder.config.hidden_size
            self.text_projection = nn.Linear(encoder_dim, hidden_dim)

            # Apply LoRA if requested
            if use_lora:
                print(f"Applying LoRA adaptation with r={lora_r}, alpha={lora_alpha}")
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["query", "key", "value", "dense"]
                )
                self.text_encoder = get_peft_model(self.text_encoder, lora_config)
                print(f"LoRA parameters: {sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)}")
            else:
                # Freeze all parameters if not using LoRA
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        else:
            # Simple text encoder trained from scratch (fair comparison)
            print(f"Using simple text encoder trained from scratch (arch: {eeg_arch})")
            self.text_encoder = SimpleTextEncoder(
                vocab_size=30522,  # Will be updated with actual tokenizer size
                hidden_dim=hidden_dim,
                arch=eeg_arch  # Use same architecture as EEG encoder for fairness
            )
            encoder_dim = hidden_dim
            self.text_projection = nn.Identity()  # No additional projection needed

        # EEG encoder will be created dynamically
        self.eeg_encoder = None
        self.eeg_projection = nn.Linear(hidden_dim, hidden_dim)

        # Components for CLS pooling
        if pooling_strategy == 'cls':
            # Learnable CLS token for EEG sequences
            self.eeg_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

            # CLS transformer for attention-based aggregation
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.eeg_cls_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            print(f"Initialized learnable EEG CLS token: {self.eeg_cls_token.shape}")

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        print(f"Initialized Simplified model with {colbert_model_name if use_pretrained_text else 'SimpleTextEncoder'}")
        print(f"Pooling strategy: {pooling_strategy}")
        print(f"LoRA enabled: {use_lora}")
        print(f"Use pretrained text: {use_pretrained_text}")
        print(f"Use sequence concat: {use_sequence_concat}")  # ADD THIS LINE
        print(f"Text encoder dimension: {encoder_dim} -> {hidden_dim}")

    def _apply_temporal_spatial_decomposition(self, eeg_input):
        """Apply temporal-spatial decomposition to EEG data"""
        batch_size, num_words, time_samples, channels = eeg_input.shape

        if self.decomp_level == 'word':
            # Word-level: decompose each word separately
            temporal_features = torch.max(eeg_input, dim=3)[0]  # [batch, words, time]
            spatial_features = torch.max(eeg_input, dim=2)[0]  # [batch, words, channels]

            return temporal_features, spatial_features

        else:  # sequence level
            # Sequence-level: concatenate all words then decompose
            eeg_concat = eeg_input.view(batch_size, num_words * time_samples, channels)

            temporal_features = torch.max(eeg_concat, dim=2)[0]  # [batch, total_time]
            spatial_features = torch.max(eeg_concat, dim=1)[0]  # [batch, channels]

            # Reshape for consistent interface
            temporal_features = temporal_features.unsqueeze(1)  # [batch, 1, total_time]
            spatial_features = spatial_features.unsqueeze(1)  # [batch, 1, channels]

            return temporal_features, spatial_features

    def set_tokenizer_vocab_size(self, tokenizer_vocab_size):
        """Update text encoder vocab size after tokenizer is created"""
        if not self.use_pretrained_text:
            self.text_encoder.embedding = nn.Embedding(
                tokenizer_vocab_size,
                self.hidden_dim,
                padding_idx=0
            ).to(next(self.parameters()).device)  # ADD .to(device) HERE

    def _create_eeg_encoder(self, input_size, device, is_cnn_preprocessed=False):
        """Create EEG encoder based on architecture choice with CNN preprocessing support"""

        if self.use_cnn_preprocessing and not is_cnn_preprocessed:
            # Create CNN preprocessor and positional embeddings
            if isinstance(input_size, tuple):
                # For word/sequence level: input_size is (time_samples, channels)
                time_samples, channels = input_size

                self.cnn_preprocessor = LaBramCNNPreprocessor(
                    input_dim=channels,
                    hidden_dim=self.hidden_dim,
                    num_layers=2 if self.eeg_arch == 'simple' else 3
                ).to(device)

                self.positional_embeddings = EEGPositionalEmbeddings(
                    hidden_dim=self.hidden_dim
                ).to(device)

                print(f"Created CNN preprocessor with {channels} input channels")

                # After CNN preprocessing, we feed hidden_dim features to transformer
                effective_input_size = self.hidden_dim
            else:
                # Legacy path for flattened input
                effective_input_size = input_size
        else:
            effective_input_size = input_size if not is_cnn_preprocessed else self.hidden_dim

        # Create the main encoder based on architecture
        if self.eeg_arch == 'simple':
            encoder = nn.Sequential(
                nn.Linear(effective_input_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        elif self.eeg_arch == 'complex':
            encoder = nn.Sequential(
                nn.Linear(effective_input_size, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.BatchNorm1d(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
        elif self.eeg_arch == 'transformer':
            # For CNN preprocessing, we adjust layers to match parameter budget
            if self.use_cnn_preprocessing:
                num_layers = 2  # Fewer layers since we have CNN preprocessing
            elif self.use_temporal_spatial_decomp:
                num_layers = 2  # Decomposition uses 2 separate encoders
            elif self.pooling_strategy == 'cls':
                num_layers = 3  # CLS has additional transformer
            else:
                num_layers = 4  # Standard baseline

            encoder = EEGTransformerEncoder(effective_input_size, self.hidden_dim, num_layers=num_layers)
            print(
                f"Created transformer with {num_layers} layers ({'CNN + transformer' if self.use_cnn_preprocessing else 'transformer only'})")
        else:
            raise ValueError(f"Unknown EEG architecture: {self.eeg_arch}")

        print(f"Created {self.eeg_arch.upper()} EEG encoder with input size {effective_input_size}")
        return encoder.to(device)

    def _apply_cnn_preprocessing(self, eeg_input):
        """Apply CNN preprocessing EXACTLY following LaBraM process"""
        batch_size, num_words, time_samples, channels = eeg_input.shape

        # LaBraM patches: segment each channel into patches
        patch_length = 100  # Equivalent to LaBraM's w=200 at their sampling rate
        num_patches_per_channel = time_samples // patch_length

        if self.use_sequence_concat:
            # Sequence-level: treat all words as one continuous sequence
            eeg_concat = eeg_input.view(batch_size, num_words * time_samples, channels)
            total_time = num_words * time_samples
            num_time_patches = total_time // patch_length

            # Create patches: [batch, channels, patches_per_channel, patch_length]
            patches = []
            for c in range(channels):
                for t in range(num_time_patches):
                    start_t = t * patch_length
                    end_t = start_t + patch_length
                    if end_t <= total_time:
                        patch = eeg_concat[:, start_t:end_t, c]  # [batch, patch_length]
                        patches.append(patch)

            # Stack patches: [batch, total_patches, patch_length]
            if patches:
                eeg_patches = torch.stack(patches, dim=1)  # [batch, C*num_time_patches, patch_length]
            else:
                eeg_patches = torch.zeros(batch_size, 1, patch_length, device=eeg_input.device)

            # STEP 1: Temporal encoder (CNN) per patch
            patch_embeddings = []
            for p in range(eeg_patches.shape[1]):
                patch = eeg_patches[:, p, :].unsqueeze(-1)  # [batch, patch_length, 1] ✅
                # Apply 1D CNN
                patch_emb = self.cnn_preprocessor(patch)  # [batch, new_time, hidden_dim]
                patch_emb = torch.mean(patch_emb, dim=1)  # Pool to [batch, hidden_dim]
                patch_embeddings.append(patch_emb)

            patch_features = torch.stack(patch_embeddings, dim=1)  # [batch, total_patches, hidden_dim]

            # STEP 2: Add spatial embeddings (channel-specific)
            spatial_embeddings = []
            for c in range(channels):
                for t in range(num_time_patches):
                    # Each patch gets the spatial embedding of its channel
                    spatial_emb = self.positional_embeddings.spatial_embeddings(
                        torch.tensor(c, device=eeg_input.device)
                    )  # [hidden_dim]
                    spatial_embeddings.append(spatial_emb)

            if spatial_embeddings:
                spatial_emb_tensor = torch.stack(spatial_embeddings, dim=0)  # [total_patches, hidden_dim]
                spatial_emb_tensor = spatial_emb_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                patch_features = patch_features + spatial_emb_tensor

            # STEP 3: Add temporal embeddings (time-position specific)
            temporal_embeddings = []
            for c in range(channels):
                for t in range(num_time_patches):
                    # Each patch gets temporal embedding based on its time position
                    temporal_emb = self.positional_embeddings.temporal_embeddings(
                        torch.tensor(t, device=eeg_input.device)
                    )  # [hidden_dim]
                    temporal_embeddings.append(temporal_emb)

            if temporal_embeddings:
                temporal_emb_tensor = torch.stack(temporal_embeddings, dim=0)  # [total_patches, hidden_dim]
                temporal_emb_tensor = temporal_emb_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                patch_features = patch_features + temporal_emb_tensor

            return patch_features, 'sequence'

        else:
            # Word-level: process each word separately but with proper LaBraM patching
            word_features = []

            for word_idx in range(num_words):
                word_eeg = eeg_input[:, word_idx, :, :]  # [batch, time, channels]
                word_time = word_eeg.shape[1]
                num_time_patches = word_time // patch_length

                # Create patches for this word
                word_patches = []
                word_spatial_embs = []
                word_temporal_embs = []

                for c in range(channels):
                    for t in range(num_time_patches):
                        start_t = t * patch_length
                        end_t = start_t + patch_length
                        if end_t <= word_time:
                            # Extract patch
                            patch = word_eeg[:, start_t:end_t, c]  # [batch, patch_length]
                            word_patches.append(patch)

                            # Store spatial embedding index (channel)
                            word_spatial_embs.append(c)

                            # Store temporal embedding index (time position)
                            word_temporal_embs.append(t)

                if word_patches:
                    # Stack patches for this word
                    word_patch_tensor = torch.stack(word_patches, dim=1)  # [batch, patches, patch_length]

                    # STEP 1: Apply temporal encoder (CNN) to each patch
                    word_patch_embeddings = []
                    for p in range(word_patch_tensor.shape[1]):
                        patch = word_patch_tensor[:, p, :].unsqueeze(-1)  # [batch, patch_length, 1] ✅
                        patch_emb = self.cnn_preprocessor(patch)  # [batch, new_time, hidden_dim]
                        patch_emb = torch.mean(patch_emb, dim=1)  # Pool to [batch, hidden_dim]
                        word_patch_embeddings.append(patch_emb)

                    word_patch_features = torch.stack(word_patch_embeddings, dim=1)  # [batch, patches, hidden_dim]

                    # STEP 2: Add spatial embeddings (channel-specific)
                    for p, channel_idx in enumerate(word_spatial_embs):
                        spatial_emb = self.positional_embeddings.spatial_embeddings(
                            torch.tensor(channel_idx, device=eeg_input.device)
                        )  # [hidden_dim]
                        word_patch_features[:, p, :] += spatial_emb

                    # STEP 3: Add temporal embeddings (time-position specific)
                    for p, time_idx in enumerate(word_temporal_embs):
                        temporal_emb = self.positional_embeddings.temporal_embeddings(
                            torch.tensor(time_idx, device=eeg_input.device)
                        )  # [hidden_dim]
                        word_patch_features[:, p, :] += temporal_emb

                    # Pool patches within word (mean pooling)
                    word_feature = torch.mean(word_patch_features, dim=1)  # [batch, hidden_dim]
                    word_features.append(word_feature)
                else:
                    # Fallback for empty word
                    word_features.append(torch.zeros(batch_size, self.hidden_dim, device=eeg_input.device))

            # Stack all word features
            final_features = torch.stack(word_features, dim=1)  # [batch, num_words, hidden_dim]

            # Add word-level positional embeddings
            word_positions = torch.arange(num_words, device=eeg_input.device)
            word_pos_emb = self.positional_embeddings.word_embeddings(word_positions)  # [num_words, hidden_dim]
            word_pos_emb = word_pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            final_features = final_features + word_pos_emb

            return final_features, 'word'

    def encode_text(self, input_ids, attention_mask):
        """Encode text with CLS or Multi-Vector pooling"""

        # Get contextual representations (handle both pretrained and simple encoders)
        if self.use_pretrained_text:
            if self.use_lora:
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
            else:
                with torch.no_grad():
                    outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state
        else:
            # Simple encoder trained from scratch
            hidden_states = self.text_encoder(input_ids, attention_mask)

        # Project to target dimension
        projected = self.text_projection(hidden_states)
        projected = self.dropout(projected)

        batch_size = input_ids.size(0)

        if self.pooling_strategy == 'multi':
            # ColBERT-style: Use all positions except [CLS]

            multi_vectors = []

            for i in range(batch_size):
                # Skip [CLS] token (position 0), use all other positions where attention_mask = 1
                valid_positions = torch.where(attention_mask[i] == 1)[0][1:]  # Skip position 0 ([CLS])

                if len(valid_positions) > 0:
                    sample_vectors = projected[i, valid_positions]
                else:
                    # Fallback: single zero vector (shouldn't happen with proper augmentation)
                    sample_vectors = torch.zeros(1, projected.size(-1), device=projected.device)

                multi_vectors.append(sample_vectors)

            return multi_vectors

        elif self.pooling_strategy == 'max':
            # Max pooling: take max across sequence dimension, excluding padding
            max_vectors = []
            for i in range(batch_size):
                # Get valid positions (exclude padding)
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_representations = projected[i, valid_mask]  # [valid_len, hidden_dim]
                    max_vector = torch.max(valid_representations, dim=0)[0]  # [hidden_dim]
                else:
                    # Fallback: zero vector
                    max_vector = torch.zeros(projected.size(-1), device=projected.device)
                max_vectors.append(max_vector.unsqueeze(0))  # [1, hidden_dim]

            return torch.stack(max_vectors)  # [batch, 1, hidden_dim]

        elif self.pooling_strategy == 'mean':
            # Mean pooling: average across sequence dimension, excluding padding
            mean_vectors = []
            for i in range(batch_size):
                # Get valid positions (exclude padding)
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_representations = projected[i, valid_mask]  # [valid_len, hidden_dim]
                    mean_vector = torch.mean(valid_representations, dim=0)  # [hidden_dim]
                else:
                    # Fallback: zero vector
                    mean_vector = torch.zeros(projected.size(-1), device=projected.device)
                mean_vectors.append(mean_vector.unsqueeze(0))  # [1, hidden_dim]

            return torch.stack(mean_vectors)  # [batch, 1, hidden_dim]

        elif self.pooling_strategy == 'cls':
            # Use [CLS] token only
            cls_vectors = projected[:, 0:1, :]  # [batch, 1, hidden_dim]
            return cls_vectors

    def encode_eeg(self, eeg_input, eeg_mv_mask):
        """Encode EEG with optional CNN preprocessing, sequence concatenation and/or temporal-spatial decomposition"""

        # CNN Preprocessing Path
        if self.use_cnn_preprocessing:
            return self._encode_eeg_cnn(eeg_input, eeg_mv_mask)

        # Original paths (unchanged)
        if not self.use_sequence_concat and not self.use_temporal_spatial_decomp:
            return self._encode_eeg_original(eeg_input, eeg_mv_mask)

        batch_size, num_words, time_samples, channels = eeg_input.shape

        if self.use_sequence_concat and not self.use_temporal_spatial_decomp:
            return self._encode_eeg_sequence_concat(eeg_input, eeg_mv_mask)

        if self.use_temporal_spatial_decomp:
            # Apply decomposition
            temporal_features, spatial_features = self._apply_temporal_spatial_decomposition(eeg_input)

            # Create encoders if needed
            if self.temporal_eeg_encoder is None:
                temporal_input_size = temporal_features.shape[-1]
                spatial_input_size = spatial_features.shape[-1]

                self.temporal_eeg_encoder = self._create_eeg_encoder(temporal_input_size, eeg_input.device)
                self.spatial_eeg_encoder = self._create_eeg_encoder(spatial_input_size, eeg_input.device)

            # Encode temporal and spatial separately
            temporal_representations = self._encode_decomposed_features(temporal_features, self.temporal_eeg_encoder,
                                                                        self.temporal_projection)
            spatial_representations = self._encode_decomposed_features(spatial_features, self.spatial_eeg_encoder,
                                                                       self.spatial_projection)

            # Apply pooling strategy to both
            temporal_vectors = self._apply_pooling_to_decomposed(temporal_representations, eeg_input, 'temporal')
            spatial_vectors = self._apply_pooling_to_decomposed(spatial_representations, eeg_input, 'spatial')

            return {
                'temporal_vectors': temporal_vectors,
                'spatial_vectors': spatial_vectors,
                'combined_vectors': None  # Will be handled in training
            }

    # REPLACE the _encode_eeg_sequence_concat method in mv_models.py:

    def _encode_eeg_cnn(self, eeg_input, eeg_mv_mask):
        """NEW: Encode EEG with CNN preprocessing (LaBram-style)"""

        batch_size, num_words, time_samples, channels = eeg_input.shape

        # Create CNN preprocessor if needed - DIRECT CREATION
        if self.cnn_preprocessor is None:
            self.cnn_preprocessor = LaBramCNNPreprocessor(
                input_dim=1,  # ✅ Each patch is from a single channel
                hidden_dim=self.hidden_dim,
                num_layers=2 if self.eeg_arch == 'simple' else 3
            ).to(eeg_input.device)

            self.positional_embeddings = EEGPositionalEmbeddings(
                hidden_dim=self.hidden_dim,
                max_channels=channels  # Use actual number of channels from your data
            ).to(eeg_input.device)

            print(f"Created CNN preprocessor with {channels} input channels")

        # Apply CNN preprocessing - CORRECTED LOGIC
        cnn_features, level = self._apply_cnn_preprocessing(eeg_input)

        # Create main encoder if needed (works with CNN-preprocessed features)
        if self.eeg_encoder is None:
            self.eeg_encoder = self._create_eeg_encoder(self.hidden_dim, eeg_input.device, is_cnn_preprocessed=True)

        # Process CNN features through main encoder
        if level == 'sequence':
            # Sequence-level: [batch, total_patches, hidden_dim]
            if self.eeg_arch == 'transformer':
                sequence_representations = self.eeg_encoder(cnn_features)
            else:
                # For simple/complex: flatten and process
                batch_size, seq_len, hidden_dim = cnn_features.shape
                cnn_flat = cnn_features.view(batch_size * seq_len, hidden_dim)
                encoded_flat = self.eeg_encoder(cnn_flat)
                sequence_representations = encoded_flat.view(batch_size, seq_len, hidden_dim)

            # Apply pooling (similar to sequence_concat)
            if self.pooling_strategy == 'max':
                # Max pooling across patches
                pooled_representation = torch.max(sequence_representations, dim=1)[0].unsqueeze(1)
            elif self.pooling_strategy == 'cls':
                # Use CLS pooling with attention
                cls_tokens = self.eeg_cls_token.expand(batch_size, -1, -1)
                cls_sequence = torch.cat([cls_tokens, sequence_representations], dim=1)
                attended_sequence = self.eeg_cls_transformer(cls_sequence)
                pooled_representation = attended_sequence[:, 0:1, :]
            else:
                raise ValueError(f"CNN preprocessing only supports 'cls' and 'max' pooling for sequence-level")

        else:
            # Word-level: [batch, num_words, hidden_dim] (already processed by _apply_cnn_preprocessing)
            word_representations = cnn_features  # This is already [batch, words, hidden_dim]

            # Apply original word-level pooling logic
            if self.pooling_strategy == 'multi':
                multi_vectors = []
                for i in range(batch_size):
                    # Find non-zero EEG word positions
                    word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                    active_positions = torch.where(word_mask)[0]

                    if len(active_positions) > 0:
                        sample_vectors = word_representations[i, active_positions]
                    else:
                        sample_vectors = torch.zeros(1, self.hidden_dim, device=eeg_input.device)

                    multi_vectors.append(sample_vectors)

                pooled_representation = multi_vectors

            elif self.pooling_strategy == 'max':
                # Max pooling over words
                pooled_representation = torch.max(word_representations, dim=1)[0].unsqueeze(1)

            elif self.pooling_strategy == 'cls':
                # CLS pooling with attention over words
                cls_tokens = self.eeg_cls_token.expand(batch_size, -1, -1)
                cls_word_sequence = torch.cat([cls_tokens, word_representations], dim=1)
                word_mask = (eeg_input.abs().sum(dim=(2, 3)) > 0)
                cls_mask = torch.ones(batch_size, 1, device=eeg_input.device)
                full_mask = torch.cat([cls_mask, word_mask], dim=1)

                attended_sequence = self.eeg_cls_transformer(
                    cls_word_sequence,
                    src_key_padding_mask=~full_mask.bool()
                )

                pooled_representation = attended_sequence[:, 0:1, :]

            else:
                raise ValueError(f"Pooling strategy {self.pooling_strategy} not supported with CNN preprocessing")

        # Apply final projection and dropout
        if not isinstance(pooled_representation, list):
            pooled_representation = self.eeg_projection(pooled_representation)
            pooled_representation = self.dropout(pooled_representation)

        return pooled_representation

    def _encode_eeg_sequence_concat(self, eeg_input, eeg_mv_mask):
        """NEW: Encode EEG by concatenating words into sequences with MAX or CLS pooling"""

        batch_size, num_words, time_samples, channels = eeg_input.shape

        # Concatenate all words into a single sequence
        # Reshape: [batch, num_words, time, channels] -> [batch, num_words*time, channels]
        eeg_concat = eeg_input.view(batch_size, num_words * time_samples, channels)

        # Create mask for valid time steps (where we have actual EEG data)
        word_mask = (eeg_input.abs().sum(dim=(2, 3)) > 0)  # [batch, num_words]
        time_mask = word_mask.unsqueeze(-1).repeat(1, 1, time_samples)  # [batch, num_words, time]
        sequence_mask = time_mask.view(batch_size, num_words * time_samples)  # [batch, total_time]

        # Determine input size for encoder
        input_size = channels

        # Create EEG encoder if needed - always use sequence processing for consistent pooling
        if self.eeg_encoder is None:
            self.eeg_encoder = self._create_eeg_encoder(input_size, eeg_input.device)
            print(f"Created sequence-level EEG encoder ({self.eeg_arch}) for concatenated input")

        # Process concatenated sequence through encoder
        if self.eeg_arch == 'transformer':
            # Transformer can handle the sequence directly
            # eeg_concat: [batch, total_time, channels]
            sequence_representations = self.eeg_encoder(eeg_concat)  # [batch, total_time, hidden_dim]
        else:
            # For simple/complex: process each time step individually, then reshape back
            eeg_flat = eeg_concat.view(batch_size * (num_words * time_samples),
                                       channels)  # [batch*total_time, channels]
            encoded_flat = self.eeg_encoder(eeg_flat)  # [batch*total_time, hidden_dim]
            sequence_representations = encoded_flat.view(batch_size, num_words * time_samples,
                                                         self.hidden_dim)  # [batch, total_time, hidden_dim]

        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            # CLS POOLING: Use learnable CLS token with attention
            cls_tokens = self.eeg_cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]

            # Concatenate CLS token with sequence representations
            cls_sequence = torch.cat([cls_tokens, sequence_representations], dim=1)  # [batch, 1+total_time, hidden_dim]

            # Create attention mask: CLS can attend to all, sequence attends based on valid data
            cls_mask = torch.ones(batch_size, 1, device=eeg_input.device)  # CLS is never masked
            full_mask = torch.cat([cls_mask, sequence_mask], dim=1)  # [batch, 1+total_time]

            # Apply CLS transformer
            attended_sequence = self.eeg_cls_transformer(
                cls_sequence,
                src_key_padding_mask=~full_mask.bool()  # True = masked
            )

            # Extract CLS token representation
            pooled_representation = attended_sequence[:, 0:1, :]  # [batch, 1, hidden_dim]

        elif self.pooling_strategy == 'max':
            # MAX POOLING: Take max across sequence dimension with masking
            max_vectors = []
            for i in range(batch_size):
                # Get valid positions (exclude padding)
                valid_mask = sequence_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_representations = sequence_representations[i, valid_mask]  # [valid_len, hidden_dim]
                    max_vector = torch.max(valid_representations, dim=0)[0]  # [hidden_dim]
                else:
                    # Fallback: zero vector
                    max_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)
                max_vectors.append(max_vector.unsqueeze(0))  # [1, hidden_dim]

            pooled_representation = torch.stack(max_vectors)  # [batch, 1, hidden_dim]

        else:
            raise ValueError(
                f"Sequence concatenation only supports 'cls' and 'max' pooling, got: {self.pooling_strategy}")

        # Project to final dimension
        pooled_representation = self.eeg_projection(pooled_representation)
        pooled_representation = self.dropout(pooled_representation)

        # Handle multi-vector interface (return as list for consistency)
        if self.pooling_strategy == 'multi':
            # This shouldn't happen since we check above, but for safety
            multi_vectors = []
            for i in range(batch_size):
                sample_vectors = pooled_representation[i:i + 1]  # [1, hidden_dim]
                multi_vectors.append(sample_vectors)
            return multi_vectors
        else:
            # Return tensor for CLS/max pooling
            return pooled_representation  # [batch, 1, hidden_dim]

    def _encode_eeg_original(self, eeg_input, eeg_mv_mask):
        """Original EEG encoding logic (unchanged)"""
        batch_size, num_words, time_samples, channels = eeg_input.shape
        input_size = time_samples * channels

        # Create EEG encoder if needed
        if self.eeg_encoder is None:
            self.eeg_encoder = self._create_eeg_encoder(input_size, eeg_input.device)

        # Encode EEG words
        if self.eeg_arch == 'transformer':
            eeg_reshaped = eeg_input.view(batch_size, num_words, input_size)
            word_representations = self.eeg_encoder(eeg_reshaped)
        else:
            eeg_flat = eeg_input.view(batch_size * num_words, input_size)
            encoded = self.eeg_encoder(eeg_flat)
            word_representations = encoded.view(batch_size, num_words, self.hidden_dim)

        # Project to final dimension
        word_representations = self.eeg_projection(word_representations)
        word_representations = self.dropout(word_representations)

        if self.pooling_strategy == 'multi':
            # Multi-vector: use all valid EEG word representations
            multi_vectors = []

            for i in range(batch_size):
                # Find non-zero EEG word positions
                word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                active_positions = torch.where(word_mask)[0]

                if len(active_positions) > 0:
                    sample_vectors = word_representations[i, active_positions]
                else:
                    # Fallback: single zero vector
                    sample_vectors = torch.zeros(1, self.hidden_dim, device=eeg_input.device)

                multi_vectors.append(sample_vectors)

            return multi_vectors

        elif self.pooling_strategy == 'max':
            # Max pooling over EEG word representations
            max_vectors = []
            for i in range(batch_size):
                # Find non-zero EEG word positions
                word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                active_positions = torch.where(word_mask)[0]

                if len(active_positions) > 0:
                    active_representations = word_representations[i, active_positions]  # [active_words, hidden_dim]
                    max_vector = torch.max(active_representations, dim=0)[0]  # [hidden_dim]
                else:
                    # Fallback: zero vector
                    max_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)

                max_vectors.append(max_vector.unsqueeze(0))  # [1, hidden_dim]

            return torch.stack(max_vectors)  # [batch, 1, hidden_dim]

        elif self.pooling_strategy == 'mean':
            # Mean pooling over EEG word representations
            mean_vectors = []
            for i in range(batch_size):
                # Find non-zero EEG word positions
                word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                active_positions = torch.where(word_mask)[0]

                if len(active_positions) > 0:
                    active_representations = word_representations[i, active_positions]  # [active_words, hidden_dim]
                    mean_vector = torch.mean(active_representations, dim=0)  # [hidden_dim]
                else:
                    # Fallback: zero vector
                    mean_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)

                mean_vectors.append(mean_vector.unsqueeze(0))  # [1, hidden_dim]

            return torch.stack(mean_vectors)  # [batch, 1, hidden_dim]

        elif self.pooling_strategy == 'cls':
            # CLS pooling: use learnable CLS token with attention
            cls_tokens = self.eeg_cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]

            # Concatenate CLS token with word representations
            cls_word_sequence = torch.cat([cls_tokens, word_representations], dim=1)

            # Create attention mask: CLS can attend to all, words attend based on EEG data
            word_mask = (eeg_input.abs().sum(dim=(2, 3)) > 0)  # [batch, num_words]
            cls_mask = torch.ones(batch_size, 1, device=eeg_input.device)  # CLS is never masked
            full_mask = torch.cat([cls_mask, word_mask], dim=1)  # [batch, 1+num_words]

            # Apply CLS transformer
            attended_sequence = self.eeg_cls_transformer(
                cls_word_sequence,
                src_key_padding_mask=~full_mask.bool()  # True = masked
            )

            # Extract CLS token representation
            pooled_vectors = attended_sequence[:, 0:1, :]  # [batch, 1, hidden_dim]
            return pooled_vectors

    def _encode_decomposed_features(self, features, encoder, projection):
        """Encode temporal or spatial features"""
        batch_size = features.shape[0]

        if self.eeg_arch == 'transformer':
            # For transformer, features are already in the right shape
            encoded = encoder(features)
        else:
            # For simple/complex architectures, flatten and encode
            if len(features.shape) == 3:
                # Reshape for batch processing
                num_elements = features.shape[1]
                feature_dim = features.shape[2]
                flat_features = features.view(batch_size * num_elements, feature_dim)
                encoded_flat = encoder(flat_features)
                encoded = encoded_flat.view(batch_size, num_elements, self.hidden_dim)
            else:
                # 2D features - add sequence dimension
                encoded = encoder(features).unsqueeze(1)

        # Apply projection and dropout
        projected = projection(encoded)
        projected = self.dropout(projected)

        return projected

    def _apply_pooling_to_decomposed(self, representations, original_eeg_input, component_type):
        """Apply pooling strategy to decomposed representations"""
        batch_size = representations.shape[0]

        if self.pooling_strategy == 'multi':
            # Multi-vector: use all valid representations
            multi_vectors = []

            for i in range(batch_size):
                if len(representations.shape) == 3 and representations.shape[1] > 1:
                    # Multiple elements (word-level decomposition)
                    if component_type == 'temporal':
                        # Use same mask as original EEG
                        word_mask = (original_eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                        active_positions = torch.where(word_mask)[0]

                        if len(active_positions) > 0:
                            sample_vectors = representations[i, active_positions]
                        else:
                            sample_vectors = torch.zeros(1, self.hidden_dim, device=original_eeg_input.device)
                    else:  # spatial
                        # Similar logic for spatial
                        word_mask = (original_eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                        active_positions = torch.where(word_mask)[0]

                        if len(active_positions) > 0:
                            sample_vectors = representations[i, active_positions]
                        else:
                            sample_vectors = torch.zeros(1, self.hidden_dim, device=original_eeg_input.device)
                else:
                    # Single element (sequence-level decomposition)
                    # FIX: Remove extra dimension for sequence-level decomposition
                    if len(representations.shape) == 3:
                        sample_vectors = representations[i].squeeze(0)  # [1, 768] -> [768]
                        sample_vectors = sample_vectors.unsqueeze(0)  # [768] -> [1, 768]
                    else:
                        sample_vectors = representations[i:i + 1]

                multi_vectors.append(sample_vectors)

            return multi_vectors

        elif self.pooling_strategy in ['max', 'mean', 'cls']:
            # Single vector pooling
            if len(representations.shape) == 3 and representations.shape[1] > 1:
                # Multiple elements - need to pool
                pooled_vectors = []
                for i in range(batch_size):
                    if self.pooling_strategy == 'max':
                        pooled = torch.max(representations[i], dim=0)[0]
                    elif self.pooling_strategy == 'mean':
                        pooled = torch.mean(representations[i], dim=0)
                    else:  # cls - use first element as proxy
                        pooled = representations[i, 0]

                    pooled_vectors.append(pooled.unsqueeze(0))

                return torch.stack(pooled_vectors)
            else:
                # Already single vectors - but handle extra dimension for sequence-level
                if len(representations.shape) == 3:
                    # Remove extra dimension: [batch, 1, hidden_dim] -> [batch, hidden_dim] -> [batch, 1, hidden_dim]
                    return representations.squeeze(1).unsqueeze(1)
                else:
                    return representations

    def forward(self, eeg_queries, text_queries, docs, eeg_mv_masks):
        """
        Complete forward pass with toggleable query type
        """

        # Always encode documents
        doc_vectors = self.encode_text(
            docs['input_ids'],
            docs['attention_mask']
        )

        # Conditionally encode query based on query_type
        if self.query_type == 'eeg':
            eeg_output = self.encode_eeg(eeg_queries, eeg_mv_masks)

            # Handle temporal-spatial decomposition vs normal encoding
            if self.use_temporal_spatial_decomp and isinstance(eeg_output, dict):
                # Decomposition case: return the structure expected by training
                return {
                    'temporal_vectors': eeg_output['temporal_vectors'],
                    'spatial_vectors': eeg_output['spatial_vectors'],
                    'doc_vectors': doc_vectors,
                    'query_vectors': None,  # Not used in decomposition mode
                    'eeg_vectors': None  # For backward compatibility
                }
            else:
                # Normal case: return standard structure
                return {
                    'query_vectors': eeg_output,
                    'doc_vectors': doc_vectors,
                    'eeg_vectors': eeg_output  # For backward compatibility
                }
        else:  # text query
            query_vectors = self.encode_text(
                text_queries['input_ids'],
                text_queries['attention_mask']
            )

            return {
                'query_vectors': query_vectors,
                'doc_vectors': doc_vectors,
                'eeg_vectors': None  # For backward compatibility
            }


class EEGTransformerEncoder(nn.Module):
    """Transformer encoder for EEG sequences with configurable layer count"""

    def __init__(self, input_size, hidden_dim=768, num_heads=8, num_layers=2):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        # Use configurable number of layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.num_layers = num_layers  # Store for debugging

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, num_words, input_size] or [batch, sequence_length, input_size]
        Returns:
            [batch, num_words, hidden_dim] or [batch, sequence_length, hidden_dim]
        """
        # Project input
        x = self.input_projection(x)

        # Create padding mask (True = padded position)
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Final projection
        x = self.output_projection(x)

        return x


class CrossEncoderBrainRetrieval(nn.Module):
    """
    Cross-Encoder for Brain Retrieval with EEG-Document Cross-Attention
    """

    def __init__(self, colbert_model_name='colbert-ir/colbertv2.0',
                 hidden_dim=768, eeg_arch='simple', dropout=0.1,
                 use_lora=True, lora_r=16, lora_alpha=32,
                 query_type='eeg', use_pretrained_text=True):
        super().__init__()

        self.query_type = query_type
        self.hidden_dim = hidden_dim
        self.eeg_arch = eeg_arch
        self.use_lora = use_lora
        self.use_pretrained_text = use_pretrained_text
        self.pooling_strategy = 'cross'

        # Text encoder - either pretrained or simple from-scratch
        if use_pretrained_text:
            # Original pretrained approach
            print(f"Loading pretrained ColBERT model: {colbert_model_name}")
            try:
                self.text_encoder = AutoModel.from_pretrained(colbert_model_name)
            except:
                print(f"ColBERT model not found, falling back to bert-base-uncased")
                self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
                colbert_model_name = 'bert-base-uncased'

            encoder_dim = self.text_encoder.config.hidden_size
            self.text_projection = nn.Linear(encoder_dim, hidden_dim)

            # Apply LoRA if requested
            if use_lora:
                print(f"Applying LoRA adaptation with r={lora_r}, alpha={lora_alpha}")
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["query", "key", "value", "dense"]
                )
                self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            else:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        else:
            # Simple text encoder trained from scratch (fair comparison)
            print(f"Using simple text encoder trained from scratch (arch: {eeg_arch})")
            self.text_encoder = SimpleTextEncoder(
                vocab_size=30522,  # Will be updated with actual tokenizer size
                hidden_dim=hidden_dim,
                arch=eeg_arch  # Use same architecture as EEG encoder for fairness
            )
            encoder_dim = hidden_dim
            self.text_projection = nn.Identity()  # No additional projection needed

        # EEG encoder (created dynamically)
        self.eeg_encoder = None
        self.eeg_projection = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single score output
        )

        self.dropout = nn.Dropout(dropout)

        print(f"Initialized Cross-Encoder with {colbert_model_name if use_pretrained_text else 'SimpleTextEncoder'}")
        print(f"Use pretrained text: {use_pretrained_text}")
        print(f"Cross-attention heads: 8, Hidden dim: {hidden_dim}")

    def set_tokenizer_vocab_size(self, tokenizer_vocab_size):
        """Update text encoder vocab size after tokenizer is created"""
        if not self.use_pretrained_text:
            self.text_encoder.embedding = nn.Embedding(
                tokenizer_vocab_size,
                self.hidden_dim,
                padding_idx=0
            ).to(next(self.parameters()).device)  # ADD .to(device) HERE

    def _create_eeg_encoder(self, input_size, device):
        """Create EEG encoder (same as dual encoder)"""
        if self.eeg_arch == 'simple':
            encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        elif self.eeg_arch == 'complex':
            encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.BatchNorm1d(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
        elif self.eeg_arch == 'transformer':
            encoder = EEGTransformerEncoder(input_size, self.hidden_dim)
        else:
            raise ValueError(f"Unknown EEG architecture: {self.eeg_arch}")

        print(f"Created {self.eeg_arch.upper()} EEG encoder with input size {input_size}")
        return encoder.to(device)

    def forward(self, eeg_queries, text_queries, docs):
        """
        Cross-encoder forward pass with toggleable query type
        """

        batch_size = docs['input_ids'].size(0)

        # Encode documents (handle both pretrained and simple text encoders)
        if self.use_pretrained_text:
            if self.use_lora:
                doc_outputs = self.text_encoder(
                    input_ids=docs['input_ids'],
                    attention_mask=docs['attention_mask']
                )
                doc_representations = doc_outputs.last_hidden_state
            else:
                with torch.no_grad():
                    doc_outputs = self.text_encoder(
                        input_ids=docs['input_ids'],
                        attention_mask=docs['attention_mask']
                    )
                    doc_representations = doc_outputs.last_hidden_state
        else:
            # Simple encoder for documents (fair comparison)
            doc_representations = self.text_encoder(docs['input_ids'], docs['attention_mask'])

        doc_representations = self.text_projection(doc_representations)
        doc_representations = self.dropout(doc_representations)

        # Conditionally encode query based on query_type
        if self.query_type == 'eeg':
            # EEG encoding (existing logic)
            num_words, time_samples, channels = eeg_queries.shape[1:]
            input_size = time_samples * channels

            if self.eeg_encoder is None:
                self.eeg_encoder = self._create_eeg_encoder(input_size, eeg_queries.device)

            if self.eeg_arch == 'transformer':
                eeg_reshaped = eeg_queries.view(batch_size, num_words, input_size)
                query_representations = self.eeg_encoder(eeg_reshaped)
            else:
                eeg_flat = eeg_queries.view(batch_size * num_words, input_size)
                encoded = self.eeg_encoder(eeg_flat)
                query_representations = encoded.view(batch_size, num_words, self.hidden_dim)

            query_representations = self.eeg_projection(query_representations)
            query_representations = self.dropout(query_representations)
            query_mask = (eeg_queries.abs().sum(dim=(2, 3)) > 0)  # [batch, num_words]

        else:  # text query
            # Text query encoding (handle both pretrained and simple)
            if self.use_pretrained_text:
                if self.use_lora:
                    query_outputs = self.text_encoder(
                        input_ids=text_queries['input_ids'],
                        attention_mask=text_queries['attention_mask']
                    )
                    query_representations = query_outputs.last_hidden_state
                else:
                    with torch.no_grad():
                        query_outputs = self.text_encoder(
                            input_ids=text_queries['input_ids'],
                            attention_mask=text_queries['attention_mask']
                        )
                        query_representations = query_outputs.last_hidden_state
            else:
                # Simple encoder for text queries (fair comparison)
                query_representations = self.text_encoder(text_queries['input_ids'], text_queries['attention_mask'])

            query_representations = self.text_projection(query_representations)
            query_representations = self.dropout(query_representations)
            query_mask = text_queries['attention_mask'].bool()  # [batch, seq_len]

        # Cross-attention: Query attends to Document
        attended_query, _ = self.cross_attention(
            query=query_representations,
            key=doc_representations,
            value=doc_representations,
            key_padding_mask=~docs['attention_mask'].bool(),  # True = ignore
            attn_mask=None
        )

        # Pool query representations (mean over valid positions)
        query_mask_expanded = query_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        pooled_query = (attended_query * query_mask_expanded).sum(dim=1) / (query_mask_expanded.sum(dim=1) + 1e-8)

        # Get compatibility score
        scores = self.classifier(pooled_query)  # [batch, 1]

        return scores



def compute_similarity(query_vectors, doc_vectors, pooling_strategy, temperature=1.0):
    """Compute similarity based on pooling strategy"""

    if pooling_strategy == 'multi':
        # Multi-vector MaxSim similarity (ColBERT-style)
        return compute_multi_vector_similarity(query_vectors, doc_vectors, temperature)
    elif pooling_strategy in ['cls', 'max', 'mean']:
        # Simple cosine similarity for single-vector strategies
        return compute_cls_similarity(query_vectors, doc_vectors, temperature)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

def compute_multi_vector_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute ColBERT-style MaxSim similarity for multi-vectors"""

    if isinstance(query_vectors, list):
        # Handle list of variable-length tensors
        similarities = []

        for i in range(len(query_vectors)):
            q_vecs = query_vectors[i]
            d_vecs = doc_vectors[i]

            # Normalize vectors
            q_vecs = F.normalize(q_vecs, p=2, dim=1)
            d_vecs = F.normalize(d_vecs, p=2, dim=1)

            # Remove zero vectors
            q_nonzero = q_vecs[q_vecs.norm(dim=1) > 1e-6]
            d_nonzero = d_vecs[d_vecs.norm(dim=1) > 1e-6]

            if len(q_nonzero) == 0 or len(d_nonzero) == 0:
                similarities.append(torch.tensor(0.0, device=q_vecs.device))
                continue

            # Compute MaxSim: for each query vector, find max similarity with document vectors
            sim_matrix = torch.matmul(q_nonzero, d_nonzero.t())
            max_sims = sim_matrix.max(dim=1)[0]  # Max similarity for each query vector
            sim = max_sims.sum()  # Sum all max similarities
            similarities.append(sim)

        return torch.stack(similarities) / temperature
    else:
        raise ValueError("Multi-vector similarity requires list input")


def compute_cls_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute cosine similarity for CLS tokens"""

    if isinstance(query_vectors, list):
        # Handle list input (consistent with multi-vector interface)
        similarities = []

        for i in range(len(query_vectors)):
            # Extract vectors (remove singleton dimensions if needed)
            q_vec = query_vectors[i].squeeze()  # [hidden_dim]
            d_vec = doc_vectors[i].squeeze()  # [hidden_dim]

            # Normalize and compute cosine similarity
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_norm = F.normalize(d_vec, p=2, dim=0)

            sim = torch.dot(q_norm, d_norm)
            similarities.append(sim)

        return torch.stack(similarities) / temperature
    else:
        # Handle tensor input (legacy path)
        batch_size = query_vectors.size(0)
        similarities = []

        for i in range(batch_size):
            # Extract vectors (remove singleton dimensions if needed)
            q_vec = query_vectors[i].squeeze()  # [hidden_dim]
            d_vec = doc_vectors[i].squeeze()  # [hidden_dim]

            # Normalize and compute cosine similarity
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_norm = F.normalize(d_vec, p=2, dim=0)

            sim = torch.dot(q_norm, d_norm)
            similarities.append(sim)

        return torch.stack(similarities) / temperature


def create_model(colbert_model_name='colbert-ir/colbertv2.0', hidden_dim=768,
                 eeg_arch='simple', device='cuda', use_lora=True, lora_r=16,
                 lora_alpha=32, pooling_strategy='multi', encoder_type='dual',
                 global_eeg_dims=None, query_type='eeg',
                 use_pretrained_text=True, use_temporal_spatial_decomp=False,
                 decomp_level='word', use_dual_loss=False,
                 lambda_temporal=1.0, lambda_spatial=1.0, use_sequence_concat=False,
                 use_cnn_preprocessing=False):  # ADD THIS PARAMETER
    if encoder_type == 'dual':
        model = SimplifiedBrainRetrieval(
            colbert_model_name=colbert_model_name,
            hidden_dim=hidden_dim,
            eeg_arch=eeg_arch,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            pooling_strategy=pooling_strategy,
            query_type=query_type,
            use_pretrained_text=use_pretrained_text,
            use_temporal_spatial_decomp=use_temporal_spatial_decomp,
            decomp_level=decomp_level,
            use_sequence_concat=use_sequence_concat,
            use_cnn_preprocessing=use_cnn_preprocessing  # ADD THIS LINE
        )
        # Add dual loss parameters as attributes
        model.use_dual_loss = use_dual_loss
        model.lambda_temporal = lambda_temporal
        model.lambda_spatial = lambda_spatial

    elif encoder_type == 'cross':
        model = CrossEncoderBrainRetrieval(
            colbert_model_name=colbert_model_name,
            hidden_dim=hidden_dim,
            eeg_arch=eeg_arch,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            query_type=query_type,
            use_pretrained_text=use_pretrained_text
        )

    return model.to(device)