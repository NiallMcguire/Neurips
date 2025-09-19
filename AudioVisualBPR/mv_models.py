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
                 use_pretrained_text=True):
        super().__init__()

        self.query_type = query_type
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        self.eeg_arch = eeg_arch
        self.use_lora = use_lora
        self.use_pretrained_text = use_pretrained_text

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
        print(f"Text encoder dimension: {encoder_dim} -> {hidden_dim}")

    def set_tokenizer_vocab_size(self, tokenizer_vocab_size):
        """Update text encoder vocab size after tokenizer is created"""
        if not self.use_pretrained_text:
            self.text_encoder.embedding = nn.Embedding(
                tokenizer_vocab_size,
                self.hidden_dim,
                padding_idx=0
            ).to(next(self.parameters()).device)  # ADD .to(device) HERE

    def _create_eeg_encoder(self, input_size, device):
        """Create EEG encoder based on architecture choice"""

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
        """Encode EEG with CLS or Multi-Vector pooling"""

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
            query_vectors = self.encode_eeg(eeg_queries, eeg_mv_masks)
        else:  # text
            query_vectors = self.encode_text(
                text_queries['input_ids'],
                text_queries['attention_mask']
            )

        return {
            'query_vectors': query_vectors,
            'doc_vectors': doc_vectors,
            'eeg_vectors': None if self.query_type == 'text' else query_vectors  # For backward compatibility
        }


class EEGTransformerEncoder(nn.Module):
    """Transformer encoder for EEG sequences"""

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, num_words, input_size]
        Returns:
            [batch, num_words, hidden_dim]
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
                 use_pretrained_text=True):
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
            use_pretrained_text=use_pretrained_text
        )
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