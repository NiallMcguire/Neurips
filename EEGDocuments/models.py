#!/usr/bin/env python3
"""
EEG Alignment Models for EEG-EEG vs EEG-Text Alignment
Clean dual-encoder architecture with toggleable document types
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType


class EEGEncoder(nn.Module):
    """EEG encoder with different architecture options and STRONG regularization"""

    def __init__(self, input_size, hidden_dim=768, arch='simple', dropout=0.3):  # INCREASED from 0.1
        super().__init__()
        self.arch = arch
        self.hidden_dim = hidden_dim

        if arch == 'simple':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Added LayerNorm
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Added LayerNorm
                nn.ReLU(),
                nn.Dropout(dropout),  # Added dropout
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif arch == 'complex':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d
                nn.ReLU(),
                nn.Dropout(dropout * 2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),  # Changed from BatchNorm1d
                nn.ReLU(),
                nn.Dropout(dropout * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)  # Added final dropout
            )
        elif arch == 'transformer':
            self.input_projection = nn.Linear(input_size, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)  # Added normalization

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True  # Pre-norm architecture for stability
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
            self.output_dropout = nn.Dropout(dropout)  # Added dropout
        else:
            raise ValueError(f"Unknown EEG architecture: {arch}")

    def forward(self, x):
        """
        Args:
            x: [batch, num_words, input_size] or [batch*num_words, input_size]
        Returns:
            [batch, num_words, hidden_dim] or [batch*num_words, hidden_dim]
        """
        if self.arch == 'transformer':
            x = self.input_projection(x)
            x = self.input_norm(x)  # Normalize after projection

            # Create padding mask
            padding_mask = (x.abs().sum(dim=-1) == 0)
            x = self.transformer(x, src_key_padding_mask=padding_mask)
            x = self.output_projection(x)
            x = self.output_dropout(x)  # Apply dropout
        else:
            x = self.encoder(x)

        return x


class TextEncoder(nn.Module):
    """Text encoder using pretrained models with optional LoRA"""

    def __init__(self, model_name='bert-base-uncased', hidden_dim=768,
                 use_lora=True, lora_r=16, lora_alpha=32):
        super().__init__()

        print(f"Loading text encoder: {model_name}")
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
        except:
            print(f"Model {model_name} not found, falling back to bert-base-uncased")
            self.encoder = AutoModel.from_pretrained('bert-base-uncased')

        encoder_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_dim, hidden_dim)

        # Apply LoRA if requested
        if use_lora:
            print(f"Applying LoRA adaptation (r={lora_r}, alpha={lora_alpha})")
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"]
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
        else:
            # Freeze parameters if not using LoRA
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """Encode text tokens"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        projected = self.projection(hidden_states)
        return projected


class EEGAlignmentModel(nn.Module):
    """
    Dual encoder for EEG-EEG or EEG-Text alignment

    Modes:
    - EEG-Text: EEG encoder for queries, Text encoder for documents
    - EEG-EEG: EEG encoder for both queries and documents
    """

    def __init__(self, document_type='text', colbert_model_name='bert-base-uncased',
                 hidden_dim=768, eeg_arch='simple', pooling_strategy='multi',
                 use_lora=True, lora_r=16, lora_alpha=32, dropout=0.3):  # INCREASED from 0.1
        super().__init__()

        self.document_type = document_type
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        self.eeg_arch = eeg_arch

        print(f"Creating EEG-{document_type.upper()} alignment model")
        print(f"Pooling strategy: {pooling_strategy}")
        print(f"EEG architecture: {eeg_arch}")
        print(f"Dropout: {dropout}")  # Log dropout rate

        # EEG encoder for queries (always needed)
        self.eeg_encoder = None
        self.eeg_projection = nn.Linear(hidden_dim, hidden_dim)
        self.eeg_proj_dropout = nn.Dropout(dropout)  # Added dropout after projection

        # Document encoder based on type
        if document_type == 'text':
            self.text_encoder = TextEncoder(
                model_name=colbert_model_name,
                hidden_dim=hidden_dim,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            )
            self.doc_eeg_encoder = None
        elif document_type == 'eeg':
            self.text_encoder = None
            self.doc_eeg_encoder = None
        else:
            raise ValueError(f"document_type must be 'text' or 'eeg', got {document_type}")

        # Pooling components
        if pooling_strategy == 'cls':
            # Learnable CLS tokens
            self.eeg_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            if document_type == 'eeg':
                self.doc_eeg_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

            # CLS transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True  # Pre-norm
            )
            self.cls_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.dropout = nn.Dropout(dropout)

        print(f"Model initialized for {document_type.upper()} documents with dropout={dropout}")

    def set_tokenizer_vocab_size(self, vocab_size):
        """Update tokenizer vocab size - only relevant for text documents with custom tokenizers"""
        # This method is called by the controller but not needed since we use pretrained models
        # with fixed vocab sizes. Just pass for compatibility.
        pass

    def _create_eeg_encoder(self, input_size, device):
        """Create EEG encoder based on input size"""
        encoder = EEGEncoder(input_size, self.hidden_dim, self.eeg_arch)
        print(f"Created EEG encoder ({self.eeg_arch}) with input size {input_size}")
        return encoder.to(device)

    def encode_eeg(self, eeg_input, is_document=False):
        """
        Encode EEG sequences with specified pooling strategy
        """
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

        # Project and apply dropout
        word_representations = self.eeg_projection(word_representations)
        word_representations = self.eeg_proj_dropout(word_representations)  # Use separate dropout
        word_representations = self.dropout(word_representations)  # Additional dropout

        # Apply pooling strategy
        return self._apply_pooling(word_representations, eeg_input, is_document)

    def encode_text(self, input_ids, attention_mask):
        """Encode text documents"""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not initialized - model configured for EEG documents")

        # Get text representations
        text_representations = self.text_encoder(input_ids, attention_mask)
        text_representations = self.dropout(text_representations)

        # Apply pooling strategy for text
        return self._apply_text_pooling(text_representations, attention_mask)

    def _apply_pooling(self, word_representations, eeg_input, is_document=False):
        """Apply pooling strategy to EEG representations"""
        batch_size = word_representations.size(0)

        if self.pooling_strategy == 'multi':
            # Multi-vector: return all valid word representations
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
            # Max pooling over valid positions
            pooled_vectors = []
            for i in range(batch_size):
                word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                active_positions = torch.where(word_mask)[0]

                if len(active_positions) > 0:
                    active_reps = word_representations[i, active_positions]
                    max_vector = torch.max(active_reps, dim=0)[0]
                else:
                    max_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)

                pooled_vectors.append(max_vector.unsqueeze(0))

            return torch.stack(pooled_vectors)

        elif self.pooling_strategy == 'mean':
            # Mean pooling over valid positions
            pooled_vectors = []
            for i in range(batch_size):
                word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                active_positions = torch.where(word_mask)[0]

                if len(active_positions) > 0:
                    active_reps = word_representations[i, active_positions]
                    mean_vector = torch.mean(active_reps, dim=0)
                else:
                    mean_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)

                pooled_vectors.append(mean_vector.unsqueeze(0))

            return torch.stack(pooled_vectors)

        elif self.pooling_strategy == 'cls':
            # CLS pooling with learnable token
            cls_token = self.doc_eeg_cls_token if (is_document and self.document_type == 'eeg') else self.eeg_cls_token
            cls_tokens = cls_token.expand(batch_size, -1, -1)

            # Concatenate CLS token with word representations
            cls_word_sequence = torch.cat([cls_tokens, word_representations], dim=1)

            # Create attention mask
            word_mask = (eeg_input.abs().sum(dim=(2, 3)) > 0)
            cls_mask = torch.ones(batch_size, 1, device=eeg_input.device)
            full_mask = torch.cat([cls_mask, word_mask], dim=1)

            # Apply CLS transformer
            attended_sequence = self.cls_transformer(
                cls_word_sequence,
                src_key_padding_mask=~full_mask.bool()
            )

            # Extract CLS token representation
            return attended_sequence[:, 0:1, :]  # [batch, 1, hidden_dim]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def _apply_text_pooling(self, text_representations, attention_mask):
        """Apply pooling strategy to text representations"""
        batch_size = text_representations.size(0)

        if self.pooling_strategy == 'multi':
            # Multi-vector: use all valid positions except [CLS]
            multi_vectors = []
            for i in range(batch_size):
                valid_positions = torch.where(attention_mask[i] == 1)[0][1:]  # Skip [CLS]
                if len(valid_positions) > 0:
                    sample_vectors = text_representations[i, valid_positions]
                else:
                    sample_vectors = torch.zeros(1, text_representations.size(-1), device=text_representations.device)
                multi_vectors.append(sample_vectors)
            return multi_vectors

        elif self.pooling_strategy == 'max':
            # Max pooling over valid positions
            pooled_vectors = []
            for i in range(batch_size):
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_reps = text_representations[i, valid_mask]
                    max_vector = torch.max(valid_reps, dim=0)[0]
                else:
                    max_vector = torch.zeros(text_representations.size(-1), device=text_representations.device)
                pooled_vectors.append(max_vector.unsqueeze(0))
            return torch.stack(pooled_vectors)

        elif self.pooling_strategy == 'mean':
            # Mean pooling over valid positions
            pooled_vectors = []
            for i in range(batch_size):
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_reps = text_representations[i, valid_mask]
                    mean_vector = torch.mean(valid_reps, dim=0)
                else:
                    mean_vector = torch.zeros(text_representations.size(-1), device=text_representations.device)
                pooled_vectors.append(mean_vector.unsqueeze(0))
            return torch.stack(pooled_vectors)

        elif self.pooling_strategy == 'cls':
            # Use [CLS] token only
            return text_representations[:, 0:1, :]  # [batch, 1, hidden_dim]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(self, batch):
        """
        Forward pass for either EEG-EEG or EEG-Text alignment

        Args:
            batch: Dictionary containing query_eegs and either doc_eegs or doc_text_tokens
        """
        query_eegs = batch['query_eegs']
        document_type = batch['document_type']

        # Encode queries (always EEG)
        query_vectors = self.encode_eeg(query_eegs, is_document=False)

        # Encode documents based on type
        if document_type == 'eeg':
            # EEG-EEG alignment
            doc_eegs = batch['doc_eegs']
            doc_vectors = self.encode_eeg(doc_eegs, is_document=True)
        elif document_type == 'text':
            # EEG-Text alignment
            doc_text_tokens = batch['doc_text_tokens']
            doc_vectors = self.encode_text(
                doc_text_tokens['input_ids'],
                doc_text_tokens['attention_mask']
            )
        else:
            raise ValueError(f"Unknown document type: {document_type}")

        return {
            'query_vectors': query_vectors,
            'doc_vectors': doc_vectors,
            'document_type': document_type
        }


def compute_similarity(query_vectors, doc_vectors, pooling_strategy, temperature=1.0):
    """Compute similarity based on pooling strategy"""

    if pooling_strategy == 'multi':
        return compute_multi_vector_similarity(query_vectors, doc_vectors, temperature)
    else:  # cls, max, mean
        return compute_single_vector_similarity(query_vectors, doc_vectors, temperature)


def compute_multi_vector_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute ColBERT-style MaxSim similarity for multi-vectors"""
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


def compute_single_vector_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute cosine similarity for single vectors (cls, max, mean pooling)"""

    # Handle list input (convert to tensors)
    if isinstance(query_vectors, list):
        query_vectors = torch.stack([qv for qv in query_vectors])
    if isinstance(doc_vectors, list):
        doc_vectors = torch.stack([dv for dv in doc_vectors])

    batch_size = query_vectors.size(0)
    similarities = []

    for i in range(batch_size):
        # Extract vectors (remove singleton dimensions)
        q_vec = query_vectors[i].squeeze()  # [hidden_dim]
        d_vec = doc_vectors[i].squeeze()  # [hidden_dim]

        # Normalize and compute cosine similarity
        q_norm = F.normalize(q_vec, p=2, dim=0)
        d_norm = F.normalize(d_vec, p=2, dim=0)

        sim = torch.dot(q_norm, d_norm)
        similarities.append(sim)

    return torch.stack(similarities) / temperature


def create_alignment_model(document_type='text', colbert_model_name='bert-base-uncased',
                           hidden_dim=768, eeg_arch='simple', pooling_strategy='multi',
                           global_eeg_dims=None, device='cuda', dropout=0.3):  # NEW parameter
    """Create EEG alignment model"""

    model = EEGAlignmentModel(
        document_type=document_type,
        colbert_model_name=colbert_model_name,
        hidden_dim=hidden_dim,
        eeg_arch=eeg_arch,
        pooling_strategy=pooling_strategy,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        dropout=dropout  # NEW: pass dropout
    )

    return model.to(device)