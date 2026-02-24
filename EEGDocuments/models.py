#!/usr/bin/env python3
"""
EEG Alignment Models for EEG-EEG vs EEG-Text Alignment
Clean dual-encoder architecture with toggleable document types

v2.1 additions:
  doc_encoder_type : 'bert' (default) | 'eeg'
      When 'eeg' and document_type='text', the document is encoded by an
      EEGEncoder fed with frozen BERT token embeddings rather than by the
      full TextEncoder.  This isolates the encoder *architecture* effect
      from the document *modality* effect.
  freeze_doc_encoder : bool (default False)
      Freeze the document-side encoder weights after initialisation.
      Works for all four conditions:
        EEG-Text  / bert encoder  → freezes TextEncoder
        EEG-Text  / eeg  encoder  → freezes text_doc_eeg_encoder
        EEG-EEG              → creates a separate doc_eeg_encoder and freezes it
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType


class EEGEncoder(nn.Module):
    """EEG encoder with different architecture options and STRONG regularization"""

    def __init__(self, input_size, hidden_dim=768, arch='simple', dropout=0.3):
        super().__init__()
        self.arch = arch
        self.hidden_dim = hidden_dim

        if arch == 'simple':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif arch == 'complex':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
        elif arch == 'transformer':
            self.input_projection = nn.Linear(input_size, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
            self.output_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown EEG architecture: {arch}")

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
               For EEG: seq_len = num_words, input_size = time * channels
               For text: seq_len = num_tokens, input_size = embed_dim
        Returns:
            [batch, seq_len, hidden_dim]
        """
        if self.arch == 'transformer':
            x = self.input_projection(x)
            x = self.input_norm(x)
            # Padding mask: positions where the full vector is zero
            padding_mask = (x.abs().sum(dim=-1) == 0)
            x = self.transformer(x, src_key_padding_mask=padding_mask)
            x = self.output_projection(x)
            x = self.output_dropout(x)
        else:
            # simple / complex: process all positions independently
            batch_size, seq_len, input_size = x.shape
            x_flat = x.view(batch_size * seq_len, input_size)
            x_flat = self.encoder(x_flat)
            x = x_flat.view(batch_size, seq_len, self.hidden_dim)

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
    Dual encoder for EEG-EEG or EEG-Text alignment.

    document_type   doc_encoder_type   Description
    ─────────────   ────────────────   ───────────────────────────────────────────
    'text'          'bert'             Existing: EEG query → BERT/LoRA text doc
    'text'          'eeg'              NEW: EEG query → EEGEncoder text doc
                                       (frozen BERT token embeddings → EEGEncoder)
    'eeg'           'bert' (ignored)   Existing: EEG query → EEG doc (shared enc.)
    'eeg'           'eeg'  (default)   EEG query → EEG doc, optionally frozen doc

    freeze_doc_encoder=True freezes whatever encoder is on the document side.
    """

    def __init__(self, document_type='text', colbert_model_name='bert-base-uncased',
                 hidden_dim=768, eeg_arch='simple', pooling_strategy='multi',
                 use_lora=True, lora_r=16, lora_alpha=32, dropout=0.3,
                 global_eeg_dims=None, doc_encoder_type='bert',
                 freeze_doc_encoder=False):
        super().__init__()

        self.document_type = document_type
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        self.eeg_arch = eeg_arch
        self.doc_encoder_type = doc_encoder_type
        self.freeze_doc_encoder = freeze_doc_encoder

        print(f"Creating EEG-{document_type.upper()} alignment model")
        print(f"Pooling strategy: {pooling_strategy}")
        print(f"EEG architecture: {eeg_arch}")
        print(f"Dropout: {dropout}")
        print(f"Document encoder type: {doc_encoder_type.upper()}")
        print(f"Freeze document encoder: {'YES' if freeze_doc_encoder else 'NO'}")

        # ── EEG query encoder (always needed) ────────────────────────────────
        self.eeg_encoder = None
        self.eeg_projection = nn.Linear(hidden_dim, hidden_dim)
        self.eeg_proj_dropout = nn.Dropout(dropout)

        if global_eeg_dims is not None:
            max_words, max_time, max_channels = global_eeg_dims
            input_size = max_time * max_channels
            self.eeg_encoder = EEGEncoder(input_size, hidden_dim, eeg_arch, dropout)
            print(f"Initialized EEG query encoder with input size {input_size}")

        # ── Document encoder ─────────────────────────────────────────────────
        self.text_encoder = None           # BERT-based text encoder
        self.text_doc_eeg_encoder = None   # EEG-style encoder for text documents
        self.doc_token_embeddings = None   # Frozen BERT token embeddings (for text→eeg path)
        self.doc_eeg_encoder = None        # Separate frozen EEG doc encoder (for eeg→frozen path)

        if document_type == 'text':
            if doc_encoder_type == 'bert':
                # ── Existing path ──────────────────────────────────────────
                self.text_encoder = TextEncoder(
                    model_name=colbert_model_name,
                    hidden_dim=hidden_dim,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
                if freeze_doc_encoder:
                    print("Freezing TextEncoder (BERT) document encoder")
                    for param in self.text_encoder.parameters():
                        param.requires_grad = False

            elif doc_encoder_type == 'eeg':
                # ── New path: EEGEncoder architecture for text documents ──
                # Load frozen BERT token embeddings to convert token IDs → vectors.
                # This is the only BERT component retained; everything else is EEGEncoder.
                print(f"Loading frozen token embeddings from: {colbert_model_name}")
                try:
                    _bert = AutoModel.from_pretrained(colbert_model_name)
                except:
                    print(f"Falling back to bert-base-uncased for token embeddings")
                    _bert = AutoModel.from_pretrained('bert-base-uncased')

                self.doc_token_embeddings = _bert.embeddings.word_embeddings
                embed_dim = self.doc_token_embeddings.embedding_dim
                # Always freeze the embedding lookup — it is purely a converter
                for param in self.doc_token_embeddings.parameters():
                    param.requires_grad = False
                print(f"Frozen token embedding dim: {embed_dim}")
                del _bert  # Release the rest of the BERT model

                # EEGEncoder for text documents: input_size = embed_dim (typically 768)
                self.text_doc_eeg_encoder = EEGEncoder(
                    input_size=embed_dim,
                    hidden_dim=hidden_dim,
                    arch=eeg_arch,
                    dropout=dropout
                )
                print(f"Initialized EEG-style text document encoder "
                      f"(input_size={embed_dim}, arch={eeg_arch})")

                if freeze_doc_encoder:
                    print("Freezing EEG-style text document encoder")
                    for param in self.text_doc_eeg_encoder.parameters():
                        param.requires_grad = False
            else:
                raise ValueError(
                    f"doc_encoder_type must be 'bert' or 'eeg', got '{doc_encoder_type}'")

        elif document_type == 'eeg':
            if freeze_doc_encoder:
                # Create a *separate* document EEG encoder so we can freeze it
                # independently of the query encoder.
                if global_eeg_dims is not None:
                    max_words, max_time, max_channels = global_eeg_dims
                    input_size = max_time * max_channels
                    self.doc_eeg_encoder = EEGEncoder(
                        input_size, hidden_dim, eeg_arch, dropout)
                    print("Initialized separate (frozen) EEG document encoder")
                    for param in self.doc_eeg_encoder.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(
                f"document_type must be 'text' or 'eeg', got '{document_type}'")

        # ── Pooling components ────────────────────────────────────────────────
        if pooling_strategy == 'cls':
            self.eeg_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            if document_type == 'eeg':
                self.doc_eeg_cls_token = nn.Parameter(
                    torch.randn(1, 1, hidden_dim) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.cls_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.dropout = nn.Dropout(dropout)

        print(f"Model initialized for {document_type.upper()} documents "
              f"(doc_encoder={doc_encoder_type}, frozen={freeze_doc_encoder})")

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def set_tokenizer_vocab_size(self, vocab_size):
        """Compatibility shim — not required for pretrained models."""
        pass

    def _create_eeg_encoder(self, input_size, device):
        """Lazy-create query EEG encoder if global_eeg_dims was not available at init."""
        encoder = EEGEncoder(input_size, self.hidden_dim, self.eeg_arch)
        print(f"Created EEG encoder ({self.eeg_arch}) with input size {input_size}")
        return encoder.to(device)

    # ──────────────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────────────

    def encode_eeg(self, eeg_input, is_document=False):
        """
        Encode EEG sequences with specified pooling strategy.

        When is_document=True and a separate frozen doc_eeg_encoder exists
        (freeze_doc_encoder=True for EEG-EEG condition), routes through that
        encoder instead of the shared query encoder.
        """
        batch_size, num_words, time_samples, channels = eeg_input.shape
        input_size = time_samples * channels

        # Select encoder: separate frozen doc encoder if available, else shared
        if is_document and self.doc_eeg_encoder is not None:
            encoder = self.doc_eeg_encoder
        else:
            if self.eeg_encoder is None:
                self.eeg_encoder = self._create_eeg_encoder(
                    input_size, eeg_input.device)
            encoder = self.eeg_encoder

        # Encode EEG words
        if self.eeg_arch == 'transformer':
            eeg_reshaped = eeg_input.view(batch_size, num_words, input_size)
            word_representations = encoder(eeg_reshaped)
        else:
            eeg_flat = eeg_input.view(batch_size * num_words, input_size)
            encoded = encoder(eeg_flat)
            word_representations = encoded.view(batch_size, num_words, self.hidden_dim)

        # Project and regularise
        word_representations = self.eeg_projection(word_representations)
        word_representations = self.eeg_proj_dropout(word_representations)
        word_representations = self.dropout(word_representations)

        return self._apply_pooling(word_representations, eeg_input, is_document)

    def encode_text(self, input_ids, attention_mask):
        """
        Encode text documents.

        Routes to BERT TextEncoder (doc_encoder_type='bert') or to the
        EEG-style encoder fed with frozen token embeddings (doc_encoder_type='eeg').
        """
        if self.doc_encoder_type == 'bert':
            return self._encode_text_bert(input_ids, attention_mask)
        else:
            return self._encode_text_eeg(input_ids, attention_mask)

    def _encode_text_bert(self, input_ids, attention_mask):
        """Existing BERT / LoRA text encoding path."""
        if self.text_encoder is None:
            raise RuntimeError(
                "TextEncoder not initialised — model was not configured for BERT text encoding")
        text_representations = self.text_encoder(input_ids, attention_mask)
        text_representations = self.dropout(text_representations)
        return self._apply_text_pooling(text_representations, attention_mask)

    def _encode_text_eeg(self, input_ids, attention_mask):
        """
        EEG-style text encoding path.

        1. Convert token IDs → dense vectors via frozen BERT token embeddings.
        2. Zero-mask padding positions so the EEGEncoder's internal zero-detector
           correctly ignores them (exactly analogous to zero-padded EEG words).
        3. Pass through EEGEncoder (same architecture as query encoder).
        4. Project, dropout, pool — reusing the shared eeg_projection and
           _apply_text_pooling, minimising architectural differences.
        """
        if self.text_doc_eeg_encoder is None or self.doc_token_embeddings is None:
            raise RuntimeError(
                "EEG-style text encoder not initialised — set doc_encoder_type='eeg'")

        # Step 1: token IDs → embeddings  [batch, seq_len, embed_dim]
        with torch.no_grad():
            token_embeds = self.doc_token_embeddings(input_ids)

        # Step 2: zero out padding positions  [batch, seq_len, embed_dim]
        # Mirrors how EEG padding is handled (zero-padded words are masked out)
        token_embeds = token_embeds * attention_mask.unsqueeze(-1).float()

        # Step 3: EEGEncoder over token sequence  [batch, seq_len, hidden_dim]
        representations = self.text_doc_eeg_encoder(token_embeds)

        # Step 4: shared projection + dropout (same as EEG query path)
        representations = self.eeg_projection(representations)
        representations = self.eeg_proj_dropout(representations)
        representations = self.dropout(representations)

        # Step 5: pool using attention_mask (same _apply_text_pooling as BERT path)
        return self._apply_text_pooling(representations, attention_mask)

    # ──────────────────────────────────────────────────────────────────────────
    # Pooling
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_pooling(self, word_representations, eeg_input, is_document=False):
        """Apply pooling strategy to EEG representations."""
        batch_size = word_representations.size(0)

        if self.pooling_strategy == 'multi':
            multi_vectors = []
            for i in range(batch_size):
                word_mask = (eeg_input[i].abs().sum(dim=(1, 2)) > 0)
                active_positions = torch.where(word_mask)[0]
                if len(active_positions) > 0:
                    sample_vectors = word_representations[i, active_positions]
                else:
                    sample_vectors = torch.zeros(
                        1, self.hidden_dim, device=eeg_input.device)
                multi_vectors.append(sample_vectors)
            return multi_vectors

        elif self.pooling_strategy == 'max':
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
            cls_token = (self.doc_eeg_cls_token
                         if (is_document and self.document_type == 'eeg')
                         else self.eeg_cls_token)
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            cls_word_sequence = torch.cat([cls_tokens, word_representations], dim=1)

            word_mask = (eeg_input.abs().sum(dim=(2, 3)) > 0)
            cls_mask = torch.ones(batch_size, 1, device=eeg_input.device)
            full_mask = torch.cat([cls_mask, word_mask], dim=1)

            attended_sequence = self.cls_transformer(
                cls_word_sequence,
                src_key_padding_mask=~full_mask.bool()
            )
            return attended_sequence[:, 0:1, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def _apply_text_pooling(self, text_representations, attention_mask):
        """Apply pooling strategy to text representations (BERT or EEG-style)."""
        batch_size = text_representations.size(0)

        if self.pooling_strategy == 'multi':
            multi_vectors = []
            for i in range(batch_size):
                valid_positions = torch.where(attention_mask[i] == 1)[0][1:]  # skip [CLS]
                if len(valid_positions) > 0:
                    sample_vectors = text_representations[i, valid_positions]
                else:
                    sample_vectors = torch.zeros(
                        1, text_representations.size(-1),
                        device=text_representations.device)
                multi_vectors.append(sample_vectors)
            return multi_vectors

        elif self.pooling_strategy == 'max':
            pooled_vectors = []
            for i in range(batch_size):
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_reps = text_representations[i, valid_mask]
                    max_vector = torch.max(valid_reps, dim=0)[0]
                else:
                    max_vector = torch.zeros(
                        text_representations.size(-1),
                        device=text_representations.device)
                pooled_vectors.append(max_vector.unsqueeze(0))
            return torch.stack(pooled_vectors)

        elif self.pooling_strategy == 'mean':
            pooled_vectors = []
            for i in range(batch_size):
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    valid_reps = text_representations[i, valid_mask]
                    mean_vector = torch.mean(valid_reps, dim=0)
                else:
                    mean_vector = torch.zeros(
                        text_representations.size(-1),
                        device=text_representations.device)
                pooled_vectors.append(mean_vector.unsqueeze(0))
            return torch.stack(pooled_vectors)

        elif self.pooling_strategy == 'cls':
            return text_representations[:, 0:1, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, batch):
        """
        Forward pass for either EEG-EEG or EEG-Text alignment.
        """
        query_eegs = batch['query_eegs']
        document_type = batch['document_type']

        # Encode queries (always EEG)
        query_vectors = self.encode_eeg(query_eegs, is_document=False)

        # Encode documents
        if document_type == 'eeg':
            doc_eegs = batch['doc_eegs']
            doc_vectors = self.encode_eeg(doc_eegs, is_document=True)
        elif document_type == 'text':
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


# ──────────────────────────────────────────────────────────────────────────────
# Similarity functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_similarity(query_vectors, doc_vectors, pooling_strategy, temperature=1.0):
    """Compute similarity based on pooling strategy."""
    if pooling_strategy == 'multi':
        return compute_multi_vector_similarity(query_vectors, doc_vectors, temperature)
    else:
        return compute_single_vector_similarity(query_vectors, doc_vectors, temperature)


def compute_multi_vector_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute ColBERT-style MaxSim similarity for multi-vectors."""
    similarities = []
    for i in range(len(query_vectors)):
        q_vecs = query_vectors[i]
        d_vecs = doc_vectors[i]

        q_vecs = F.normalize(q_vecs, p=2, dim=1)
        d_vecs = F.normalize(d_vecs, p=2, dim=1)

        q_nonzero = q_vecs[q_vecs.norm(dim=1) > 1e-6]
        d_nonzero = d_vecs[d_vecs.norm(dim=1) > 1e-6]

        if len(q_nonzero) == 0 or len(d_nonzero) == 0:
            similarities.append(torch.tensor(0.0, device=q_vecs.device))
            continue

        sim_matrix = torch.matmul(q_nonzero, d_nonzero.t())
        max_sims = sim_matrix.max(dim=1)[0]
        sim = max_sims.sum()
        similarities.append(sim)

    return torch.stack(similarities) / temperature


def compute_single_vector_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute cosine similarity for single vectors (cls, max, mean pooling)."""
    if isinstance(query_vectors, list):
        query_vectors = torch.stack([qv for qv in query_vectors])
    if isinstance(doc_vectors, list):
        doc_vectors = torch.stack([dv for dv in doc_vectors])

    batch_size = query_vectors.size(0)
    similarities = []

    for i in range(batch_size):
        q_vec = query_vectors[i].squeeze()
        d_vec = doc_vectors[i].squeeze()

        q_norm = F.normalize(q_vec, p=2, dim=0)
        d_norm = F.normalize(d_vec, p=2, dim=0)

        sim = torch.dot(q_norm, d_norm)
        similarities.append(sim)

    return torch.stack(similarities) / temperature


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_alignment_model(document_type='text', colbert_model_name='bert-base-uncased',
                           hidden_dim=768, eeg_arch='simple', pooling_strategy='multi',
                           global_eeg_dims=None, device='cuda', dropout=0.3,
                           doc_encoder_type='bert', freeze_doc_encoder=False):
    model = EEGAlignmentModel(
        document_type=document_type,
        colbert_model_name=colbert_model_name,
        hidden_dim=hidden_dim,
        eeg_arch=eeg_arch,
        pooling_strategy=pooling_strategy,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        dropout=dropout,
        global_eeg_dims=global_eeg_dims,
        doc_encoder_type=doc_encoder_type,
        freeze_doc_encoder=freeze_doc_encoder
    )
    return model.to(device)