#!/usr/bin/env python3
"""
Memory-Efficient Training for Brain Passage Retrieval with DYNAMIC MULTI-MASKING VALIDATION
Uses single dataloader that changes masking probability on-the-fly
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from mv_models import compute_similarity
from mv_dataloader import create_positive_negative_pairs


# ==========================================
# CROSS-ENCODER SPECIFIC FUNCTIONS
# ==========================================

def compute_bce_loss(scores, labels):
    """Compute BCE loss for cross-encoder"""
    return F.binary_cross_entropy_with_logits(scores.squeeze(), labels.float())


def compute_dual_contrastive_loss(temporal_vectors, spatial_vectors, doc_vectors,
                                  pooling_strategy, temperature=0.07,
                                  lambda_temporal=1.0, lambda_spatial=1.0):
    """Compute dual contrastive loss for temporal and spatial components"""

    # Compute separate losses
    temporal_loss, temporal_sims = compute_contrastive_loss(
        temporal_vectors, doc_vectors, pooling_strategy, temperature
    )

    spatial_loss, spatial_sims = compute_contrastive_loss(
        spatial_vectors, doc_vectors, pooling_strategy, temperature
    )

    # Combine losses
    total_loss = lambda_temporal * temporal_loss + lambda_spatial * spatial_loss

    return total_loss, temporal_loss, spatial_loss, temporal_sims, spatial_sims


def cross_encoder_train_step(model, batch, optimizer, device, step_num, debug=False):
    """Training step for cross-encoder"""

    # Move to device
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}

    # Create positive pairs
    positive_labels, negative_pairs, negative_labels = create_positive_negative_pairs(batch)
    positive_labels = positive_labels.to(device)
    negative_labels = negative_labels.to(device)

    if debug:
        query_type = model.query_type
        print(f"[DEBUG] Cross-encoder training step {step_num} (query_type: {query_type})")
        print(f"  Positive pairs: {len(positive_labels)}")
        print(f"  Negative pairs: {len(negative_labels)}")

    # Forward pass for positive pairs
    positive_scores = model(eeg_queries, text_queries, docs)
    positive_loss = compute_bce_loss(positive_scores, positive_labels)

    # Forward pass for negative pairs
    if len(negative_pairs) > 0:
        neg_eeg = torch.stack([eeg_queries[pair['eeg_idx']] for pair in negative_pairs])
        neg_text_queries = {
            'input_ids': torch.stack([text_queries['input_ids'][pair['eeg_idx']] for pair in negative_pairs]),
            'attention_mask': torch.stack([text_queries['attention_mask'][pair['eeg_idx']] for pair in negative_pairs])
        }
        neg_docs = {
            'input_ids': torch.stack([docs['input_ids'][pair['doc_idx']] for pair in negative_pairs]),
            'attention_mask': torch.stack([docs['attention_mask'][pair['doc_idx']] for pair in negative_pairs])
        }

        negative_scores = model(neg_eeg, neg_text_queries, neg_docs)
        negative_loss = compute_bce_loss(negative_scores, negative_labels)

        total_loss = positive_loss + negative_loss

        # Compute negative accuracy too
        neg_acc = ((torch.sigmoid(negative_scores.squeeze()) > 0.5) == negative_labels.to(device)).float().mean()
    else:
        total_loss = positive_loss
        neg_acc = torch.tensor(0.0)

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Log metrics
    pos_acc = ((torch.sigmoid(positive_scores.squeeze()) > 0.5) == positive_labels).float().mean()

    wandb.log({
        'train/loss': total_loss.item(),
        'train/positive_loss': positive_loss.item(),
        'train/positive_accuracy': pos_acc.item(),
        'train/negative_accuracy': neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc,
        'train/grad_norm': grad_norm.item(),
        'train/step': step_num
    })

    if debug:
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Positive accuracy: {pos_acc.item():.4f}")
        print(f"  Negative accuracy: {neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc:.4f}")

    return total_loss.item(), {'accuracy': pos_acc.item(), 'negative_accuracy': neg_acc.item() if torch.is_tensor(
        neg_acc) else neg_acc}, grad_norm.item()


def cross_encoder_validation_step(model, batch, device):
    """Validation step for cross-encoder"""

    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}

    # Create positive pairs for validation
    positive_labels, negative_pairs, negative_labels = create_positive_negative_pairs(batch)
    positive_labels = positive_labels.to(device)
    negative_labels = negative_labels.to(device)

    with torch.no_grad():
        # Forward pass for positive pairs
        positive_scores = model(eeg_queries, text_queries, docs)
        positive_loss = compute_bce_loss(positive_scores, positive_labels)

        # Compute accuracy
        pos_acc = ((torch.sigmoid(positive_scores.squeeze()) > 0.5) == positive_labels).float().mean()

        # For negative pairs if any
        total_loss = positive_loss
        neg_acc = torch.tensor(0.0)
        if len(negative_pairs) > 0:
            neg_eeg = torch.stack([eeg_queries[pair['eeg_idx']] for pair in negative_pairs])
            neg_text_queries = {
                'input_ids': torch.stack([text_queries['input_ids'][pair['eeg_idx']] for pair in negative_pairs]),
                'attention_mask': torch.stack(
                    [text_queries['attention_mask'][pair['eeg_idx']] for pair in negative_pairs])
            }
            neg_docs = {
                'input_ids': torch.stack([docs['input_ids'][pair['doc_idx']] for pair in negative_pairs]),
                'attention_mask': torch.stack([docs['attention_mask'][pair['doc_idx']] for pair in negative_pairs])
            }
            negative_scores = model(neg_eeg, neg_text_queries, neg_docs)
            negative_loss = compute_bce_loss(negative_scores, negative_labels)
            total_loss = positive_loss + negative_loss
            neg_acc = ((torch.sigmoid(negative_scores.squeeze()) > 0.5) == negative_labels.to(device)).float().mean()

    return total_loss.item(), {'accuracy': pos_acc.item(),
                               'negative_accuracy': neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc}


# ==========================================
# DUAL-ENCODER SPECIFIC FUNCTIONS
# ==========================================

def compute_contrastive_loss(query_vectors, doc_vectors, pooling_strategy, temperature=0.07):
    """Compute contrastive loss for retrieval - works with any query type"""

    # Handle both list and tensor returns - check actual type instead of pooling strategy
    if isinstance(query_vectors, list):
        batch_size = len(query_vectors)
        device = query_vectors[0].device
    else:
        batch_size = query_vectors.size(0)
        device = query_vectors.device

    # Compute similarities between queries and documents
    query_to_doc_sims = []
    for i in range(batch_size):
        if isinstance(query_vectors, list):
            query_i = query_vectors[i]
            if isinstance(doc_vectors, list):
                doc_i = doc_vectors[i]
            else:
                doc_i = doc_vectors[i:i + 1]
        else:
            query_i = query_vectors[i:i + 1]
            if isinstance(doc_vectors, list):
                doc_i = doc_vectors[i]
            else:
                doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_to_doc_sims.append(sim[0])

    similarities = torch.stack(query_to_doc_sims)

    # Create contrastive loss using in-batch negatives
    logits = torch.zeros(batch_size, batch_size, device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            if isinstance(query_vectors, list):
                query_i = query_vectors[i]
            else:
                query_i = query_vectors[i:i + 1]

            if isinstance(doc_vectors, list):
                doc_j = doc_vectors[j]
            else:
                doc_j = doc_vectors[j:j + 1]

            sim = compute_similarity([query_i], [doc_j], pooling_strategy, temperature=1.0)
            logits[i, j] = sim[0] / temperature

    # Labels: positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss, similarities


def compute_alignment_metrics(query_vectors, doc_vectors, pooling_strategy, query_type='eeg'):
    """Compute alignment metrics between representations"""

    # Handle both list and tensor returns - check actual type instead of pooling strategy
    if isinstance(query_vectors, list):
        batch_size = len(query_vectors)
    else:
        batch_size = query_vectors.size(0)

    # Query-to-doc similarities (always computed)
    query_doc_sims = []
    for i in range(batch_size):
        if isinstance(query_vectors, list):
            query_i = query_vectors[i]
            if isinstance(doc_vectors, list):
                doc_i = doc_vectors[i]
            else:
                doc_i = doc_vectors[i:i + 1]
        else:
            query_i = query_vectors[i:i + 1]
            if isinstance(doc_vectors, list):
                doc_i = doc_vectors[i]
            else:
                doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_doc_sims.append(sim[0].item())

    metrics = {
        'query_doc_similarity': np.mean(query_doc_sims),
        'query_doc_similarity_std': np.std(query_doc_sims),
    }

    # Add query type specific names for compatibility
    if query_type == 'eeg':
        metrics['eeg_query_similarity'] = metrics['query_doc_similarity']
        metrics['eeg_query_similarity_std'] = metrics['query_doc_similarity_std']
        metrics['eeg_doc_similarity'] = metrics['query_doc_similarity']
        metrics['eeg_doc_similarity_std'] = metrics['query_doc_similarity_std']
    else:  # text
        metrics['text_query_similarity'] = metrics['query_doc_similarity']
        metrics['text_query_similarity_std'] = metrics['query_doc_similarity_std']
        metrics['text_doc_similarity'] = metrics['query_doc_similarity']
        metrics['text_doc_similarity_std'] = metrics['query_doc_similarity_std']

    return metrics


def dual_encoder_train_step(model, batch, optimizer, device, step_num, debug=False):
    """Training step for dual encoder"""

    # Move batch to device
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}
    eeg_mv_masks = batch['eeg_mv_masks'].to(device)

    if debug:
        print(f"[DEBUG] Dual encoder training step {step_num} (query_type: {model.query_type})")
        print(f"  EEG queries: {eeg_queries.shape}")
        print(f"  Text query IDs: {text_queries['input_ids'].shape}")
        print(f"  Doc IDs: {docs['input_ids'].shape}")
        print(f"  Pooling strategy: {model.pooling_strategy}")

    # Forward pass
    outputs = model(eeg_queries, text_queries, docs, eeg_mv_masks)

    # Check if using temporal-spatial decomposition
    if model.use_temporal_spatial_decomp and model.query_type == 'eeg':
        if hasattr(model, 'use_dual_loss') and model.use_dual_loss:
            # Dual loss path
            loss, temporal_loss, spatial_loss, temporal_sims, spatial_sims = compute_dual_contrastive_loss(
                outputs['temporal_vectors'], outputs['spatial_vectors'],
                outputs['doc_vectors'], model.pooling_strategy,
                lambda_temporal=getattr(model, 'lambda_temporal', 1.0),
                lambda_spatial=getattr(model, 'lambda_spatial', 1.0)
            )

            # Extended metrics
            metrics = {
                'query_doc_similarity': (temporal_sims.mean().item() + spatial_sims.mean().item()) / 2,
                'temporal_similarity': temporal_sims.mean().item(),
                'spatial_similarity': spatial_sims.mean().item(),
                'temporal_loss': temporal_loss.item(),
                'spatial_loss': spatial_loss.item()
            }

            # Add query type specific names for compatibility with logging
            if model.query_type == 'eeg':
                metrics['eeg_query_similarity'] = metrics['query_doc_similarity']
                metrics['eeg_doc_similarity'] = metrics['query_doc_similarity']
                metrics['eeg_query_similarity_std'] = 0.0  # Not computed in dual loss
                metrics['eeg_doc_similarity_std'] = 0.0
            else:  # text
                metrics['text_query_similarity'] = metrics['query_doc_similarity']
                metrics['text_doc_similarity'] = metrics['query_doc_similarity']
                metrics['text_query_similarity_std'] = 0.0
                metrics['text_doc_similarity_std'] = 0.0
        else:
            # Single loss: concatenate temporal and spatial
            combined_vectors = []
            for i in range(len(outputs['temporal_vectors'])):
                temp_vec = outputs['temporal_vectors'][i]
                spat_vec = outputs['spatial_vectors'][i]
                combined = torch.cat([temp_vec, spat_vec], dim=-1)  # [batch_vectors, 1536]
                # Project back to original dimension
                projected = model.combined_projection(combined)  # [batch_vectors, 768]
                combined_vectors.append(projected)

            loss, query_sims = compute_contrastive_loss(
                combined_vectors, outputs['doc_vectors'], model.pooling_strategy
            )

            metrics = compute_alignment_metrics(
                combined_vectors, outputs['doc_vectors'],
                model.pooling_strategy, model.query_type
            )
    else:
        # Original path - no changes
        loss, query_sims = compute_contrastive_loss(
            outputs['query_vectors'], outputs['doc_vectors'], model.pooling_strategy
        )

        metrics = compute_alignment_metrics(
            outputs['query_vectors'], outputs['doc_vectors'],
            model.pooling_strategy, model.query_type
        )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Log to wandb - conditional on query type
    log_dict = {
        'train/loss': loss.item(),
        'train/query_doc_similarity': metrics['query_doc_similarity'],
        'train/grad_norm': grad_norm.item(),
        'train/step': step_num
    }

    # Add query-type specific metrics
    if model.query_type == 'eeg':
        log_dict.update({
            'train/eeg_doc_similarity': metrics['eeg_doc_similarity'],
            'train/eeg_query_similarity': metrics['eeg_query_similarity'],
        })
    else:  # text
        log_dict.update({
            'train/text_doc_similarity': metrics['text_doc_similarity'],
            'train/text_query_similarity': metrics['text_query_similarity'],
        })

    wandb.log(log_dict)

    if debug:
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Query-Doc similarity: {metrics['query_doc_similarity']:.4f}")
        print(f"  Grad norm: {grad_norm.item():.4f}")
        meta = batch['metadata'][0]
        print(f"  Sample query: '{meta['query_text'][:50]}...'")
        print(f"  Sample doc: '{meta['document_text'][:50]}...'")
        print(f"  Participant: {meta['participant_id']}")

    # Return with query-type appropriate metric name
    main_metric_key = 'eeg_query_similarity' if model.query_type == 'eeg' else 'text_query_similarity'
    return loss.item(), metrics, grad_norm.item()


def dual_encoder_validation_step(model, batch, device):
    """Validation step for dual encoder"""

    # Move batch to device
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}
    eeg_mv_masks = batch['eeg_mv_masks'].to(device)

    # Forward pass (no gradients)
    with torch.no_grad():
        outputs = model(eeg_queries, text_queries, docs, eeg_mv_masks)

        # Check if using temporal-spatial decomposition (same as training)
        if model.use_temporal_spatial_decomp and model.query_type == 'eeg':
            if hasattr(model, 'use_dual_loss') and model.use_dual_loss:
                # Dual loss path
                loss, temporal_loss, spatial_loss, temporal_sims, spatial_sims = compute_dual_contrastive_loss(
                    outputs['temporal_vectors'], outputs['spatial_vectors'],
                    outputs['doc_vectors'], model.pooling_strategy,
                    lambda_temporal=getattr(model, 'lambda_temporal', 1.0),
                    lambda_spatial=getattr(model, 'lambda_spatial', 1.0)
                )

                # Extended metrics
                metrics = {
                    'query_doc_similarity': (temporal_sims.mean().item() + spatial_sims.mean().item()) / 2,
                    'temporal_similarity': temporal_sims.mean().item(),
                    'spatial_similarity': spatial_sims.mean().item(),
                    'temporal_loss': temporal_loss.item(),
                    'spatial_loss': spatial_loss.item()
                }

                # Add query type specific names for compatibility with logging
                if model.query_type == 'eeg':
                    metrics['eeg_query_similarity'] = metrics['query_doc_similarity']
                    metrics['eeg_doc_similarity'] = metrics['query_doc_similarity']
                    metrics['eeg_query_similarity_std'] = 0.0  # Not computed in dual loss
                    metrics['eeg_doc_similarity_std'] = 0.0
                else:  # text
                    metrics['text_query_similarity'] = metrics['query_doc_similarity']
                    metrics['text_doc_similarity'] = metrics['query_doc_similarity']
                    metrics['text_query_similarity_std'] = 0.0
                    metrics['text_doc_similarity_std'] = 0.0


            else:

                # Single loss: concatenate temporal and spatial

                combined_vectors = []
                for i in range(len(outputs['temporal_vectors'])):
                    temp_vec = outputs['temporal_vectors'][i]
                    spat_vec = outputs['spatial_vectors'][i]
                    combined = torch.cat([temp_vec, spat_vec], dim=-1)  # [batch_vectors, 1536]
                    # Project back to original dimension
                    projected = model.combined_projection(combined)  # [batch_vectors, 768]
                    combined_vectors.append(projected)

                loss, query_sims = compute_contrastive_loss(
                    combined_vectors, outputs['doc_vectors'], model.pooling_strategy
                )

                metrics = compute_alignment_metrics(
                    combined_vectors, outputs['doc_vectors'],
                    model.pooling_strategy, model.query_type
                )
        else:
            # Original path - no changes
            loss, query_sims = compute_contrastive_loss(
                outputs['query_vectors'],
                outputs['doc_vectors'],
                model.pooling_strategy
            )

            metrics = compute_alignment_metrics(
                outputs['query_vectors'],
                outputs['doc_vectors'],
                model.pooling_strategy,
                model.query_type
            )

    return loss.item(), metrics


# ==========================================
# DISPATCH FUNCTIONS
# ==========================================

def train_step(model, batch, optimizer, device, step_num, debug=False):
    """Single training step - dispatches to dual or cross encoder"""

    if hasattr(model, 'cross_attention'):  # Cross-encoder
        return cross_encoder_train_step(model, batch, optimizer, device, step_num, debug)
    else:  # Dual encoder
        return dual_encoder_train_step(model, batch, optimizer, device, step_num, debug)


def validation_step(model, batch, device):
    """Single validation step - dispatches based on model type"""

    if hasattr(model, 'cross_attention'):  # Cross-encoder
        return cross_encoder_validation_step(model, batch, device)
    else:  # Dual encoder
        return dual_encoder_validation_step(model, batch, device)


# ==========================================
# RANKING VALIDATION (DUAL ENCODER ONLY)
# ==========================================

def build_document_database(val_dataloader):
    """Build unique document database for ranking evaluation"""

    print("Building document database for ranking evaluation...")

    unique_docs = {}  # text -> doc_info
    query_to_doc_mapping = {}  # query_idx -> unique_doc_idx
    query_idx = 0

    for batch in val_dataloader:
        for sample_idx, metadata in enumerate(batch['metadata']):
            doc_text = metadata['document_text'].strip()

            if doc_text:
                # Add to unique docs if not already present
                if doc_text not in unique_docs:
                    unique_doc_idx = len(unique_docs)
                    unique_docs[doc_text] = {
                        'idx': unique_doc_idx,
                        'text': doc_text,
                        'input_ids': batch['docs']['input_ids'][sample_idx].clone(),
                        'attention_mask': batch['docs']['attention_mask'][sample_idx].clone()
                    }
                else:
                    unique_doc_idx = unique_docs[doc_text]['idx']

                query_to_doc_mapping[query_idx] = unique_doc_idx
            query_idx += 1

    # Convert to list for easier batch processing
    doc_list = [None] * len(unique_docs)
    for text, doc_info in unique_docs.items():
        doc_list[doc_info['idx']] = doc_info

    print(f"Found {len(doc_list)} unique documents for {len(query_to_doc_mapping)} queries")
    return doc_list, query_to_doc_mapping


def generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size=100, seed=42):
    """Generate consistent document subsets for fair comparison between encoders"""

    print(f"Generating consistent document subsets (subset_size={subset_size}, seed={seed})...")

    # Set seed for reproducible subsets
    random.seed(seed)

    query_subsets = {}  # query_idx -> list of doc_indices

    for query_idx, correct_doc_idx in query_to_doc_mapping.items():
        # Always include correct document
        doc_subset_indices = [correct_doc_idx]

        # Sample random negatives
        negative_candidates = [i for i in range(len(doc_list)) if i != correct_doc_idx]
        if negative_candidates:
            random_negatives = random.sample(negative_candidates,
                                             min(subset_size - 1, len(negative_candidates)))
            doc_subset_indices.extend(random_negatives)

        query_subsets[query_idx] = doc_subset_indices

    print(f"Generated {len(query_subsets)} consistent subsets")
    return query_subsets


def batch_encode_documents(model, doc_list, device, batch_size=32):
    """Encode all unique documents in batches"""

    print(f"Batch encoding {len(doc_list)} unique documents (batch_size={batch_size})...")

    model.eval()
    all_doc_vectors = []

    with torch.no_grad():
        for i in range(0, len(doc_list), batch_size):
            batch_docs = doc_list[i:i + batch_size]

            # Stack document tensors for batch processing
            doc_input_ids = torch.stack([doc['input_ids'] for doc in batch_docs]).to(device)
            doc_attention_mask = torch.stack([doc['attention_mask'] for doc in batch_docs]).to(device)

            # Batch encode documents
            batch_doc_vectors = model.encode_text(doc_input_ids, doc_attention_mask)

            # Handle different pooling strategies
            if isinstance(batch_doc_vectors, list):
                # Multi-vector pooling
                all_doc_vectors.extend(batch_doc_vectors)
            else:
                # CLS pooling - split batch back to individual documents
                for j in range(len(batch_docs)):
                    all_doc_vectors.append(batch_doc_vectors[j:j + 1])

            # Progress update
            print(f"  Encoded {min(i + batch_size, len(doc_list))}/{len(doc_list)} documents")

    print(f"Document encoding complete. Got {len(all_doc_vectors)} document representations")
    return all_doc_vectors


def batch_similarity_computation(eeg_vectors, doc_vectors_list, pooling_strategy):
    """Compute similarities between one EEG query and all documents"""

    similarities = []

    if pooling_strategy == 'multi':
        # Multi-vector similarity computation
        for doc_vectors in doc_vectors_list:
            sim = compute_similarity([eeg_vectors], [doc_vectors], pooling_strategy, temperature=1.0)
            similarities.append(sim[0].item())

    elif pooling_strategy == 'cls':
        # CLS similarity - can be batched efficiently
        if len(doc_vectors_list) > 0:
            # Stack all document vectors for batch cosine similarity
            doc_stack = torch.stack([doc_vec[0] for doc_vec in doc_vectors_list])

            # Create batch input for similarity computation
            eeg_batch = eeg_vectors.repeat(len(doc_vectors_list), 1, 1)

            # Batch cosine similarity
            batch_similarities = compute_similarity(eeg_batch, doc_stack, pooling_strategy, temperature=1.0)
            similarities = batch_similarities.tolist()

    return similarities


def rank_documents_for_query(model, eeg_query, eeg_mv_mask, doc_vectors_list, pooling_strategy):
    """Rank pre-encoded documents for a single EEG query"""

    model.eval()

    with torch.no_grad():
        # Encode EEG query
        eeg_vectors = model.encode_eeg(eeg_query, eeg_mv_mask)

        # Handle different return types
        if isinstance(eeg_vectors, list):
            eeg_vectors = eeg_vectors[0]  # Get first (and only) element for single query
        else:
            eeg_vectors = eeg_vectors[0:1]  # Keep as batch of 1

        # Compute similarities efficiently
        doc_scores = batch_similarity_computation(eeg_vectors, doc_vectors_list, pooling_strategy)

    # Sort documents by score (descending)
    doc_indices_and_scores = list(enumerate(doc_scores))
    doc_indices_and_scores.sort(key=lambda x: x[1], reverse=True)

    ranked_doc_indices = [idx for idx, score in doc_indices_and_scores]
    ranked_scores = [score for idx, score in doc_indices_and_scores]

    return ranked_doc_indices, ranked_scores


def collect_eeg_queries(val_dataloader, device):
    """Collect all EEG queries and text queries from validation set"""

    print("Collecting EEG and text queries from validation set...")

    queries = []  # Changed name to be more general
    query_idx = 0

    for batch in val_dataloader:
        batch_size = batch['eeg_queries'].size(0)
        for sample_idx in range(batch_size):
            eeg_query = batch['eeg_queries'][sample_idx:sample_idx + 1].to(device)
            eeg_mv_mask = batch['eeg_mv_masks'][sample_idx:sample_idx + 1].to(device)
            text_query = {
                'input_ids': batch['text_queries']['input_ids'][sample_idx:sample_idx + 1].to(device),
                'attention_mask': batch['text_queries']['attention_mask'][sample_idx:sample_idx + 1].to(device)
            }
            queries.append((eeg_query, eeg_mv_mask, text_query))
            query_idx += 1

    print(f"Collected {len(queries)} query sets (EEG + text)")
    return queries


def compute_ranking_metrics(ranked_doc_indices, correct_doc_idx, k_values=[1, 5, 10, 20]):
    """
    Compute comprehensive ranking metrics for a single query

    Args:
        ranked_doc_indices: List of document indices ranked by relevance (best first)
        correct_doc_idx: Index of the correct/relevant document
        k_values: List of K values for Hit@K, Precision@K, Recall@K

    Returns:
        dict: Comprehensive ranking metrics
    """
    metrics = {}

    # Find rank of correct document (1-indexed)
    try:
        correct_rank = ranked_doc_indices.index(correct_doc_idx) + 1
    except ValueError:
        correct_rank = len(ranked_doc_indices) + 1  # Not found

    # Mean Reciprocal Rank
    metrics['rr'] = 1.0 / correct_rank if correct_rank <= len(ranked_doc_indices) else 0.0
    metrics['rank_of_correct'] = correct_rank

    # Hit@K, Precision@K, and Recall@K for all K values
    for k in k_values:
        if k <= len(ranked_doc_indices):
            # Hit@K: is correct doc in top-k?
            hit_at_k = 1.0 if correct_rank <= k else 0.0
            metrics[f'hit_at_{k}'] = hit_at_k

            # Precision@K: for single relevant doc, precision = hit@k / k
            precision_at_k = hit_at_k / k
            metrics[f'precision_at_{k}'] = precision_at_k

            # Recall@K: for single relevant doc, recall = hit@k (since there's only 1 relevant doc)
            recall_at_k = hit_at_k
            metrics[f'recall_at_{k}'] = recall_at_k
        else:
            metrics[f'hit_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0

    return metrics

def perform_dual_encoder_test_ranking_at_masking_level(model, test_dataloader, device, masking_level):
    """Perform ranking evaluation at specific masking level for dual encoder"""

    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        return {}

    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    if not all_metrics:
        return {}

    # Aggregate metrics
    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/masking_level': masking_level,
        'test_ranking/main/query_type': model.query_type
    })

    return test_metrics


def perform_cross_encoder_test_ranking_at_masking_level(model, test_dataloader, device, masking_level):
    """Perform ranking evaluation at specific masking level for cross encoder"""

    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        return {}

    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    if not all_metrics:
        return {}

    # Aggregate metrics
    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/masking_level': masking_level,
        'test_ranking/main/encoder_type': 'cross'
    })

    return test_metrics

def calculate_masking_performance_trends(all_metrics, test_masking_levels):
    """Calculate performance trends across masking levels"""

    trend_metrics = {}

    # Extract MRR and Hit@K across all masking levels
    mrr_values = []
    hit1_values = []
    hit10_values = []
    hit20_values = []

    for masking_level in test_masking_levels:
        mrr_key = f'test_ranking/masking_{masking_level}/rr'
        hit1_key = f'test_ranking/masking_{masking_level}/hit_at_1'
        hit10_key = f'test_ranking/masking_{masking_level}/hit_at_10'
        hit20_key = f'test_ranking/masking_{masking_level}/hit_at_20'

        if mrr_key in all_metrics:
            mrr_values.append(all_metrics[mrr_key])
        if hit1_key in all_metrics:
            hit1_values.append(all_metrics[hit1_key])
        if hit10_key in all_metrics:
            hit10_values.append(all_metrics[hit10_key])
        if hit20_key in all_metrics:
            hit20_values.append(all_metrics[hit20_key])

    if mrr_values:
        trend_metrics.update({
            'test_ranking/trends/mrr_mean': np.mean(mrr_values),
            'test_ranking/trends/mrr_std': np.std(mrr_values),
            'test_ranking/trends/mrr_min': np.min(mrr_values),
            'test_ranking/trends/mrr_max': np.max(mrr_values),
            'test_ranking/trends/mrr_range': np.max(mrr_values) - np.min(mrr_values)
        })

    if hit1_values:
        trend_metrics.update({
            'test_ranking/trends/hit1_mean': np.mean(hit1_values),
            'test_ranking/trends/hit1_std': np.std(hit1_values),
            'test_ranking/trends/hit1_min': np.min(hit1_values),
            'test_ranking/trends/hit1_max': np.max(hit1_values),
            'test_ranking/trends/hit1_range': np.max(hit1_values) - np.min(hit1_values)
        })

    if hit10_values:
        trend_metrics.update({
            'test_ranking/trends/hit10_mean': np.mean(hit10_values),
            'test_ranking/trends/hit10_std': np.std(hit10_values),
            'test_ranking/trends/hit10_min': np.min(hit10_values),
            'test_ranking/trends/hit10_max': np.max(hit10_values),
            'test_ranking/trends/hit10_range': np.max(hit10_values) - np.min(hit10_values)
        })

    if hit20_values:
        trend_metrics.update({
            'test_ranking/trends/hit20_mean': np.mean(hit20_values),
            'test_ranking/trends/hit20_std': np.std(hit20_values),
            'test_ranking/trends/hit20_min': np.min(hit20_values),
            'test_ranking/trends/hit20_max': np.max(hit20_values),
            'test_ranking/trends/hit20_range': np.max(hit20_values) - np.min(hit20_values)
        })

    return trend_metrics


def print_multi_masking_test_summary(all_metrics, test_masking_levels, primary_masking_level):
    """Print comprehensive multi-masking test summary"""

    print(f"\n=== MULTI-MASKING TEST EVALUATION SUMMARY ===")
    print(f"Test masking levels: {test_masking_levels}%")
    print(f"Primary masking level: {primary_masking_level}%")

    print(f"\nPerformance across masking levels:")
    print(f"{'Masking':>8} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'Hit@20':>8}")
    print(f"{'-' * 55}")

    for masking_level in test_masking_levels:
        mrr = all_metrics.get(f'test_ranking/masking_{masking_level}/rr', 0)
        hit1 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_1', 0)
        hit5 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_5', 0)
        hit10 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_10', 0)
        hit20 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_20', 0)

        primary_marker = " *" if masking_level == primary_masking_level else ""
        print(f"{masking_level:>6}%{primary_marker:>2} {mrr:>8.4f} {hit1:>8.4f} {hit5:>8.4f} {hit10:>8.4f} {hit20:>8.4f}")

    # Noise baseline comparison
    noise_mrr = all_metrics.get('test_ranking/noise/rr', 0)
    noise_hit1 = all_metrics.get('test_ranking/noise/hit_at_1', 0)

    if noise_mrr > 0:
        primary_mrr = all_metrics.get(f'test_ranking/masking_{primary_masking_level}/rr', 0)
        primary_hit1 = all_metrics.get(f'test_ranking/masking_{primary_masking_level}/hit_at_1', 0)

        print(f"\nNoise Baseline: MRR {noise_mrr:.4f}, Hit@1 {noise_hit1:.4f}")
        print(f"Primary vs Noise: MRR +{primary_mrr - noise_mrr:.4f}, Hit@1 +{primary_hit1 - noise_hit1:.4f}")

    # Performance trends
    if 'test_ranking/trends/mrr_range' in all_metrics:
        mrr_range = all_metrics['test_ranking/trends/mrr_range']
        hit1_range = all_metrics['test_ranking/trends/hit1_range']
        print(f"\nRobustness Analysis:")
        print(f"  MRR range across masking levels: {mrr_range:.4f}")
        print(f"  Hit@1 range across masking levels: {hit1_range:.4f}")

    print(f"\nAll multi-masking test results logged to wandb under:")
    for masking_level in test_masking_levels:
        print(f"  - test_ranking/masking_{masking_level}/ (performance at {masking_level}%)")
    print(f"  - test_ranking/trends/ (cross-masking analysis)")
    print(f"  - test_ranking/noise/ (noise baseline)")
    print(f"  - test_ranking/summary/ (summary statistics)")
    print("=" * 60)

def test_model(model, test_dataloader, device, debug=False,
               test_masking_levels=[0, 25, 50, 75, 90, 100],
               primary_masking_level=90):
    """
    Comprehensive test evaluation with MULTI-MASKING support

    Args:
        model: Trained model to test
        test_dataloader: Test dataloader with dynamic masking capability
        device: Device to run testing on
        debug: Enable debug prints
        test_masking_levels: List of masking levels to test (0-100)
        primary_masking_level: Primary masking level for main results
    """

    print(f"=== COMPREHENSIVE MULTI-MASKING TEST SET EVALUATION ===")
    print(f"Test set size: {len(test_dataloader.dataset)} samples")
    print(f"Test masking levels: {test_masking_levels}%")
    print(f"Primary masking level: {primary_masking_level}%")

    model.eval()
    is_cross_encoder = hasattr(model, 'cross_attention')

    # Store original masking probability
    original_probability = test_dataloader.dataset.get_current_masking_probability()
    all_test_metrics = {}

    try:
        # Test at each masking level
        for masking_level in test_masking_levels:
            masking_probability = masking_level / 100.0
            is_primary = (masking_level == primary_masking_level)

            print(f"\n=== TESTING AT {masking_level}% MASKING {'(PRIMARY)' if is_primary else ''} ===")

            # Set masking probability for this level
            test_dataloader.dataset.set_masking_probability(masking_probability)

            # Main ranking evaluation at this masking level
            if is_cross_encoder:
                main_ranking_results = perform_cross_encoder_test_ranking_at_masking_level(
                    model, test_dataloader, device, masking_level
                )
            else:
                main_ranking_results = perform_dual_encoder_test_ranking_at_masking_level(
                    model, test_dataloader, device, masking_level
                )

            # Add masking level to all metric names
            masking_prefix = f'test_ranking/masking_{masking_level}'

            # Rename main results
            main_masking_results = {}
            for key, value in main_ranking_results.items():
                new_key = key.replace('test_ranking/main', masking_prefix)
                main_masking_results[new_key] = value

            # Store results for this masking level
            all_test_metrics.update(main_masking_results)

            # Print summary for this masking level
            if main_ranking_results:
                main_mrr = main_ranking_results.get('test_ranking/main/rr', 0)
                main_hit1 = main_ranking_results.get('test_ranking/main/hit_at_1', 0)
                main_hit10 = main_ranking_results.get('test_ranking/main/hit_at_10', 0)
                print(f"  {masking_level}% Masking: MRR {main_mrr:.4f}, Hit@1 {main_hit1:.4f}, Hit@10 {main_hit10:.4f}")

        # Noise baseline evaluation (at primary masking level)
        print(f"\n=== NOISE BASELINE AT {primary_masking_level}% MASKING ===")
        test_dataloader.dataset.set_masking_probability(primary_masking_level / 100.0)
        noise_ranking_results = perform_noise_ranking_evaluation(
            model, test_dataloader, device, primary_masking_level
        )
        all_test_metrics.update(noise_ranking_results)

        # Calculate cross-masking performance comparisons
        masking_comparison_metrics = calculate_masking_performance_trends(
            all_test_metrics, test_masking_levels
        )
        all_test_metrics.update(masking_comparison_metrics)

        # Add comprehensive summary statistics
        summary_metrics = {
            'test_ranking/summary/test_masking_levels': test_masking_levels,
            'test_ranking/summary/primary_masking_level': primary_masking_level,
            'test_ranking/summary/num_test_samples': len(test_dataloader.dataset),
            'test_ranking/summary/encoder_type': 'cross' if is_cross_encoder else 'dual'
        }

        if hasattr(model, 'query_type'):
            summary_metrics['test_ranking/summary/query_type'] = model.query_type

        all_test_metrics.update(summary_metrics)

        # Log all metrics to wandb
        wandb.log(all_test_metrics)

        # Print comprehensive multi-masking summary
        print_multi_masking_test_summary(all_test_metrics, test_masking_levels, primary_masking_level)

        return all_test_metrics

    finally:
        # Always restore original masking probability
        test_dataloader.dataset.set_masking_probability(original_probability)

def generate_eeg_noise_baseline(eeg_queries, noise_type='gaussian'):
    """Generate random noise with same shape as EEG queries for baseline comparison"""
    if noise_type == 'gaussian':
        # Generate Gaussian noise with same shape as real EEG
        noise = torch.randn_like(eeg_queries)
        # Optional: match the scale of real EEG data
        eeg_std = eeg_queries.std()
        eeg_mean = eeg_queries.mean()
        noise = noise * eeg_std + eeg_mean
    return noise


def perform_noise_ranking_evaluation(model, test_dataloader, device, primary_masking_level=90):
    """
    Perform ranking evaluation with noise queries as baseline

    Args:
        model: Model to evaluate
        test_dataloader: Test dataloader
        device: Device to run on
        primary_masking_level: Masking level to use
    """

    print(f"\n=== NOISE BASELINE RANKING EVALUATION ===")
    print(f"Testing ranking performance with Gaussian noise instead of EEG queries...")

    # Set masking probability
    original_probability = test_dataloader.dataset.get_current_masking_probability()
    test_dataloader.dataset.set_masking_probability(primary_masking_level / 100.0)

    try:
        # Build document database
        doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

        if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
            print("Warning: No valid query-document pairs found for noise ranking")
            return {}

        # Use all documents
        subset_size = len(doc_list)
        query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
        queries = collect_eeg_queries(test_dataloader, device)

        print(f"Noise ranking: {len(queries)} noise queries against {subset_size} documents each")

        all_metrics = []
        is_cross_encoder = hasattr(model, 'cross_attention')

        for query_idx, (real_eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
            if query_idx not in query_to_doc_mapping:
                continue

            # Replace EEG with noise
            noise_eeg_query = generate_eeg_noise_baseline(real_eeg_query).to(device)

            correct_doc_idx = query_to_doc_mapping[query_idx]
            doc_subset_indices = query_subsets[query_idx]

            # Perform ranking with noise query
            if is_cross_encoder:
                ranked_indices, scores = rank_cross_encoder_subset(
                    model, noise_eeg_query, text_query, doc_list, doc_subset_indices, device
                )
            else:
                ranked_indices, scores = rank_dual_encoder_subset(
                    model, noise_eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
                )

            query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
            all_metrics.append(query_metrics)

            if (query_idx + 1) % 50 == 0:
                print(f"  Processed {query_idx + 1}/{len(queries)} noise queries...")

        if not all_metrics:
            print("Warning: No noise ranking metrics computed")
            return {}

        # Aggregate noise ranking metrics
        noise_metrics = {}
        metric_names = all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            noise_metrics[f'test_ranking/noise/{metric_name}'] = np.mean(values)
            if metric_name != 'rank_of_correct':
                noise_metrics[f'test_ranking/noise/{metric_name}_std'] = np.std(values)

        noise_metrics.update({
            'test_ranking/noise/num_unique_documents': len(doc_list),
            'test_ranking/noise/num_queries_evaluated': len(all_metrics),
            'test_ranking/noise/subset_size': subset_size,
            'test_ranking/noise/masking_level': primary_masking_level
        })

        # Print noise baseline results
        print(f"Noise Baseline Ranking Results:")
        print(f"  Queries Evaluated: {len(all_metrics)}")
        print(f"  MRR: {noise_metrics['test_ranking/noise/rr']:.4f}")
        print(f"  Hit@1: {noise_metrics['test_ranking/noise/hit_at_1']:.4f}")
        print(f"  Hit@5: {noise_metrics['test_ranking/noise/hit_at_5']:.4f}")
        print(f"  Hit@10: {noise_metrics['test_ranking/noise/hit_at_10']:.4f}")
        print(f"  Hit@20: {noise_metrics['test_ranking/noise/hit_at_20']:.4f}")

        return noise_metrics

    finally:
        # Restore original masking probability
        test_dataloader.dataset.set_masking_probability(original_probability)


def perform_ranking_validation(model, val_dataloader, device, epoch_num, subset_size=100, use_full_set=False):
    """Fair ranking validation using same subsets as cross-encoder or full set"""

    print(f"\n=== DUAL-ENCODER RANKING VALIDATION (Epoch {epoch_num}) ===")

    # Step 1: Build unique document database
    doc_list, query_to_doc_mapping = build_document_database(val_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    # Step 2: Handle full set vs subset
    if use_full_set:
        print(f"Using FULL validation set: all {len(doc_list)} documents for each query")
        # Create "subsets" that contain all documents
        query_subsets = {}
        all_doc_indices = list(range(len(doc_list)))
        for query_idx in query_to_doc_mapping:
            query_subsets[query_idx] = all_doc_indices
        subset_size = len(doc_list)
    else:
        if len(doc_list) < subset_size:
            print(f"Warning: Only {len(doc_list)} documents available, using all")
            subset_size = len(doc_list)
        # Generate consistent subsets (SAME AS CROSS-ENCODER)
        query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)

    # Step 3: Collect queries
    queries = collect_eeg_queries(val_dataloader, device)

    print(f"Dual-encoder ranking: {len(queries)} queries against {subset_size} documents each")
    print(f"Query type: {model.query_type}")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        # Rank this subset using dual encoder
        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        # Compute metrics
        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    # Rest of function unchanged...
    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    aggregated_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated_metrics[f'ranking/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            aggregated_metrics[f'ranking/{metric_name}_std'] = np.std(values)

    # Add dataset info
    aggregated_metrics['ranking/num_unique_documents'] = len(doc_list)
    aggregated_metrics['ranking/num_queries_evaluated'] = len(all_metrics)
    aggregated_metrics['ranking/epoch_num'] = epoch_num
    aggregated_metrics['ranking/pooling_strategy'] = model.pooling_strategy
    aggregated_metrics['ranking/subset_size'] = subset_size
    aggregated_metrics['ranking/query_type'] = model.query_type
    aggregated_metrics['ranking/use_full_set'] = use_full_set  # Add this

    # Log to wandb
    wandb.log(aggregated_metrics)

    # Print summary
    print(f"Dual-Encoder Ranking Results:")
    print(f"  Query Type: {model.query_type}")
    print(f"  {'Full Set' if use_full_set else 'Subset'} Size: {subset_size}")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {aggregated_metrics['ranking/rr']:.4f}")
    print(f"  Hit@1: {aggregated_metrics['ranking/hit_at_1']:.4f}")
    print(f"  Hit@5: {aggregated_metrics['ranking/hit_at_5']:.4f}")
    print(f"  Precision@1: {aggregated_metrics['ranking/precision_at_1']:.4f}")
    print(f"  Precision@5: {aggregated_metrics['ranking/precision_at_5']:.4f}")
    print(f"  Avg Rank of Correct: {aggregated_metrics['ranking/rank_of_correct']:.1f}")
    print("=" * 60)

    return aggregated_metrics


def rank_dual_encoder_subset(model, eeg_query, eeg_mv_mask, doc_list, doc_indices, device, text_query=None):
    """EFFICIENT: Batch encode all documents at once for dual-encoder"""
    model.eval()

    with torch.no_grad():
        # Encode query once (based on model's query type)
        if model.query_type == 'eeg':
            eeg_output = model.encode_eeg(eeg_query, eeg_mv_mask)

            # Handle temporal-spatial decomposition case
            if model.use_temporal_spatial_decomp and isinstance(eeg_output, dict):
                # Handle decomposition case
                if hasattr(model, 'use_dual_loss') and model.use_dual_loss:
                    # For dual loss, we need to compute similarities for both components
                    # This is complex for ranking, so we'll combine them for now
                    temporal_vectors = eeg_output['temporal_vectors']
                    spatial_vectors = eeg_output['spatial_vectors']

                    # Combine temporal and spatial vectors
                    if isinstance(temporal_vectors, list) and isinstance(spatial_vectors, list):
                        combined_vectors = []
                        for i in range(len(temporal_vectors)):
                            temp_vec = temporal_vectors[i]
                            spat_vec = spatial_vectors[i]
                            combined = torch.cat([temp_vec, spat_vec], dim=-1)
                            # Project back to original dimension
                            projected = model.combined_projection(combined)
                            combined_vectors.append(projected)
                        query_vectors = combined_vectors[0]  # Single query
                    else:
                        # Tensor case
                        temp_vec = temporal_vectors[0:1]
                        spat_vec = spatial_vectors[0:1]
                        combined = torch.cat([temp_vec, spat_vec], dim=-1)
                        query_vectors = model.combined_projection(combined)
                else:
                    # Single loss case - already combined
                    temporal_vectors = eeg_output['temporal_vectors']
                    spatial_vectors = eeg_output['spatial_vectors']

                    # Combine temporal and spatial vectors
                    if isinstance(temporal_vectors, list) and isinstance(spatial_vectors, list):
                        combined_vectors = []
                        for i in range(len(temporal_vectors)):
                            temp_vec = temporal_vectors[i]
                            spat_vec = spatial_vectors[i]
                            combined = torch.cat([temp_vec, spat_vec], dim=-1)
                            projected = model.combined_projection(combined)
                            combined_vectors.append(projected)
                        query_vectors = combined_vectors[0]  # Single query
                    else:
                        # Tensor case
                        temp_vec = temporal_vectors[0:1]
                        spat_vec = spatial_vectors[0:1]
                        combined = torch.cat([temp_vec, spat_vec], dim=-1)
                        query_vectors = model.combined_projection(combined)
            else:
                # Normal case - no decomposition
                query_vectors = eeg_output

                # Handle different return types
                if isinstance(query_vectors, list):
                    query_vectors = query_vectors[0]  # Get first element for single query
                else:
                    query_vectors = query_vectors[0:1]  # Keep as batch of 1

        else:  # text query
            if text_query is None:
                raise ValueError("text_query is required when model.query_type='text'")
            query_vectors = model.encode_text(text_query['input_ids'], text_query['attention_mask'])

            # Handle different return types
            if isinstance(query_vectors, list):
                query_vectors = query_vectors[0]  # Get first element for single query
            else:
                query_vectors = query_vectors[0:1]  # Keep as batch of 1

        # BATCH encode ALL documents in the subset at once
        doc_input_ids = torch.stack([doc_list[doc_idx]['input_ids'] for doc_idx in doc_indices]).to(device)
        doc_attention_masks = torch.stack([doc_list[doc_idx]['attention_mask'] for doc_idx in doc_indices]).to(device)

        # Single batch encoding for ALL documents
        batch_doc_vectors = model.encode_text(doc_input_ids, doc_attention_masks)

        # Handle different pooling strategies
        if isinstance(batch_doc_vectors, list):
            # Multi-vector pooling
            doc_vectors_list = batch_doc_vectors
        else:
            # CLS pooling - split batch back to individual documents
            doc_vectors_list = [batch_doc_vectors[i:i + 1] for i in range(batch_doc_vectors.size(0))]

        # Compute similarities in batch
        scores = []
        for i, doc_vectors in enumerate(doc_vectors_list):
            sim = compute_similarity([query_vectors], [doc_vectors], model.pooling_strategy, temperature=1.0)
            scores.append((doc_indices[i], sim[0].item()))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [doc_idx for doc_idx, score in scores]
    ranked_scores = [score for doc_idx, score in scores]

    return ranked_indices, ranked_scores


def perform_cross_encoder_ranking_validation(model, val_dataloader, device, epoch_num, subset_size=100,
                                             use_full_set=False):
    """Cross-encoder ranking validation using SAME subsets as dual encoder or full set"""

    print(f"\n=== CROSS-ENCODER RANKING VALIDATION (Epoch {epoch_num}) ===")

    # Step 1: Build document database (same as dual encoder)
    doc_list, query_to_doc_mapping = build_document_database(val_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    # Step 2: Handle full set vs subset
    if use_full_set:
        print(f"Using FULL validation set: all {len(doc_list)} documents for each query")
        # Create "subsets" that contain all documents
        query_subsets = {}
        all_doc_indices = list(range(len(doc_list)))
        for query_idx in query_to_doc_mapping:
            query_subsets[query_idx] = all_doc_indices
        subset_size = len(doc_list)
    else:
        if len(doc_list) < subset_size:
            print(f"Warning: Only {len(doc_list)} documents available, using all")
            subset_size = len(doc_list)
        # Generate SAME consistent subsets as dual encoder
        query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)

    # Step 3: Collect queries (both EEG and text)
    queries = collect_eeg_queries(val_dataloader, device)

    print(f"Cross-encoder ranking: {len(queries)} queries against {subset_size} documents each")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]  # Now contains all docs if use_full_set=True

        # Rank this subset using cross-encoder
        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        # Compute metrics (identical to dual encoder)
        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    # Aggregate results (identical format to dual encoder)
    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    aggregated_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated_metrics[f'ranking/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            aggregated_metrics[f'ranking/{metric_name}_std'] = np.std(values)

    # Add dataset info
    aggregated_metrics['ranking/num_unique_documents'] = len(doc_list)
    aggregated_metrics['ranking/num_queries_evaluated'] = len(all_metrics)
    aggregated_metrics['ranking/epoch_num'] = epoch_num
    aggregated_metrics['ranking/pooling_strategy'] = 'cross'
    aggregated_metrics['ranking/subset_size'] = subset_size
    aggregated_metrics['ranking/use_full_set'] = use_full_set  # Add this

    # Log to wandb
    wandb.log(aggregated_metrics)

    # Print summary (identical format)
    print(f"Cross-Encoder Ranking Results:")
    print(f"  {'Full Set' if use_full_set else 'Subset'} Size: {subset_size}")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {aggregated_metrics['ranking/rr']:.4f}")
    print(f"  Hit@1: {aggregated_metrics['ranking/hit_at_1']:.4f}")
    print(f"  Hit@5: {aggregated_metrics['ranking/hit_at_5']:.4f}")
    print(f"  Precision@1: {aggregated_metrics['ranking/precision_at_1']:.4f}")
    print(f"  Precision@5: {aggregated_metrics['ranking/precision_at_5']:.4f}")
    print(f"  Avg Rank of Correct: {aggregated_metrics['ranking/rank_of_correct']:.1f}")
    print("=" * 60)

    return aggregated_metrics


def rank_cross_encoder_subset(model, eeg_query, text_query, doc_list, doc_indices, device):
    """EFFICIENT: Batch process all documents at once for cross-encoder"""
    model.eval()

    with torch.no_grad():
        # Stack all documents in the subset for batch processing
        doc_input_ids = torch.stack([doc_list[doc_idx]['input_ids'] for doc_idx in doc_indices]).to(device)
        doc_attention_masks = torch.stack([doc_list[doc_idx]['attention_mask'] for doc_idx in doc_indices]).to(device)

        batch_size = len(doc_indices)

        # Replicate query for each document in the batch
        batch_eeg_queries = eeg_query.repeat(batch_size, 1, 1, 1)  # [batch_size, words, time, channels]
        batch_text_queries = {
            'input_ids': text_query['input_ids'].repeat(batch_size, 1),
            'attention_mask': text_query['attention_mask'].repeat(batch_size, 1)
        }
        batch_docs = {
            'input_ids': doc_input_ids,
            'attention_mask': doc_attention_masks
        }

        # Single forward pass for ALL documents at once
        batch_scores = model(batch_eeg_queries, batch_text_queries, batch_docs)
        scores = batch_scores.squeeze().cpu().tolist()

        # Handle single document case
        if isinstance(scores, float):
            scores = [scores]

    # Create (doc_idx, score) pairs and sort
    doc_score_pairs = list(zip(doc_indices, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    ranked_indices = [doc_idx for doc_idx, score in doc_score_pairs]
    ranked_scores = [score for doc_idx, score in doc_score_pairs]

    return ranked_indices, ranked_scores


# ==========================================
# MAIN TRAINING FUNCTIONS WITH DYNAMIC MULTI-MASKING
# ==========================================

def train_epoch(model, dataloader, optimizer, device, epoch_num, total_epochs, debug=False):
    """Train for a single epoch"""

    model.train()
    total_loss = 0
    num_batches = 0

    # Determine if cross-encoder for metric tracking
    is_cross_encoder = hasattr(model, 'cross_attention')

    # Accumulate metrics
    if is_cross_encoder:
        epoch_accuracies = []
    else:
        epoch_similarities = []

    epoch_grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        # Debug first batch of first epoch
        debug_this_batch = debug and epoch_num == 1 and batch_idx == 0

        # Calculate global step
        step_num = (epoch_num - 1) * len(dataloader) + batch_idx

        # Training step
        loss, metrics, grad_norm = train_step(
            model, batch, optimizer, device, step_num, debug=debug_this_batch
        )

        total_loss += loss
        num_batches += 1

        if is_cross_encoder:
            epoch_accuracies.append(metrics['accuracy'])
        else:
            # Use appropriate similarity metric based on query type
            if hasattr(model, 'query_type') and model.query_type == 'text':
                similarity_key = 'text_query_similarity'
                display_name = 'Text-Query Sim'
            else:
                similarity_key = 'eeg_query_similarity'
                display_name = 'EEG-Query Sim'

            epoch_similarities.append(metrics[similarity_key])

        epoch_grad_norms.append(grad_norm)

        # Progress logging with appropriate labels
        if batch_idx % 20 == 0:
            if is_cross_encoder:
                print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_display = f"Text-Query Sim: {metrics['text_query_similarity']:.4f}"
                else:
                    sim_display = f"EEG-Query Sim: {metrics['eeg_query_similarity']:.4f}"
                print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}, {sim_display}")

    # Compute epoch statistics
    avg_loss = total_loss / num_batches
    if is_cross_encoder:
        avg_metric = np.mean(epoch_accuracies)
        metric_name = "Accuracy"
    else:
        avg_metric = np.mean(epoch_similarities)
        if hasattr(model, 'query_type') and model.query_type == 'text':
            metric_name = "Text-Query Sim"
        else:
            metric_name = "EEG-Query Sim"

    avg_grad_norm = np.mean(epoch_grad_norms)

    print(f"Epoch {epoch_num} training completed. "
          f"Avg Loss: {avg_loss:.4f}, Avg {metric_name}: {avg_metric:.4f}")

    return avg_loss, avg_metric, avg_grad_norm


def validate_single_masking_level(model, val_dataloader, device, masking_probability, masking_level_name,
                                  is_primary=False):
    """Validate model on a single masking level using DYNAMIC masking"""

    # DYNAMIC: Change masking probability on the fly
    original_probability = val_dataloader.dataset.get_current_masking_probability()
    val_dataloader.dataset.set_masking_probability(masking_probability)

    try:
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        # Determine if cross-encoder for metric tracking
        is_cross_encoder = hasattr(model, 'cross_attention')

        # Accumulate validation metrics
        if is_cross_encoder:
            val_accuracies = []
            val_neg_accuracies = []
        else:
            val_similarities = []
            val_query_doc_sims = []
            val_eeg_doc_sims = []

        for batch_idx, batch in enumerate(val_dataloader):
            val_loss, val_metrics = validation_step(model, batch, device)

            total_val_loss += val_loss
            num_val_batches += 1

            if is_cross_encoder:
                val_accuracies.append(val_metrics['accuracy'])
                val_neg_accuracies.append(val_metrics['negative_accuracy'])
            else:
                # Use appropriate similarity key based on query type
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    similarity_key = 'text_query_similarity'
                else:
                    similarity_key = 'eeg_query_similarity'

                val_similarities.append(val_metrics[similarity_key])
                val_query_doc_sims.append(val_metrics['query_doc_similarity'])

                # Use appropriate doc similarity key
                doc_similarity_key = 'text_doc_similarity' if model.query_type == 'text' else 'eeg_doc_similarity'
                val_eeg_doc_sims.append(val_metrics[doc_similarity_key])

        # Compute validation statistics
        avg_val_loss = total_val_loss / num_val_batches

        if is_cross_encoder:
            avg_val_accuracy = np.mean(val_accuracies)
            avg_val_neg_accuracy = np.mean(val_neg_accuracies)
            main_metric = avg_val_accuracy

            metrics_dict = {
                'loss': avg_val_loss,
                'accuracy': avg_val_accuracy,
                'negative_accuracy': avg_val_neg_accuracy
            }
        else:
            avg_val_similarity = np.mean(val_similarities)
            avg_query_doc_sim = np.mean(val_query_doc_sims)
            avg_eeg_doc_sim = np.mean(val_eeg_doc_sims)
            main_metric = avg_val_similarity

            metrics_dict = {
                'loss': avg_val_loss,
                'query_doc_similarity': avg_query_doc_sim,
                'eeg_doc_similarity': avg_eeg_doc_sim
            }

            # Add query-type specific metrics
            if hasattr(model, 'query_type') and model.query_type == 'text':
                metrics_dict.update({
                    'text_query_similarity': avg_val_similarity,
                    'text_query_similarity_std': np.std(val_similarities),
                    'text_doc_similarity': avg_eeg_doc_sim,
                })
            else:
                metrics_dict.update({
                    'eeg_query_similarity': avg_val_similarity,
                    'eeg_query_similarity_std': np.std(val_similarities),
                    'eeg_doc_similarity': avg_eeg_doc_sim,
                })

        return metrics_dict, main_metric

    finally:
        # DYNAMIC: Always restore original masking probability
        val_dataloader.dataset.set_masking_probability(original_probability)


def validate_epoch(model, val_dataloader, device, epoch_num,
                   enable_multi_masking_validation=False,
                   validation_masking_levels=[0, 25, 50, 75, 90, 100],
                   multi_masking_frequency=3,
                   primary_masking_level=90):
    """
    Enhanced validate_epoch with DYNAMIC multi-masking support

    Args:
        model: The model to validate
        val_dataloader: Single validation dataloader with dynamic masking capability
        device: Device to run validation on
        epoch_num: Current epoch number
        enable_multi_masking_validation: Whether to enable multi-masking validation
        validation_masking_levels: List of masking percentages to test
        multi_masking_frequency: Run multi-masking validation every N epochs
        primary_masking_level: Masking level to use for early stopping (default: 90)
    """

    # Always validate on primary masking level (90% by default)
    print(f"Running standard validation for epoch {epoch_num}...")

    primary_masking_prob = primary_masking_level / 100.0
    primary_metrics, primary_main_metric = validate_single_masking_level(
        model, val_dataloader, device, primary_masking_prob, f"masking_{primary_masking_level}%", is_primary=True
    )

    # Log primary validation metrics to wandb
    log_dict = {}
    for metric_name, value in primary_metrics.items():
        log_dict[f'val/{metric_name}'] = value
    log_dict['val/epoch_num'] = epoch_num

    wandb.log(log_dict)

    # Determine if cross-encoder for display
    is_cross_encoder = hasattr(model, 'cross_attention')

    if is_cross_encoder:
        print(f"Standard validation completed. Val Loss: {primary_metrics['loss']:.4f}, "
              f"Val Accuracy: {primary_metrics['accuracy']:.4f}")
    else:
        if hasattr(model, 'query_type') and model.query_type == 'text':
            display_name = "Text-Query Sim"
            similarity_key = 'text_query_similarity'
        else:
            display_name = "EEG-Query Sim"
            similarity_key = 'eeg_query_similarity'

        print(f"Standard validation completed. Val Loss: {primary_metrics['loss']:.4f}, "
              f"{display_name}: {primary_metrics[similarity_key]:.4f}")

    # DYNAMIC Multi-masking validation (if enabled and appropriate epoch)
    if (enable_multi_masking_validation and epoch_num % multi_masking_frequency == 0):

        print(f"\n=== DYNAMIC MULTI-MASKING VALIDATION (Epoch {epoch_num}) ===")
        print(f"Testing masking levels: {validation_masking_levels}%")
        print(f"Using SINGLE dataloader with dynamic masking probability")

        all_masking_metrics = {}

        for masking_level in validation_masking_levels:
            masking_probability = masking_level / 100.0
            masking_level_name = f"{masking_level}%"

            print(f"  Validating on {masking_level_name} masking...")

            masking_metrics, masking_main_metric = validate_single_masking_level(
                model, val_dataloader, device, masking_probability, masking_level_name
            )

            # Store metrics for this masking level
            all_masking_metrics[masking_level_name] = {
                'metrics': masking_metrics,
                'main_metric': masking_main_metric
            }

            # Log to wandb with masking level prefix
            masking_log_dict = {}
            for metric_name, value in masking_metrics.items():
                masking_log_dict[f'val/masking_{masking_level}_{metric_name}'] = value
            masking_log_dict[f'val/masking_{masking_level}_epoch_num'] = epoch_num

            wandb.log(masking_log_dict)

            # Print summary for this masking level
            if is_cross_encoder:
                print(f"    {masking_level_name}: Loss {masking_metrics['loss']:.4f}, "
                      f"Accuracy {masking_metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_key = 'text_query_similarity'
                else:
                    sim_key = 'eeg_query_similarity'

                print(f"    {masking_level_name}: Loss {masking_metrics['loss']:.4f}, "
                      f"Similarity {masking_metrics[sim_key]:.4f}")

        # Print multi-masking summary
        print(f"\n  DYNAMIC multi-masking validation summary:")
        for masking_level, data in all_masking_metrics.items():
            metrics = data['metrics']
            main_metric = data['main_metric']

            if is_cross_encoder:
                print(f"    {masking_level}: Accuracy {metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_key = 'text_query_similarity'
                else:
                    sim_key = 'eeg_query_similarity'
                print(f"    {masking_level}: Similarity {metrics[sim_key]:.4f}")

        print("=" * 60)

    # Standard ranking validation every 3rd epoch
    if epoch_num % 3 == 0:
        if is_cross_encoder:
            print(f"Running cross-encoder ranking validation...")
            ranking_metrics = perform_cross_encoder_ranking_validation(
                model, val_dataloader, device, epoch_num, subset_size=100, use_full_set=True
            )
        else:
            print(f"Running dual-encoder ranking validation...")
            ranking_metrics = perform_ranking_validation(
                model, val_dataloader, device, epoch_num, subset_size=100, use_full_set=True
            )

    # Return primary metrics for early stopping
    return primary_metrics['loss'], primary_main_metric


def initialize_wandb(config):
    """Initialize wandb logging"""

    # Create descriptive run name
    encoder_type = config.get('encoder_type', 'dual')
    query_type = config.get('query_type', 'eeg')

    if encoder_type == 'cross':
        pooling_strategy = 'cross'
        eeg_arch = config.get('eeg_arch', 'simple')
        use_lora = config.get('use_lora', True)
        model_suffix = f"_lora{config.get('lora_r', 16)}" if use_lora else "_frozen"
        run_name = f"cross_{query_type}_{eeg_arch}{model_suffix}"
    else:
        pooling_strategy = config.get('pooling_strategy', 'multi')
        eeg_arch = config.get('eeg_arch', 'simple')
        use_lora = config.get('use_lora', True)
        model_suffix = f"_lora{config.get('lora_r', 16)}" if use_lora else "_frozen"

        # ADD: sequence concatenation info to run name
        processing_suffix = ""
        if config.get('use_sequence_concat', False):
            processing_suffix = "_seq_concat"
        elif config.get('use_temporal_spatial_decomp', False):
            decomp_level = config.get('decomp_level', 'word')
            loss_type = 'dual' if config.get('use_dual_loss', False) else 'single'
            processing_suffix = f"_{decomp_level}_decomp_{loss_type}"
        else:
            processing_suffix = "_word_level"

        run_name = f"dual_{query_type}_{pooling_strategy}_{eeg_arch}{model_suffix}{processing_suffix}"

    # Add multi-masking info to run name if enabled
    if config.get('enable_multi_masking_validation', False):
        run_name += "_dynamic_mask"

    # Build tags list without None values
    tags = [
        'brain-retrieval', 'eeg', 'colbert',
        'lora' if use_lora else 'frozen',
        f'arch-{eeg_arch}',
        f'encoder-{encoder_type}',
        f'pooling-{pooling_strategy}' if encoder_type == 'dual' else 'cross-encoder',
        # Processing strategy tags
        'sequence-concat' if config.get('use_sequence_concat', False) else (
            'temporal-spatial-decomp' if config.get('use_temporal_spatial_decomp', False) else 'word-level'
        ),
        'dual-loss' if config.get('use_dual_loss', False) else 'single-loss'
    ]

    # Add multi-masking tag
    if config.get('enable_multi_masking_validation', False):
        tags.append('dynamic-multi-masking-validation')

    # Add decomposition level tag only if decomposition is enabled
    if config.get('use_temporal_spatial_decomp', False):
        tags.append(f"decomp-{config.get('decomp_level', 'word')}")

    wandb.init(
        project="SpatialTemporalLearning",
        name=run_name,
        config={
            'encoder_type': encoder_type,
            'query_type': query_type,
            'pooling_strategy': pooling_strategy,

            # Model config
            'colbert_model_name': config.get('colbert_model_name', 'colbert-ir/colbertv2.0'),
            'eeg_arch': eeg_arch,
            'hidden_dim': config.get('hidden_dim', 768),
            'num_vectors': config.get('num_vectors', 32),

            # LoRA config
            'use_lora': use_lora,
            'lora_r': config.get('lora_r', 16),
            'lora_alpha': config.get('lora_alpha', 32),
            'trainable_params': config.get('trainable_params', 0),
            'total_params': config.get('total_params', 0),

            # Training config
            'batch_size': config.get('batch_size', 8),
            'learning_rate': config.get('learning_rate', 1e-4),
            'epochs': config.get('epochs', 50),
            'patience': config.get('patience', 10),

            # Data config
            'max_text_len': config.get('max_text_len', 256),
            'max_eeg_len': config.get('max_eeg_len', 50),
            'train_samples': config.get('train_samples', 0),
            'val_samples': config.get('val_samples', 0),

            # DYNAMIC Multi-masking validation config
            'enable_multi_masking_validation': config.get('enable_multi_masking_validation', False),
            'validation_masking_levels': config.get('validation_masking_levels', []),
            'multi_masking_frequency': config.get('multi_masking_frequency', 3),
            'primary_masking_level': config.get('primary_masking_level', 90),
            'training_masking_level': config.get('training_masking_level', 90),
            'dynamic_masking': True,  # NEW: indicates this uses dynamic masking

            # Processing strategy config
            'use_sequence_concat': config.get('use_sequence_concat', False),
            'use_temporal_spatial_decomp': config.get('use_temporal_spatial_decomp', False),
            'decomp_level': config.get('decomp_level', 'word'),
            'use_dual_loss': config.get('use_dual_loss', False),
            'lambda_temporal': config.get('lambda_temporal', 1.0),
            'lambda_spatial': config.get('lambda_spatial', 1.0),

            # Experiment config
            'seed': config.get('seed', 42),
            'loss_type': 'bce' if encoder_type == 'cross' else 'contrastive',
            'similarity_function': f'{pooling_strategy}_similarity'
        },
        tags=tags
    )


def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs,
                patience=10, device='cuda', debug=False, config=None,
                enable_multi_masking_validation=False, multi_masking_frequency=3,
                validation_masking_levels=[0, 25, 50, 75, 90, 100],
                primary_masking_level=90):
    """
    Complete training loop with validation and early stopping + DYNAMIC MULTI-MASKING VALIDATION

    Args:
        model: Model to train
        train_dataloader: Training dataloader (with dynamic masking)
        val_dataloader: Single validation dataloader (with dynamic masking capability)
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
        debug: Enable debug prints
        config: Configuration dict for wandb
        enable_multi_masking_validation: Enable dynamic multi-masking validation
        multi_masking_frequency: Run multi-masking validation every N epochs
        validation_masking_levels: List of masking percentages to test
        primary_masking_level: Masking level to use for early stopping decisions
    """

    # Initialize wandb
    if config:
        initialize_wandb(config)

    is_cross_encoder = hasattr(model, 'cross_attention')
    encoder_type = 'Cross-Encoder' if is_cross_encoder else 'Dual-Encoder'
    pooling_info = model.pooling_strategy if hasattr(model, 'pooling_strategy') else 'unknown'

    print(f"Starting {encoder_type} training with early stopping (patience={patience})...")
    print(f"Pooling strategy: {pooling_info}")
    print(f"Training masking level: {config.get('training_masking_level', 90)}%")

    # DYNAMIC Multi-masking validation info
    if enable_multi_masking_validation:
        print(f"DYNAMIC multi-masking validation: ENABLED")
        print(f"  Validation masking levels: {validation_masking_levels}%")
        print(f"  Multi-masking frequency: every {multi_masking_frequency} epochs")
        print(f"  Primary masking level: {primary_masking_level}% (for early stopping)")
        print(f"  Memory efficient: Uses SINGLE dataloader with dynamic masking")
    else:
        print(f"Multi-masking validation: DISABLED")

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_metric = -1.0 if not is_cross_encoder else 0.0  # similarity vs accuracy
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    early_stopped = False

    for epoch in range(num_epochs):
        epoch_num = epoch + 1

        # Training epoch
        train_loss, train_metric, train_grad_norm = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch_num,
            total_epochs=num_epochs,
            debug=debug
        )

        # Enhanced validation epoch with DYNAMIC multi-masking support
        val_loss, val_metric = validate_epoch(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            epoch_num=epoch_num,
            enable_multi_masking_validation=enable_multi_masking_validation,
            validation_masking_levels=validation_masking_levels,
            multi_masking_frequency=multi_masking_frequency,
            primary_masking_level=primary_masking_level
        )

        # Early stopping logic - use loss for primary criterion
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metric = val_metric
            best_epoch = epoch_num
            epochs_without_improvement = 0

            # Save best model state
            best_model_state = model.state_dict().copy()

            print(f"New best validation loss: {best_val_loss:.4f} (epoch {epoch_num})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch_num} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                early_stopped = True

        # Log epoch-level metrics
        metric_name = 'accuracy' if is_cross_encoder else 'similarity'
        wandb.log({
            'epoch/train_loss': train_loss,
            'epoch/val_loss': val_loss,
            f'epoch/train_{metric_name}': train_metric,
            f'epoch/val_{metric_name}': val_metric,
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch/epoch_num': epoch_num,
            'epoch/epochs_without_improvement': epochs_without_improvement,
            'epoch/best_val_loss': best_val_loss
        })

        # Print epoch summary
        metric_display = 'Accuracy' if is_cross_encoder else 'EEG-Query Sim'
        print(f"\nEpoch {epoch_num}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, {metric_display}: {train_metric:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, {metric_display}: {val_metric:.4f}")
        print(f"  Best  - Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"  Early Stopping: {epochs_without_improvement}/{patience} epochs without improvement")

        # Enhanced logging for multi-masking epochs
        if (enable_multi_masking_validation and epoch_num % multi_masking_frequency == 0):
            print(f"  Note: DYNAMIC multi-masking validation performed this epoch ({validation_masking_levels}%)")
        elif epoch_num % 3 == 0:
            print(f"  Note: Standard ranking validation performed this epoch")

        print("-" * 70)

        if early_stopped:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")

    # Log training summary
    wandb.log({
        'training/completed_epochs': epoch_num,
        'training/early_stopped': early_stopped,
        'training/best_epoch': best_epoch,
        'training/best_val_loss': best_val_loss,
        'training/final_val_loss': val_loss
    })

    print(f"\n{encoder_type} training completed!")
    if early_stopped:
        print(f"Stopped early after {epoch_num} epochs")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if enable_multi_masking_validation:
        print(f"DYNAMIC multi-masking validation was performed every {multi_masking_frequency} epochs")
        print(f"Memory efficient: Used single dataloader with dynamic masking")
        print(f"Check wandb logs for performance trends across masking levels: {validation_masking_levels}%")

    return model


def finish_wandb():
    """Finish wandb run"""
    wandb.finish()