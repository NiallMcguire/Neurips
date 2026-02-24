#!/usr/bin/env python3
"""
Training for EEG-EEG vs EEG-Text Alignment Experiments
Version: 2.1
Focus: Track dataset name, subject mode, and model version
New: text_loss_mode support ('standard', 'masked', 'multi_positive') for EEG-Text condition
"""

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import random
from typing import Dict, List, Tuple
from models import compute_similarity


def compute_contrastive_loss(query_vectors, doc_vectors, pooling_strategy, temperature=0.07,
                             batch_metadata=None, multi_positive_train=False, debug_step=False,
                             text_loss_mode='standard'):
    """
    Compute contrastive loss for EEG alignment using in-batch negatives.

    For EEG-EEG: supports multi_positive_train to handle multiple subjects reading
    the same sentence as co-positives.

    For EEG-Text: supports three loss modes via text_loss_mode:
        'standard'      - Standard CE. Same-sentence duplicates in batch treated as hard
                          negatives. Gradient impact is degenerate (identical doc vectors)
                          but loss is technically inconsistent.
        'masked'        - CE with same-sentence off-diagonal positions masked out of the
                          denominator. Neither positive nor negative — excluded from loss.
        'multi_positive'- Same multi-positive CE used for EEG-EEG. Same-sentence entries
                          in batch marked as co-positives in the labels matrix.
    """
    if pooling_strategy == 'multi':
        batch_size = len(query_vectors)
        device = query_vectors[0].device
    else:
        batch_size = query_vectors.size(0)
        device = query_vectors.device

    # Compute similarities between queries and documents (diagonal)
    query_to_doc_sims = []
    for i in range(batch_size):
        if pooling_strategy == 'multi':
            query_i = query_vectors[i]
            doc_i = doc_vectors[i]
        else:
            query_i = query_vectors[i:i + 1]
            doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_to_doc_sims.append(sim[0])

    similarities = torch.stack(query_to_doc_sims)

    # Build full batch similarity matrix (logits)
    logits = torch.zeros(batch_size, batch_size, device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            if pooling_strategy == 'multi':
                query_i = query_vectors[i]
                doc_j = doc_vectors[j]
            else:
                query_i = query_vectors[i:i + 1]
                doc_j = doc_vectors[j:j + 1]

            sim = compute_similarity([query_i], [doc_j], pooling_strategy, temperature=1.0)
            logits[i, j] = sim[0] / temperature

    # ── DEBUG LOGGING ────────────────────────────────────────────────────────
    if debug_step:
        print("\n" + "=" * 80)
        print("🔍 CONTRASTIVE LOSS DEBUG")
        print("=" * 80)

        print(f"\n📊 RAW SIMILARITIES (temperature=1.0):")
        print(f"  Diagonal (positive pairs):  {torch.diagonal(logits * temperature).detach().cpu().numpy()}")
        print(f"  Min: {(logits * temperature).min().item():.4f}")
        print(f"  Max: {(logits * temperature).max().item():.4f}")
        print(f"  Mean: {(logits * temperature).mean().item():.4f}")
        print(f"  Std: {(logits * temperature).std().item():.4f}")

        print(f"\n🌡️  LOGITS (after dividing by temperature={temperature}):")
        print(f"  Diagonal (positive pairs):  {torch.diagonal(logits).detach().cpu().numpy()}")
        print(f"  Min: {logits.min().item():.4f}")
        print(f"  Max: {logits.max().item():.4f}")
        print(f"  Mean: {logits.mean().item():.4f}")
        print(f"  Std: {logits.std().item():.4f}")

        exp_logits = torch.exp(logits)
        print(f"\n💥 EXPONENTIALS (exp(logits)):")
        print(f"  Diagonal (positive pairs):  {torch.diagonal(exp_logits).detach().cpu().numpy()}")
        print(f"  Min: {exp_logits.min().item():.2e}")
        print(f"  Max: {exp_logits.max().item():.2e}")
        print(f"  Mean: {exp_logits.mean().item():.2e}")
        print(f"  RANGE (max/min): {(exp_logits.max() / exp_logits.min()).item():.2e}")

        if exp_logits.max().item() > 1e10:
            print(f"  ⚠️  WARNING: Exponentials > 1e10 (numerical instability risk!)")
        if (exp_logits.max() / exp_logits.min()).item() > 1e8:
            print(f"  ⚠️  WARNING: Exponential range > 1e8 (gradient explosion risk!)")

    # ── DETERMINE DOCUMENT TYPE ───────────────────────────────────────────────
    document_type = 'text'
    if batch_metadata is not None:
        document_type = batch_metadata[0].get('document_type', 'text')

    # ── EEG-EEG: MULTI-POSITIVE LOSS ─────────────────────────────────────────
    if multi_positive_train and batch_metadata is not None and document_type == 'eeg':
        sentence_ids = [meta.get('sentence_id', -1) for meta in batch_metadata]
        query_participant_ids = [meta.get('query_participant_id', 'unknown') for meta in batch_metadata]
        doc_participant_ids = [meta.get('doc_participant_id', 'unknown') for meta in batch_metadata]
        subject_mode = batch_metadata[0].get('subject_mode', 'within-subject')

        # labels_matrix[i,j] = 1 if doc j is a valid positive for query i.
        # inclusion_mask[i,j] = True if doc j should appear in query i's denominator.
        #
        # In cross-subject mode, any doc j whose participant matches query i's
        # participant is excluded from BOTH — it is neither a positive nor a
        # negative for query i, and does not contribute to the loss at all.
        labels_matrix = torch.zeros(batch_size, batch_size, device=device)
        inclusion_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)

        for i in range(batch_size):
            for j in range(batch_size):
                if subject_mode == 'cross-subject' and doc_participant_ids[j] == query_participant_ids[i]:
                    # Own-subject doc: completely invisible to this query.
                    inclusion_mask[i, j] = False
                elif sentence_ids[i] == sentence_ids[j] and sentence_ids[i] != -1:
                    labels_matrix[i, j] = 1.0

        if debug_step:
            print(f"\n🎯 MULTI-POSITIVE TRAINING (EEG-EEG, {subject_mode}):")
            positives_per_query = labels_matrix.sum(dim=1)
            excluded_per_query = (~inclusion_mask).sum(dim=1)
            print(f"  Positives per query:        {positives_per_query.detach().cpu().numpy()}")
            print(f"  Excluded (own-subj) per row: {excluded_per_query.detach().cpu().numpy()}")
            print(f"  Avg positives per query:    {positives_per_query.mean().item():.2f}")
            print(f"  Sentence IDs in batch:      {set(sentence_ids)}")

        exp_logits = torch.exp(logits)
        losses = []
        for i in range(batch_size):
            positive_mask = labels_matrix[i] > 0
            num_positives = positive_mask.sum()

            if num_positives > 0:
                positive_sum = (exp_logits[i] * positive_mask).sum()
                # Denominator only includes positions not masked out —
                # own-subject docs are completely absent from this sum.
                all_sum = (exp_logits[i] * inclusion_mask[i]).sum()
                loss_i = -torch.log(positive_sum / all_sum)
                losses.append(loss_i)

                if debug_step:
                    print(f"\n  Query {i}:")
                    print(f"    Positives:    {num_positives.item()}")
                    print(f"    Excluded:     {(~inclusion_mask[i]).sum().item()}")
                    print(f"    Positive sum: {positive_sum.item():.2e}")
                    print(f"    All sum:      {all_sum.item():.2e}")
                    print(f"    Ratio:        {(positive_sum / all_sum).item():.4f}")
                    print(f"    Loss:         {loss_i.item():.4f}")

        if losses:
            loss = torch.stack(losses).mean()
        else:
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(logits, labels)

    # ── EEG-TEXT: MASKED CE ───────────────────────────────────────────────────
    elif document_type == 'text' and text_loss_mode == 'masked' and batch_metadata is not None:
        sentence_ids = [meta.get('sentence_id', -1) for meta in batch_metadata]

        # inclusion_mask[i,j] = True  → include position j in row i's denominator
        # inclusion_mask[i,j] = False → exclude (same sentence, off-diagonal)
        inclusion_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and sentence_ids[i] == sentence_ids[j] and sentence_ids[i] != -1:
                    inclusion_mask[i, j] = False

        if debug_step:
            masked_count = (~inclusion_mask).sum().item() - 0  # off-diagonal masks
            print(f"\n🎭 MASKED CE (EEG-TEXT):")
            print(f"  Sentence IDs in batch: {sentence_ids}")
            print(f"  Off-diagonal positions masked: {(~inclusion_mask).sum().item()}")
            print(f"  text_loss_mode: masked")

        losses = []
        for i in range(batch_size):
            valid_mask = inclusion_mask[i]           # [batch_size] bool
            valid_logits = logits[i][valid_mask]     # subset of logit row
            positive_logit = logits[i, i]            # diagonal always included
            loss_i = -positive_logit + torch.logsumexp(valid_logits, dim=0)
            losses.append(loss_i)

        loss = torch.stack(losses).mean()

    # ── EEG-TEXT: MULTI-POSITIVE CE ───────────────────────────────────────────
    elif document_type == 'text' and text_loss_mode == 'multi_positive' and batch_metadata is not None:
        sentence_ids = [meta.get('sentence_id', -1) for meta in batch_metadata]

        labels_matrix = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if sentence_ids[i] == sentence_ids[j] and sentence_ids[i] != -1:
                    labels_matrix[i, j] = 1.0

        if debug_step:
            print(f"\n🎯 MULTI-POSITIVE TRAINING (EEG-TEXT):")
            positives_per_query = labels_matrix.sum(dim=1)
            print(f"  Positives per query: {positives_per_query.detach().cpu().numpy()}")
            print(f"  Avg positives per query: {positives_per_query.mean().item():.2f}")
            print(f"  Sentence IDs in batch: {set(sentence_ids)}")

        exp_logits = torch.exp(logits)
        losses = []
        for i in range(batch_size):
            positive_mask = labels_matrix[i] > 0
            num_positives = positive_mask.sum()

            if num_positives > 0:
                positive_sum = (exp_logits[i] * positive_mask).sum()
                all_sum = exp_logits[i].sum()
                loss_i = -torch.log(positive_sum / all_sum)
                losses.append(loss_i)

                if debug_step:
                    print(f"\n  Query {i}:")
                    print(f"    Positives: {num_positives}")
                    print(f"    Positive sum: {positive_sum.item():.2e}")
                    print(f"    All sum: {all_sum.item():.2e}")
                    print(f"    Ratio: {(positive_sum / all_sum).item():.4f}")
                    print(f"    Loss: {loss_i.item():.4f}")

        if losses:
            loss = torch.stack(losses).mean()
        else:
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(logits, labels)

    # ── STANDARD CE (default for all other cases) ─────────────────────────────
    else:
        labels = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(logits, labels)

        if debug_step:
            print(f"\n🎯 STANDARD CONTRASTIVE LOSS:")
            print(f"  document_type: {document_type}")
            print(f"  text_loss_mode: {text_loss_mode}")
            print(f"  Labels: {labels.detach().cpu().numpy()}")
            print(f"  Loss: {loss.item():.4f}")

    # ── FINAL DEBUG ───────────────────────────────────────────────────────────
    if debug_step:
        print(f"\n📉 FINAL LOSS:")
        print(f"  Value: {loss.item():.4f}")
        print(f"  Requires grad: {loss.requires_grad}")
        print("=" * 80 + "\n")

    return loss, similarities


def compute_alignment_metrics(query_vectors, doc_vectors, pooling_strategy, document_type):
    """Compute alignment metrics between query and document representations"""

    if pooling_strategy == 'multi':
        batch_size = len(query_vectors)
    else:
        batch_size = query_vectors.size(0)

    query_doc_sims = []
    for i in range(batch_size):
        if pooling_strategy == 'multi':
            query_i = query_vectors[i]
            doc_i = doc_vectors[i]
        else:
            query_i = query_vectors[i:i + 1]
            doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_doc_sims.append(sim[0].item())

    metrics = {
        f'eeg_{document_type}_similarity': np.mean(query_doc_sims),
        f'eeg_{document_type}_similarity_std': np.std(query_doc_sims),
        'query_doc_similarity': np.mean(query_doc_sims),
        'query_doc_similarity_std': np.std(query_doc_sims)
    }

    return metrics


def train_step(model, batch, optimizer, device, step_num, debug=False,
               multi_positive_train=False, text_loss_mode='standard'):
    """Single training step WITH GRADIENT ANALYSIS"""

    # Move batch to device
    if batch['doc_eegs'] is not None:
        batch['doc_eegs'] = batch['doc_eegs'].to(device)
    if batch['doc_text_tokens'] is not None:
        batch['doc_text_tokens'] = {k: v.to(device) for k, v in batch['doc_text_tokens'].items()}
    batch['query_eegs'] = batch['query_eegs'].to(device)

    document_type = batch['document_type']

    debug_this_step = debug or (step_num % 50 == 0)

    if debug_this_step:
        subject_mode = batch['metadata'][0]['subject_mode']
        print(f"\n{'=' * 80}")
        print(f"🔍 DEBUG STEP {step_num} (EEG-{document_type.upper()}, {subject_mode})")
        print(f"{'=' * 80}")
        print(f"  Query EEGs: {batch['query_eegs'].shape}")
        if document_type == 'eeg':
            print(f"  Doc EEGs: {batch['doc_eegs'].shape}")
        else:
            print(f"  Doc text tokens: {batch['doc_text_tokens']['input_ids'].shape}")
        print(f"  Pooling strategy: {model.pooling_strategy}")
        print(f"  Multi-positive training (EEG-EEG): {'ENABLED' if multi_positive_train else 'DISABLED'}")
        if document_type == 'text':
            print(f"  Text loss mode: {text_loss_mode.upper()}")

    # Forward pass
    outputs = model(batch)

    if debug_this_step:
        print(f"\n📤 MODEL OUTPUTS:")
        print(f"  Query vectors type: {type(outputs['query_vectors'])}")
        print(f"  Doc vectors type: {type(outputs['doc_vectors'])}")
        if isinstance(outputs['query_vectors'], list):
            print(f"  Query vector 0 shape: {outputs['query_vectors'][0].shape}")
            print(f"  Doc vector 0 shape: {outputs['doc_vectors'][0].shape}")
        else:
            print(f"  Query vectors shape: {outputs['query_vectors'].shape}")
            print(f"  Doc vectors shape: {outputs['doc_vectors'].shape}")

    # Compute contrastive loss
    loss, query_sims = compute_contrastive_loss(
        outputs['query_vectors'],
        outputs['doc_vectors'],
        model.pooling_strategy,
        batch_metadata=batch['metadata'],
        multi_positive_train=multi_positive_train,
        debug_step=debug_this_step,
        text_loss_mode=text_loss_mode
    )

    # Compute alignment metrics
    metrics = compute_alignment_metrics(
        outputs['query_vectors'],
        outputs['doc_vectors'],
        model.pooling_strategy,
        document_type
    )

    if debug_this_step:
        print(f"\n⏪ BEFORE BACKWARD:")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss requires_grad: {loss.requires_grad}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    if debug_this_step:
        print(f"\n🔬 GRADIENT ANALYSIS (before clipping):")
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        grad_info = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_max = param.grad.data.abs().max().item()
                param_min = param.grad.data.abs().min().item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, param_max)
                min_grad = min(min_grad, param_min)

                if 'eeg_encoder' in name or 'projection' in name:
                    grad_info[name] = {
                        'norm': param_norm,
                        'max': param_max,
                        'min': param_min,
                        'mean': param.grad.data.abs().mean().item()
                    }

        total_norm = total_norm ** 0.5
        print(f"  Total gradient norm (unclipped): {total_norm:.4f}")
        print(f"  Max gradient value: {max_grad:.6f}")
        print(f"  Min gradient value: {min_grad:.6f}")

        if min_grad > 0:
            grad_range = max_grad / min_grad
            print(f"  Gradient range (max/min): {grad_range:.2e}")
        else:
            print(f"  Gradient range (max/min): inf (some gradients are zero)")

        print(f"\n  Key layer gradients:")
        for name, info in list(grad_info.items())[:5]:
            print(f"    {name[:50]:50s} norm={info['norm']:.4f} max={info['max']:.6f}")

        if total_norm > 10.0:
            print(f"  ⚠️  WARNING: Large gradients detected (norm={total_norm:.2f})")
        if max_grad > 1.0:
            print(f"  ⚠️  WARNING: Gradient values > 1.0 (max={max_grad:.4f})")
        if min_grad > 0 and (max_grad / min_grad) > 1e6:
            print(f"  ⚠️  WARNING: Huge gradient range (may cause instability)")

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    if debug_this_step:
        print(f"\n✂️  AFTER GRADIENT CLIPPING:")
        print(f"  Gradient norm (clipped): {grad_norm.item():.4f}")
        if grad_norm.item() > 1.0:
            print(f"  ⚠️  Clipping applied! (was > 1.0)")

    optimizer.step()

    log_dict = {
        'train/loss': loss.item(),
        f'train/eeg_{document_type}_similarity': metrics[f'eeg_{document_type}_similarity'],
        'train/query_doc_similarity': metrics['query_doc_similarity'],
        'train/grad_norm': grad_norm.item(),
        'train/step': step_num
    }

    wandb.log(log_dict)

    if debug_this_step:
        print(f"\n📊 METRICS:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  EEG-{document_type.upper()} similarity: {metrics[f'eeg_{document_type}_similarity']:.4f}")
        print(f"  Grad norm: {grad_norm.item():.4f}")
        meta = batch['metadata'][0]
        print(f"  Sample query: '{meta['query_text'][:50]}...'")
        print(f"  Query participant: {meta['query_participant_id']}")
        print(f"  Doc participant: {meta['doc_participant_id']}")
        print(f"{'=' * 80}\n")

    return loss.item(), metrics, grad_norm.item()


def debug_temperature_sensitivity(model, batch, device, temperatures=[0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]):
    """
    Test how different temperatures affect the loss landscape
    Call this during training to diagnose temperature issues
    """
    print("\n" + "=" * 80)
    print("🌡️  TEMPERATURE SENSITIVITY ANALYSIS")
    print("=" * 80)

    model.eval()

    if batch['doc_eegs'] is not None:
        batch['doc_eegs'] = batch['doc_eegs'].to(device)
    if batch['doc_text_tokens'] is not None:
        batch['doc_text_tokens'] = {k: v.to(device) for k, v in batch['doc_text_tokens'].items()}
    batch['query_eegs'] = batch['query_eegs'].to(device)

    with torch.no_grad():
        outputs = model(batch)

        print(f"\nTesting {len(temperatures)} different temperatures:")
        print(f"{'Temperature':>12} {'Loss':>10} {'Max Logit':>12} {'Max Exp':>15} {'Exp Range':>15}")
        print("-" * 80)

        results = []
        for temp in temperatures:
            loss, sims = compute_contrastive_loss(
                outputs['query_vectors'],
                outputs['doc_vectors'],
                model.pooling_strategy,
                temperature=temp,
                batch_metadata=batch['metadata'],
                multi_positive_train=False,
                debug_step=False,
                text_loss_mode='standard'
            )

            batch_size = len(outputs['query_vectors']) if isinstance(outputs['query_vectors'], list) else \
                outputs['query_vectors'].size(0)

            logits = torch.zeros(batch_size, batch_size, device=device)
            for i in range(batch_size):
                for j in range(batch_size):
                    if isinstance(outputs['query_vectors'], list):
                        q_i = outputs['query_vectors'][i]
                        d_j = outputs['doc_vectors'][j]
                    else:
                        q_i = outputs['query_vectors'][i:i + 1]
                        d_j = outputs['doc_vectors'][j:j + 1]
                    sim = compute_similarity([q_i], [d_j], model.pooling_strategy, temperature=1.0)
                    logits[i, j] = sim[0] / temp

            exp_logits = torch.exp(logits)
            max_logit = logits.max().item()
            max_exp = exp_logits.max().item()
            exp_range = (exp_logits.max() / exp_logits.min()).item()

            results.append({
                'temp': temp,
                'loss': loss.item(),
                'max_logit': max_logit,
                'max_exp': max_exp,
                'exp_range': exp_range
            })

            stability = "✓" if exp_range < 1e6 and max_exp < 1e8 else "⚠️" if exp_range < 1e8 else "❌"
            print(
                f"{temp:>12.3f} {loss.item():>10.4f} {max_logit:>12.2f} {max_exp:>15.2e} {exp_range:>15.2e} {stability}")

        print("\n💡 RECOMMENDATIONS:")
        stable_temps = [r for r in results if r['exp_range'] < 1e6 and r['max_exp'] < 1e8]
        if stable_temps:
            print(f"  ✓ Stable temperatures (exp_range < 1e6): {[r['temp'] for r in stable_temps]}")
            print(f"  📌 Recommended: temperature >= {min(r['temp'] for r in stable_temps):.2f}")
        else:
            print(f"  ⚠️  All temperatures show instability - consider even higher temperatures")

        unstable_temps = [r for r in results if r['exp_range'] > 1e8 or r['max_exp'] > 1e10]
        if unstable_temps:
            print(f"  ❌ Unstable temperatures (risk of explosion): {[r['temp'] for r in unstable_temps]}")

        print("=" * 80 + "\n")

    model.train()
    return results


def validation_step(model, batch, device, text_loss_mode='standard'):
    """Single validation step"""

    if batch['doc_eegs'] is not None:
        batch['doc_eegs'] = batch['doc_eegs'].to(device)
    if batch['doc_text_tokens'] is not None:
        batch['doc_text_tokens'] = {k: v.to(device) for k, v in batch['doc_text_tokens'].items()}
    batch['query_eegs'] = batch['query_eegs'].to(device)

    document_type = batch['document_type']

    with torch.no_grad():
        outputs = model(batch)

        loss, query_sims = compute_contrastive_loss(
            outputs['query_vectors'],
            outputs['doc_vectors'],
            model.pooling_strategy,
            batch_metadata=batch['metadata'],
            multi_positive_train=False,   # validation always uses single-positive diagonal
            debug_step=False,
            text_loss_mode=text_loss_mode
        )

        metrics = compute_alignment_metrics(
            outputs['query_vectors'],
            outputs['doc_vectors'],
            model.pooling_strategy,
            document_type
        )

    return loss.item(), metrics


# ==========================================
# RANKING EVALUATION
# ==========================================

def build_document_database(dataloader, multi_positive_eval=False):
    """Build unique document database for ranking evaluation"""

    print("Building document database for ranking evaluation...")

    unique_docs = {}
    query_to_doc_mapping = {}
    sentence_to_doc_indices = {}
    query_idx = 0

    for batch in dataloader:
        batch_size = len(batch['metadata'])
        document_type = batch['document_type']

        for sample_idx in range(batch_size):
            metadata = batch['metadata'][sample_idx]
            sentence_id = metadata.get('sentence_id', -1)

            if document_type == 'text':
                doc_key = metadata['doc_text'].strip()
                doc_data = {
                    'input_ids': batch['doc_text_tokens']['input_ids'][sample_idx].clone(),
                    'attention_mask': batch['doc_text_tokens']['attention_mask'][sample_idx].clone()
                }
            else:  # eeg
                doc_key = f"{metadata['doc_participant_id']}_{metadata['sentence_id']}"
                doc_data = {
                    'eeg': batch['doc_eegs'][sample_idx].clone()
                }

            if doc_key and doc_key not in unique_docs:
                unique_doc_idx = len(unique_docs)
                unique_docs[doc_key] = {
                    'idx': unique_doc_idx,
                    'key': doc_key,
                    'type': document_type,
                    'data': doc_data,
                    'sentence_id': sentence_id,
                    'participant_id': metadata.get('doc_participant_id', 'unknown')
                }

                if sentence_id not in sentence_to_doc_indices:
                    sentence_to_doc_indices[sentence_id] = []
                sentence_to_doc_indices[sentence_id].append(unique_doc_idx)
            else:
                unique_doc_idx = unique_docs[doc_key]['idx']

            # Multi-positive eval: applies to EEG-EEG only.
            # For EEG-Text, text deduplicates to one doc per sentence so
            # multi_positive_eval has no practical effect — single int mapping is correct.
            if multi_positive_eval and document_type == 'eeg':
                query_to_doc_mapping[query_idx] = {
                    'sentence_id': sentence_id,
                    'primary_doc_idx': unique_doc_idx,
                    'query_participant_id': metadata.get('query_participant_id', 'unknown'),
                    'subject_mode': metadata.get('subject_mode', 'within-subject')
                }
            else:
                query_to_doc_mapping[query_idx] = unique_doc_idx

            query_idx += 1

    doc_list = [None] * len(unique_docs)
    for doc_key, doc_info in unique_docs.items():
        doc_list[doc_info['idx']] = doc_info

    if multi_positive_eval:
        expanded_mapping = {}
        query_eval_metadata = {}  # q_idx -> {query_participant_id, subject_mode}
        own_doc_filtered_count = 0

        for q_idx, mapping_info in query_to_doc_mapping.items():
            if isinstance(mapping_info, dict):
                sentence_id = mapping_info['sentence_id']
                query_participant_id = mapping_info.get('query_participant_id', 'unknown')
                subject_mode = mapping_info.get('subject_mode', 'within-subject')
                all_doc_indices = sentence_to_doc_indices.get(sentence_id, [mapping_info['primary_doc_idx']])

                if subject_mode == 'cross-subject':
                    # In cross-subject eval, exclude the query subject's own EEG reading
                    # from the positive set. The task is to retrieve OTHER subjects'
                    # readings of the same sentence — not to recognise your own signal.
                    relevant_doc_indices = [
                        doc_idx for doc_idx in all_doc_indices
                        if doc_list[doc_idx]['participant_id'] != query_participant_id
                    ]
                    own_doc_filtered_count += len(all_doc_indices) - len(relevant_doc_indices)

                    if not relevant_doc_indices:
                        # Should not happen in a well-formed cross-subject dataset,
                        # but fall back to the paired doc to avoid a zero-positive query.
                        relevant_doc_indices = [mapping_info['primary_doc_idx']]
                else:
                    # Within-subject: keep all docs for this sentence as positives.
                    relevant_doc_indices = all_doc_indices

                expanded_mapping[q_idx] = relevant_doc_indices
                query_eval_metadata[q_idx] = {
                    'query_participant_id': query_participant_id,
                    'subject_mode': subject_mode
                }
            else:
                expanded_mapping[q_idx] = [mapping_info]
                query_eval_metadata[q_idx] = {
                    'query_participant_id': 'unknown',
                    'subject_mode': 'within-subject'
                }

        query_to_doc_mapping = expanded_mapping

        print(f"Multi-positive evaluation enabled:")
        print(f"  Found {len(sentence_to_doc_indices)} unique sentences")
        avg_docs_per_sentence = np.mean([len(docs) for docs in sentence_to_doc_indices.values()])
        print(f"  Average documents per sentence: {avg_docs_per_sentence:.2f}")
        if own_doc_filtered_count > 0:
            print(f"  Cross-subject: filtered out {own_doc_filtered_count} own-subject doc(s) from positive sets")
    else:
        query_eval_metadata = {}

    print(f"Found {len(doc_list)} unique documents for {len(query_to_doc_mapping)} queries")
    return doc_list, query_to_doc_mapping, sentence_to_doc_indices if multi_positive_eval else {}, query_eval_metadata


def generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size=100, seed=42,
                                multi_positive_eval=False, query_eval_metadata=None):
    """Generate consistent document subsets for fair ranking evaluation.

    In cross-subject mode the query subject's own EEG document is excluded from
    BOTH the positive set (handled in build_document_database) AND the negative
    candidate pool here. It must be completely invisible to that query — not
    reclassified as a negative, simply absent from the pool entirely.
    """

    print(f"Generating consistent document subsets (subset_size={subset_size}, seed={seed})...")

    random.seed(seed)

    if query_eval_metadata is None:
        query_eval_metadata = {}

    query_subsets = {}

    for query_idx, correct_doc_info in query_to_doc_mapping.items():
        if multi_positive_eval and isinstance(correct_doc_info, list):
            correct_doc_indices = correct_doc_info
        else:
            correct_doc_indices = [correct_doc_info] if isinstance(correct_doc_info, int) else correct_doc_info

        doc_subset_indices = correct_doc_indices.copy()

        # Retrieve per-query subject info so we can filter the negative pool.
        meta = query_eval_metadata.get(query_idx, {})
        query_participant_id = meta.get('query_participant_id', None)
        subject_mode = meta.get('subject_mode', 'within-subject')

        # In cross-subject mode, exclude the query subject's own doc from the
        # negative pool entirely — it must not appear anywhere in the pool.
        if subject_mode == 'cross-subject' and query_participant_id is not None:
            negative_candidates = [
                i for i in range(len(doc_list))
                if i not in correct_doc_indices
                and doc_list[i]['participant_id'] != query_participant_id
            ]
        else:
            negative_candidates = [i for i in range(len(doc_list)) if i not in correct_doc_indices]

        if negative_candidates:
            num_negatives_needed = max(0, subset_size - len(correct_doc_indices))
            random_negatives = random.sample(negative_candidates,
                                             min(num_negatives_needed, len(negative_candidates)))
            doc_subset_indices.extend(random_negatives)

        query_subsets[query_idx] = doc_subset_indices

    print(f"Generated {len(query_subsets)} consistent subsets")
    return query_subsets


def collect_queries(dataloader, device):
    """Collect all queries from dataloader"""

    print(f"Collecting queries from dataloader...")

    queries = []
    query_idx = 0

    for batch in dataloader:
        batch_size = batch['query_eegs'].size(0)
        for sample_idx in range(batch_size):
            query_eeg = batch['query_eegs'][sample_idx:sample_idx + 1].to(device)
            queries.append(query_eeg)
            query_idx += 1

    print(f"Collected {len(queries)} queries")
    return queries


def rank_documents_for_query(model, query_eeg, doc_list, doc_indices, device):
    """Rank pre-selected documents for a single query"""

    model.eval()

    with torch.no_grad():
        if model.document_type == 'eeg':
            query_vectors = model.encode_eeg(query_eeg, is_document=False)
        else:
            query_vectors = model.encode_eeg(query_eeg, is_document=False)

        if isinstance(query_vectors, list):
            query_vectors = query_vectors[0]
        else:
            query_vectors = query_vectors[0:1]

        doc_scores = []

        for doc_idx in doc_indices:
            doc_info = doc_list[doc_idx]

            if doc_info['type'] == 'text':
                input_ids = doc_info['data']['input_ids'].unsqueeze(0).to(device)
                attention_mask = doc_info['data']['attention_mask'].unsqueeze(0).to(device)
                doc_vectors = model.encode_text(input_ids, attention_mask)
            else:
                doc_eeg = doc_info['data']['eeg'].unsqueeze(0).to(device)
                doc_vectors = model.encode_eeg(doc_eeg, is_document=True)

            if isinstance(doc_vectors, list):
                doc_vectors = doc_vectors[0]
            else:
                doc_vectors = doc_vectors[0:1]

            sim = compute_similarity([query_vectors], [doc_vectors], model.pooling_strategy, temperature=1.0)
            doc_scores.append((doc_idx, sim[0].item()))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [doc_idx for doc_idx, score in doc_scores]
    ranked_scores = [score for doc_idx, score in doc_scores]

    return ranked_indices, ranked_scores


def compute_ranking_metrics(ranked_doc_indices, correct_doc_indices, k_values=[1, 5, 10, 20]):
    """
    Compute ranking metrics for a single query

    Args:
        ranked_doc_indices: List of doc indices in ranked order
        correct_doc_indices: List of all correct doc indices (can be single or multiple)
    """

    metrics = {}

    if isinstance(correct_doc_indices, int):
        correct_doc_indices = [correct_doc_indices]

    num_relevant = len(correct_doc_indices)

    first_correct_rank = float('inf')
    for correct_idx in correct_doc_indices:
        try:
            rank = ranked_doc_indices.index(correct_idx) + 1
            first_correct_rank = min(first_correct_rank, rank)
        except ValueError:
            continue

    if first_correct_rank == float('inf'):
        first_correct_rank = len(ranked_doc_indices) + 1

    metrics['rr'] = 1.0 / first_correct_rank if first_correct_rank <= len(ranked_doc_indices) else 0.0
    metrics['rank_of_correct'] = first_correct_rank

    for k in k_values:
        if k <= len(ranked_doc_indices):
            top_k = ranked_doc_indices[:k]
            num_relevant_in_top_k = sum(1 for doc_idx in top_k if doc_idx in correct_doc_indices)

            hit_at_k = 1.0 if num_relevant_in_top_k > 0 else 0.0
            precision_at_k = num_relevant_in_top_k / k
            recall_at_k = num_relevant_in_top_k / num_relevant if num_relevant > 0 else 0.0

            metrics[f'hit_at_{k}'] = hit_at_k
            metrics[f'precision_at_{k}'] = precision_at_k
            metrics[f'recall_at_{k}'] = recall_at_k
        else:
            metrics[f'hit_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0

    return metrics


def perform_ranking_evaluation(model, dataloader, device, epoch_num, subset_size=100, multi_positive_eval=False):
    """Perform ranking evaluation"""

    print(f"\n=== RANKING EVALUATION (Epoch {epoch_num}) ===")
    print(f"Multi-positive evaluation: {'ENABLED ✅' if multi_positive_eval else 'DISABLED ❌'}")

    doc_list, query_to_doc_mapping, sentence_to_doc_indices, query_eval_metadata = build_document_database(
        dataloader, multi_positive_eval)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    if len(doc_list) < subset_size:
        print(f"Warning: Only {len(doc_list)} documents available, using all")
        subset_size = len(doc_list)

    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size,
                                                multi_positive_eval=multi_positive_eval,
                                                query_eval_metadata=query_eval_metadata)
    queries = collect_queries(dataloader, device)

    print(f"Ranking evaluation: {len(queries)} queries against {subset_size} documents each")
    print(f"Document type: {model.document_type.upper()}")

    all_metrics = []

    for query_idx, query_eeg in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_info = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_documents_for_query(
            model, query_eeg, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_info)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    ranking_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        ranking_metrics[f'ranking/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            ranking_metrics[f'ranking/{metric_name}_std'] = np.std(values)

    ranking_metrics.update({
        'ranking/num_unique_documents': len(doc_list),
        'ranking/num_queries_evaluated': len(all_metrics),
        'ranking/epoch_num': epoch_num,
        'ranking/subset_size': subset_size,
        'ranking/document_type': model.document_type,
        'ranking/multi_positive_eval': multi_positive_eval
    })

    print(f"Ranking Results (EEG-{model.document_type.upper()}):")
    if multi_positive_eval:
        print(f"  [Multi-Positive Mode] All recordings of same sentence treated as relevant")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {ranking_metrics['ranking/rr']:.4f}")
    print(f"  Hit@1: {ranking_metrics['ranking/hit_at_1']:.4f}")
    print(f"  Hit@5: {ranking_metrics['ranking/hit_at_5']:.4f}")
    print(f"  Hit@10: {ranking_metrics['ranking/hit_at_10']:.4f}")
    print(f"  Hit@20: {ranking_metrics['ranking/hit_at_20']:.4f}")

    wandb.log(ranking_metrics)

    return ranking_metrics


# ==========================================
# MAIN TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, dataloader, optimizer, device, epoch_num, total_epochs, debug=False,
                multi_positive_train=False, text_loss_mode='standard'):
    """Train for a single epoch WITH PERIODIC DEBUGGING"""

    model.train()
    total_loss = 0
    num_batches = 0
    epoch_similarities = []
    epoch_grad_norms = []

    document_type = None

    if epoch_num == 1 and num_batches == 0:
        first_batch = next(iter(dataloader))
        print("\n🔬 Running temperature sensitivity analysis on first batch...")
        debug_temperature_sensitivity(model, first_batch, device)

    for batch_idx, batch in enumerate(dataloader):
        debug_this_batch = debug and epoch_num == 1 and batch_idx == 0

        step_num = (epoch_num - 1) * len(dataloader) + batch_idx

        if document_type is None:
            document_type = batch['document_type']

        loss, metrics, grad_norm = train_step(
            model, batch, optimizer, device, step_num,
            debug=debug_this_batch,
            multi_positive_train=multi_positive_train,
            text_loss_mode=text_loss_mode
        )

        total_loss += loss
        num_batches += 1
        epoch_similarities.append(metrics[f'eeg_{document_type}_similarity'])
        epoch_grad_norms.append(grad_norm)

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss:.4f}, EEG-{document_type.upper()} Sim: {metrics[f'eeg_{document_type}_similarity']:.4f}, "
                  f"Grad Norm: {grad_norm:.2f}")

    avg_loss = total_loss / num_batches
    avg_similarity = np.mean(epoch_similarities)
    avg_grad_norm = np.mean(epoch_grad_norms)
    max_grad_norm = np.max(epoch_grad_norms)

    print(f"\n📊 Epoch {epoch_num} Summary:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg EEG-{document_type.upper()} Sim: {avg_similarity:.4f}")
    print(f"  Avg Grad Norm: {avg_grad_norm:.2f}")
    print(f"  Max Grad Norm: {max_grad_norm:.2f}")

    if max_grad_norm > 5.0:
        print(f"  ⚠️  WARNING: High gradient norms detected (max={max_grad_norm:.2f})")

    return avg_loss, avg_similarity, avg_grad_norm


def validate_epoch(model, val_dataloader, device, epoch_num, text_loss_mode='standard'):
    """Validate for a single epoch"""

    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    val_similarities = []

    document_type = None

    for batch in val_dataloader:
        if document_type is None:
            document_type = batch['document_type']

        val_loss, val_metrics = validation_step(model, batch, device, text_loss_mode=text_loss_mode)

        total_val_loss += val_loss
        num_val_batches += 1
        val_similarities.append(val_metrics[f'eeg_{document_type}_similarity'])

    avg_val_loss = total_val_loss / num_val_batches
    avg_val_similarity = np.mean(val_similarities)

    wandb.log({
        'val/loss': avg_val_loss,
        f'val/eeg_{document_type}_similarity': avg_val_similarity,
        'val/epoch_num': epoch_num
    })

    print(f"Validation completed. Val Loss: {avg_val_loss:.4f}, "
          f"Val EEG-{document_type.upper()} Sim: {avg_val_similarity:.4f}")

    return avg_val_loss, avg_val_similarity


def initialize_wandb(config):
    """Initialize wandb logging with dataset name and subject mode"""

    dataset_name = config.get('dataset_name', 'Unknown')
    document_type = config.get('document_type', 'text')
    subject_mode = config.get('subject_mode', 'within-subject')
    holdout_subjects = config.get('holdout_subjects', False)
    eeg_arch = config.get('eeg_arch', 'simple')
    pooling_strategy = config.get('pooling_strategy', 'multi')
    version = config.get('version', 'v1')
    multi_positive_eval = config.get('multi_positive_eval', False)
    multi_positive_train = config.get('multi_positive_train', False)
    text_loss_mode = config.get('text_loss_mode', 'standard')
    doc_encoder_type = config.get('doc_encoder_type', 'bert')
    freeze_doc_encoder = config.get('freeze_doc_encoder', False)

    alignment_type = f"eeg_{document_type}"
    subject_mode_short = subject_mode.replace('-', '_')
    split_suffix = "_holdout" if holdout_subjects else "_random"

    mp_suffix = ""
    if multi_positive_eval:
        mp_suffix += "_MPeval"
    if multi_positive_train:
        mp_suffix += "_MPtrain"
    # Only append text_loss_mode suffix when non-standard and using text documents
    if document_type == 'text' and text_loss_mode != 'standard':
        mp_suffix += f"_TLM{text_loss_mode}"
    if doc_encoder_type != 'bert':
        mp_suffix += f"_DE{doc_encoder_type}"
    if freeze_doc_encoder:
        mp_suffix += "_frozenDoc"

    run_name = (f"{dataset_name}_{alignment_type}_{subject_mode_short}_"
                f"{pooling_strategy}_{eeg_arch}{split_suffix}{mp_suffix}")

    tags = [
        'eeg-alignment',
        f'eeg-{document_type}',
        f'{subject_mode}',
        f'arch-{eeg_arch}',
        f'pooling-{pooling_strategy}',
        f'dataset-{dataset_name}',
        f'version-{version}',
        'holdout-subjects' if holdout_subjects else 'random-split'
    ]

    if multi_positive_eval:
        tags.append('multi-positive-eval')
    if multi_positive_train:
        tags.append('multi-positive-train')
    if document_type == 'text' and text_loss_mode != 'standard':
        tags.append(f'text-loss-{text_loss_mode}')

    # Determine loss type label for wandb config
    if document_type == 'eeg' and multi_positive_train:
        loss_type = 'multi_positive_contrastive'
    elif document_type == 'text' and text_loss_mode == 'masked':
        loss_type = 'masked_contrastive'
    elif document_type == 'text' and text_loss_mode == 'multi_positive':
        loss_type = 'multi_positive_contrastive'
    else:
        loss_type = 'contrastive'

    wandb.init(
        project="EEG-Alignment",
        name=run_name,
        config={
            'version': version,
            'dataset_name': dataset_name,
            'dataset_path': config.get('dataset_path', 'unknown'),

            'alignment_type': f'EEG-{document_type.upper()}',
            'document_type': document_type,
            'subject_mode': subject_mode,
            'alignment_method': subject_mode,
            'pooling_strategy': pooling_strategy,
            'holdout_subjects': holdout_subjects,
            'split_method': 'holdout_subjects' if holdout_subjects else 'random_samples',
            'multi_positive_eval': multi_positive_eval,
            'multi_positive_train': multi_positive_train,
            'text_loss_mode': text_loss_mode,
            'doc_encoder_type': doc_encoder_type,
            'freeze_doc_encoder': freeze_doc_encoder,

            'colbert_model_name': config.get('colbert_model_name', 'bert-base-uncased'),
            'eeg_arch': eeg_arch,
            'hidden_dim': config.get('hidden_dim', 768),

            'batch_size': config.get('batch_size', 8),
            'learning_rate': config.get('learning_rate', 1e-4),
            'epochs': config.get('epochs', 50),
            'patience': config.get('patience', 10),

            'max_text_len': config.get('max_text_len', 256),
            'max_eeg_len': config.get('max_eeg_len', 50),
            'train_samples': config.get('train_samples', 0),
            'val_samples': config.get('val_samples', 0),
            'test_samples': config.get('test_samples', 0),
            'train_subjects': config.get('train_subjects', 0),
            'val_subjects': config.get('val_subjects', 0),
            'test_subjects': config.get('test_subjects', 0),

            'seed': config.get('seed', 42),
            'loss_type': loss_type,
            'similarity_function': f'{pooling_strategy}_similarity',
            'trainable_params': config.get('trainable_params', 0),
            'total_params': config.get('total_params', 0)
        },
        tags=tags
    )


def train_alignment_model(model, train_dataloader, val_dataloader, test_dataloader,
                          optimizer, num_epochs, patience=10, device='cuda',
                          debug=False, config=None, multi_positive_eval=False,
                          multi_positive_train=False, text_loss_mode='standard'):
    """
    Complete training loop with validation and early stopping.

    Args:
        text_loss_mode: Loss formulation for EEG-Text condition.
            'standard'       - Standard CE (existing behaviour).
            'masked'         - CE with same-sentence off-diagonal positions
                               masked out of the softmax denominator.
            'multi_positive' - Multi-positive CE matching EEG-EEG formulation.
            Ignored when document_type is 'eeg'.
    """

    if config:
        initialize_wandb(config)

    document_type = model.document_type
    pooling_strategy = model.pooling_strategy
    subject_mode = config.get('subject_mode', 'within-subject') if config else 'within-subject'
    dataset_name = config.get('dataset_name', 'Unknown') if config else 'Unknown'

    print(f"Starting EEG-{document_type.upper()} alignment training...")
    print(f"Dataset: {dataset_name}")
    print(f"Subject mode: {subject_mode}")
    print(f"Pooling strategy: {pooling_strategy}")
    print(f"Multi-positive evaluation: {'ENABLED ✅' if multi_positive_eval else 'DISABLED ❌'}")
    print(f"Multi-positive training: {'ENABLED ✅' if multi_positive_train else 'DISABLED ❌'}")
    if document_type == 'text':
        mode_labels = {
            'standard': 'Standard CE (same-sentence treated as hard negative)',
            'masked': 'Masked CE (same-sentence excluded from denominator)',
            'multi_positive': 'Multi-positive CE (same-sentence treated as co-positives)'
        }
        print(f"Text loss mode: {text_loss_mode.upper()} → {mode_labels.get(text_loss_mode, text_loss_mode)}")
    print(f"Early stopping patience: {patience}")

    best_val_loss = float('inf')
    best_val_similarity = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    early_stopped = False

    for epoch in range(num_epochs):
        epoch_num = epoch + 1

        train_loss, train_similarity, train_grad_norm = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch_num,
            total_epochs=num_epochs,
            debug=debug,
            multi_positive_train=multi_positive_train,
            text_loss_mode=text_loss_mode
        )

        val_loss, val_similarity = validate_epoch(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            epoch_num=epoch_num,
            text_loss_mode=text_loss_mode
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_similarity = val_similarity
            best_epoch = epoch_num
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation loss: {best_val_loss:.4f} (epoch {epoch_num})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch_num} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                early_stopped = True

        if epoch_num % 3 == 0:
            ranking_metrics = perform_ranking_evaluation(
                model, val_dataloader, device, epoch_num, subset_size=300,
                multi_positive_eval=multi_positive_eval
            )

        wandb.log({
            'epoch/train_loss': train_loss,
            'epoch/val_loss': val_loss,
            f'epoch/train_eeg_{document_type}_similarity': train_similarity,
            f'epoch/val_eeg_{document_type}_similarity': val_similarity,
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch/epoch_num': epoch_num,
            'epoch/epochs_without_improvement': epochs_without_improvement,
            'epoch/best_val_loss': best_val_loss
        })

        print(f"\nEpoch {epoch_num}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, EEG-{document_type.upper()} Sim: {train_similarity:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, EEG-{document_type.upper()} Sim: {val_similarity:.4f}")
        print(f"  Best  - Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"  Early Stopping: {epochs_without_improvement}/{patience} epochs without improvement")
        print("-" * 70)

        if early_stopped:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")

    print(f"\n=== FINAL TEST EVALUATION ===")
    test_ranking_metrics = perform_ranking_evaluation(
        model, test_dataloader, device, epoch_num=-1, subset_size=300,
        multi_positive_eval=multi_positive_eval
    )

    test_metrics = {}
    for key, value in test_ranking_metrics.items():
        if key.startswith('ranking/'):
            new_key = key.replace('ranking/', 'test_ranking/')
            test_metrics[new_key] = value

    wandb.log(test_metrics)

    wandb.log({
        'training/completed_epochs': epoch_num,
        'training/early_stopped': early_stopped,
        'training/best_epoch': best_epoch,
        'training/best_val_loss': best_val_loss,
        'training/final_val_loss': val_loss
    })

    print(f"\nEEG-{document_type.upper()} alignment training completed!")
    print(f"Dataset: {dataset_name}")
    print(f"Subject mode: {subject_mode}")
    print(f"Multi-positive eval: {'ENABLED ✅' if multi_positive_eval else 'DISABLED ❌'}")
    print(f"Multi-positive train: {'ENABLED ✅' if multi_positive_train else 'DISABLED ❌'}")
    if document_type == 'text':
        print(f"Text loss mode: {text_loss_mode.upper()}")
    if early_stopped:
        print(f"Stopped early after {epoch_num} epochs")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return model


def finish_wandb():
    """Finish wandb run"""
    wandb.finish()


# Import aliases for the controller
train_simplified_model = train_alignment_model
create_simplified_model = lambda **kwargs: None  # Placeholder