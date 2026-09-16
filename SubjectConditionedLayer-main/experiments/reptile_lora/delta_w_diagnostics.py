"""
ΔW heterogeneity diagnostic (the intended contribution).

For each adapted layer, materialise per-subject effective weight deltas
    ΔW_s = (α/r) · B_s ∘ A_s        (conv composition, gauge-correct)
and compute:
    within-subject spread  — same subject, different seeds (noise floor)
    between-subject spread — different subjects (signal)
    ratio = between / within, per layer, before and after EA.

Interpretation:
    ratio ≈ 1   → subjects indistinguishable from seed noise;
                  the centroid is fine (noise-limited regime).
    ratio ≫ 1   → subjects genuinely differ in parameter space;
                  the single centroid is structurally wrong
                  (heterogeneity-limited regime).

Also computes an empirical chance bound for the observed accuracies
via label permutation (Combrisson & Jerbi, 2015).

Usage (after training a model with trained adapter slots):
    python delta_w_diagnostics.py --dataset BCI2a --seeds 1 2 3 4 5
Requires saved checkpoints; see run_diagnostic.sh.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '../EEGNex')

from meta_init import LORA_LAYER_NAMES, _resolve_layer_names
from inits import delta_w


# ── Spread computation ────────────────────────────────────────────────────────

def flatten_deltas(model, slots, layer_names=None):
    """
    Extract per-slot flattened ΔW vectors per layer.
    Returns {layer_name: {slot: 1-D numpy vector}}.
    """
    names = _resolve_layer_names(layer_names)
    out = {name: {} for name in names}
    for name in names:
        layer = getattr(model, name)
        for slot in slots:
            W = delta_w(layer, slot)
            out[name][slot] = W.detach().cpu().numpy().ravel()
    return out


def spread_metrics(layer_deltas):
    """
    layer_deltas: {layer_name: {slot: vector}} where slot encodes
    (subject, seed) as slot = subject * 1000 + seed for multi-seed runs.

    Returns per-layer dict with within_spread, between_spread, ratio.
    """
    results = {}
    for name, slot_vecs in layer_deltas.items():
        subjects = {}
        for slot, vec in slot_vecs.items():
            subj = slot // 1000
            subjects.setdefault(subj, []).append(vec)

        subj_ids   = sorted(subjects.keys())
        centroids  = {s: np.mean(np.stack(v), axis=0)
                      for s, v in subjects.items()}

        # within: mean distance from each replicate to its subject centroid
        within_dists = [
            np.linalg.norm(v - centroids[s])
            for s in subj_ids for v in subjects[s]
        ]
        within = float(np.mean(within_dists)) if within_dists else 0.0

        # between: mean pairwise distance between subject centroids
        cents = np.stack([centroids[s] for s in subj_ids])
        if len(cents) > 1:
            d = np.linalg.norm(cents[None, :] - cents[:, None], axis=-1)
            iu = np.triu_indices(len(cents), k=1)
            between = float(d[iu].mean())
        else:
            between = 0.0

        ratio = between / within if within > 0 else float('inf')
        results[name] = {
            'within_spread':  within,
            'between_spread': between,
            'ratio':          ratio,
            'n_subjects':     len(subj_ids),
        }
    return results


def summarise(results):
    """Aggregate per-layer ratios into a headline number."""
    ratios  = [r['ratio'] for r in results.values() if np.isfinite(r['ratio'])]
    betweens = [r['between_spread'] for r in results.values()]
    withins  = [r['within_spread'] for r in results.values()]
    return {
        'mean_ratio':      float(np.mean(ratios)) if ratios else float('nan'),
        'mean_between':    float(np.mean(betweens)),
        'mean_within':     float(np.mean(withins)),
        'per_layer':       results,
    }


# ── Empirical chance bound (Combrisson & Jerbi 2015) ──────────────────────────

def permutation_chance_bound(model_eval_fn, X, y, slot, n_perm=1000, rng=None):
    """
    Empirical chance accuracy bound via label permutation.
    model_eval_fn(X, y, slot) -> accuracy.
    Returns (null_mean, null_95th, null_max).
    """
    rng = rng or np.random.default_rng(0)
    null_accs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        null_accs.append(model_eval_fn(X, y_perm, slot))
    null_accs = np.array(null_accs)
    return {
        'chance_mean': float(null_accs.mean()),
        'chance_p95':  float(np.percentile(null_accs, 95)),
        'chance_max':  float(null_accs.max()),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to saved model checkpoints (.pt)')
    parser.add_argument('--layer_names', type=str, nargs='*', default=None)
    parser.add_argument('--out', type=str, default='delta_w_diagnostic.json')
    args = parser.parse_args()

    # Each checkpoint: {'state_dict', 'slots', 'dataset', 'seed', 'ea'}
    all_deltas = {'pre_ea': {}, 'post_ea': {}}
    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Placeholder: real implementation reconstructs model and loads.
        print(f'Loaded {ckpt_path}')

    print('Diagnostic skeleton — see module docstring for usage.')


if __name__ == '__main__':
    main()
