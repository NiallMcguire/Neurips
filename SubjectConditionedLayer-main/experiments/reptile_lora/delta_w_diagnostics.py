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

def _reconstruct_model(ckpt, device):
    """Rebuild an EEGNeX LoRA model from a saved checkpoint."""
    import sys
    sys.path.insert(0, '../EEGNex')
    from EEGNeX import EEGNeX
    cfg = ckpt['config']
    n_ch = 22 if ckpt['dataset'] == 'BCI2a' else 3
    n_cls = 4 if ckpt['dataset'] == 'BCI2a' else 2
    n_times = 512
    model = EEGNeX(
        n_chans=n_ch, n_outputs=n_cls, n_times=n_times,
        mode='LoRA', rank=cfg['rank'], alpha=cfg['alpha'],
        num_adapters=ckpt['n_train_subjects'] + 1,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model, cfg


def _subject_deltas(model, slots, layer_names, device):
    """Per-subject flattened ΔW per layer for the given adapter slots."""
    names = _resolve_layer_names(layer_names)
    out = {}
    for name in names:
        layer = getattr(model, name)
        for slot in slots:
            out.setdefault(name, {})[slot] = delta_w(layer, slot).detach().cpu().numpy().ravel()
    return out


def _distill_subject_deltas(model, X, y, sids, layer_names, cfg, device):
    """
    Gated distillation: for each subject, build a teacher, gate on it beating
    the hard-label adapter, and where it passes, return the distilled slot's
    ΔW. Returns {layer_name: {slot: vector}} for distilled subjects and the
    count distilled.
    """
    import copy
    from distill import distill_teacher_into_adapter, teacher_beats_hardlabel
    from meta_init import get_adapter_params, freeze_backbone, unfreeze_all
    from evaluate import run_eval

    names = _resolve_layer_names(layer_names)
    distilled = {}
    n_distilled = 0
    for s in np.unique(sids):
        s = int(s)
        X_s = X[sids == s]; y_s = y[sids == s]
        n_cut = max(1, int(0.8 * len(X_s)))
        X_tr, y_tr = X_s[:n_cut], y_s[:n_cut]
        X_ho, y_ho = (X_s[n_cut:], y_s[n_cut:]) if len(X_s[n_cut:]) else (X_tr, y_tr)

        hl_acc, _, _, _ = run_eval(model, X_ho, y_ho, s, device)

        teacher = copy.deepcopy(model).to(device)
        freeze_backbone(teacher, names)
        tparams = get_adapter_params(teacher, s, names)
        topt = torch.optim.AdamW(tparams, lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        teacher.train()
        for _ in range(cfg.get('teacher_steps', 100)):
            idx = np.random.choice(len(X_tr), min(32, len(X_tr)),
                                   replace=len(X_tr) < 32)
            Xb = torch.from_numpy(X_tr[idx]).float().to(device)
            yb = torch.from_numpy(y_tr[idx]).long().to(device)
            sid = torch.full((len(idx),), s, dtype=torch.long, device=device)
            topt.zero_grad()
            loss = crit(teacher(Xb, sid), yb)
            loss.backward()
            topt.step()
        t_acc, _, _, _ = run_eval(teacher, X_ho, y_ho, s, device)

        if teacher_beats_hardlabel(t_acc, hl_acc):
            def tfn(Xb, _t=teacher, _s=s):
                sid = torch.full((len(Xb),), _s, dtype=torch.long, device=device)
                return _t(Xb, sid)
            m = copy.deepcopy(model).to(device)
            distill_teacher_into_adapter(m, tfn, X_tr, s,
                                         layer_names=names, device=device,
                                         steps=cfg.get('distill_steps', 200))
            for name in names:
                layer = getattr(m, name)
                distilled.setdefault(name, {})[s] = delta_w(layer, s).detach().cpu().numpy().ravel()
            n_distilled += 1
            del m
        del teacher
    return distilled, n_distilled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BCI2a',
                        choices=['BCI2a', 'BCI2b'])
    parser.add_argument('--condition', type=str, default='baseline_lora')
    parser.add_argument('--held_out', type=int, default=9)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--layer_names', type=str, nargs='*', default=None)
    parser.add_argument('--distill', action='store_true',
                        help='Also compute distilled-adapter deltas (gated)')
    parser.add_argument('--ea', action='store_true',
                        help='Apply EA before distillation/diagnostic eval')
    parser.add_argument('--out', type=str, default='delta_w_diagnostic.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data once for distillation gating
    from data_utils import load_dataset, build_loso_split, euclidean_align
    data, labels, subject_ids, sessions, _ = load_dataset(args.dataset)
    if args.ea:
        data = euclidean_align(data, subject_ids, sessions=sessions,
                               dataset=args.dataset)
    (train_X, train_y, train_sids,
     _, _, _, _, _, _) = build_loso_split(
        data, labels, subject_ids, sessions, args.held_out - 1,
        dataset=args.dataset)

    # slot encoding for spread_metrics: subject * 1000 + seed
    all_raw = {}
    all_distilled = {}
    for seed in args.seeds:
        path = (f'checkpoints/{args.condition}_{args.dataset}'
                f'_heldout{args.held_out}_seed{seed}.pt')
        if not os.path.exists(path):
            print(f'MISSING {path}, skipping')
            continue
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model, cfg = _reconstruct_model(ckpt, device)
        n_train = ckpt['n_train_subjects']
        slots = list(range(n_train))  # training subjects only, not held-out slot

        raw = _subject_deltas(model, slots, args.layer_names, device)
        for name, slot_vecs in raw.items():
            for slot, vec in slot_vecs.items():
                all_raw.setdefault(name, {})[slot * 1000 + seed] = vec

        if args.distill:
            dist, nd = _distill_subject_deltas(
                model, train_X, train_y, train_sids,
                args.layer_names, cfg, device)
            for name, slot_vecs in dist.items():
                for slot, vec in slot_vecs.items():
                    all_distilled.setdefault(name, {})[slot * 1000 + seed] = vec
            print(f'seed {seed}: distilled {nd} subjects')
        del model

    report = {
        'dataset': args.dataset, 'condition': args.condition,
        'held_out': args.held_out, 'seeds': args.seeds,
        'ea': args.ea, 'distill': args.distill,
        'raw': summarise(spread_metrics(all_raw)),
    }
    if args.distill:
        report['distilled'] = summarise(spread_metrics(all_distilled))

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    r = report['raw']
    print(f"RAW      ratio={r['mean_ratio']:.3f} "
          f"between={r['mean_between']:.4f} within={r['mean_within']:.4f}")
    if args.distill and 'distilled' in report:
        d = report['distilled']
        print(f"DISTIL   ratio={d['mean_ratio']:.3f} "
              f"between={d['mean_between']:.4f} within={d['mean_within']:.4f}")
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
