"""
Analysis for hypernet_objective results: summary tables and paired,
Holm-corrected Wilcoxon signed-rank tests with rank-biserial effect sizes.

Usage:
    python analyze_hypernet_objective.py --csv hypernet_objective_results.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy import stats


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r['ea'] = r['ea'] in ('True', 'true', '1')
            r['seed'] = int(r['seed'])
            r['held_out_subject'] = int(r['held_out_subject'])
            r['k'] = int(r['k'])
            r['balanced_accuracy'] = float(r['balanced_accuracy'])
            rows.append(r)
    return rows


def summary_table(rows):
    agg = defaultdict(list)
    for r in rows:
        key = (r['condition'], r['descriptor_control'], r['ea'], r['k'])
        agg[key].append(r['balanced_accuracy'])
    print(f"\n{'condition':24s} {'control':9s} {'EA':>5s} {'k':>3s} "
         f"{'n':>3s} {'mean':>8s} {'sd':>8s}")
    for key, vals in sorted(agg.items(), key=lambda x: (x[0][0], x[0][3], x[0][2], x[0][1])):
        cond, ctrl, ea, k = key
        print(f'{cond:24s} {ctrl:9s} {str(ea):>5s} {k:3d} '
             f'{len(vals):3d} {np.mean(vals):8.4f} {np.std(vals):8.4f}')


def wilcoxon_rank_biserial(x, y):
    """Paired Wilcoxon signed-rank p-value and rank-biserial effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    nz = diff[diff != 0]
    if len(nz) == 0:
        return float('nan'), float('nan')
    _, p = stats.wilcoxon(x, y)
    ranks = stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    denom = r_plus + r_minus
    r_rb = (r_plus - r_minus) / denom if denom > 0 else float('nan')
    return float(p), float(r_rb)


def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order of `pvals`."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def paired_by_subject(rows, cond_a, ctrl_a, cond_b, ctrl_b, ea, k):
    """Average over seeds per held-out subject, then pair on subject id."""
    a = defaultdict(list)
    b = defaultdict(list)
    for r in rows:
        if r['ea'] != ea or r['k'] != k:
            continue
        if r['condition'] == cond_a and r['descriptor_control'] == ctrl_a:
            a[r['held_out_subject']].append(r['balanced_accuracy'])
        if r['condition'] == cond_b and r['descriptor_control'] == ctrl_b:
            b[r['held_out_subject']].append(r['balanced_accuracy'])
    subjects = sorted(set(a.keys()) & set(b.keys()))
    xa = np.array([np.mean(a[s]) for s in subjects])
    xb = np.array([np.mean(b[s]) for s in subjects])
    return xa, xb, subjects


def run_comparisons(rows, ea, k):
    comparisons = [
        ('recon_product', 'recon_factors'),
        ('end_to_end', 'recon_factors'),
        ('recon_then_e2e', 'recon_factors'),
        ('recon_factors', 'joint_shared_adapter'),
        ('recon_product', 'joint_shared_adapter'),
        ('end_to_end', 'joint_shared_adapter'),
        ('recon_then_e2e', 'joint_shared_adapter'),
    ]
    results = []
    for cond_a, cond_b in comparisons:
        xa, xb, subjects = paired_by_subject(rows, cond_a, 'none', cond_b, 'none', ea, k)
        if len(subjects) < 3:
            results.append((cond_a, cond_b, len(subjects), float('nan'), float('nan')))
            continue
        p, r_rb = wilcoxon_rank_biserial(xa, xb)
        results.append((cond_a, cond_b, len(subjects), p, r_rb))

    pvals = np.array([r[3] for r in results])
    valid = ~np.isnan(pvals)
    adj = np.full(len(pvals), float('nan'))
    if valid.sum() > 0:
        adj[valid] = holm_correct(pvals[valid])

    print(f'\n=== Paired Wilcoxon (Holm-corrected), EA={ea}, k={k} ===')
    print(f"{'A vs B':45s} {'n':>3s} {'p':>8s} {'p_holm':>8s} {'r_rb':>7s}")
    for (cond_a, cond_b, n, p, r_rb), padj in zip(results, adj):
        label = f'{cond_a} vs {cond_b}'
        print(f'{label:45s} {n:3d} {p:8.4f} {padj:8.4f} {r_rb:7.3f}')


def descriptor_control_check(rows, ea, k):
    print(f'\n=== Descriptor control check, EA={ea}, k={k} ===')
    for cond in ['recon_product', 'end_to_end', 'recon_then_e2e']:
        xa, xb, subjects = paired_by_subject(rows, cond, 'none', cond, 'shuffled', ea, k)
        if len(subjects) >= 3:
            p, r_rb = wilcoxon_rank_biserial(xa, xb)
            informative = 'INFORMATIVE' if (p < 0.05 and xa.mean() > xb.mean()) else 'NOT CLEARLY INFORMATIVE'
            print(f'{cond}: real={xa.mean():.4f} shuffled={xb.mean():.4f} '
                 f'p={p:.4f} r_rb={r_rb:.3f} -> descriptor {informative}')
        xa, xb, subjects = paired_by_subject(rows, cond, 'none', cond, 'mean', ea, k)
        if len(subjects) >= 3:
            p, r_rb = wilcoxon_rank_biserial(xa, xb)
            informative = 'INFORMATIVE' if (p < 0.05 and xa.mean() > xb.mean()) else 'NOT CLEARLY INFORMATIVE'
            print(f'{cond}: real={xa.mean():.4f} mean_desc={xb.mean():.4f} '
                 f'p={p:.4f} r_rb={r_rb:.3f} -> descriptor {informative}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    rows = load_rows(args.csv)

    summary_table(rows)
    ks = sorted(set(r['k'] for r in rows))
    eas = sorted(set(r['ea'] for r in rows))
    for ea in eas:
        for k in ks:
            run_comparisons(rows, ea, k)
            descriptor_control_check(rows, ea, k)


if __name__ == '__main__':
    main()
