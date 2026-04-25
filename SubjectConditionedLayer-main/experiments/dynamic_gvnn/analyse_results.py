"""
Results Analysis: pulls all experiment runs from wandb and produces
summary tables and plots for the paper.

Usage:
  python analyse_results.py --entity YOUR_WANDB_ENTITY

Outputs (saved to results/):
  ablation_table.csv        — mean ± std accuracy and kappa per condition
  loso_table.csv            — zero-shot and few-shot accuracy per condition
  fewshot_curve.csv         — accuracy vs N curves per condition
  ablation_table.tex        — LaTeX table ready for paper
  fewshot_curve.png         — adaptation curve plot
  per_subject_heatmap.png   — per-subject accuracy heatmap
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import wandb
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print('Warning: matplotlib/seaborn not available, skipping plots')

os.makedirs('results', exist_ok=True)

CONDITIONS_ORDER = [
    'vanilla', 'static_lora', 'gvnn_pop',
    'gvnn_subject', 'gvnn_additive', 'full'
]
CONDITION_LABELS = {
    'vanilla':        'Vanilla EEGNeX',
    'static_lora':    'Static LoRA (baseline)',
    'gvnn_pop':       'GVNN Population',
    'gvnn_subject':   'GVNN Subject (ours)',
    'gvnn_additive':  'GVNN Additive (ablation)',
    'full':           'GVNN + LoRA (full)',
}


def pull_ablation_results(entity, project='dynamic_gvnn_ablation'):
    """Pull ablation study runs from wandb."""
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}')
    records = []
    for run in runs:
        cfg = run.config
        summary = run.summary
        records.append({
            'condition': cfg.get('condition'),
            'dataset':   cfg.get('dataset'),
            'seed':      cfg.get('seed'),
            'final_test_acc':  summary.get('final_test_acc'),
            'best_test_acc':   summary.get('best_test_acc'),
            'final_kappa':     summary.get('final_kappa'),
            'n_params':        summary.get('n_params'),
            'mean_epoch_time': summary.get('mean_epoch_time'),
            **{k: summary.get(k) for k in summary.keys()
               if k.startswith('final_acc_subject_')},
        })
    return pd.DataFrame(records)


def pull_loso_results(entity, project='dynamic_gvnn_loso'):
    """Pull LOSO results from wandb."""
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}')
    records = []
    for run in runs:
        cfg = run.config
        summary = run.summary
        base = {
            'condition':      cfg.get('condition'),
            'held_out':       cfg.get('held_out'),
            'seed':           cfg.get('seed'),
            'zero_shot_acc':  summary.get('zero_shot_acc'),
            'zero_shot_kappa':summary.get('zero_shot_kappa'),
        }
        for N in [5, 10, 20, 50]:
            base[f'few_shot_acc_N{N}']   = summary.get(f'few_shot_acc_N{N}')
            base[f'few_shot_kappa_N{N}'] = summary.get(f'few_shot_kappa_N{N}')
        records.append(base)
    return pd.DataFrame(records)


def pull_fewshot_results(entity, project='dynamic_gvnn_fewshot'):
    """Pull few-shot adaptation results from wandb."""
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}')
    records = []
    for run in runs:
        cfg     = run.config
        history = run.history(keys=['few_shot_N', 'few_shot_acc',
                                    'few_shot_kappa', 'finetune_time'])
        for _, row in history.iterrows():
            records.append({
                'condition':      cfg.get('condition'),
                'target_subject': cfg.get('target_subject'),
                'seed':           cfg.get('seed'),
                'N':              row.get('few_shot_N'),
                'acc':            row.get('few_shot_acc'),
                'kappa':          row.get('few_shot_kappa'),
                'finetune_time':  row.get('finetune_time'),
            })
    return pd.DataFrame(records)


def make_ablation_table(df):
    """Aggregate ablation results: mean ± std across seeds."""
    rows = []
    for cond in CONDITIONS_ORDER:
        for dataset in ['BCI2a', 'BCI2b']:
            sub = df[(df['condition'] == cond) & (df['dataset'] == dataset)]
            if len(sub) == 0:
                continue
            accs   = sub['final_test_acc'].dropna() * 100
            kappas = sub['final_kappa'].dropna()
            n_params = sub['n_params'].dropna().mean()
            epoch_t  = sub['mean_epoch_time'].dropna().mean()
            rows.append({
                'Condition':   CONDITION_LABELS.get(cond, cond),
                'Dataset':     dataset,
                'Acc (%)':     f'{accs.mean():.2f} ± {accs.std():.2f}',
                'Kappa':       f'{kappas.mean():.4f} ± {kappas.std():.4f}',
                'Params':      f'{int(n_params):,}' if not np.isnan(n_params) else '—',
                'Epoch (s)':   f'{epoch_t:.1f}' if not np.isnan(epoch_t) else '—',
            })
    return pd.DataFrame(rows)


def make_latex_table(table_df):
    """Convert summary table to LaTeX."""
    latex = table_df.to_latex(index=False, escape=False,
                              column_format='llrrrr')
    return latex


def make_loso_table(df):
    """Aggregate LOSO results: mean across subjects and seeds."""
    rows = []
    for cond in ['static_lora', 'gvnn_pop', 'gvnn_subject', 'full']:
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        zs_acc = sub['zero_shot_acc'].dropna() * 100
        row = {
            'Condition': CONDITION_LABELS.get(cond, cond),
            'Zero-shot Acc (%)': f'{zs_acc.mean():.2f} ± {zs_acc.std():.2f}',
        }
        for N in [5, 10, 20, 50]:
            col = f'few_shot_acc_N{N}'
            vals = sub[col].dropna() * 100
            row[f'N={N} Acc (%)'] = (
                f'{vals.mean():.2f} ± {vals.std():.2f}'
                if len(vals) > 0 else '—'
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fewshot_curves(df):
    """Plot accuracy vs N calibration trials per condition."""
    if not PLOT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette('tab10', n_colors=4)
    conditions = ['static_lora', 'gvnn_subject', 'gvnn_additive', 'full']

    for i, cond in enumerate(conditions):
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        grouped = sub.groupby('N')['acc'].agg(['mean', 'std'])
        grouped = grouped.reset_index()
        grouped['mean'] *= 100
        grouped['std']  *= 100
        ax.errorbar(grouped['N'], grouped['mean'], yerr=grouped['std'],
                    label=CONDITION_LABELS.get(cond, cond),
                    color=palette[i], marker='o', linewidth=2, capsize=4)

    ax.set_xlabel('N calibration trials', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Few-Shot Adaptation: Accuracy vs Calibration Trials', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig('results/fewshot_curve.png', dpi=150)
    plt.close()
    print('Saved: results/fewshot_curve.png')


def plot_per_subject_heatmap(df, dataset='BCI2a'):
    """Heatmap of per-subject accuracy across conditions."""
    if not PLOT_AVAILABLE:
        return

    sub_df = df[df['dataset'] == dataset]
    conditions = [c for c in CONDITIONS_ORDER if c in sub_df['condition'].values]
    subjects   = list(range(1, 10))
    matrix = np.full((len(conditions), len(subjects)), np.nan)

    for i, cond in enumerate(conditions):
        for j, sid in enumerate(subjects):
            col = f'final_acc_subject_{sid}'
            if col in sub_df.columns:
                vals = sub_df[sub_df['condition'] == cond][col].dropna() * 100
                if len(vals) > 0:
                    matrix[i, j] = vals.mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(matrix, ax=ax,
                xticklabels=[f'S{s}' for s in subjects],
                yticklabels=[CONDITION_LABELS.get(c, c) for c in conditions],
                annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=30, vmax=80, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title(f'Per-Subject Accuracy — {dataset}', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/per_subject_heatmap.png', dpi=150)
    plt.close()
    print('Saved: results/per_subject_heatmap.png')


def main(args):
    print('Pulling results from wandb...')

    # Ablation
    print('  Ablation study...')
    abl_df = pull_ablation_results(args.entity)
    abl_df.to_csv('results/ablation_raw.csv', index=False)
    abl_table = make_ablation_table(abl_df)
    abl_table.to_csv('results/ablation_table.csv', index=False)
    with open('results/ablation_table.tex', 'w') as f:
        f.write(make_latex_table(abl_table))
    print(abl_table.to_string(index=False))

    # LOSO
    print('\n  LOSO cross-subject...')
    loso_df = pull_loso_results(args.entity)
    loso_df.to_csv('results/loso_raw.csv', index=False)
    loso_table = make_loso_table(loso_df)
    loso_table.to_csv('results/loso_table.csv', index=False)
    print(loso_table.to_string(index=False))

    # Few-shot
    print('\n  Few-shot adaptation...')
    fs_df = pull_fewshot_results(args.entity)
    fs_df.to_csv('results/fewshot_raw.csv', index=False)

    # Plots
    plot_fewshot_curves(fs_df)
    plot_per_subject_heatmap(abl_df, dataset='BCI2a')

    print('\nAll results saved to results/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, required=True,
                        help='WandB entity (username or team name)')
    args = parser.parse_args()
    main(args)
