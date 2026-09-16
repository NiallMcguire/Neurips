"""
Small-sample statistics for LOSO EEG results (reviewer request 4).

Implements:
- Empirical chance bound via the binomial distribution, following
  Combrisson & Jerbi (2015): the maximum accuracy achievable under the
  null hypothesis of independent guessing, at a given significance level.
  This is the correct chance floor for small N — not 1/n_classes.
- Bootstrap confidence intervals on the LOSO accuracy distribution.
- Paired comparisons (Wilcoxon signed-rank, paired t) already used in
  the paper, kept here for a single import location.

All functions are pure numpy/scipy — no model dependency.
"""

import numpy as np
from scipy import stats


def binomial_chance_bound(n_trials, n_classes, alpha=0.05):
    """
    Combrisson & Jerbi (2015) empirical chance bound.

    Under the null, the number of correct classifications out of
    `n_trials` independent trials is Binomial(n_trials, 1/n_classes).
    The bound is the (1-alpha) quantile of that distribution divided by
    n_trials — the accuracy that would be exceeded with probability
    `alpha` by pure chance.

    Returns (chance_level, chance_bound):
        chance_level  = 1 / n_classes
        chance_bound  = quantile(1 - alpha) / n_trials
    """
    from scipy.stats import binom
    p0 = 1.0 / n_classes
    bound = binom.ppf(1.0 - alpha, n_trials, p0) / n_trials
    return p0, float(bound)


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    """
    Percentile bootstrap confidence interval on the mean of `values`
    (e.g. per-subject LOSO accuracies).
    Returns (mean, lo, hi).
    """
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lo), float(hi)


def paired_comparison(x, y):
    """
    Paired t-test and Wilcoxon signed-rank between two matched
    per-subject accuracy vectors (e.g. MetaSub vs baseline).
    Returns dict with both p-values and the mean difference.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    t_stat, t_p = stats.ttest_rel(x, y)
    # Wilcoxon requires non-zero diffs
    nz = diff[diff != 0]
    if len(nz) >= 1:
        w_stat, w_p = stats.wilcoxon(x, y)
    else:
        w_stat, w_p = float('nan'), float('nan')
    return {
        'mean_diff':  float(diff.mean()),
        'paired_t_p': float(t_p),
        'wilcoxon_p': float(w_p),
        'n_pairs':    int(len(x)),
    }


def report(accuracies, n_trials, n_classes, label=''):
    """
    One-shot summary for a LOSO accuracy vector.
    Prints mean, bootstrap CI, chance level and chance bound.
    """
    mean, lo, hi = bootstrap_ci(accuracies)
    p0, bound = binomial_chance_bound(n_trials, n_classes)
    print(f'{label:30s} mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  '
          f'chance={p0:.3f}  chance-bound(95%)={bound:.3f}')
    return {'mean': mean, 'ci_lo': lo, 'ci_hi': hi,
            'chance': p0, 'chance_bound': bound}
