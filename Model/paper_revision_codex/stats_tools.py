"""User-level paired inference with family-wise error control.

Uncertainty sources are reported separately: the bootstrap and sign-flip test
resample evaluation users (seed-averaged per user), seed variation is the
standard deviation of seed-level means, and temporal variation comes from
separate rolling-origin folds.
"""
import numpy as np


def paired_bootstrap_ci(differences, samples=10000, seed=2026, level=.95):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError('Paired differences must be a finite one-dimensional sample')
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 500):
        take = min(500, samples - start)
        means[start:start + take] = values[rng.integers(len(values), size=(take, len(values)))].mean(axis=1)
    tail = (1 - level) / 2
    return float(values.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def sign_flip_pvalue(differences, samples=10000, seed=2027):
    """Two-sided paired randomization test of mean difference = 0.

    Under the null each user's difference is symmetric around zero. The p-value
    includes the observed statistic, so it is never exactly zero.
    """
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError('Paired differences must be finite')
    observed = abs(values.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, samples, 500):
        take = min(500, samples - start)
        signs = rng.choice((-1.0, 1.0), size=(take, len(values)))
        extreme += int((np.abs((signs * values).mean(axis=1)) >= observed - 1e-15).sum())
    return (extreme + 1) / (samples + 1)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or ((p < 0) | (p > 1)).any():
        raise ValueError('p-values must lie in [0, 1]')
    m = len(p)
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def compare_family(per_user, reference, competitors, metrics=('ndcg', 'recall'), alpha=.05,
                   samples=10000):
    """``per_user``: model -> DataFrame indexed by visitorid with metric columns.

    Every comparison in the returned list belongs to one Holm family.
    """
    rows = []
    base = per_user[reference]
    for model in competitors:
        other = per_user[model].reindex(base.index)
        if other.isna().any().any():
            raise ValueError(f'{model} is missing evaluation users')
        for metric in metrics:
            diff = (base[metric] - other[metric]).to_numpy()
            mean, low, high = paired_bootstrap_ci(diff, samples)
            rows.append({'comparison': f'{reference} - {model}', 'metric': metric, 'difference': mean,
                         'ci_low': low, 'ci_high': high, 'p_value': sign_flip_pvalue(diff, samples),
                         'users': len(diff), 'users_better': int((diff > 0).sum()),
                         'users_worse': int((diff < 0).sum()), 'users_tied': int((diff == 0).sum())})
    return _apply_holm(rows, alpha)


def _apply_holm(rows, alpha):
    adjusted = holm([row['p_value'] for row in rows])
    for row, value in zip(rows, adjusted):
        row['p_holm'] = float(value)
        row['significant_holm'] = bool(value < alpha)
        row['claim'] = ('higher' if row['difference'] > 0 else 'lower') if value < alpha else 'not significant'
    return rows


def compare_families(per_user, references, competitors, metrics=('ndcg', 'recall'), alpha=.05, samples=10000):
    """Every reference against every competitor; all comparisons of all references form one Holm family."""
    rows = [row for reference in references
            for row in compare_family(per_user, reference, competitors, metrics, alpha, samples)]
    return _apply_holm(rows, alpha)
