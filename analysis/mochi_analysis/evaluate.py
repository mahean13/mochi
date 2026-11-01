"""Out-of-sample evaluation, uncertainty and weight recovery.

Every procedure here treats the *session* as the unit of independence. Picks in one
session share a rater, a generation of creatures and a time of day, so they are
correlated. Splitting or resampling individual picks would let near-duplicate
information leak between training and test data and would make intervals too
narrow. We therefore:

* cross-validate with whole sessions held out (grouped K-fold),
* bootstrap by resampling whole sessions,
* build learning curves by adding whole sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .io import PairData, Truth
from .model import DEFAULT_LAMBDA, FitResult, fit_bradley_terry, model_from_parameters

Z_95 = stats.norm.ppf(0.975)


# ---------------------------------------------------------------- scoring rules
def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Share of picks where the model's more likely creature was the one chosen."""
    return float(np.mean((p > 0.5) == (y == 1)))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood per pick (natural log). Chance is ln 2 = 0.693."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the predicted probability. Chance (p = 0.5) is 0.25."""
    return float(np.mean((p - y) ** 2))


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Better behaved than the normal (Wald) interval near 0 or 1 and for modest n.
    """
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def binomial_test_above_chance(successes: int, n: int) -> float:
    """Exact one-sided binomial test p-value for H1: accuracy > 0.5.

    Note: this treats held-out picks as independent trials, which is optimistic
    when picks within a session are correlated; the session bootstrap and the
    grouped CV are the more conservative checks.
    """
    return float(stats.binomtest(successes, n, 0.5, alternative="greater").pvalue)


# ---------------------------------------------------------------- grouped cross-validation
def grouped_folds(groups: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each row a fold number so that every group lies in exactly one fold.

    Groups (sessions) are shuffled with a fixed seed and dealt round-robin into
    folds, which keeps fold sizes within one session of each other. If there are
    fewer groups than folds, this becomes leave-one-group-out.

    Returns:
        (n,) integer array of fold ids in 0..k-1.
    """
    unique = np.array(sorted(set(groups)), dtype=object)
    k = min(n_folds, len(unique))
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of_group = {unique[g]: i % k for i, g in enumerate(order)}
    return np.array([fold_of_group[g] for g in groups], dtype=int)


@dataclass
class CrossValidation:
    """Held-out predictions from grouped cross-validation."""

    folds: np.ndarray  # (n,) fold id per pick
    p_heldout: np.ndarray  # (n,) P(left) from the model that did not see the pick
    models: list[FitResult] = field(default_factory=list)  # one model per fold


def cross_validate(data: PairData, n_folds: int = 5, seed: int = 0,
                   lam: float = DEFAULT_LAMBDA) -> CrossValidation:
    """Grouped K-fold CV by session: every pick is predicted once, by a model
    trained only on other sessions."""
    folds = grouped_folds(data.session, n_folds, seed)
    p = np.empty(data.n_picks)
    models = []
    for f in range(folds.max() + 1):
        test = folds == f
        model = fit_bradley_terry(data.X[~test], data.y[~test], lam=lam)
        p[test] = model.predict_proba(data.X[test])
        models.append(model)
    return CrossValidation(folds=folds, p_heldout=p, models=models)


def heldout_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Accuracy with Wilson CI, exact binomial test, log loss and Brier score."""
    n = int(y.size)
    hits = int(np.sum((p > 0.5) == (y == 1)))
    lo, hi = wilson_interval(hits, n)
    return {
        "n": n,
        "correct": hits,
        "accuracy": hits / n,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "binomial_p_one_sided": binomial_test_above_chance(hits, n),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
    }


def truth_model(truth: Truth, lineage: str) -> FitResult:
    """The data-generating model: shared true weights plus the true side bias."""
    return model_from_parameters(truth.side_bias, truth.weights[lineage])


# ---------------------------------------------------------------- session bootstrap
def session_bootstrap(data: PairData, n_boot: int = 1000, seed: int = 0,
                      lam: float = DEFAULT_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """Refit the model on B resamples of whole sessions (with replacement).

    Returns:
        (biases, weights): arrays of shape (B,) and (B, d).
    """
    sessions = data.sessions
    rows_of = {s: np.flatnonzero(data.session == s) for s in sessions}
    rng = np.random.default_rng(seed)
    biases = np.empty(n_boot)
    weights = np.empty((n_boot, data.X.shape[1]))
    for b in range(n_boot):
        drawn = rng.choice(len(sessions), size=len(sessions), replace=True)
        idx = np.concatenate([rows_of[sessions[i]] for i in drawn])
        fit = fit_bradley_terry(data.X[idx], data.y[idx], lam=lam)
        biases[b], weights[b] = fit.bias, fit.weights
    return biases, weights


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval along axis 0."""
    alpha = (1 - level) / 2
    return np.quantile(samples, alpha, axis=0), np.quantile(samples, 1 - alpha, axis=0)


# ---------------------------------------------------------------- learning curve
def learning_curve(data: PairData, n_repeats: int = 50, test_fraction: float = 0.2,
                   seed: int = 0, lam: float = DEFAULT_LAMBDA,
                   oracle: FitResult | None = None) -> dict[str, list[float]]:
    """Held-out accuracy as a function of how many sessions the model trains on.

    For each repeat r (its own seed): hold out a random ``test_fraction`` of the
    sessions, then for k = 1, 2, ... up to all remaining sessions, train on a random
    k of them and score on the held-out sessions. With 40 picks per session this
    gives training sizes 40, 80, ... picks. The oracle (true model) is scored on
    the same held-out sessions, so its line is the ceiling for that test set.

    Returns:
        Lists over k of mean training picks, mean accuracy, its 2.5% and 97.5%
        quantiles across repeats, and the mean oracle accuracy.
    """
    sessions = data.sessions
    rows_of = {s: np.flatnonzero(data.session == s) for s in sessions}
    n_test = max(1, int(round(test_fraction * len(sessions))))
    n_pool = len(sessions) - n_test
    acc = np.full((n_repeats, n_pool), np.nan)
    n_train = np.zeros((n_repeats, n_pool))
    oracle_acc = np.full(n_repeats, np.nan)
    for r in range(n_repeats):
        rng = np.random.default_rng([seed, r])
        order = rng.permutation(len(sessions))
        test_idx = np.concatenate([rows_of[sessions[i]] for i in order[:n_test]])
        pool = order[n_test:]
        if oracle is not None:
            oracle_acc[r] = accuracy(data.y[test_idx], oracle.predict_proba(data.X[test_idx]))
        for k in range(1, n_pool + 1):
            chosen = rng.choice(pool, size=k, replace=False)
            train_idx = np.concatenate([rows_of[sessions[i]] for i in chosen])
            fit = fit_bradley_terry(data.X[train_idx], data.y[train_idx], lam=lam)
            acc[r, k - 1] = accuracy(data.y[test_idx], fit.predict_proba(data.X[test_idx]))
            n_train[r, k - 1] = train_idx.size
    return {
        "train_sessions": list(range(1, n_pool + 1)),
        "train_picks": n_train.mean(axis=0).tolist(),
        "accuracy_mean": acc.mean(axis=0).tolist(),
        "accuracy_low": np.quantile(acc, 0.025, axis=0).tolist(),
        "accuracy_high": np.quantile(acc, 0.975, axis=0).tolist(),
        "oracle_accuracy_mean": float(np.nanmean(oracle_acc)) if oracle is not None else None,
        "test_sessions": n_test,
        "repeats": n_repeats,
    }


# ---------------------------------------------------------------- recovery of known weights
def weight_recovery(true_w: np.ndarray, est_w: np.ndarray,
                    ci_low: np.ndarray | None = None,
                    ci_high: np.ndarray | None = None) -> dict[str, float]:
    """How well the estimated weights match the simulation's true weights.

    * Pearson r across the 20 genes (pattern recovery), with its p-value.
    * Slope of estimated on true: below 1 means estimates are pulled toward zero
      (by the ridge penalty and by attenuation from rater heterogeneity).
    * Sign agreement on genes whose true weight is non-zero.
    * For genes whose true weight is zero: how many 95% CIs include 0.
    * For non-zero genes: how many CIs exclude 0 (detected effects).
    * Coverage: how many CIs contain the true weight.
    """
    nonzero = true_w != 0
    r, p = stats.pearsonr(true_w, est_w)
    slope = float(np.polyfit(true_w, est_w, 1)[0])
    out: dict[str, float] = {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "slope_est_on_true": slope,
        "rmse": float(np.sqrt(np.mean((est_w - true_w) ** 2))),
        "n_nonzero": int(nonzero.sum()),
        "sign_agree_nonzero": int(np.sum(np.sign(est_w[nonzero]) == np.sign(true_w[nonzero]))),
        "n_zero": int((~nonzero).sum()),
    }
    if ci_low is not None and ci_high is not None:
        covers_zero = (ci_low <= 0) & (ci_high >= 0)
        out["zero_genes_ci_include_0"] = int(np.sum(covers_zero[~nonzero]))
        out["nonzero_genes_ci_exclude_0"] = int(np.sum(~covers_zero[nonzero]))
        out["ci_cover_true"] = int(np.sum((ci_low <= true_w) & (ci_high >= true_w)))
    return out

