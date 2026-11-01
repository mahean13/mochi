"""Sensitivity analysis: which genes move the model's predictions, and by how much.

Two complementary views:

1. One-at-a-time (OAT) effect. Compare two creatures that are identical except
   that gene j is 0.75 in one and 0.25 in the other, shown without side bias. The
   gene difference is 0.5, so the model gives the first creature probability
   sigmoid(0.5 * w_j). The effect is that probability minus the 0.5 of a tie:

       effect_j = sigmoid(0.5 * w_j) - 0.5.

   It is a direct, interpretable reading of a weight on the probability scale
   (range -0.5 to 0.5), with uncertainty from the session bootstrap.

2. Permutation importance. In each cross-validation fold, shuffle gene j's
   difference column in the held-out picks and measure how much held-out accuracy
   drops. Shuffling breaks the link between that gene and the choice while keeping
   its distribution. Unlike the OAT effect, this depends on how much the gene
   actually varied in the pairs shown: a strong weight on a gene that barely varies
   between the two creatures matters little for prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .evaluate import CrossValidation, accuracy, log_loss
from .io import PairData

LOW, HIGH = 0.25, 0.75  # gene values compared in the one-at-a-time effect


def one_at_a_time_effect(weights: np.ndarray) -> np.ndarray:
    """P(picked) change when one gene goes from 0.25 to 0.75, all else equal.

    Works elementwise, so it accepts a (d,) weight vector or a (B, d) array of
    bootstrap weights.
    """
    return expit((HIGH - LOW) * np.asarray(weights)) - 0.5


def permutation_importance(data: PairData, cv: CrossValidation, n_permutations: int = 30,
                           seed: int = 0) -> dict[str, np.ndarray]:
    """Drop in pooled held-out accuracy (and rise in log loss) per permuted gene.

    For every fold model, gene j's column in that fold's held-out picks is permuted
    ``n_permutations`` times. The held-out predictions of all folds are pooled for
    each permutation, scored, and the scores averaged over permutations.

    Returns:
        dict with (d,) arrays "accuracy_drop" and "log_loss_increase", and the
        (d,) "accuracy_drop_sd" across permutations.
    """
    rng = np.random.default_rng(seed)
    n, d = data.X.shape
    base_acc = accuracy(data.y, cv.p_heldout)
    base_ll = log_loss(data.y, cv.p_heldout)
    fold_rows = [np.flatnonzero(cv.folds == f) for f in range(len(cv.models))]
    acc_drop = np.zeros((n_permutations, d))
    ll_rise = np.zeros((n_permutations, d))
    for j in range(d):
        for r in range(n_permutations):
            p = np.empty(n)
            for model, rows in zip(cv.models, fold_rows):
                X_perm = data.X[rows].copy()
                X_perm[:, j] = X_perm[rng.permutation(rows.size), j]
                p[rows] = model.predict_proba(X_perm)
            acc_drop[r, j] = base_acc - accuracy(data.y, p)
            ll_rise[r, j] = log_loss(data.y, p) - base_ll
    return {
        "accuracy_drop": acc_drop.mean(axis=0),
        "accuracy_drop_sd": acc_drop.std(axis=0, ddof=1),
        "log_loss_increase": ll_rise.mean(axis=0),
    }
