"""The model recovers known weights from data simulated directly in Python."""

import numpy as np

from mochi_analysis.evaluate import weight_recovery
from mochi_analysis.model import fit_bradley_terry


def test_recovers_sparse_weights():
    rng = np.random.default_rng(42)
    d, n = 20, 4000
    true_w = np.zeros(d)
    true_w[[2, 6, 8, 10, 12, 13]] = [2.0, 2.4, 1.6, 1.4, 1.2, -1.0]
    true_b = 0.15
    left = rng.uniform(size=(n, d))
    right = rng.uniform(size=(n, d))
    X = left - right
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(true_b + X @ true_w)))).astype(int)

    fit = fit_bradley_terry(X, y, lam=1.0)
    rec = weight_recovery(true_w, fit.weights)
    assert rec["pearson_r"] > 0.97
    assert rec["sign_agree_nonzero"] == rec["n_nonzero"]
    assert rec["rmse"] < 0.25
    assert abs(fit.bias - true_b) < 0.15
    # Zero-weight genes stay small.
    assert np.max(np.abs(fit.weights[true_w == 0])) < 0.4


def test_recovery_improves_with_data():
    rng = np.random.default_rng(7)
    d = 20
    true_w = rng.normal(scale=1.0, size=d)
    errors = []
    for n in (200, 800, 3200):
        X = rng.uniform(size=(n, d)) - rng.uniform(size=(n, d))
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ true_w)))).astype(int)
        errors.append(weight_recovery(true_w, fit_bradley_terry(X, y).weights)["rmse"])
    assert errors[0] > errors[1] > errors[2]
