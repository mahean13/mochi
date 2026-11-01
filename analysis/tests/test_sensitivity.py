"""The one-at-a-time formula equals a direct model prediction difference."""

import numpy as np

from mochi_analysis.evaluate import cross_validate
from mochi_analysis.genes import N_GENES
from mochi_analysis.io import PairData
from mochi_analysis.model import model_from_parameters
from mochi_analysis.sensitivity import HIGH, LOW, one_at_a_time_effect, permutation_importance


def test_oat_equals_prediction_difference():
    rng = np.random.default_rng(0)
    weights = rng.normal(scale=1.5, size=N_GENES)
    # Side-neutral presentation: the effect is defined without side bias.
    model = model_from_parameters(0.0, weights)
    base = rng.uniform(size=N_GENES)  # any reference creature; it cancels
    formula = one_at_a_time_effect(weights)
    for j in range(N_GENES):
        high, low = base.copy(), base.copy()
        high[j], low[j] = HIGH, LOW
        p_high_vs_low = model.predict_proba((high - low)[None, :])[0]
        p_tie = model.predict_proba((base - base)[None, :])[0]
        np.testing.assert_allclose(formula[j], p_high_vs_low - p_tie, atol=1e-12)


def test_oat_works_on_bootstrap_arrays():
    w = np.array([[0.0, 2.0], [-2.0, 4.0]])
    out = one_at_a_time_effect(w)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[1, 0], -out[0, 1])  # antisymmetric in w


def test_permutation_importance_finds_the_only_informative_gene():
    rng = np.random.default_rng(1)
    n = 1200
    X = rng.uniform(size=(n, N_GENES)) - rng.uniform(size=(n, N_GENES))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-4 * X[:, 3]))).astype(int)
    session = np.repeat([f"s{i}" for i in range(30)], 40).astype(object)
    data = PairData("home", X, y, session, session.copy(), np.zeros(n, dtype=int))
    cv = cross_validate(data, n_folds=5, seed=0)
    imp = permutation_importance(data, cv, n_permutations=10, seed=0)["accuracy_drop"]
    assert np.argmax(imp) == 3
    assert imp[3] > 0.1
    assert np.max(np.abs(np.delete(imp, 3))) < 0.03
