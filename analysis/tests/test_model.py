"""The Newton/IRLS fit against independent implementations.

Reference: scikit-learn's LogisticRegression with the lbfgs solver. With that
solver sklearn does NOT penalise the intercept, and its objective
C * sum(log loss) + 0.5 * ||w||^2 has the same minimiser as ours,
sum(log loss) + (lambda / 2) * ||w||^2, when C = 1 / lambda. So two exact
comparisons are possible:
  * unpenalised: lambda = 0 against C = inf;
  * the analysis setting: lambda = 1 against C = 1 (only w penalised in both).
The tolerance reflects lbfgs's own convergence, not ours (our gradient is ~1e-12).
We also check that the intercept really is unpenalised in sklearn, by showing a
large true side bias is recovered equally by both fits.
"""

import json
import shutil
import subprocess

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from conftest import REPO_DIR
from mochi_analysis.io import load_pairs
from mochi_analysis.model import fit_bradley_terry


def simulated_design(n=800, d=20, bias=0.8, seed=0):
    """Gene differences of two uniform creatures, with a strong side bias."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d)) - rng.uniform(size=(n, d))
    w = rng.normal(scale=1.5, size=d)
    p = 1 / (1 + np.exp(-(bias + X @ w)))
    return X, (rng.uniform(size=n) < p).astype(int)


def sklearn_fit(X, y, C):
    model = LogisticRegression(C=C, solver="lbfgs", tol=1e-12, max_iter=100_000)
    model.fit(X, y)
    return model.intercept_[0], model.coef_[0]


@pytest.mark.parametrize("lam, C", [(0.0, np.inf), (1.0, 1.0), (5.0, 0.2)])
def test_matches_sklearn(lam, C):
    X, y = simulated_design()
    ours = fit_bradley_terry(X, y, lam=lam)
    b, w = sklearn_fit(X, y, C)
    assert ours.converged
    assert ours.grad_norm < 1e-8
    np.testing.assert_allclose(ours.bias, b, atol=1e-5)
    np.testing.assert_allclose(ours.weights, w, atol=1e-5)


def test_side_bias_is_not_shrunk():
    """A strong penalty shrinks w towards 0 but leaves the side bias near its MLE."""
    X, y = simulated_design(bias=1.0, seed=3)
    heavy = fit_bradley_terry(X, y, lam=1e6)
    assert np.max(np.abs(heavy.weights)) < 1e-3
    # With w ~ 0 the bias is the logit of the observed left-choice rate.
    rate = y.mean()
    np.testing.assert_allclose(heavy.bias, np.log(rate / (1 - rate)), atol=1e-3)


def test_objective_decreases_and_converges_fast():
    X, y = simulated_design(seed=5)
    fit = fit_bradley_terry(X, y, lam=1.0)
    assert fit.converged and fit.n_iter <= 15
    assert fit.objective <= -fit.log_likelihood + 0.5 * fit.weights @ fit.weights + 1e-9


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_matches_engine_js(tmp_path):
    """The Python fit equals fitFeatureBT in web/engine.js on the same test picks."""
    subprocess.run(["node", "sim/simulate.mjs", str(tmp_path), "3", "4", "7"], cwd=REPO_DIR,
                   capture_output=True, text=True, check=True)
    script = (
        "import fs from 'node:fs';"
        "import { picksToRows, fitFeatureBT } from './web/engine.js';"
        f"const s = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'state.json'))}));"
        "const out = {};"
        "for (const lin of ['home', 'trust']) out[lin] = fitFeatureBT(picksToRows(s, lin));"
        "console.log(JSON.stringify(out));"
    )
    run = subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO_DIR,
                         capture_output=True, text=True, check=True)
    js = json.loads(run.stdout)
    for lin, data in load_pairs(tmp_path).items():
        ours = fit_bradley_terry(data.X, data.y, lam=1.0)
        np.testing.assert_allclose(ours.bias, js[lin]["bias"], atol=1e-6)
        np.testing.assert_allclose(ours.weights, js[lin]["weights"], atol=1e-6)
