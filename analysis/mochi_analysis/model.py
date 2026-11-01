"""Feature-based Bradley-Terry preference model, fitted by Newton's method (IRLS).

Model
-----
A volunteer sees creature L on the left and creature R on the right. Each creature
has an unobserved "appeal" that is linear in its genes, u(x) = w . x, with
x = genes - 0.5. Bradley-Terry says the chance of picking L is a logistic function
of the appeal difference. We add an intercept b for a left/right side bias:

    P(left chosen) = sigmoid(b + w . (x_L - x_R)).

So the model is a logistic regression on the gene *differences* with an intercept.
The intercept is identifiable only because the engine randomises which creature
appears on which side.

Estimation
----------
We minimise the penalised negative log-likelihood

    J(b, w) = -sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + (lambda / 2) ||w||^2,

with lambda = 1 by default. The ridge penalty acts on w only, never on b, so the
side bias is estimated freely. J is strictly convex, so it has one minimum.

Newton's method uses the gradient g and Hessian H of J:

    g = Z^T (p - y) + lambda * P theta,     H = Z^T diag(p (1 - p)) Z + lambda * P,

where Z = [1, X] is the design, theta = (b, w) and P = diag(0, 1, ..., 1) masks
the intercept out of the penalty. Each step solves H delta = -g. This is the same
as iteratively reweighted least squares (IRLS). We add step halving: if a full
step does not lower J, the step is halved until it does, which guarantees descent.
This mirrors fitFeatureBT in web/engine.js (same objective), with a line search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_expit

DEFAULT_LAMBDA = 1.0


@dataclass(frozen=True)
class FitResult:
    """A fitted model plus convergence diagnostics.

    Attributes:
        bias: side-bias intercept b, in logits (positive favours the left creature).
        weights: (d,) gene weights w, in logits per unit of gene difference.
        n_iter: Newton iterations used.
        converged: True if the largest parameter change fell below ``tol``.
        grad_norm: max |gradient of J| at the solution (should be close to 0).
        objective: penalised negative log-likelihood J at the solution.
        log_likelihood: unpenalised log-likelihood at the solution.
        lam: ridge strength used.
    """

    bias: float
    weights: np.ndarray
    n_iter: int
    converged: bool
    grad_norm: float
    objective: float
    log_likelihood: float
    lam: float

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """Logit of P(left chosen) for gene-difference rows X."""
        return self.bias + X @ self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(left chosen) for gene-difference rows X."""
        return expit(self.linear_predictor(X))


def _design(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (the side-bias intercept) to X."""
    return np.column_stack([np.ones(X.shape[0]), X])


def _objective(Z: np.ndarray, y: np.ndarray, theta: np.ndarray, lam: float) -> tuple[float, float]:
    """Return (penalised objective J, unpenalised log-likelihood)."""
    eta = Z @ theta
    # log p = log_expit(eta), log(1 - p) = log_expit(-eta): stable for large |eta|.
    loglik = float(np.sum(y * log_expit(eta) + (1 - y) * log_expit(-eta)))
    penalty = 0.5 * lam * float(theta[1:] @ theta[1:])
    return -loglik + penalty, loglik


def fit_bradley_terry(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = DEFAULT_LAMBDA,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> FitResult:
    """Fit the feature Bradley-Terry model by penalised Newton/IRLS.

    Args:
        X: (n, d) gene differences, left minus right.
        y: (n,) 1 if left was chosen, else 0.
        lam: ridge strength on the gene weights (not on the side bias). lam = 0
            gives the unpenalised maximum-likelihood fit.
        max_iter: maximum Newton iterations.
        tol: stop when the largest absolute parameter change is below this.

    Returns:
        FitResult with the estimates and convergence diagnostics.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = _design(X)
    d = Z.shape[1]
    penalty_mask = np.ones(d)
    penalty_mask[0] = 0.0  # the side bias is not penalised

    theta = np.zeros(d)
    obj, _ = _objective(Z, y, theta, lam)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        p = expit(Z @ theta)
        grad = Z.T @ (p - y) + lam * penalty_mask * theta
        # Hessian: Fisher information of the logistic likelihood plus the ridge term.
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(lam * penalty_mask)
        step = -np.linalg.solve(hess, grad)

        # Step halving guarantees the objective never increases.
        t = 1.0
        while True:
            candidate = theta + t * step
            new_obj, _ = _objective(Z, y, candidate, lam)
            if new_obj <= obj + 1e-12 or t < 1e-8:
                break
            t *= 0.5
        theta, obj = candidate, new_obj
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break

    p = expit(Z @ theta)
    grad = Z.T @ (p - y) + lam * penalty_mask * theta
    obj, loglik = _objective(Z, y, theta, lam)
    return FitResult(
        bias=float(theta[0]),
        weights=theta[1:].copy(),
        n_iter=n_iter,
        converged=converged,
        grad_norm=float(np.max(np.abs(grad))),
        objective=obj,
        log_likelihood=loglik,
        lam=lam,
    )


def model_from_parameters(bias: float, weights: np.ndarray) -> FitResult:
    """Wrap known parameters (e.g. the simulation truth) as a FitResult for prediction."""
    return FitResult(bias=float(bias), weights=np.asarray(weights, dtype=float), n_iter=0,
                     converged=True, grad_norm=float("nan"), objective=float("nan"),
                     log_likelihood=float("nan"), lam=float("nan"))
