"""The full per-lineage analysis, assembled from the building blocks.

``analyse_lineage`` runs steps 1 to 6 of the analysis plan for one lineage and
returns plain Python data (lists, floats), ready for JSON. Truth-dependent parts
(oracle accuracy, weight recovery) are included only when a Truth is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluate import (
    accuracy, cross_validate, heldout_metrics, learning_curve, percentile_interval,
    session_bootstrap, truth_model, weight_recovery,
)
from .genes import GENE_KEYS, GENE_LABELS
from .io import PairData, Truth
from .model import DEFAULT_LAMBDA, fit_bradley_terry
from .sensitivity import one_at_a_time_effect, permutation_importance


@dataclass(frozen=True)
class Settings:
    """Analysis settings. Fixed seeds make every number reproducible."""

    lam: float = DEFAULT_LAMBDA
    n_folds: int = 5
    cv_seed: int = 1
    n_boot: int = 1000
    boot_seed: int = 2
    n_permutations: int = 30
    perm_seed: int = 3
    lc_repeats: int = 50
    lc_seed: int = 4
    lc_test_fraction: float = 0.2


def analyse_lineage(data: PairData, truth: Truth | None, settings: Settings = Settings(),
                    with_learning_curve: bool = True) -> dict[str, Any]:
    """Fit, evaluate, bootstrap and probe the preference model for one lineage."""
    oracle = truth_model(truth, data.lineage) if truth is not None else None
    true_w = truth.weights[data.lineage] if truth is not None else None

    # 1. Fit on all picks.
    fit = fit_bradley_terry(data.X, data.y, lam=settings.lam)

    # 2. Held-out performance from grouped (by session) cross-validation.
    cv = cross_validate(data, settings.n_folds, settings.cv_seed, settings.lam)
    heldout = heldout_metrics(data.y, cv.p_heldout)
    heldout["n_folds"] = int(cv.folds.max() + 1)
    heldout["in_sample_accuracy"] = accuracy(data.y, fit.predict_proba(data.X))
    if oracle is not None:
        oracle_metrics = heldout_metrics(data.y, oracle.predict_proba(data.X))
        heldout["oracle_accuracy"] = oracle_metrics["accuracy"]
        heldout["oracle_accuracy_ci_low"] = oracle_metrics["accuracy_ci_low"]
        heldout["oracle_accuracy_ci_high"] = oracle_metrics["accuracy_ci_high"]
        heldout["oracle_log_loss"] = oracle_metrics["log_loss"]
    heldout["meets_target_60"] = bool(heldout["accuracy"] > 0.60)

    # 3. Session bootstrap for weight uncertainty.
    boot_bias, boot_w = session_bootstrap(data, settings.n_boot, settings.boot_seed, settings.lam)
    w_lo, w_hi = percentile_interval(boot_w)
    b_lo, b_hi = percentile_interval(boot_bias)

    # 4. Sensitivity: OAT effect (with bootstrap CI) and permutation importance.
    effect = one_at_a_time_effect(fit.weights)
    eff_lo, eff_hi = percentile_interval(one_at_a_time_effect(boot_w))
    perm = permutation_importance(data, cv, settings.n_permutations, settings.perm_seed)

    # Collinearity diagnostic: creatures of one generation share elite parents, so the
    # left-right differences of different genes can be correlated. A gene whose
    # difference column tracks another's cannot be told apart from it well.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(data.X.T)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr)

    genes = []
    for j, key in enumerate(GENE_KEYS):
        partner = int(np.argmax(np.abs(corr[j])))
        genes.append({
            "gene": key,
            "label": GENE_LABELS[key],
            "weight": float(fit.weights[j]),
            "ci_low": float(w_lo[j]),
            "ci_high": float(w_hi[j]),
            "true_weight": float(true_w[j]) if true_w is not None else None,
            "ooa_effect": float(effect[j]),
            "ooa_ci_low": float(eff_lo[j]),
            "ooa_ci_high": float(eff_hi[j]),
            "perm_importance": float(perm["accuracy_drop"][j]),
            "perm_importance_sd": float(perm["accuracy_drop_sd"][j]),
            "perm_log_loss_increase": float(perm["log_loss_increase"][j]),
            # Spread of this gene's left-right differences: how much it varied in the pairs shown.
            "diff_sd": float(np.std(data.X[:, j])),
            "most_correlated_gene": GENE_KEYS[partner],
            "max_abs_diff_corr": float(abs(corr[j, partner])),
        })

    result: dict[str, Any] = {
        "lineage": data.lineage,
        "n_picks": data.n_picks,
        "n_sessions": int(len(data.sessions)),
        "n_participants": int(len(set(data.participant))),
        "left_chosen_rate": float(data.y.mean()),
        "fit": {
            "side_bias": fit.bias,
            "side_bias_ci_low": float(b_lo),
            "side_bias_ci_high": float(b_hi),
            "true_side_bias": truth.side_bias if truth is not None else None,
            "lambda": fit.lam,
            "iterations": fit.n_iter,
            "converged": fit.converged,
            "max_abs_gradient": fit.grad_norm,
            "penalised_objective": fit.objective,
            "log_likelihood": fit.log_likelihood,
        },
        "heldout": heldout,
        "bootstrap": {"B": settings.n_boot, "seed": settings.boot_seed, "unit": "session"},
        "genes": genes,
    }

    # 5. Recovery of the known weights (simulation only).
    if true_w is not None:
        result["recovery"] = weight_recovery(true_w, fit.weights, w_lo, w_hi)

    # 6. Learning curve.
    if with_learning_curve:
        result["learning_curve"] = learning_curve(
            data, settings.lc_repeats, settings.lc_test_fraction, settings.lc_seed,
            settings.lam, oracle)
    return result


def quick_summary(data: PairData, truth: Truth | None, settings: Settings = Settings()) -> dict[str, Any]:
    """The few headline numbers used in the multi-seed robustness check."""
    fit = fit_bradley_terry(data.X, data.y, lam=settings.lam)
    cv = cross_validate(data, settings.n_folds, settings.cv_seed, settings.lam)
    out: dict[str, Any] = {"n_picks": data.n_picks, "heldout_accuracy": accuracy(data.y, cv.p_heldout)}
    if truth is not None:
        true_w = truth.weights[data.lineage]
        rec = weight_recovery(true_w, fit.weights)
        out["oracle_accuracy"] = accuracy(data.y, truth_model(truth, data.lineage).predict_proba(data.X))
        out["recovery_r"] = rec["pearson_r"]
        out["sign_agree_nonzero"] = rec["sign_agree_nonzero"]
        out["n_nonzero"] = rec["n_nonzero"]
    return out
