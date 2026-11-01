"""Writing results: results.json, weights_<lineage>.csv and summary.md."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

WEIGHT_COLUMNS = ["gene", "label", "weight", "ci_low", "ci_high", "true_weight",
                  "ooa_effect", "ooa_ci_low", "ooa_ci_high", "perm_importance"]


def tidy(value: Any, digits: int = 4) -> Any:
    """Make a nested result JSON-safe and round floats sensibly.

    Ordinary floats keep ``digits`` decimals; very small magnitudes (such as
    p-values) keep 3 significant figures instead of being rounded to zero. NaN
    becomes None (JSON null).
    """
    if isinstance(value, dict):
        return {k: tidy(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tidy(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return tidy(value.tolist(), digits)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if math.isnan(x):
            return None
        if x != 0 and abs(x) < 10 ** -(digits - 1):
            return float(f"{x:.3g}")
        return round(x, digits)
    return value


def write_json(results: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(tidy(results), indent=2) + "\n")


def write_weight_tables(results: dict[str, Any], out_dir: Path) -> None:
    """One CSV per lineage with weights, CIs, truth, OAT effects and permutation importance."""
    for lin, res in results["lineages"].items():
        with (out_dir / f"weights_{lin}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=WEIGHT_COLUMNS)
            writer.writeheader()
            for g in res["genes"]:
                writer.writerow({k: ("" if g[k] is None else tidy(g[k])) for k in WEIGHT_COLUMNS})


# ---------------------------------------------------------------- markdown summary
def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _pval(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def _ci(lo: float, hi: float, pct: bool = False, digits: int = 2) -> str:
    if pct:
        return f"[{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def write_summary(results: dict[str, Any], path: Path) -> None:
    """A plain, factual summary of every headline number."""
    meta = results["meta"]
    lines = ["# Mochi preference analysis: summary", ""]
    lines.append(f"Data folder: `{meta['data_dir']}`. Truth file: "
                 f"{'`' + meta['truth'] + '`' if meta['truth'] else 'none (real export; truth-dependent results skipped)'}.")
    lines.append(f"Model: feature Bradley-Terry, P(left) = sigmoid(b + w . (x_left - x_right)), "
                 f"ridge lambda = {meta['settings']['lam']} on w only, fitted by Newton/IRLS. "
                 f"Cross-validation: {meta['settings']['n_folds']} folds grouped by session. "
                 f"Bootstrap: B = {meta['settings']['n_boot']} session resamples.")
    lines.append("")

    # Held-out performance table.
    lines += ["## Held-out performance (grouped cross-validation by session)", "",
              "| Lineage | Picks | Sessions | Accuracy | 95% CI (Wilson) | One-sided binomial p | Log loss | Brier | Oracle accuracy | Above 60% target |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for lin, r in results["lineages"].items():
        h = r["heldout"]
        oracle = _pct(h["oracle_accuracy"]) if "oracle_accuracy" in h else "n/a"
        lines.append(f"| {lin} | {r['n_picks']} | {r['n_sessions']} | {_pct(h['accuracy'])} | "
                     f"{_ci(h['accuracy_ci_low'], h['accuracy_ci_high'], pct=True)} | {_pval(h['binomial_p_one_sided'])} | "
                     f"{h['log_loss']:.3f} | {h['brier']:.3f} | {oracle} | {'yes' if h['meets_target_60'] else 'no'} |")
    lines += ["", "Chance is 50% accuracy, log loss 0.693 and Brier 0.250. The oracle uses the true "
              "shared weights and true side bias on the same held-out picks." if meta["truth"] else
              "Chance is 50% accuracy, log loss 0.693 and Brier 0.250.", ""]

    # Fit diagnostics.
    lines += ["## Model fit on all picks", ""]
    for lin, r in results["lineages"].items():
        f, h = r["fit"], r["heldout"]
        true_b = f" (true {f['true_side_bias']})" if f["true_side_bias"] is not None else ""
        lines.append(f"- {lin}: side bias b = {f['side_bias']:.3f}, 95% CI {_ci(f['side_bias_ci_low'], f['side_bias_ci_high'], digits=3)}{true_b}; "
                     f"left chosen in {_pct(r['left_chosen_rate'])} of picks. Converged in {f['iterations']} Newton iterations "
                     f"(max |gradient| {f['max_abs_gradient']:.1e}). In-sample accuracy {_pct(h['in_sample_accuracy'])}.")
    lines.append("")

    # Recovery.
    if meta["truth"]:
        lines += ["## Recovery of the true weights", ""]
        for lin, r in results["lineages"].items():
            rec = r["recovery"]
            lines.append(f"- {lin}: Pearson r = {rec['pearson_r']:.3f} (p = {_pval(rec['pearson_p'])}); slope of estimate on truth "
                         f"{rec['slope_est_on_true']:.2f}; signs correct on {rec['sign_agree_nonzero']} of {rec['n_nonzero']} non-zero genes; "
                         f"{rec['zero_genes_ci_include_0']} of {rec['n_zero']} zero-weight genes have a 95% CI that includes 0; "
                         f"{rec['nonzero_genes_ci_exclude_0']} of {rec['n_nonzero']} non-zero genes have a CI that excludes 0; "
                         f"{rec['ci_cover_true']} of 20 CIs contain the true weight.")
        lines.append("")

    # Sensitivity.
    lines += ["## Sensitivity", "",
              "One-at-a-time effect: change in the probability of being picked when one gene moves from 0.25 to 0.75, "
              "all else equal and without side bias, in percentage points (pp), with 95% bootstrap CI. "
              "Permutation importance: drop in pooled held-out accuracy when that gene's difference column is shuffled "
              f"({meta['settings']['n_permutations']} permutations).", ""]
    for lin, r in results["lineages"].items():
        genes = sorted(r["genes"], key=lambda g: -abs(g["ooa_effect"]))[:5]
        lines.append(f"Top 5 genes for {lin} by one-at-a-time effect:")
        lines.append("")
        lines.append("| Gene | Effect (pp) | 95% CI (pp) | Weight | True weight | Permutation importance (pp) |")
        lines.append("|---|---|---|---|---|---|")
        for g in genes:
            tw = f"{g['true_weight']:.2f}" if g["true_weight"] is not None else "n/a"
            lines.append(f"| {g['label']} | {100 * g['ooa_effect']:+.1f} | {_ci(g['ooa_ci_low'], g['ooa_ci_high'], pct=True)} | "
                         f"{g['weight']:.2f} | {tw} | {100 * g['perm_importance']:.1f} |")
        lines.append("")

    # Learning curve.
    lines += ["## Learning curve", ""]
    for lin, r in results["lineages"].items():
        lc = r.get("learning_curve")
        if not lc:
            continue
        acc = lc["accuracy_mean"]
        picks = lc["train_picks"]
        first = next((p for p, a in zip(picks, acc) if a > 0.60), None)
        lines.append(f"- {lin}: mean held-out accuracy {_pct(acc[0])} with {picks[0]:.0f} training picks and {_pct(acc[-1])} with "
                     f"{picks[-1]:.0f} (95% interval {_ci(lc['accuracy_low'][-1], lc['accuracy_high'][-1], pct=True)}; "
                     f"{lc['repeats']} repeats, {lc['test_sessions']} held-out sessions each). "
                     + (f"Mean first exceeds 60% at {first:.0f} training picks." if first else "Mean never exceeds 60%.")
                     + (f" Oracle on the same test sets: {_pct(lc['oracle_accuracy_mean'])}." if lc["oracle_accuracy_mean"] is not None else ""))
    lines.append("")

    # Lineage history.
    if any("drift" in r for r in results["lineages"].values()):
        lines += ["## Lineage drift and archive", ""]
        for lin, r in results["lineages"].items():
            if "drift" not in r:
                continue
            d, a = r["drift"], r["archive"]
            first, last = np.array(d["mean_genes"][0]), np.array(d["mean_genes"][-1])
            change = last - first
            order = np.argsort(-np.abs(change))[:3]
            moved = "; ".join(f"{r['genes'][j]['label']} {first[j]:.2f} to {last[j]:.2f}" for j in order)
            lines.append(f"- {lin}: generations 0 to {d['generations'][-1]}. Largest changes in the active-population mean: {moved}. "
                         f"Archive cells filled: {a['history']['filled_cells'][0]} after generation 0, "
                         f"{a['history']['filled_cells'][-1]} of 25 at the end.")
        lines.append("")

    # Robustness.
    rob = results.get("robustness") or {}
    if rob:
        lines += ["## Robustness across simulation seeds", "",
                  "| Design | Seeds | Lineage | Held-out accuracy mean (min, max) | Share of seeds above 60% | Oracle mean | Recovery r mean (min, max) |",
                  "|---|---|---|---|---|---|---|"]
        for design, s in rob.items():
            for lin, st in s["lineages"].items():
                acc = st["heldout_accuracy"]
                r_ = st.get("recovery_r")
                o = st.get("oracle_accuracy")
                lines.append(f"| {design} | {s['n_seeds']} | {lin} | {_pct(acc['mean'])} ({_pct(acc['min'])}, {_pct(acc['max'])}) | "
                             f"{_pct(st['share_above_60'], 0)} | {_pct(o['mean']) if o else 'n/a'} | "
                             + (f"{r_['mean']:.3f} ({r_['min']:.3f}, {r_['max']:.3f})" if r_ else "n/a") + " |")
        lines += ["", "Design 3x10 is 3 simulated raters with 10 sessions each; 15x2 is 15 raters with 2 sessions each. "
                  "Both give 600 picks per lineage.", ""]

    notes = results.get("notes") or []
    if notes:
        lines += ["## Notes and limitations", ""] + [f"- {n}" for n in notes] + [""]
    path.write_text("\n".join(lines))
