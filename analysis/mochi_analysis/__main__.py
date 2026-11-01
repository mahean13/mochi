"""Command-line entry point.

Run from the ``analysis`` folder:

    python3 -m mochi_analysis ../data/sim --out results --figures ../docs/figures \\
        --seeds ../data/sim_seeds --scale-seeds ../data/sim_scale

``--truth`` defaults to ``<data_dir>/truth.json`` when that file exists. For a real
export (no truth file) every truth-dependent result is skipped. If ``state.json``
is missing, the lineage drift and archive parts are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import figures, lineage, report
from .genes import GENE_LABELS, LINEAGE_QUESTIONS
from .io import load_pairs, load_state, load_truth
from .pipeline import Settings, analyse_lineage
from .robustness import summarise_design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mochi_analysis", description="Mochi preference analysis.")
    parser.add_argument("data_dir", type=Path, help="folder with picks.csv (and optionally state.json, truth.json)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="folder for results.json, summary.md, CSVs")
    parser.add_argument("--figures", type=Path, default=None, help="folder for figures (omit to skip figures)")
    parser.add_argument("--truth", type=Path, default=None, help="truth.json (default: <data_dir>/truth.json if present)")
    parser.add_argument("--seeds", type=Path, default=None, help="folder of seed_<k> runs, 3 raters x 10 sessions")
    parser.add_argument("--scale-seeds", type=Path, default=None, help="folder of seed_<k> runs, 15 raters x 2 sessions")
    parser.add_argument("--n-boot", type=int, default=1000, help="session bootstrap resamples (default 1000)")
    return parser.parse_args()


def descriptive_history(state: dict[str, Any], lin: str) -> dict[str, Any]:
    """Lineage drift and MAP-Elites archive results for one lineage."""
    grid = lineage.archive_grid(state, lin)
    return {
        "drift": lineage.gene_drift(state, lin),
        "archive": {
            "history": lineage.archive_history(state, lin),
            "grid_scores": grid["scores"],
            "grid_elites": grid["elites"],
            "filled": grid["filled"],
            "layout": "grid_scores[row][col]: row = machineness bin (0 low), col = babyness bin (0 low)",
        },
    }


def data_driven_notes(results: dict[str, Any]) -> list[str]:
    """Limitations and observations computed from the results (no hard-coded numbers)."""
    notes = []
    for lin, r in results["lineages"].items():
        h = r["heldout"]
        if "oracle_accuracy" in h and h["oracle_accuracy"] < 0.65:
            notes.append(f"{lin}: the true model itself reaches only {100 * h['oracle_accuracy']:.1f}% on these picks, "
                         f"so the 60% target sits close to the ceiling for this lineage; the fitted model reaches "
                         f"{100 * h['accuracy']:.1f}%.")
        if "recovery" in r:
            med_sd = float(np.median([g["diff_sd"] for g in r["genes"]]))

            def describe(g: dict[str, Any]) -> str:
                return (f"{g['label']} (true {g['true_weight']:+.1f}, estimate {g['weight']:+.2f}, "
                        f"CI [{g['ci_low']:+.2f}, {g['ci_high']:+.2f}], SD of pair difference {g['diff_sd']:.2f}, "
                        f"largest difference correlation {g['max_abs_diff_corr']:.2f} with "
                        f"{GENE_LABELS[g['most_correlated_gene']]})")

            missed = [g for g in r["genes"] if g["true_weight"] and g["ci_low"] <= 0 <= g["ci_high"]]
            if missed:
                notes.append(f"{lin}: non-zero genes whose 95% CI includes 0: "
                             + "; ".join(describe(g) for g in missed) + ".")
            false_pos = [g for g in r["genes"] if not g["true_weight"] and not g["ci_low"] <= 0 <= g["ci_high"]]
            if false_pos:
                notes.append(f"{lin}: zero-weight genes whose 95% CI excludes 0 (false positives): "
                             + "; ".join(describe(g) for g in false_pos) + ".")
            off = [g for g in r["genes"] if not g["ci_low"] <= g["true_weight"] <= g["ci_high"]
                   and g not in missed and g not in false_pos]
            if off:
                notes.append(f"{lin}: other CIs that miss the true weight: " + "; ".join(describe(g) for g in off) + ".")
            if missed or false_pos or off:
                notes.append(f"{lin}: median SD of pair differences across genes is {med_sd:.2f} (random pairs: about "
                             f"0.41). Genes that vary little between the two creatures shown carry little "
                             f"information about their weight, and correlated differences (creatures of one "
                             f"generation share elite parents) let weight move between genes.")
            slope = r["recovery"]["slope_est_on_true"]
            if slope < 0.9:
                notes.append(f"{lin}: estimates are pulled toward zero (slope of estimate on truth {slope:.2f}). "
                             f"Ridge shrinkage contributes, and so does rater heterogeneity: averaging logistic "
                             f"models over raters with different weights attenuates the pooled weights.")
    for design, summary in (results.get("robustness") or {}).items():
        for lin, st in summary["lineages"].items():
            if lin not in results["lineages"]:
                continue
            seed_acc = np.array([run["heldout_accuracy"] for run in st["runs"]])
            main_acc = results["lineages"][lin]["heldout"]["accuracy"]
            below = int(np.sum(seed_acc < main_acc))
            notes.append(f"{lin}: the main run's held-out accuracy ({100 * main_acc:.1f}%) is above {below} of "
                         f"{seed_acc.size} seeds of design {design} (seed mean {100 * seed_acc.mean():.1f}%).")
    notes.append("The one-sided binomial test treats held-out picks as independent; picks within a session share a "
                 "rater and a generation, so its p-value is optimistic. The session bootstrap and the grouped "
                 "cross-validation respect that clustering.")
    if results["meta"]["truth"]:
        notes.append("The truth is the shared mean weight vector. Each simulated rater deviates from it (SD "
                     f"{results['meta']['rater_spread']} per gene), so with few raters the target the data actually "
                     "carry differs from the shared truth; recovery r is measured against the shared truth.")
    notes.append("The learning curve draws training sessions at random regardless of generation; in the real "
                 "study, later sessions show more similar creatures because the lineage converges.")
    return notes


def main() -> None:
    args = parse_args()
    started = time.time()
    data_dir = args.data_dir
    truth_path = args.truth or (data_dir / "truth.json" if (data_dir / "truth.json").exists() else None)
    truth = load_truth(truth_path)
    pairs = load_pairs(data_dir)
    state = load_state(data_dir)
    settings = Settings(n_boot=args.n_boot)

    results: dict[str, Any] = {
        "meta": {
            "data_dir": str(data_dir),
            "truth": str(truth_path) if truth is not None else None,
            "rater_spread": truth.rater_spread if truth is not None else None,
            "settings": settings.__dict__,
            "questions": LINEAGE_QUESTIONS,
        },
        "lineages": {},
    }
    for lin, data in pairs.items():
        print(f"[{lin}] {data.n_picks} picks, {len(data.sessions)} sessions: fitting, CV, bootstrap, learning curve")
        res = analyse_lineage(data, truth, settings)
        if state is not None and lin in state.get("lineages", {}):
            res.update(descriptive_history(state, lin))
        results["lineages"][lin] = res

    robustness = {}
    for name, folder in (("3x10", args.seeds), ("15x2", args.scale_seeds)):
        if folder is not None and folder.exists():
            print(f"[robustness] design {name} from {folder}")
            summary = summarise_design(folder, settings)
            if summary:
                robustness[name] = summary
    results["robustness"] = robustness
    results["notes"] = data_driven_notes(results)
    results["meta"]["runtime_seconds"] = time.time() - started

    args.out.mkdir(parents=True, exist_ok=True)
    report.write_json(results, args.out / "results.json")
    report.write_weight_tables(results, args.out)
    report.write_summary(results, args.out / "summary.md")
    print(f"Wrote {args.out / 'results.json'}, {args.out / 'summary.md'} and weight tables")

    if args.figures is not None:
        # Draw from the rounded JSON so the figures show exactly what results.json reports.
        tidy = json.loads((args.out / "results.json").read_text())
        for path in figures.make_all(tidy, args.figures):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
