"""Robustness of the headline results across simulation seeds and study designs.

One simulation is one draw of rater weights, pair schedules and choices. To check
that the results are not a lucky draw, the simulator is rerun with many seeds and
each run is analysed with the same pipeline (cross-validated accuracy and weight
recovery only; no bootstrap, to keep it quick).

Two designs are checked:
* "3x10": 3 raters x 10 sessions each (the main simulation's design), 20 seeds.
* "15x2": 15 raters x 2 sessions each (the full planned study scale), 5 seeds.
Both give 1200 picks, 600 per lineage.

Generate the runs (needs Node.js), from the analysis folder:
    python3 -m mochi_analysis.robustness --simulator ../sim/simulate.mjs --out ../data
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .genes import LINEAGES
from .io import load_pairs, load_truth
from .pipeline import Settings, quick_summary

#: name -> (output folder under the data root, raters, sessions per rater, seeds)
DESIGNS: dict[str, tuple[str, int, int, list[int]]] = {
    "3x10": ("sim_seeds", 3, 10, list(range(1, 21))),
    "15x2": ("sim_scale", 15, 2, list(range(1, 6))),
}


def generate_runs(simulator: Path, data_root: Path) -> None:
    """Run the Node simulator for every design and seed (skips existing runs)."""
    for folder, raters, sessions, seeds in DESIGNS.values():
        for seed in seeds:
            out = data_root / folder / f"seed_{seed}"
            if (out / "picks.csv").exists():
                continue
            subprocess.run(["node", str(simulator), str(out), str(raters), str(sessions), str(seed)],
                           check=True, cwd=simulator.parent.parent)


def summarise_design(design_dir: Path, settings: Settings = Settings()) -> dict[str, Any] | None:
    """Analyse every ``seed_<k>`` folder of one design and summarise per lineage."""
    runs = sorted(design_dir.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    if not runs:
        return None
    per_run: dict[str, list[dict[str, Any]]] = {lin: [] for lin in LINEAGES}
    for run in runs:
        truth = load_truth(run / "truth.json")
        for lin, data in load_pairs(run).items():
            row = quick_summary(data, truth, settings)
            row["seed"] = int(run.name.split("_")[1])
            per_run[lin].append(row)

    summary: dict[str, Any] = {"n_seeds": len(runs), "lineages": {}}
    for lin, rows in per_run.items():
        stats: dict[str, Any] = {"runs": rows}
        for metric in ("heldout_accuracy", "oracle_accuracy", "recovery_r"):
            values = np.array([r[metric] for r in rows if metric in r], dtype=float)
            if values.size:
                stats[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                                 "min": float(values.min()), "max": float(values.max())}
        acc = np.array([r["heldout_accuracy"] for r in rows])
        stats["share_above_60"] = float(np.mean(acc > 0.60))
        summary["lineages"][lin] = stats
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the multi-seed simulation runs.")
    parser.add_argument("--simulator", type=Path, required=True, help="path to sim/simulate.mjs")
    parser.add_argument("--out", type=Path, required=True, help="data root, e.g. ../data")
    args = parser.parse_args()
    generate_runs(args.simulator.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
