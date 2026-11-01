"""Loading a Mochi data folder (a simulation run or a real export).

A data folder contains:

* ``picks.csv`` (required): one row per pick, with both creatures' genes.
* ``state.json`` (optional): the full engine state, used for lineage drift and the
  MAP-Elites archive. Without it those descriptive parts are skipped.
* ``truth.json`` (simulations only): the true preference weights and side bias.
  A real export has no truth file, and every truth-dependent result is skipped.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .genes import GENE_KEYS, LINEAGES, N_GENES


@dataclass(frozen=True)
class PairData:
    """Paired-comparison data for one lineage, in the form the model needs.

    Attributes:
        X: (n, 20) matrix of gene differences, left minus right. Genes are centred
            at 0.5 in the model (x = genes - 0.5), but the centring cancels in a
            difference, so X is simply ``genes_left - genes_right``.
        y: (n,) array, 1 if the left creature was chosen, else 0.
        session: (n,) session identifier per pick (the unit of grouping).
        participant: (n,) participant code per pick.
        gen: (n,) generation in which the pick was made.
    """

    lineage: str
    X: np.ndarray
    y: np.ndarray
    session: np.ndarray
    participant: np.ndarray
    gen: np.ndarray

    @property
    def n_picks(self) -> int:
        return int(self.y.shape[0])

    @property
    def sessions(self) -> np.ndarray:
        """Unique session identifiers, in order of first appearance."""
        _, first = np.unique(self.session, return_index=True)
        return self.session[np.sort(first)]

    def subset(self, mask: np.ndarray) -> "PairData":
        """Return the picks selected by a boolean mask or an index array."""
        return PairData(self.lineage, self.X[mask], self.y[mask], self.session[mask],
                        self.participant[mask], self.gen[mask])


@dataclass(frozen=True)
class Truth:
    """Known generating parameters of a simulation (see sim/simulate.mjs)."""

    weights: dict[str, np.ndarray]  # lineage -> (20,) shared true weights
    side_bias: float  # logit added in favour of the left creature
    rater_spread: float  # SD of each rater's deviation from the shared weights


def read_picks(data_dir: Path) -> list[dict[str, str]]:
    """Read ``picks.csv`` as a list of row dictionaries."""
    path = Path(data_dir) / "picks.csv"
    if not path.exists():
        raise FileNotFoundError(f"No picks.csv in {data_dir}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pair_data_from_rows(rows: list[dict[str, str]], lineage: str) -> PairData:
    """Build the design for one lineage from picks.csv rows.

    The response is taken from the ``chosen`` column compared with ``left``, not
    from ``side``, so the two are cross-checked: a mismatch raises an error.
    """
    rows = [r for r in rows if r["lineage"] == lineage]
    n = len(rows)
    X = np.empty((n, N_GENES))
    y = np.empty(n, dtype=int)
    for i, r in enumerate(rows):
        left = np.array([float(r[f"left_{k}"]) for k in GENE_KEYS])
        right = np.array([float(r[f"right_{k}"]) for k in GENE_KEYS])
        X[i] = left - right
        y[i] = int(r["chosen"] == r["left"])
        if r.get("side") and (r["side"] == "left") != bool(y[i]):
            raise ValueError(f"Pick {r['n']}: 'side' disagrees with 'chosen'")
    return PairData(
        lineage=lineage,
        X=X,
        y=y,
        session=np.array([r["session"] for r in rows], dtype=object),
        participant=np.array([r["participant"] for r in rows], dtype=object),
        gen=np.array([int(r["gen"]) for r in rows], dtype=int),
    )


def load_pairs(data_dir: Path) -> dict[str, PairData]:
    """Load every lineage that has at least one pick, keyed by lineage id."""
    rows = read_picks(data_dir)
    present = {r["lineage"] for r in rows}
    return {lin: pair_data_from_rows(rows, lin) for lin in LINEAGES if lin in present}


def load_state(data_dir: Path) -> dict[str, Any] | None:
    """Load ``state.json`` if present, else None."""
    path = Path(data_dir) / "state.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_truth(path: Path | None) -> Truth | None:
    """Load a ``truth.json`` written by the simulator, or None if absent."""
    if path is None or not Path(path).exists():
        return None
    raw = json.loads(Path(path).read_text())
    if list(raw["genes"]) != list(GENE_KEYS):
        raise ValueError("truth.json gene order differs from the analysis gene order")
    return Truth(
        weights={lin: np.asarray(w, dtype=float) for lin, w in raw["weights"].items()},
        side_bias=float(raw["side_bias"]),
        rater_spread=float(raw.get("rater_spread", float("nan"))),
    )
