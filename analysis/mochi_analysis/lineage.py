"""Descriptive history of each lineage, read from the engine's state.json.

The engine records, at the end of every generation, the mean genome of that
generation's active creatures and the number of filled MAP-Elites cells. The
generation still being played (the current ``active`` list) has no history entry
yet, so its mean is computed here from the creature table.

Archive cells are keyed "i,j" in engine.js, where i is the babyness bin and j the
machineness bin, each in 0..4 (see ``descriptors`` and ``cellOf``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

GRID = 5  # MAP-Elites grid size per descriptor (engine.js GRID)


def gene_drift(state: dict[str, Any], lineage: str) -> dict[str, Any]:
    """Mean of every gene over the active creatures of each generation.

    Returns:
        {"generations": [g0, g1, ...], "mean_genes": (G, 20) nested list}. The last
        generation is the current, not yet completed one.
    """
    L = state["lineages"][lineage]
    gens = [h["gen"] for h in L["history"]]
    means = [h["meanGenes"] for h in L["history"]]
    active = [state["creatures"][cid]["genes"] for cid in L["active"]]
    if active:
        gens.append(L["gen"])
        means.append(np.mean(active, axis=0).tolist())
    return {"generations": gens, "mean_genes": means}


def archive_history(state: dict[str, Any], lineage: str) -> dict[str, list]:
    """Number of filled archive cells after each completed generation."""
    L = state["lineages"][lineage]
    return {
        "generations": [h["gen"] for h in L["history"]],
        "filled_cells": [h["archiveCells"] for h in L["history"]],
    }


def archive_grid(state: dict[str, Any], lineage: str) -> dict[str, Any]:
    """Final archive as a 5 x 5 grid of elite scores (NaN where a cell is empty).

    Rows index machineness (row 0 = lowest), columns index babyness (column 0 =
    lowest). Scores are the creature-level Bradley-Terry log-strengths that the
    engine stored at its last generation step.
    """
    L = state["lineages"][lineage]
    scores = np.full((GRID, GRID), np.nan)
    elites: list[list[str | None]] = [[None] * GRID for _ in range(GRID)]
    for cell, cid in L["archive"].items():
        baby, machine = (int(v) for v in cell.split(","))
        scores[machine, baby] = L["scores"][cid]
        elites[machine][baby] = cid
    return {"scores": scores, "elites": elites, "filled": int(np.isfinite(scores).sum())}
