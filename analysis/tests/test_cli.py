"""End to end: the CLI runs on a real-style export with no truth.json and no state.json."""

import csv
import json
import subprocess
import sys

import shutil

import pytest

from conftest import ANALYSIS_DIR, REPO_DIR


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_cli_without_truth(tmp_path):
    gen = tmp_path / "generated"
    subprocess.run(["node", "sim/simulate.mjs", str(gen), "3", "4", "11"], cwd=REPO_DIR, check=True, capture_output=True)
    data = tmp_path / "export"
    data.mkdir()
    # Keep only picks.csv, as a real export without truth might.
    rows = list(csv.reader((gen / "picks.csv").open()))
    with (data / "picks.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    out = tmp_path / "results"
    subprocess.run([sys.executable, "-m", "mochi_analysis", str(data), "--out", str(out), "--n-boot", "50",
                    "--figures", str(tmp_path / "figs")],
                   cwd=ANALYSIS_DIR, check=True, capture_output=True)
    results = json.loads((out / "results.json").read_text())
    assert results["meta"]["truth"] is None
    for lin in ("home", "trust"):
        res = results["lineages"][lin]
        assert "recovery" not in res
        assert "oracle_accuracy" not in res["heldout"]
        assert res["genes"][0]["true_weight"] is None
        assert "drift" not in res
    assert (out / "weights_home.csv").exists()
    assert "n/a" in (out / "summary.md").read_text()
    figs = {p.name for p in (tmp_path / "figs").iterdir()}
    assert "fig_sensitivity.png" in figs and "fig_recovery.png" not in figs
