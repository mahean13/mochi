"""Publication figures (PNG at 300 dpi and SVG) for the APA-style report.

Style: white background, no titles (captions live in the document), 9 pt text at
print size, no top/right spines, light grey gridlines only where they help. The
two lineages always differ by marker shape (home = circle, trust = square) or by
direct labels as well as by colour, so no figure relies on colour alone.

Every function takes the ``results`` dictionary written to results.json, so the
figures can be redrawn from that file without refitting anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .genes import GENE_KEYS, GENE_LABELS  # noqa: E402

# ---------------------------------------------------------------- house style
COLOURS = {"home": "#B8501F", "trust": "#3B6A8C"}
INK, MUTED, LIGHT = "#2A231E", "#6E6157", "#E4D8C8"
MARKERS = {"home": "o", "trust": "s"}
NAMES = {"home": "Home", "trust": "Trust"}
FULL_WIDTH, HALF_WIDTH = 6.5, 3.1  # inches at print size
FONT_SIZE = 9


def apply_style() -> None:
    """Set matplotlib defaults for every figure in this module."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in available else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "svg.fonttype": "path",  # text as outlines: the SVG renders the same everywhere
        "lines.solid_capstyle": "round",
    })


def save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    """Save as PNG (300 dpi) and SVG; return the paths written."""
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _panel_label(ax: plt.Axes, text: str) -> None:
    """Lineage name in the top-left corner of a panel (a label, not a title)."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FONT_SIZE, fontweight="bold", color=INK)


def _lineages(results: dict[str, Any]) -> list[str]:
    return list(results["lineages"].keys())


def _fmt_signed(value: float, digits: int = 1) -> str:
    """Format with a true minus sign and without a negative zero."""
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def _callout_labels(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, tops: np.ndarray,
                    labels: list[str]) -> None:
    """Label selected points without overlaps, in a fixed, predictable layout.

    Points with x > 0 are labelled in a column just right of the rightmost point,
    each joined to its point by a thin leader line; the column positions are pushed
    apart vertically so labels never touch. Other points get a label centred just
    above the top of their error bar. Call after the layout is final.
    """
    fig = ax.figure
    fig.canvas.draw()
    # One text line (with a little air) expressed in data units on the y axis.
    to_data = ax.transData.inverted()
    line_px = FONT_SIZE * 1.35 * fig.dpi / 72
    gap = abs(to_data.transform((0, line_px))[1] - to_data.transform((0, 0))[1])

    right = [i for i in range(len(xs)) if xs[i] > 0]
    if right:
        column_x = max(xs) + 0.35
        order = sorted(right, key=lambda i: ys[i])
        label_y = [ys[i] for i in order]
        for k in range(1, len(label_y)):
            label_y[k] = max(label_y[k], label_y[k - 1] + gap)
        # If the column runs off the top, shift it down as a block.
        overflow = label_y[-1] + gap / 2 - ax.get_ylim()[1]
        if overflow > 0:
            label_y = [v - overflow for v in label_y]
        for i, ly in zip(order, label_y):
            ax.plot([xs[i] + 0.08, column_x - 0.06], [ys[i], ly], lw=0.5, color=MUTED, zorder=2)
            ax.text(column_x, ly, labels[i], ha="left", va="center", fontsize=FONT_SIZE, color=INK)
    for i in range(len(xs)):
        if i not in right:
            ax.text(xs[i], tops[i] + gap * 0.3, labels[i], ha="center", va="bottom",
                    fontsize=FONT_SIZE, color=INK)


# ---------------------------------------------------------------- recovery
def fig_recovery(results: dict[str, Any], out_dir: Path, n_labels: int = 4) -> list[Path]:
    """True vs estimated weight per gene, one panel per lineage, with 95% CIs."""
    lins = [l for l in _lineages(results) if "recovery" in results["lineages"][l]]
    if not lins:
        return []
    fig, axes = plt.subplots(1, len(lins), figsize=(FULL_WIDTH, 3.1), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    all_vals = [v for l in lins for g in results["lineages"][l]["genes"]
                for v in (g["true_weight"], g["ci_low"], g["ci_high"])]
    lim = (min(all_vals) - 0.3, max(all_vals) + 0.3)
    # Extra room on the right so the largest genes can be labelled beside their point.
    xlim = (lim[0], lim[1] + 1.1)
    panels = []
    for ax, lin in zip(axes, lins):
        res = results["lineages"][lin]
        genes = res["genes"]
        true = np.array([g["true_weight"] for g in genes])
        est = np.array([g["weight"] for g in genes])
        lo = np.array([g["ci_low"] for g in genes])
        hi = np.array([g["ci_high"] for g in genes])
        # The 12 zero-weight genes share x = 0; spread them slightly so their CIs stay
        # readable. Only these points are displaced, by at most 0.12 on the x axis.
        x = true.copy()
        zero = np.flatnonzero(true == 0)
        x[zero] = np.linspace(-0.12, 0.12, zero.size)

        ax.plot(lim, lim, ls="--", lw=0.8, color=MUTED, zorder=1)  # identity, y = x
        # Identity label in the empty lower-left corner, along the line.
        ax.text(lim[0] + 0.15, lim[0] + 0.25, "Identity", rotation=45, rotation_mode="anchor",
                ha="left", va="bottom", color=MUTED, fontsize=FONT_SIZE, transform_rotates_text=True)
        ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
        ax.axvline(0, color=LIGHT, lw=0.6, zorder=0)
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt=MARKERS[lin], ms=4.5,
                    color=COLOURS[lin], mec="white", mew=0.6, elinewidth=0.9, capsize=0, zorder=3)
        r = res["recovery"]["pearson_r"]
        ax.text(0.04, 0.96, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="top", color=INK)
        _panel_label(ax, NAMES[lin])
        ax.set_xlim(xlim)
        ax.set_ylim(lim)
        ax.set_xticks(np.arange(np.ceil(lim[0]), np.floor(lim[1]) + 1))  # ticks only where data can be
        ax.set_xlabel("True weight (logits)")
        panels.append((ax, x, true, est, lo, hi))
    axes[0].set_ylabel("Estimated weight (logits)")
    fig.tight_layout(w_pad=2.0)

    # Label the genes with the largest true weights once the layout is final.
    for ax, x, true, est, lo, hi in panels:
        top = np.argsort(-np.abs(true), kind="stable")[:n_labels]
        _callout_labels(ax, x[top], est[top], hi[top], [GENE_LABELS[GENE_KEYS[j]] for j in top])
    return save(fig, out_dir, "fig_recovery")


# ---------------------------------------------------------------- learning curve
def fig_learning_curve(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Held-out accuracy vs training picks with 95% bands and reference lines."""
    lins = [l for l in _lineages(results) if results["lineages"][l].get("learning_curve")]
    fig, axes = plt.subplots(1, len(lins), figsize=(FULL_WIDTH, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lin in zip(axes, lins):
        lc = results["lineages"][lin]["learning_curve"]
        x = np.array(lc["train_picks"])
        mean = 100 * np.array(lc["accuracy_mean"])
        ax.fill_between(x, 100 * np.array(lc["accuracy_low"]), 100 * np.array(lc["accuracy_high"]),
                        color=COLOURS[lin], alpha=0.16, lw=0)
        ax.plot(x, mean, color=COLOURS[lin], lw=1.6, marker=MARKERS[lin], ms=3.8, mec="white", mew=0.5)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        xmax = x.max() * 1.02
        # Chance and target labels sit just below their lines at the left end, where
        # the curve has not started yet; the oracle label sits above its line at the
        # right end, so labels never collide even when the oracle is near 60%.
        for y, label in [(50, "Chance 50%"), (60, "Target 60%")]:
            ax.axhline(y, ls="--", lw=0.8, color=MUTED, zorder=1)
            ax.text(xmax * 0.01, y - 0.5, label, ha="left", va="top", color=MUTED)
        oracle = lc["oracle_accuracy_mean"]
        if oracle is not None:
            ax.axhline(100 * oracle, ls=":", lw=1.2, color=COLOURS[lin], zorder=1)
            ax.text(xmax, 100 * oracle + 0.5, f"Oracle {100 * oracle:.1f}%", ha="right", va="bottom", color=INK)
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.arange(0, x.max() + 1, 80))
        ax.set_xlabel("Training picks")
        _panel_label(ax, NAMES[lin])
    lows = [100 * min(results["lineages"][l]["learning_curve"]["accuracy_low"]) for l in lins]
    highs = [100 * max(results["lineages"][l]["learning_curve"]["accuracy_high"]) for l in lins]
    axes[0].set_ylim(5 * np.floor((min(lows + [48]) - 1) / 5), 5 * np.ceil((max(highs) + 1) / 5))
    axes[0].set_ylabel("Held-out accuracy (%)")
    fig.tight_layout(w_pad=1.5)
    return save(fig, out_dir, "fig_learning_curve")


# ---------------------------------------------------------------- sensitivity
def fig_sensitivity(results: dict[str, Any], out_dir: Path, top: int = 10) -> list[Path]:
    """One-at-a-time effects (pp) with 95% CIs for the top genes of each lineage."""
    lins = _lineages(results)
    fig, axes = plt.subplots(1, len(lins), figsize=(FULL_WIDTH, 3.3), sharex=True)
    axes = np.atleast_1d(axes)
    span = max(max(abs(g["ooa_ci_low"]), abs(g["ooa_ci_high"]))
               for l in lins for g in results["lineages"][l]["genes"]) * 100
    for ax, lin in zip(axes, lins):
        genes = sorted(results["lineages"][lin]["genes"], key=lambda g: -abs(g["ooa_effect"]))[:top]
        genes = genes[::-1]  # largest at the top
        y = np.arange(len(genes))
        eff = np.array([100 * g["ooa_effect"] for g in genes])
        lo = np.array([100 * g["ooa_ci_low"] for g in genes])
        hi = np.array([100 * g["ooa_ci_high"] for g in genes])
        ax.barh(y, eff, height=0.62, color=COLOURS[lin], lw=0)
        ax.errorbar(eff, y, xerr=[eff - lo, hi - eff], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(y, [g["label"] for g in genes])
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlim(-span * 1.08, span * 1.08)
        ax.set_xlabel("Change in P(picked) (pp)")
        _panel_label(ax, NAMES[lin])
        ax.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    return save(fig, out_dir, "fig_sensitivity")


# ---------------------------------------------------------------- drift
def drift_genes(results: dict[str, Any], n: int = 6) -> list[int]:
    """Indices of the genes with the largest |weight| in either lineage.

    Uses the true weights when they exist, else the estimated ones. Ties keep
    gene order, so the choice is deterministic.
    """
    key = "true_weight" if all(g["true_weight"] is not None
                               for r in results["lineages"].values() for g in r["genes"]) else "weight"
    strength = np.max([[abs(g[key]) for g in r["genes"]] for r in results["lineages"].values()], axis=0)
    return list(np.argsort(-strength, kind="stable")[:n])


def fig_drift(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Active-population mean of the strongest genes over generations, both lineages."""
    lins = [l for l in _lineages(results) if "drift" in results["lineages"][l]]
    if not lins:
        return []
    genes = drift_genes(results)
    ncol = 3
    nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH, 1.75 * nrow + 0.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, j in zip(axes, genes):
        for lin in lins:
            d = results["lineages"][lin]["drift"]
            gens = np.array(d["generations"])
            vals = np.array(d["mean_genes"])[:, j]
            ax.plot(gens, vals, color=COLOURS[lin], lw=1.4, marker=MARKERS[lin], ms=3, mec="white", mew=0.4,
                    label=NAMES[lin])
        ax.axhline(0.5, ls="--", lw=0.7, color=MUTED, zorder=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.text(0.03, 0.97, GENE_LABELS[GENE_KEYS[j]], transform=ax.transAxes, ha="left", va="top",
                color=INK, fontweight="bold")
    axes[0].text(0.97, 0.47, "Midpoint 0.5", transform=axes[0].transAxes,
                 ha="right", va="top", color=MUTED)
    for ax in axes[len(genes):]:
        ax.set_visible(False)
    for ax in axes[(nrow - 1) * ncol:]:
        ax.set_xlabel("Generation")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("Mean gene value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(lins), bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=1.0)
    return save(fig, out_dir, "fig_drift")


# ---------------------------------------------------------------- archive
def fig_archive(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Final 5 x 5 archives (heatmaps of elite scores) and filled cells per generation."""
    lins = [l for l in _lineages(results) if "archive" in results["lineages"][l]]
    if not lins:
        return []
    grids = {l: np.array(results["lineages"][l]["archive"]["grid_scores"], dtype=float) for l in lins}
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    vmin, vmax = float(finite.min()), float(finite.max())
    # One neutral sequential ramp (light to dark ink) shared by both lineages, so equal
    # shades mean equal scores; the score is also printed in each cell.
    cmap = LinearSegmentedColormap.from_list("mochi_seq", ["#F6F0E8", "#C9B49B", "#7A6552", INK])

    # Fixed layout in inches: two square heatmaps, a colour bar, then the line chart.
    height = 2.55
    fig = plt.figure(figsize=(FULL_WIDTH, height))

    def place(left: float, bottom: float, width: float, top_height: float) -> plt.Axes:
        return fig.add_axes((left / FULL_WIDTH, bottom / height, width / FULL_WIDTH, top_height / height))

    side, bottom = 1.62, 0.62
    heat_lefts = [0.52, 0.52 + side + 0.16]
    image = None
    for k, lin in enumerate(lins):
        ax = place(heat_lefts[k], bottom, side, side)
        grid = grids[lin]
        image = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                          extent=(-0.5, 4.5, -0.5, 4.5))
        for i in range(5):  # rows: machine cues
            for j in range(5):  # columns: baby schema
                v = grid[i, j]
                if np.isfinite(v):
                    dark = (v - vmin) / (vmax - vmin) > 0.55
                    ax.text(j, i, _fmt_signed(v), ha="center", va="center", fontsize=FONT_SIZE,
                            color="white" if dark else INK)
                else:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor=LIGHT, lw=0))
        ax.set_xticks(range(5), ["1", "2", "3", "4", "5"])
        ax.set_yticks(range(5), ["1", "2", "3", "4", "5"] if k == 0 else [])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Baby schema (low to high)")
        if k == 0:
            ax.set_ylabel("Machine cues (low to high)")
        _panel_label(ax, NAMES[lin])

    cbar_left = heat_lefts[-1] + side + 0.1
    cax = place(cbar_left, bottom, 0.08, side)
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_signed(v)))
    cbar.set_label("Elite score (log-strength)")

    line_left = cbar_left + 1.2
    ax = place(line_left, bottom, FULL_WIDTH - line_left - 0.1, side)
    ends = []
    for lin in lins:
        h = results["lineages"][lin]["archive"]["history"]
        ax.plot(h["generations"], h["filled_cells"], color=COLOURS[lin], lw=1.4, marker=MARKERS[lin], ms=3.5,
                mec="white", mew=0.4)
        ends.append([h["generations"][-1], h["filled_cells"][-1], lin])
    # Direct end labels, nudged apart vertically when the two lines end close together.
    ends.sort(key=lambda e: e[1])
    for k in range(1, len(ends)):
        ends[k][1] = max(ends[k][1], ends[k - 1][1] + 2.2)
    for gx, gy, lin in ends:
        ax.text(gx + 0.6, gy, NAMES[lin], color=INK, va="center", ha="left")
    ax.axhline(25, ls="--", lw=0.7, color=MUTED)
    ax.text(0, 24.2, "All 25 cells", color=MUTED, va="top", ha="left")
    ax.set_ylim(0, 26.5)
    gmax = max(e[0] for e in ends)
    ax.set_xlim(-0.5, gmax + 7.5)
    ax.set_xticks(np.arange(0, gmax + 1, 5))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Filled cells")
    return save(fig, out_dir, "fig_archive")


# ---------------------------------------------------------------- seeds
def fig_seeds(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Held-out accuracy and recovery r across simulation seeds and designs."""
    rob = results.get("robustness") or {}
    if not rob:
        return []
    has_r = any("recovery_r" in st for s in rob.values() for st in s["lineages"].values())
    metrics = [("heldout_accuracy", "Held-out accuracy (%)", 100)]
    if has_r:
        metrics.append(("recovery_r", "Recovery r (true vs estimated)", 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(FULL_WIDTH, 3.0))
    axes = np.atleast_1d(axes)
    groups = [(lin, design) for lin in ("home", "trust") for design in rob]
    rng = np.random.default_rng(0)  # jitter only, for legibility
    for ax, (metric, label, scale) in zip(axes, metrics):
        for k, (lin, design) in enumerate(groups):
            runs = rob[design]["lineages"][lin]["runs"]
            vals = scale * np.array([r[metric] for r in runs])
            jitter = rng.uniform(-0.17, 0.17, vals.size)
            filled = design == list(rob)[0]
            ax.scatter(k + jitter, vals, s=16, marker=MARKERS[lin], linewidths=0.9, zorder=3,
                       facecolors=COLOURS[lin] if filled else "white", edgecolors=COLOURS[lin], alpha=0.9)
            ax.hlines(vals.mean(), k - 0.3, k + 0.3, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(len(groups)),
                      [NAMES[l] + "\n" + d.replace("x", " \u00d7 ") for l, d in groups])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylabel(label)
        ax.set_xlabel("Lineage and design (raters \u00d7 sessions)")
        if metric == "heldout_accuracy":
            for y, text in [(60, "Target 60%"), (50, "Chance 50%")]:
                ax.axhline(y, ls="--", lw=0.8, color=MUTED, zorder=1)
                ax.text(-0.55, y + 0.3, text, ha="left", va="bottom", color=MUTED)
            lo = min(48, ax.get_ylim()[0])
            ax.set_ylim(lo, max(80, ax.get_ylim()[1]))
        else:
            ax.set_ylim(min(0.5, ax.get_ylim()[0]), 1.0)
    fig.tight_layout(w_pad=2.5)
    return save(fig, out_dir, "fig_seeds")


def make_all(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Draw every figure that the available results support."""
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for draw in (fig_recovery, fig_learning_curve, fig_sensitivity, fig_drift, fig_archive, fig_seeds):
        paths += draw(results, out_dir)
    return paths
