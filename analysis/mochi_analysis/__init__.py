"""Statistical analysis of Mochi paired-comparison data.

Modules:
    genes        the 20-gene genome and the two lineages
    io           loading picks.csv, state.json and truth.json
    model        feature Bradley-Terry model fitted by penalised Newton/IRLS
    evaluate     grouped cross-validation, scoring rules, session bootstrap,
                 learning curves and weight recovery
    sensitivity  one-at-a-time effects and permutation importance
    lineage      gene drift and MAP-Elites archive history from state.json
    robustness   reruns over simulation seeds and designs
    pipeline     the per-lineage analysis assembled from the above
    report       results.json, CSV tables and summary.md
    figures      publication figures
"""

__version__ = "1.0.0"
