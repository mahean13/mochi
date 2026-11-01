"""The Mochi genome: 20 genes on [0, 1], in the order used by web/engine.js.

This list mirrors ``GENES`` in ``web/engine.js`` (key and human label only). The
test ``tests/test_genes.py`` checks that the two stay in sync, so the Python
analysis can run on an exported data folder without needing the JavaScript code.
"""

from __future__ import annotations

from typing import NamedTuple


class Gene(NamedTuple):
    key: str
    label: str


GENES: tuple[Gene, ...] = (
    Gene("headWidth", "Head width"),
    Gene("headHeight", "Head height"),
    Gene("headRoundness", "Head roundness"),
    Gene("earShape", "Ear shape"),
    Gene("earSize", "Ear size"),
    Gene("earSpread", "Ear spread"),
    Gene("eyeSize", "Eye size"),
    Gene("eyeSpacing", "Eye spacing"),
    Gene("eyeHeight", "Eye height"),
    Gene("eyeShine", "Eye shine"),
    Gene("mouth", "Mouth"),
    Gene("mouthWidth", "Mouth width"),
    Gene("blush", "Blush"),
    Gene("bodySize", "Body size"),
    Gene("limbLength", "Leg length"),
    Gene("antenna", "Antenna"),
    Gene("facePlate", "Face plate"),
    Gene("hue", "Hue"),
    Gene("saturation", "Saturation"),
    Gene("lightness", "Lightness"),
)

GENE_KEYS: tuple[str, ...] = tuple(g.key for g in GENES)
GENE_LABELS: dict[str, str] = {g.key: g.label for g in GENES}
N_GENES: int = len(GENES)

#: The two lineages (questions) of the study, in reporting order.
LINEAGES: tuple[str, ...] = ("home", "trust")
LINEAGE_QUESTIONS: dict[str, str] = {
    "home": "Which would you rather have at home?",
    "trust": "Which would you trust?",
}
