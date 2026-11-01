"""The Python gene list mirrors GENES in web/engine.js."""

import re

from conftest import REPO_DIR
from mochi_analysis.genes import GENES


def test_gene_list_matches_engine():
    source = (REPO_DIR / "web" / "engine.js").read_text()
    block = source[source.index("export const GENES = ["):source.index("];", source.index("export const GENES = ["))]
    pairs = re.findall(r'key: "(\w+)", label: "([^"]+)"', block)
    assert [(g.key, g.label) for g in GENES] == pairs
