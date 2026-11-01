"""Make the package importable when pytest runs from the repository or analysis folder."""

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ANALYSIS_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))
