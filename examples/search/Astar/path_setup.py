"""Adds the examples and repo roots to sys.path for direct A* execution."""

from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_ROOT = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]

for search_path in (REPO_ROOT, EXAMPLES_ROOT):
    resolved = str(search_path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
