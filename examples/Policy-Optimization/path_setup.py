"""Adds the PPO script, examples, and repo roots to sys.path for direct execution."""

from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_ROOT = THIS_DIR.parent
REPO_ROOT = EXAMPLES_ROOT.parent

for search_path in (THIS_DIR, EXAMPLES_ROOT, REPO_ROOT):
    resolved = str(search_path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
