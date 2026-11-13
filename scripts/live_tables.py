"""Print Tables 2a, 2b, 3 recomputed from a results CSV (default: results.csv at repo root)."""

import sys

from _live_common import ROOT  # noqa: F401  (adds src/ to sys.path)
from relay.live.tables import main

if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "results.csv"))
