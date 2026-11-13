import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # so `from helpers import ...` works under pytest
sys.path.insert(0, str(HERE.parents[1] / "src"))   # relay package
