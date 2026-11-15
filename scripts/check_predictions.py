from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemstone_trends.evaluation.io import main

raise SystemExit(main())
