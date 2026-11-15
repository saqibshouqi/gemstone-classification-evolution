from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gemstone_trends.evaluation.report import _generation_chart


class GenerationChartTests(unittest.TestCase):
    def test_chart_reads_the_scoreboard_top1_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trend.svg"
            _generation_chart(
                output,
                "labelfree",
                [{"arm": "g2", "top1_accuracy_mean": 0.68}],
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("68.0%", rendered)


if __name__ == "__main__":
    unittest.main()