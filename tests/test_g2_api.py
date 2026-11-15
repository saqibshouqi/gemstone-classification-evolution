from __future__ import annotations

import unittest
from pathlib import Path

from g2.api import G2Predictor
from g2.features import FEATURE_LENGTH, FEATURE_NAME
from g2.training import train_one


class G2OracleGuardTests(unittest.TestCase):
    def test_training_rejects_published_masks_without_explicit_oracle_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "uses the known class label"):
            train_one(
                root=Path("."),
                model="rf",
                mode="published",
                seed=0,
                jobs=1,
                refresh_features=False,
            )

    def test_inference_rejects_class_conditioned_published_model(self) -> None:
        artifact = {
            "format": "gemstone-trends-g2",
            "format_version": 1,
            "feature_name": FEATURE_NAME,
            "feature_length": FEATURE_LENGTH,
            "mode": "published",
        }
        with self.assertRaisesRegex(ValueError, "cannot infer without the true class label"):
            G2Predictor(artifact)


if __name__ == "__main__":
    unittest.main()
