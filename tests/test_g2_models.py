from __future__ import annotations

import unittest

from g2.models import MODEL_NAMES, make_pipeline


class ModelGridTests(unittest.TestCase):
    def test_each_model_uses_fold_safe_smote(self) -> None:
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                pipeline, grid = make_pipeline(name, seed=0, smote_neighbors=4)
                self.assertEqual(pipeline.named_steps["smote"].k_neighbors, 4)
                self.assertIn("classifier", pipeline.named_steps)
                self.assertTrue(grid)

    def test_random_forest_grid_matches_anchor_grid_size(self) -> None:
        _pipeline, grid = make_pipeline("rf", seed=0)
        candidate_count = 1
        for values in grid.values():
            candidate_count *= len(values)
        self.assertEqual(candidate_count, 32)

    def test_svm_grid_has_one_vs_rest_parameters(self) -> None:
        _pipeline, grid = make_pipeline("svm", seed=0)
        self.assertEqual(len(grid["classifier__estimator__kernel"]), 4)
        self.assertEqual(len(grid["classifier__estimator__C"]), 3)
        self.assertEqual(len(grid["classifier__estimator__gamma"]), 2)


if __name__ == "__main__":
    unittest.main()
