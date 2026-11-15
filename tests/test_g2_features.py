from __future__ import annotations

import unittest

import numpy as np

from g2.features import (
    FEATURE_LENGTH,
    HISTOGRAM_LENGTH,
    feature_view_length,
    extract_features,
    select_feature_view,
)


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        self.rgb[8:24, 8:24] = (120, 40, 180)
        self.mask = np.zeros((32, 32), dtype=bool)
        self.mask[8:24, 8:24] = True

    def test_rgb_histogram_and_full_masked_lbp_have_expected_lengths_and_sums(self) -> None:
        vector = extract_features(self.rgb, self.mask)
        self.assertEqual(vector.shape, (FEATURE_LENGTH,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(vector[:HISTOGRAM_LENGTH].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(vector[HISTOGRAM_LENGTH:].sum()), 1.0, places=6)

    def test_lbp_includes_zero_valued_segmentation_background(self) -> None:
        partial = extract_features(self.rgb, self.mask)
        segmented = np.where(self.mask[..., None], self.rgb, 0).astype(np.uint8)
        segmented_full_mask = extract_features(segmented, np.ones_like(self.mask))
        unsegmented_full_mask = extract_features(self.rgb, np.ones_like(self.mask))
        np.testing.assert_allclose(
            partial[HISTOGRAM_LENGTH:],
            segmented_full_mask[HISTOGRAM_LENGTH:],
        )
        self.assertFalse(
            np.array_equal(
                partial[HISTOGRAM_LENGTH:],
                unsegmented_full_mask[HISTOGRAM_LENGTH:],
            )
        )

    def test_empty_foreground_has_zero_colour_bins_and_valid_lbp(self) -> None:
        vector = extract_features(self.rgb, np.zeros_like(self.mask))
        self.assertTrue(np.all(vector[:HISTOGRAM_LENGTH] == 0))
        self.assertAlmostEqual(float(vector[HISTOGRAM_LENGTH:].sum()), 1.0, places=6)

    def test_e3_views_select_histogram_lbp_and_combined_features(self) -> None:
        matrix = np.arange(3 * FEATURE_LENGTH, dtype=np.float32).reshape(3, FEATURE_LENGTH)
        histogram = select_feature_view(matrix, "rgb8")
        lbp = select_feature_view(matrix, "lbp8_r1")
        combined = select_feature_view(matrix, "rgb8_lbp8_r1")
        self.assertEqual(histogram.shape, (3, HISTOGRAM_LENGTH))
        self.assertEqual(lbp.shape, (3, FEATURE_LENGTH - HISTOGRAM_LENGTH))
        self.assertEqual(combined.shape, matrix.shape)
        self.assertEqual(feature_view_length("rgb8"), HISTOGRAM_LENGTH)
        self.assertEqual(feature_view_length("lbp8_r1"), FEATURE_LENGTH - HISTOGRAM_LENGTH)
        np.testing.assert_array_equal(histogram, matrix[:, :HISTOGRAM_LENGTH])
        np.testing.assert_array_equal(lbp, matrix[:, HISTOGRAM_LENGTH:])
        np.testing.assert_array_equal(combined, matrix)

    def test_e3_view_rejects_unknown_names_and_malformed_matrices(self) -> None:
        with self.assertRaises(ValueError):
            feature_view_length("unknown")
        with self.assertRaises(ValueError):
            select_feature_view(np.zeros((2, FEATURE_LENGTH - 1)), "rgb8")


if __name__ == "__main__":
    unittest.main()
