from __future__ import annotations

import unittest

import numpy as np

from g2.segmentation import (
    GREY_CLASSES,
    make_mask,
    select_labelfree_variant,
)


def synthetic_stone() -> np.ndarray:
    image = np.full((96, 96, 3), 240, dtype=np.uint8)
    yy, xx = np.ogrid[:96, :96]
    stone = (xx - 48) ** 2 + (yy - 48) ** 2 <= 20 ** 2
    image[stone] = (115, 45, 180)
    return image


class SegmentationTests(unittest.TestCase):
    def test_both_otsu_branches_return_binary_image_sized_masks(self) -> None:
        image = synthetic_stone()
        for mode in ("grey", "saturation", "labelfree"):
            with self.subTest(mode=mode):
                mask, variant = make_mask(image, mode)
                self.assertEqual(mask.shape, image.shape[:2])
                self.assertEqual(mask.dtype, np.bool_)
                self.assertIn(variant, ("grey", "saturation"))
                self.assertGreater(int(mask.sum()), 301)

    def test_label_free_tie_falls_back_to_saturation(self) -> None:
        empty = np.zeros((40, 40), dtype=bool)
        self.assertEqual(select_labelfree_variant(empty, empty), "saturation")

    def test_published_choice_uses_the_documented_class_map(self) -> None:
        image = synthetic_stone()
        _mask, grey_variant = make_mask(image, "published", class_name="Diamond")
        _mask, saturation_variant = make_mask(
            image, "published", class_name="Amazonite"
        )
        self.assertEqual(grey_variant, "grey")
        self.assertEqual(saturation_variant, "saturation")
        self.assertEqual(len(GREY_CLASSES), 16)

    def test_published_mode_requires_a_class_label(self) -> None:
        with self.assertRaises(ValueError):
            make_mask(synthetic_stone(), "published")


if __name__ == "__main__":
    unittest.main()
