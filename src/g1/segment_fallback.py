"""Label-free stone mask used by G1's adapter and by the 'labelfree-g1seg' mode.

THIS IS A STAND-IN. Package B owns the real label-free selector (masks/labelfree/).
It follows the pipeline described in Chow and Reyes-Aldasoro (2022) sec. 2.2.1 as
summarised in the team design doc; the exact parameters still have to be checked
against the authors' BackgroundSegmentationExecution.ipynb.

Pipeline per channel (grey: sigma 2, disc radius 8; saturation: sigma 5, radius 9):
  Gaussian blur -> Otsu -> flip if the four 20x20 corner patches are brighter than the
  image mean -> fill holes -> closing with a disc -> drop objects under 301 px ->
  fill holes -> erode with a 2x2 square -> fill holes.
Selector (no label): the candidate whose four corners are most clearly background
wins; near-ties go to saturation (better for 52 of 68 classes in the paper).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def _corner_slices(h, w, p=20):
    p = max(2, min(p, h // 4, w // 4))
    return [(slice(0, p), slice(0, p)), (slice(0, p), slice(w - p, w)),
            (slice(h - p, h), slice(0, p)), (slice(h - p, h), slice(w - p, w))]


def _remove_small(m, min_size=301):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= min_size) + 1
    return np.isin(lab, keep)


def candidate_mask(img_bgr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "grey":
        ch, sigma, radius = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 2, 8
    elif kind == "sat":
        ch, sigma, radius = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1], 5, 9
    else:
        raise ValueError(kind)
    blur = cv2.GaussianBlur(ch, (0, 0), sigma)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = blur > t
    corners = np.concatenate([blur[s].ravel() for s in _corner_slices(*blur.shape)])
    if corners.mean() > blur.mean():
        m = ~m
    m = ndi.binary_fill_holes(m)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    m = ndi.binary_fill_holes(_remove_small(m))
    m = ndi.binary_fill_holes(ndi.binary_erosion(m, structure=np.ones((2, 2))))
    return m


def corner_background_score(mask: np.ndarray) -> float:
    vals = np.concatenate([mask[s].ravel() for s in _corner_slices(*mask.shape)])
    return float(1.0 - vals.mean())


def label_free_mask(img_bgr: np.ndarray, return_kind: bool = False):
    cands = {k: candidate_mask(img_bgr, k) for k in ("sat", "grey")}
    valid = {k: m for k, m in cands.items() if 0.02 < m.mean() < 0.95}
    if not valid:
        out, kind = np.ones(img_bgr.shape[:2], bool), "full"
    else:
        kind = max(valid, key=lambda k: (round(corner_background_score(valid[k]), 2), k == "sat"))
        out = valid[kind]
    return (out, kind) if return_kind else out
