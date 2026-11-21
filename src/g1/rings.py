"""Stone extraction, scaling and concentric rings.

Follows Sinkevicius, Lipnickas and Rimkus (2015), Sec. 2.2 (preprocessing) and
Sec. 2.4 (rings of equal pixel count counted from the centre, leftover pixels
unused).

Choices that are OURS, not the paper's (they are flagged in the README too):
  * the stone is rescaled so it holds about TARGET_AREA pixels, because rings are
    counted in pixels and our photos range from 89 to 3,947 px wide;
  * rings are formed by sorting stone pixels by distance from the mask centroid
    (ties broken by angle) and cutting consecutive groups of RING_SIZE pixels.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

TARGET_AREA = 15_000   # paper: stones hold 3,000 to 30,000 px, peak near 15,000
RING_SIZE = 400        # paper: 400 px chosen
MIN_MASK_PIXELS = 50


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def read_mask(path) -> np.ndarray:
    """Mask PNG with values 0 or 255 -> boolean array (True = stone)."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return m > 127


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes and keep the largest connected component (one stone per image)."""
    m = ndi.binary_fill_holes(mask > 0)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def crop_and_scale(img: np.ndarray, mask: np.ndarray, target_area: int | None = TARGET_AREA):
    """Crop to the mask's bounding box, then resize so the stone holds ~target_area px."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img, mask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if target_area:
        area = int(mask.sum())
        s = math.sqrt(target_area / max(area, 1))
        h, w = mask.shape
        nh, nw = max(2, round(h * s)), max(2, round(w * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
        mask = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        if mask.sum() < MIN_MASK_PIXELS:   # pathological resize: undo
            raise ValueError("mask vanished after rescaling")
    return img, mask


def pixel_table(img_bgr: np.ndarray, mask: np.ndarray, circular_hue: bool = False) -> np.ndarray:
    """Stone pixels only, sorted by distance from the centroid -> (N, C) float32.

    Channels: grey, H, S, V (OpenCV scales: H 0..179, S and V 0..255). The paper kept
    HSV plus grey and found RGB and grey alone much worse. With circular_hue the H
    column becomes (sin, cos) of the hue angle (optional extra: red hues wrap around).
    """
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    g = grey[ys, xs].astype(np.float32)
    h = hsv[ys, xs, 0].astype(np.float32)
    s = hsv[ys, xs, 1].astype(np.float32)
    v = hsv[ys, xs, 2].astype(np.float32)
    if circular_hue:
        ang = 2.0 * np.pi * h / 180.0
        cols = [g, np.sin(ang), np.cos(ang), s, v]
    else:
        cols = [g, h, s, v]
    table = np.stack(cols, axis=1)
    cy, cx = ys.mean(), xs.mean()
    d = (ys - cy) ** 2 + (xs - cx) ** 2
    theta = np.arctan2(ys - cy, xs - cx)
    order = np.lexsort((theta, d))          # primary key: distance, tie-break: angle
    return table[order]


def make_rings(table: np.ndarray, ring_size: int | None) -> np.ndarray:
    """Cut the sorted pixel table into consecutive rings of `ring_size` pixels.

    Leftover pixels (last incomplete group) are dropped, as in the paper. ring_size
    None means "all stone pixels" (one group) and is also used when the stone is
    smaller than one ring. Returns (R, ring_size, C).
    """
    n = len(table)
    if ring_size in (None, "all") or n < ring_size:
        return table[None]
    r = n // ring_size
    return table[: r * ring_size].reshape(r, ring_size, -1)
