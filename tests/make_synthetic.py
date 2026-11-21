"""Fake 'repo-style' gem data to smoke-test the pipeline (NOT real gemstones).

Creates <root>/{train,test}_selected/<Class>/*.jpg and {train,test}_masked/<Class>/*.png
mimicking the hybchow/gems folder names. Classes differ by hue, with overlapping
neighbours, and by a class-specific radial colour gradient, so ring statistics matter.

    python -m tests.make_synthetic data_synth
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def one_image(rng, ci, n_classes):
    h = int(rng.integers(160, 420))
    w = int(rng.integers(160, 420))
    bg = np.clip(rng.normal(rng.integers(205, 250), 3, (h, w, 3)), 0, 255)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2 + rng.normal(0, h * 0.03), w / 2 + rng.normal(0, w * 0.03)
    a, b = rng.uniform(0.25, 0.40) * w, rng.uniform(0.25, 0.40) * h
    th = rng.uniform(0, np.pi)
    xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    mask = r < 1.0
    base_h = (ci * 170.0 / max(n_classes - 1, 1) + rng.normal(0, 7)) % 180
    s0 = 110 + 12 * (ci % 4) + rng.normal(0, 10)
    v0 = 170 + 8 * (ci % 3) + rng.normal(0, 12)
    sign = 1 if ci % 2 == 0 else -1
    hsv = np.zeros((h, w, 3), np.float32)
    hsv[..., 0] = base_h + 6 * sign * r + rng.normal(0, 2, (h, w))
    hsv[..., 1] = s0 + sign * 50 * r + rng.normal(0, 6, (h, w))
    hsv[..., 2] = v0 - (30 + 5 * (ci % 3)) * r + rng.normal(0, 8, (h, w))
    hsv[..., 0] %= 180
    hsv = np.clip(hsv, 1, 255).astype(np.uint8)
    stone = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    img = bg.copy()
    img[mask] = stone[mask]
    masked = np.zeros((h, w, 3), np.uint8)
    masked[mask] = np.maximum(stone[mask], 1).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8), masked


def make(root, n_classes=10, n_train=30, n_test=5, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for ci in range(n_classes):
        cls = f"Gem{ci:02d}"
        for split, n in (("train", n_train), ("test", n_test)):
            (root / f"{split}_selected" / cls).mkdir(parents=True, exist_ok=True)
            (root / f"{split}_masked" / cls).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img, masked = one_image(rng, ci, n_classes)
                cv2.imwrite(str(root / f"{split}_selected" / cls / f"{cls}_{i:03d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(root / f"{split}_masked" / cls / f"{cls}_{i:03d}.png"), masked)
    return root


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data_synth")
    print("done")
