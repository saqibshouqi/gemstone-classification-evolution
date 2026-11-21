"""Model adapter (contract 4): load(path) and predict(image) -> top 5 (class, probability).

    from src.g1 import api
    api.load("models/g1-hh16-rings400_published.joblib")
    api.predict("some_gem.jpg")        # or a PIL image / RGB numpy array

The saved model was trained on masks of the chosen mode, but a real user has no mask,
so predict() builds one without the label (segment_fallback.label_free_mask, a
stand-in for package B's selector). Accuracy in the demo is therefore the label-free
accuracy, not the as-published one.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np

from .committee import RingData
from .features import ring_features
from .rings import clean_mask, crop_and_scale, pixel_table
from .segment_fallback import label_free_mask

_BUNDLE = None


def load(path):
    global _BUNDLE
    _BUNDLE = joblib.load(path)
    return _BUNDLE["name"]


def _to_bgr(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image)
        return img
    arr = np.asarray(image)                       # PIL image or RGB array (Gradio gives RGB)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)


def predict(image, top_k: int = 5):
    if _BUNDLE is None:
        raise RuntimeError("call load(path) first")
    b = _BUNDLE
    img = _to_bgr(image)
    mask = clean_mask(label_free_mask(img))
    img2, mask2 = crop_and_scale(img, mask, b["target_area"])
    feats = ring_features(pixel_table(img2, mask2, b["circular_hue"]), b["ring_size"])
    k = len(b["classes"])
    data = RingData(feats.astype(np.float32), np.zeros(len(feats), dtype=int), np.array([-1]), k)
    p = b["model"].predict_proba(data)[0]
    order = np.argsort(-p, kind="stable")[:top_k]
    return [(b["classes"][i], float(p[i])) for i in order]
