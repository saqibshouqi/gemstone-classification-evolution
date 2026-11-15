"""Stable G2 inference adapter shared with the project scoreboard."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from g2.features import (
    FEATURE_LENGTH,
    FEATURE_NAME,
    FEATURE_VIEWS,
    extract_features,
    feature_view_length,
    select_feature_view,
)
from g2.models import aligned_probabilities
from g2.segmentation import make_mask


class G2Predictor:
    """Loaded G2 model with label-free, fixed-branch image preprocessing."""

    def __init__(self, artifact: dict[str, object]):
        if artifact.get("format") != "gemstone-trends-g2" or artifact.get("format_version") != 1:
            raise ValueError("Unsupported G2 model artifact format.")
        self.feature_name = str(artifact.get("feature_name", ""))
        if self.feature_name not in FEATURE_VIEWS or artifact.get("feature_length") != feature_view_length(self.feature_name):
            raise ValueError("The model's feature schema does not match this G2 implementation.")
        self.mode = str(artifact["mode"])
        if self.mode == "published":
            raise ValueError(
                "A class-conditioned published mask model cannot infer without the true class label."
            )
        if self.mode not in {"labelfree", "saturation", "grey"}:
            raise ValueError(f"Unsupported inference mask mode: {self.mode}")
        self.classes = tuple(str(name) for name in artifact["classes"])
        if len(self.classes) != 68 or len(set(self.classes)) != 68:
            raise ValueError("G2 model must contain the fixed 68-class order.")
        self.pipeline = artifact["pipeline"]

    @classmethod
    def load(cls, path: str | Path) -> "G2Predictor":
        artifact = joblib.load(Path(path))
        if not isinstance(artifact, dict):
            raise ValueError("G2 model artifact must be a dictionary.")
        return cls(artifact)

    @staticmethod
    def _image_array(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                return np.asarray(source.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Inference input must be an RGB image, grayscale image, or image path.")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("Inference image contains non-finite pixel values.")
            if array.size and float(array.max()) <= 1.0:
                array = array * 255.0
        return np.clip(np.rint(array), 0, 255).astype(np.uint8)

    def predict(
        self, image: str | Path | Image.Image | np.ndarray
    ) -> list[tuple[str, float]]:
        rgb = self._image_array(image)
        mask, _variant = make_mask(rgb, self.mode)
        full_features = extract_features(rgb, mask).reshape(1, -1)
        features = select_feature_view(full_features, self.feature_name)
        probabilities = aligned_probabilities(
            self.pipeline, features, class_count=len(self.classes)
        )[0]
        order = np.argsort(-probabilities, kind="stable")[:5]
        return [(self.classes[int(index)], float(probabilities[index])) for index in order]


_DEFAULT_PREDICTOR: G2Predictor | None = None


def load(path: str | Path) -> G2Predictor:
    """Load and register a model; the returned predictor can also be used directly."""
    global _DEFAULT_PREDICTOR
    _DEFAULT_PREDICTOR = G2Predictor.load(path)
    return _DEFAULT_PREDICTOR


def predict(
    image: str | Path | Image.Image | np.ndarray
) -> list[tuple[str, float]]:
    """Return up to five class/probability pairs using the loaded model."""
    if _DEFAULT_PREDICTOR is None:
        raise RuntimeError("Call g2.api.load(model_path) before g2.api.predict(image).")
    return _DEFAULT_PREDICTOR.predict(image)
