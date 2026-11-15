"""Stable adapter contract for the 2025 method; implementation is pending."""


def load(path):
    """Load a trained G3 model from ``path``."""
    raise NotImplementedError("G3 model loading will be implemented by the G3 owner")


def predict(image):
    """Return up to five (class name, probability) pairs for one image."""
    raise NotImplementedError("G3 prediction will be implemented by the G3 owner")
