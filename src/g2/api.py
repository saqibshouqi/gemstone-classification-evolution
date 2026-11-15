"""Stable adapter contract for the 2022 method; implementation is pending."""


def load(path):
    """Load a trained G2 model from ``path``."""
    raise NotImplementedError("G2 model loading will be implemented by the G2 owner")


def predict(image):
    """Return up to five (class name, probability) pairs for one image."""
    raise NotImplementedError("G2 prediction will be implemented by the G2 owner")
