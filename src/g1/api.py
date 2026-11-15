"""Stable adapter contract for the 2015 method; implementation is pending."""


def load(path):
    """Load a trained G1 model from ``path``."""
    raise NotImplementedError("G1 model loading will be implemented by the G1 owner")


def predict(image):
    """Return up to five (class name, probability) pairs for one image."""
    raise NotImplementedError("G1 prediction will be implemented by the G1 owner")
