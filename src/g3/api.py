"""Stable model adapter for G3 (2025) gemstone classifiers (Contract 4)."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.g3.models import build_g3_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CLASSES_TXT = _REPO_ROOT / "splits" / "classes.txt"

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_STATE = {
    "model": None,
    "model_name": None,
    "class_names": None,
    "device": None,
}


def _infer_model_name(path: Union[str, Path]) -> str:
    """Infer the G3 architecture name from the checkpoint filename."""
    stem = Path(path).stem.lower()
    for name in ("convnext_small_cbam", "convnext_small", "densenet169", "mobilenetv2"):
        if name in stem:
            return name
    raise ValueError(
        f"Could not infer G3 model architecture from filename '{Path(path).name}'. "
        "Pass model_name explicitly (e.g. 'densenet169', 'convnext_small', "
        "'convnext_small_cbam', or 'mobilenetv2')."
    )


def _load_class_names(classes_txt: Optional[Union[str, Path]] = None) -> List[str]:
    classes_path = Path(classes_txt) if classes_txt is not None else _DEFAULT_CLASSES_TXT
    if not classes_path.exists():
        raise FileNotFoundError(f"Could not find classes file at '{classes_path}'.")
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load(
    path: Union[str, Path],
    model_name: Optional[str] = None,
    classes_txt: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
):
    """Load a trained G3 model checkpoint from ``path``."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'.")

    resolved_name = model_name or _infer_model_name(ckpt_path)
    class_names = _load_class_names(classes_txt)
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, _ = build_g3_model(
        resolved_name,
        num_classes=len(class_names),
        pretrained=False,
    )
    state_dict = torch.load(ckpt_path, map_location=target_device)
    model.load_state_dict(state_dict)
    model = model.to(target_device)
    model.eval()

    _STATE["model"] = model
    _STATE["model_name"] = resolved_name
    _STATE["class_names"] = class_names
    _STATE["device"] = target_device
    return model


def _to_pil_rgb(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def predict(image: Union[str, Path, Image.Image, np.ndarray], top_k: int = 5) -> List[Tuple[str, float]]:
    """Return up to five (class_name, probability) pairs for one image."""
    if _STATE["model"] is None:
        raise RuntimeError("No G3 model is loaded. Call load(path) before predict(image).")

    pil_img = _to_pil_rgb(image)
    tensor = _EVAL_TRANSFORM(pil_img).unsqueeze(0).to(_STATE["device"])

    with torch.no_grad():
        logits = _STATE["model"](tensor)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(top_k, len(_STATE["class_names"]))
        top_probs, top_indices = torch.topk(probs, k=k)

    return [
        (_STATE["class_names"][int(idx)], float(prob))
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())
    ]