"""Shared DINOv2 runtime for all API-backed visual verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_EMBEDDING_MODEL = "facebook/dinov2-small"
VISION_DIR = Path(__file__).resolve().parent
MODELS_DIR = VISION_DIR / "models"
DEFAULT_EMBEDDING_DIR = MODELS_DIR / "dinov2-small"
REMOTE_MODELS_DIR = MODELS_DIR / "huggingface"
_DINO_RUNTIME: tuple[Any, Any, str] | None = None


def resolve_device(value: str = "auto") -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def square_product(image: Image.Image) -> Image.Image:
    """Pad instead of crop so a tall or wide package remains fully visible."""
    image = image.convert("RGB")
    side = max(image.width, image.height)
    padded = Image.new("RGB", (side, side), "white")
    padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return padded


def resolve_embedding_model(model_reference: str = DEFAULT_EMBEDDING_MODEL) -> str:
    from huggingface_hub import snapshot_download

    local_path = Path(model_reference).expanduser()
    if local_path.is_dir():
        return str(local_path)
    if model_reference == DEFAULT_EMBEDDING_MODEL:
        if (DEFAULT_EMBEDDING_DIR / "config.json").is_file() and (
            (DEFAULT_EMBEDDING_DIR / "model.safetensors").is_file()
            or (DEFAULT_EMBEDDING_DIR / "pytorch_model.bin").is_file()
        ):
            return str(DEFAULT_EMBEDDING_DIR)
        return snapshot_download(repo_id=model_reference, local_dir=DEFAULT_EMBEDDING_DIR)
    return model_reference


def embed_images(processor: Any, model: Any, images: list[Image.Image], device: str,
                 batch_size: int = 32) -> Any:
    import torch
    import torch.nn.functional as functional

    vectors = []
    for start in range(0, len(images), batch_size):
        batch = [square_product(image) for image in images[start:start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            vectors.append(functional.normalize(outputs.last_hidden_state[:, 0, :], dim=1).cpu())
    return torch.cat(vectors, dim=0)


def load_dino_runtime() -> tuple[Any, Any, str]:
    from transformers import AutoImageProcessor, AutoModel

    device = resolve_device()
    source = resolve_embedding_model()
    return (
        AutoImageProcessor.from_pretrained(source, cache_dir=REMOTE_MODELS_DIR),
        AutoModel.from_pretrained(source, cache_dir=REMOTE_MODELS_DIR).to(device).eval(),
        device,
    )


def get_dino_runtime() -> tuple[Any, Any, str]:
    global _DINO_RUNTIME
    if _DINO_RUNTIME is None:
        _DINO_RUNTIME = load_dino_runtime()
    return _DINO_RUNTIME


def preload_dino() -> tuple[Any, Any, str]:
    """Load the single API-process DINO runtime before accepting requests."""
    return get_dino_runtime()


def reference_similarity_scores(reference: Image.Image, candidates: list[Image.Image],
                                batch_size: int = 32) -> list[float]:
    if not candidates:
        return []
    processor, model, device = get_dino_runtime()
    vectors = embed_images(processor, model, [reference, *candidates], device, batch_size)
    return [float(score) for score in (vectors[1:] @ vectors[0]).tolist()]
