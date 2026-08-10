"""Local OWLv2 text retrieval with optional DINO reference verification."""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from vision.ark_grounding import BoundingBox, draw_boxes
from vision.dino import MODELS_DIR, resolve_device
from vision.vlm_sku_query import dino_box_scores, select_dino_boxes


MODEL_ID = "google/owlv2-large-patch14-ensemble"
DEFAULT_MODEL_DIR = MODELS_DIR / "owlv2-large-patch14-ensemble"
_OWLV2_RUNTIME: tuple[Any, Any, str] | None = None


def load_owlv2_runtime() -> tuple[Any, Any, str]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from huggingface_hub import snapshot_download

    device = resolve_device()
    source = str(DEFAULT_MODEL_DIR) if (DEFAULT_MODEL_DIR / "model.safetensors").is_file() else snapshot_download(
        repo_id=MODEL_ID, local_dir=DEFAULT_MODEL_DIR,
        allow_patterns=["*.json", "merges.txt", "vocab.json", "model.safetensors"],
    )
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(source).to(device).eval()
    return processor, model, device


def get_owlv2_runtime() -> tuple[Any, Any, str]:
    global _OWLV2_RUNTIME
    if _OWLV2_RUNTIME is None:
        _OWLV2_RUNTIME = load_owlv2_runtime()
    return _OWLV2_RUNTIME


def preload_owlv2() -> tuple[Any, Any, str]:
    return get_owlv2_runtime()


def _box(index: int, sku: str, coordinates: list[float], width: int, height: int) -> BoundingBox:
    x1, y1, x2, y2 = coordinates
    pixels = (
        max(0, min(width, round(x1))), max(0, min(height, round(y1))),
        max(0, min(width, round(x2))), max(0, min(height, round(y2))),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        raise ValueError("OWLv2 returned an invalid bounding box")
    normalized = tuple(round(value * 999 / size) for value, size in zip(pixels, (width, height, width, height)))
    return BoundingBox(index=index, label=sku, normalized=normalized, pixels=pixels)


def owlv2_candidates(shelf_source: Path | bytes, sku: str, prompt: str,
                     score_threshold: float, max_boxes: int | None) -> tuple[list[BoundingBox], list[float]]:
    if not 0 <= score_threshold <= 1:
        raise ValueError("owlv2_score_threshold must be between 0 and 1")
    image_input = BytesIO(shelf_source) if isinstance(shelf_source, bytes) else shelf_source
    with Image.open(image_input) as source:
        image = source.convert("RGB")
    processor, model, device = get_owlv2_runtime()
    import torch

    inputs = processor(
        text=[[prompt]], images=image, padding="max_length", truncation=True, max_length=16, return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=[(image.height, image.width)], threshold=score_threshold,
    )[0]
    raw = sorted(
        zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist()), key=lambda item: item[1], reverse=True
    )
    boxes: list[BoundingBox] = []
    scores: list[float] = []
    for coordinates, score in raw:
        try:
            boxes.append(_box(len(boxes) + 1, sku, coordinates, image.width, image.height))
            scores.append(float(score))
        except ValueError:
            continue
    if not boxes:
        return [], []
    import cv2

    keep = cv2.dnn.NMSBoxes(
        [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in (box.pixels for box in boxes)],
        scores, 0.0, 0.45,
    )
    indexes = [int(index) for index in keep] if len(keep) else []
    selected_indexes = indexes if max_boxes is None else indexes[:max_boxes]
    boxes = [replace(boxes[index], index=position + 1) for position, index in enumerate(selected_indexes)]
    return boxes, [scores[index] for index in selected_indexes]


def run_owlv2_sku_query(
    sku: str, prompt: str, target_path: Path, shelf_source: Path | bytes, output_dir: Path | None = None,
    max_boxes: int = 1, owlv2_score_threshold: float = 0.10, dino_fallback: bool = False,
    dino_confidence_threshold: float = 0.72, debug: bool = True,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("This SKU has no owlv2_prompt")
    if not 1 <= max_boxes <= 20 or not 0 <= dino_confidence_threshold <= 1:
        raise ValueError("Invalid OWLv2 query limits")
    if debug and output_dir is None:
        raise ValueError("output_dir is required when debug is enabled")
    target_path = target_path.expanduser().resolve()
    if not target_path.is_file():
        raise ValueError(f"Target image not found: {target_path}")
    if debug:
        output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # DINO needs every OWLv2 candidate above its threshold before ranking the final top N.
    candidates, owlv2_scores = owlv2_candidates(
        shelf_source, sku, prompt, owlv2_score_threshold, None if dino_fallback else max_boxes
    )
    dino_scores: list[float] = []
    boxes = candidates
    if dino_fallback:
        dino_scores = dino_box_scores(target_path, shelf_source, candidates)
        boxes = select_dino_boxes(candidates, dino_scores, confidence_threshold=dino_confidence_threshold, max_results=max_boxes)
    if not debug:
        return {
            "sku": sku, "provider": "local", "model": MODEL_ID,
            "request_seconds": time.perf_counter() - started, "total_seconds": time.perf_counter() - started,
            "detected_boxes": [asdict(box) for box in boxes],
            "owlv2_scores": [{"index": box.index, "confidence": round(score, 4)} for box, score in zip(candidates, owlv2_scores)],
            "dino_scores": [{"index": box.index, "confidence": round(score, 4)} for box, score in zip(candidates, dino_scores)],
        }

    target_copy = output_dir / f"00_target_reference{target_path.suffix.lower()}"
    shelf_copy = output_dir / f"01_shelf_source{'.png' if isinstance(shelf_source, bytes) else shelf_source.suffix.lower()}"
    shutil.copy2(target_path, target_copy)
    if isinstance(shelf_source, bytes):
        shelf_copy.write_bytes(shelf_source)
    else:
        shutil.copy2(shelf_source, shelf_copy)
    candidates_image = output_dir / "02_owlv2_candidates.png"
    draw_boxes(shelf_copy, candidates, candidates_image)
    annotated = output_dir / "03_annotated_matches.png"
    draw_boxes(shelf_copy, boxes, annotated)
    return {
        "run_id": output_dir.name, "sku": sku, "provider": "local", "model": MODEL_ID,
        "request_seconds": time.perf_counter() - started, "total_seconds": time.perf_counter() - started,
        "owlv2_detected_boxes": [asdict(box) for box in candidates],
        "owlv2_scores": [{"index": box.index, "confidence": round(score, 4)} for box, score in zip(candidates, owlv2_scores)],
        "detected_boxes": [asdict(box) for box in boxes],
        "dino_scores": [{"index": box.index, "confidence": round(score, 4)} for box, score in zip(candidates, dino_scores)],
        "config": {"max_boxes": max_boxes, "dino_fallback": dino_fallback,
                   "dino_confidence_threshold": dino_confidence_threshold,
                   "owlv2_score_threshold": owlv2_score_threshold},
        "artifacts": {"target_reference": target_copy.name, "shelf_source": shelf_copy.name,
                      "owlv2_candidates": candidates_image.name, "annotated_matches": annotated.name},
    }
