#!/usr/bin/env python3
"""Detect missing products by comparing a current photo against calibrated shelf faces.

This script:
1. Auto-finds the matching shelf face (SIFT + USAC_MAGSAC homography).
2. Reuses vision.reference_photo_align for robust alignment and quality gates.
3. Computes Lab color-distance masks with global lighting compensation.
4. Uses per-slot changed-pixel ratios to identify candidate fixed slots.
"""

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import calibration_manager
from vision.config import get_inspection_config
from vision.reference_photo_align import (
    AlignmentError,
    MatchMetrics,
    coverage,
    create_feature_extractor,
    keep_strongest,
    largest_valid_rectangle,
    mutual_ratio_matches,
    prepared_gray,
    ratio_matches,
    read_bgr,
    resize_for_matching,
    reprojection_error,
)


def is_low_confidence(
    top_sku: str | None,
    score: float,
    confidence_threshold: float,
) -> bool:
    return not top_sku or score < confidence_threshold


def is_ambiguous(score: float, second_score: float | None, ambiguity_margin: float) -> bool:
    return second_score is not None and score - second_score <= ambiguity_margin


def classify_candidate(
    expected_sku: str,
    top_sku: str | None,
    score: float,
    second_score: float | None,
    confidence_threshold: float,
    ambiguity_margin: float,
    vlm_fallback: bool,
    vlm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Turn DINO and optional Ark evidence into one slot observation."""
    low_confidence = is_low_confidence(top_sku, score, confidence_threshold)
    needs_ark = vlm_fallback and (
        low_confidence or is_ambiguous(score, second_score, ambiguity_margin)
    )
    actual_sku = top_sku if not low_confidence and not needs_ark else None
    source = "dino" if actual_sku else "shortage"
    reason = (
        "dino_expected_match" if actual_sku == expected_sku else "dino_match"
    ) if actual_sku else "low_confidence"

    if needs_ark:
        if vlm_result and vlm_result.get("kind") == "sku" and isinstance(vlm_result.get("sku"), str):
            actual_sku = vlm_result["sku"].strip() or None
            source = "ark" if actual_sku else "shortage"
            reason = "ark_match" if actual_sku else "ark_unresolved"
        else:
            reason = "ark_unresolved"

    status = "缺货" if actual_sku is None else "正常" if actual_sku == expected_sku else "摆放错误"
    return {
        "actual_sku": actual_sku,
        "status": status,
        "source": source,
        "confidence": round(float(score), 4),
        "reason": reason,
    }


def slot_status(expected_sku: str, actual_sku: str | None) -> str:
    if actual_sku is None:
        return "缺货"
    return "正常" if actual_sku == expected_sku else "摆放错误"


def slot_bbox_in_current(
    slot: dict[str, Any],
    homography: np.ndarray,
    current_width: int,
    current_height: int,
    crop_offset: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    bbox = slot.get("bbox", {})
    try:
        x, y = int(bbox["x"]), int(bbox["y"])
        width, height = int(bbox["width"]), int(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    corners = np.float32(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    left = max(0, int(transformed[:, 0].min()) - crop_offset[0])
    top = max(0, int(transformed[:, 1].min()) - crop_offset[1])
    right = min(current_width, int(transformed[:, 0].max()) - crop_offset[0])
    bottom = min(current_height, int(transformed[:, 1].max()) - crop_offset[1])
    return (left, top, right - left, bottom - top) if right > left and bottom > top else None


def reference_images_by_sku(slots: list[dict[str, Any]]) -> dict[str, list[Path]]:
    references: dict[str, list[Path]] = {}
    for slot in slots:
        slot_id = str(slot.get("slot_id", "")).strip()
        sku = str(slot.get("expected_sku", "")).strip()
        image = Path(PROJECT_ROOT) / "data" / "item_images" / slot_id / "0.png"
        if slot_id and sku and image.is_file():
            references.setdefault(sku, []).append(image)
    return references


def rank_slot_crop(crop: np.ndarray, references: dict[str, list[Path]],
                   config: dict[str, Any]) -> list[tuple[str, float]]:
    """Rank known SKU reference crops for one changed fixed slot."""
    if not references:
        return []
    from PIL import Image
    from vision.dino import reference_similarity_scores

    labels = [sku for sku, paths in references.items() for _ in paths]
    images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))]
    images.extend(Image.open(path).convert("RGB") for paths in references.values() for path in paths)
    scores = reference_similarity_scores(images[0], images[1:], int(config.get("dino_batch_size", 32)))
    best_by_sku: dict[str, float] = {}
    for sku, score in zip(labels, scores):
        best_by_sku[sku] = max(best_by_sku.get(sku, -1.0), score)
    return sorted(best_by_sku.items(), key=lambda item: item[1], reverse=True)


def image_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode inspection image")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def ark_slot_decision(crop: np.ndarray, ranked: list[tuple[str, float]],
                      references: dict[str, list[Path]], config: dict[str, Any]) -> dict[str, Any]:
    """Ask Ark to choose from the DINO-ranked SKU labels for one slot crop."""
    from PIL import Image, ImageDraw, ImageFont
    from vision.vlm_sku_query import DEFAULT_MODEL, request_ark

    choices = ranked[:int(config.get("vlm_top_k", 4))]
    if not choices:
        return {"kind": "unresolved"}
    card_width, card_height, padding = 180, 220, 12
    montage = Image.new("RGB", (padding + len(choices) * (card_width + padding), card_height + 2 * padding), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for index, (sku, score) in enumerate(choices, start=1):
        reference = Image.open(references[sku][0]).convert("RGB")
        reference.thumbnail((card_width - 12, card_height - 42))
        left = padding + (index - 1) * (card_width + padding)
        montage.paste(reference, (left + (card_width - reference.width) // 2, padding + 28))
        draw.rectangle((left, padding, left + card_width, padding + card_height), outline="black", width=2)
        draw.text((left + 6, padding + 6), f"{index}. {sku} {score:.2f}", fill="black", font=font)

    prompt = (
        "图1是当前货架固定位置的商品裁剪图，图2是候选SKU参考图。"
        "只从图2候选中选择与图1包装完全一致的一项，或判断空位/不是商品。"
        "只输出JSON：{\"kind\":\"sku\",\"sku\":\"候选SKU\"}、"
        "{\"kind\":\"empty\"} 或 {\"kind\":\"not_product\"}。"
    )
    try:
        response = request_ark(
            [
                {"type": "text", "text": "图1：当前固定位置裁剪图。"},
                {"type": "image_url", "image_url": {"url": image_data_url(crop)}},
                {"type": "text", "text": "图2：编号SKU候选参考图。"},
                {"type": "image_url", "image_url": {"url": image_data_url(cv2.cvtColor(np.array(montage), cv2.COLOR_RGB2BGR))}},
                {"type": "text", "text": prompt},
            ],
            str(config.get("ark_model", DEFAULT_MODEL)),
            int(config.get("ark_max_tokens", 128)),
        )
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        return {"kind": "unresolved"}
    sku = parsed.get("sku")
    if parsed.get("kind") == "sku" and isinstance(sku, str) and sku in dict(choices):
        return {"kind": "sku", "sku": sku}
    return {"kind": parsed.get("kind", "unresolved")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect missing products by comparing a current photo against calibrated shelf faces."
    )
    parser.add_argument("--current", required=True, type=Path, help="Path to the current photo")
    parser.add_argument("--lab-distance-threshold", type=float, default=12.0,
                        help="Lab color-distance threshold for an anomalous pixel")
    parser.add_argument("--slot-change-ratio-threshold", type=float, default=0.15,
                        help="Minimum anomalous-pixel ratio for a candidate slot")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Min IoU for missing (default 0.3)")
    parser.add_argument("--output-dir", type=Path, default=Path(VISION_DIR) / "output" / "slot_inspection")
    parser.add_argument("--max-image-side", type=int, default=1800, help="Max image side for feature matching")
    parser.add_argument("--feature", choices=("sift", "orb"), default="sift")
    parser.add_argument("--max-features", type=int, default=7000)
    parser.add_argument("--ratio-test", type=float, default=0.72)
    parser.add_argument("--reprojection-threshold", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=18)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.24)
    parser.add_argument("--min-current-coverage", type=float, default=0.05)
    parser.add_argument("--dino-confidence-threshold", type=float)
    parser.add_argument("--ambiguity-margin", type=float)
    parser.add_argument("--vlm-fallback", action="store_true")
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument("--debug", dest="debug", action="store_true")
    debug_group.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=True)
    return parser.parse_args()


def try_match_pair(
    baseline_path: Path,
    current_image: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """
    Try to match a single baseline image against the current image.
    Returns a match dict or None if quality gates fail.
    Reuses vision.reference_photo_align's feature matching + USAC_MAGSAC pipeline.
    """
    baseline_image = read_bgr(baseline_path)

    baseline_match, baseline_scale = resize_for_matching(baseline_image, args.max_image_side)
    current_match, current_scale = resize_for_matching(current_image, args.max_image_side)

    extractor, descriptor_norm = create_feature_extractor(args.feature, args.max_features)
    base_keypoints, base_descriptors = extractor.detectAndCompute(
        prepared_gray(baseline_match, True), None
    )
    current_keypoints, current_descriptors = extractor.detectAndCompute(
        prepared_gray(current_match, True), None
    )
    base_keypoints, base_descriptors = keep_strongest(base_keypoints, base_descriptors, args.max_features)
    current_keypoints, current_descriptors = keep_strongest(current_keypoints, current_descriptors, args.max_features)

    tentative = ratio_matches(base_descriptors, current_descriptors, args.ratio_test, descriptor_norm)
    mutual = mutual_ratio_matches(base_descriptors, current_descriptors, args.ratio_test, descriptor_norm)

    if len(mutual) < 4:
        return None

    source_points = np.float32(
        [base_keypoints[match.queryIdx].pt for match in mutual]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in mutual]
    ).reshape(-1, 1, 2)

    matrix_match, inlier_mask = cv2.findHomography(
        source_points, destination_points,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=args.reprojection_threshold,
        maxIters=10000, confidence=0.999,
    )
    if matrix_match is None or inlier_mask is None:
        return None

    inliers_bool = inlier_mask.reshape(-1).astype(bool)
    n_inliers = int(inliers_bool.sum())
    inlier_ratio = float(inliers_bool.mean())
    current_cov = coverage(
        destination_points[inliers_bool].reshape(-1, 2),
        current_match.shape[:2],
    )

    if n_inliers < args.min_inliers:
        return None
    if inlier_ratio < args.min_inlier_ratio:
        return None
    if current_cov < args.min_current_coverage:
        return None

    matrix_full = np.linalg.inv(current_scale) @ matrix_match @ baseline_scale

    return {
        "baseline_image": baseline_image,
        "baseline_path": str(baseline_path),
        "homography": matrix_full,
        "inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "coverage": current_cov,
    }


def find_best_match(
    current_image: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Find the best matching calibrated shelf face."""
    calibrated_shelves = calibration_manager.get_calibrated_shelves()
    if not calibrated_shelves:
        print("No calibrated shelves found in data/shelf_calibration/")
        return None

    best_match = None

    for shelf_id in calibrated_shelves:
        cal_data = calibration_manager.get_calibration(shelf_id)
        if not cal_data or "faces" not in cal_data:
            continue

        for face_str, face_data in cal_data["faces"].items():
            face = int(face_str)
            image_file = face_data.get("image_file", "")
            if not os.path.isabs(image_file):
                baseline_path = Path(PROJECT_ROOT) / image_file
            else:
                baseline_path = Path(image_file)

            if not baseline_path.is_file():
                continue

            expected_hash = face_data.get("image_hash", "")
            current_hash = calibration_manager._compute_file_hash(str(baseline_path))
            if expected_hash and expected_hash != current_hash:
                continue

            try:
                result = try_match_pair(baseline_path, current_image, args)
            except Exception:
                continue

            if result is None:
                continue

            print(
                f"  Shelf {shelf_id} face {face}: "
                f"{result['inliers']} inliers ({result['inlier_ratio']:.2f}), "
                f"coverage {result['coverage']:.3f}"
            )

            match_info = {
                **result,
                "shelf_id": shelf_id,
                "face": face,
                "calibration_data": face_data,
            }

            if best_match is None or result["inliers"] > best_match["inliers"]:
                best_match = match_info

    return best_match


def align_and_crop(
    baseline_image: np.ndarray,
    current_image: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Warp baseline to current, find largest valid rectangle, crop both.
    Reuses vision.reference_photo_align.largest_valid_rectangle.
    Returns (cropped_baseline, cropped_current, valid_mask, crop_xywh).
    """
    h, w = current_image.shape[:2]
    aligned = cv2.warpPerspective(baseline_image, H, (w, h), flags=cv2.INTER_LINEAR)
    source_mask = np.full(baseline_image.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(source_mask, H, (w, h), flags=cv2.INTER_NEAREST)

    crop_x, crop_y, crop_w, crop_h = largest_valid_rectangle(valid_mask)
    cropped_baseline = aligned[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    cropped_current = current_image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    cropped_mask = valid_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    return cropped_baseline, cropped_current, cropped_mask, (crop_x, crop_y, crop_w, crop_h)


def compute_difference_mask(
    baseline: np.ndarray,
    current: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global-lighting-compensated Lab distances and a denoised anomaly mask."""
    blurred_baseline = cv2.GaussianBlur(baseline, (11, 11), 0)
    blurred_current = cv2.GaussianBlur(current, (11, 11), 0)
    baseline_lab = cv2.cvtColor(blurred_baseline, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_lab = cv2.cvtColor(blurred_current, cv2.COLOR_BGR2LAB).astype(np.float32)
    baseline_lab[..., 0] *= 100.0 / 255.0
    current_lab[..., 0] *= 100.0 / 255.0
    baseline_lab[..., 1:] -= 128.0
    current_lab[..., 1:] -= 128.0

    delta = current_lab - baseline_lab
    delta -= np.median(delta, axis=(0, 1), keepdims=True)
    distance = np.linalg.norm(delta, axis=2)
    binary = (distance >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return distance, binary


def compute_difference_regions(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge mask contours only for CLI debug output."""

    kernel_close = np.ones((21, 21), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    min_area = 200
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append((x, y, w, h))

    merged = merge_boxes(regions)
    return merged


def slot_difference_ratio(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, width, height = box
    slot_mask = mask[y:y + height, x:x + width]
    return cv2.countNonZero(slot_mask) / float(width * height) if width and height else 0.0


def center_analysis_roi(width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    """Return a centered width-by-height analysis rectangle."""
    analysis_width = max(1, round(width * ratio))
    analysis_height = max(1, round(height * ratio))
    return (
        (width - analysis_width) // 2,
        (height - analysis_height) // 2,
        analysis_width,
        analysis_height,
    )


def box_is_inside(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
    x, y, width, height = box
    roi_x, roi_y, roi_width, roi_height = roi
    return (
        x >= roi_x
        and y >= roi_y
        and x + width <= roi_x + roi_width
        and y + height <= roi_y + roi_height
    )


def merge_boxes(boxes, overlap_thresh=0.3):
    """Merge overlapping or adjacent bounding boxes."""
    if not boxes:
        return []
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            x2_max, y2_max = x1 + w1, y1 + h1
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x3, y3, w3, h3 = boxes[j]
                x4_max, y4_max = x3 + w3, y3 + h3
                overlap_x = max(0, min(x2_max, x4_max) - max(x1, x3))
                overlap_y = max(0, min(y2_max, y4_max) - max(y1, y3))
                overlap_area = overlap_x * overlap_y
                min_a = min(w1 * h1, w3 * h3)
                if min_a > 0 and overlap_area / min_a > overlap_thresh:
                    boxes[i] = (
                        min(x1, x3), min(y1, y3),
                        max(x2_max, x4_max) - min(x1, x3),
                        max(y2_max, y4_max) - min(y1, y3),
                    )
                    used[j] = True
                    merged = True
            new_boxes.append(boxes[i])
        boxes = new_boxes
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix_min, iy_min = max(x1, x2), max(y1, y2)
    ix_max, iy_max = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = (ix_max - ix_min) * (iy_max - iy_min) if ix_min < ix_max and iy_min < iy_max else 0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list[dict[str, Any]],
    color: tuple[int, int, int] = (0, 0, 255),
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Draw bounding boxes with labels on an image."""
    result = image.copy()
    ox, oy = offset
    for item in boxes:
        bbox = item.get("bbox", {})
        x = bbox.get("x", 0) - ox
        y = bbox.get("y", 0) - oy
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        label = item.get("sku", "?")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 100)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(result, (x, y - th - 10), (x + tw + 10, y), color, -1)
        cv2.putText(result, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
    return result


def draw_analysis_roi(image: np.ndarray, roi: tuple[int, int, int, int], ratio: float) -> np.ndarray:
    """Mark the pixels that participate in difference and SKU analysis."""
    result = image.copy()
    x, y, width, height = roi
    color = (255, 255, 0)
    cv2.rectangle(result, (x, y), (x + width, y + height), color, 3)
    cv2.putText(
        result,
        f"Analysis {ratio * 100:.0f}%",
        (x + 8, min(result.shape[0] - 8, y + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return result


def run_inspection(current_source: Path | np.ndarray, config: dict[str, Any],
                   output_dir: Path | None = None, debug: bool = True) -> dict[str, Any]:
    """Inspect changed fixed slots, rendering a review run only when requested."""
    if debug:
        if output_dir is None:
            raise ValueError("output_dir is required when debug is enabled")
        output_dir.mkdir(parents=True, exist_ok=False)
    if isinstance(current_source, np.ndarray):
        current_image = current_source
    else:
        current_path = current_source.expanduser().resolve()
        if not current_path.is_file():
            raise ValueError(f"Current image not found: {current_path}")
        current_image = read_bgr(current_path)
    if current_image.size == 0:
        raise ValueError("Current image could not be decoded")
    matching_args = SimpleNamespace(
        max_image_side=int(config.get("max_image_side", 1800)),
        feature=str(config.get("feature", "sift")),
        max_features=int(config.get("max_features", 7000)),
        ratio_test=float(config.get("ratio_test", 0.72)),
        reprojection_threshold=float(config.get("reprojection_threshold", 4.0)),
        min_inliers=int(config.get("min_inliers", 18)),
        min_inlier_ratio=float(config.get("min_inlier_ratio", 0.24)),
        min_current_coverage=float(config.get("min_current_coverage", 0.05)),
    )
    match = find_best_match(current_image, matching_args)
    if match is None:
        raise RuntimeError("No matching calibrated shelf face found")

    baseline, current, _, crop = align_and_crop(
        match["baseline_image"], current_image, match["homography"]
    )
    crop_x, crop_y, _, _ = crop
    analysis_center_ratio = float(config.get("analysis_center_ratio", 0.8))
    analysis_roi = center_analysis_roi(
        current.shape[1], current.shape[0], analysis_center_ratio
    )
    distance, difference_mask = compute_difference_mask(
        baseline, current, float(config.get("lab_distance_threshold", 12.0))
    )
    roi_x, roi_y, roi_width, roi_height = analysis_roi
    analysis_mask = np.zeros_like(difference_mask)
    analysis_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = difference_mask[
        roi_y:roi_y + roi_height, roi_x:roi_x + roi_width
    ]
    slots = match["calibration_data"].get("slots", [])
    references = reference_images_by_sku(slots)
    rows: list[dict[str, Any]] = []
    candidate_boxes: list[dict[str, Any]] = []
    skipped_edge_slots = 0

    for slot in slots:
        slot_id = str(slot.get("slot_id", "")).strip()
        expected_sku = str(slot.get("expected_sku", "")).strip()
        box = slot_bbox_in_current(
            slot, match["homography"], current.shape[1], current.shape[0], (crop_x, crop_y)
        )
        if not slot_id or not expected_sku or box is None:
            continue
        if not box_is_inside(box, analysis_roi):
            skipped_edge_slots += 1
            continue
        difference_ratio = slot_difference_ratio(analysis_mask, box)
        changed = difference_ratio >= float(config.get("slot_change_ratio_threshold", 0.15))
        actual_sku = slot.get("actual_sku")
        if actual_sku is not None:
            actual_sku = str(actual_sku).strip() or None
        row = {
            "slot_id": slot_id,
            "expected_sku": expected_sku,
            "actual_sku": actual_sku,
            "status": slot_status(expected_sku, actual_sku),
            "source": "unchanged",
            "confidence": None,
            "reason": "unchanged",
            "bbox": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]},
        }
        if debug:
            row["selected"] = False
            row["difference_ratio"] = round(difference_ratio, 4)
        if changed:
            x, y, width, height = box
            crop_image = current[y:y + height, x:x + width]
            ranked = rank_slot_crop(crop_image, references, config)
            top_sku, top_score = ranked[0] if ranked else (None, 0.0)
            second_score = ranked[1][1] if len(ranked) > 1 else None
            low_confidence = is_low_confidence(
                top_sku,
                top_score,
                float(config.get("dino_confidence_threshold", 0.72)),
            )
            needs_ark = bool(config.get("vlm_fallback", False)) and (
                low_confidence
                or is_ambiguous(
                    top_score,
                    second_score,
                    float(config.get("ambiguity_margin", 0.05)),
                )
            )
            vlm_result = None
            if needs_ark:
                vlm_result = ark_slot_decision(crop_image, ranked, references, config)
            row.update(
                classify_candidate(
                    expected_sku,
                    top_sku,
                    top_score,
                    second_score,
                    float(config.get("dino_confidence_threshold", 0.72)),
                    float(config.get("ambiguity_margin", 0.05)),
                    bool(config.get("vlm_fallback", False)),
                    vlm_result,
                )
            )
            if debug:
                row["selected"] = row["status"] != "正常"
                row["dino_matches"] = [
                    {"sku": sku, "confidence": round(score, 4)}
                    for sku, score in ranked[:int(config.get("vlm_top_k", 4))]
                ]
            if row["status"] != "正常":
                if debug:
                    candidate_boxes.append({"bbox": row["bbox"], "sku": row["actual_sku"] or "缺货"})
                rows.append(row)

    if not debug:
        return {
            "shelf_id": match["shelf_id"],
            "face": match["face"],
            "slots": rows,
        }

    artifacts = {"result": "result.json", "result_overlay": "result_overlay.png"}
    cv2.imwrite(
        str(output_dir / "result_overlay.png"),
        draw_boxes_on_image(
            draw_analysis_roi(current, analysis_roi, analysis_center_ratio),
            candidate_boxes,
            (0, 165, 255),
        ),
    )
    cv2.imwrite(
        str(output_dir / "aligned_reference.png"),
        draw_analysis_roi(baseline, analysis_roi, analysis_center_ratio),
    )
    cv2.imwrite(
        str(output_dir / "current_overlap.png"),
        draw_analysis_roi(current, analysis_roi, analysis_center_ratio),
    )
    analysis_distance = np.zeros_like(distance)
    analysis_distance[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = distance[
        roi_y:roi_y + roi_height, roi_x:roi_x + roi_width
    ]
    difference = cv2.applyColorMap(
        cv2.convertScaleAbs(analysis_distance, alpha=255.0 / max(float(config.get("lab_distance_threshold", 12.0)) * 3, 1.0)),
        cv2.COLORMAP_JET,
    )
    cv2.imwrite(
        str(output_dir / "difference.png"),
        draw_analysis_roi(difference, analysis_roi, analysis_center_ratio),
    )
    cv2.imwrite(
        str(output_dir / "candidate_boxes.png"),
        draw_boxes_on_image(
            draw_analysis_roi(current, analysis_roi, analysis_center_ratio), candidate_boxes
        ),
    )
    artifacts.update({
        "aligned_reference": "aligned_reference.png",
        "current_overlap": "current_overlap.png",
        "difference": "difference.png",
        "candidate_boxes": "candidate_boxes.png",
    })

    report = {
        "run_id": output_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "shelf_id": match["shelf_id"],
        "face": match["face"],
        "slots": rows,
        "analysis": {
            "center_ratio": analysis_center_ratio,
            "roi": {"x": roi_x, "y": roi_y, "width": roi_width, "height": roi_height},
            "skipped_edge_slots": skipped_edge_slots,
        },
        "artifacts": artifacts,
        "match": {
            "inliers": match["inliers"],
            "inlier_ratio": round(match["inlier_ratio"], 4),
            "coverage": round(match["coverage"], 4),
        },
    }
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    current_path = args.current.expanduser().resolve()
    if not current_path.is_file():
        raise SystemExit(f"Error: Current image not found: {current_path}")

    created_at = datetime.now().astimezone()
    run_dir = args.output_dir.expanduser().resolve() / created_at.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    shutil.copy2(current_path, run_dir / f"00_current_source{current_path.suffix.lower()}")
    print(f"Current image: {current_path}")
    print(f"Started at: {created_at.strftime('%Y-%m-%d %H:%M:%S %z')}")
    print(f"Run directory: {run_dir}")

    # Step 1: Find matching shelf face
    print("\n[Step 1] Searching for matching shelf face...")
    current_image = read_bgr(current_path)
    best_match = find_best_match(current_image, args)

    if best_match is None:
        elapsed = time.perf_counter() - started
        report = {
            "created_at": created_at.isoformat(),
            "status": "failed",
            "error": "No matching calibrated shelf face found",
            "current_image": str(current_path),
            "total_seconds": round(elapsed, 2),
        }
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n{'='*60}")
        print("ERROR: Could not find a matching calibrated shelf face!")
        print(f"Run directory: {run_dir}")
        sys.exit(1)

    shelf_id = best_match["shelf_id"]
    face = best_match["face"]
    baseline_image = best_match["baseline_image"]
    H_full = best_match["homography"]
    calibration_data = best_match["calibration_data"]

    baseline_path = Path(best_match["baseline_path"])
    shutil.copy2(baseline_path, run_dir / f"01_baseline_full{baseline_path.suffix.lower()}")

    match_info = {
        "shelf_id": shelf_id,
        "face": face,
        "baseline_image": str(baseline_path),
        "inliers": best_match["inliers"],
        "inlier_ratio": round(best_match["inlier_ratio"], 3),
        "coverage": round(best_match["coverage"], 3),
    }
    print(f"  Shelf {shelf_id}, Face {face}: {best_match['inliers']} inliers ({best_match['inlier_ratio']:.2f}), coverage {best_match['coverage']:.3f}")

    # Step 2: Align and crop (reuses reference_photo_align.largest_valid_rectangle)
    print("\n[Step 2] Aligning and cropping images...")
    try:
        cropped_baseline, cropped_current, cropped_mask, (crop_x, crop_y, crop_w, crop_h) = align_and_crop(
            baseline_image, current_image, H_full
        )
    except Exception as e:
        raise SystemExit(f"Error during alignment/cropping: {e}")

    cv2.imwrite(str(run_dir / "02_aligned_baseline_crop.png"), cropped_baseline)
    cv2.imwrite(str(run_dir / "03_current_overlap_crop.png"), cropped_current)
    cv2.imwrite(str(run_dir / "03b_valid_overlap_mask.png"), cropped_mask)
    print(f"  Cropped size: {cropped_baseline.shape[1]}x{cropped_baseline.shape[0]}, offset: ({crop_x}, {crop_y})")

    # Step 3: Compute difference regions
    print("\n[Step 3] Computing difference regions...")
    diff_distance, diff_binary = compute_difference_mask(
        cropped_baseline, cropped_current, args.lab_distance_threshold
    )
    diff_regions = compute_difference_regions(diff_binary)
    print(f"  Found {len(diff_regions)} difference regions")

    diff_color = cv2.applyColorMap(
        cv2.convertScaleAbs(diff_distance, alpha=255.0 / max(args.lab_distance_threshold * 3, 1.0)),
        cv2.COLORMAP_JET,
    )
    cv2.imwrite(str(run_dir / "04_diff_heatmap.png"), diff_color)
    cv2.imwrite(str(run_dir / "05_diff_binary.png"), diff_binary)

    diff_overlay = cropped_current.copy()
    for i, (x, y, w, h) in enumerate(diff_regions):
        cv2.rectangle(diff_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(diff_overlay, str(i), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(run_dir / "06_diff_regions.png"), diff_overlay)

    if diff_regions:
        print("  Regions (x, y, w, h):")
        for i, (x, y, w, h) in enumerate(diff_regions):
            print(f"    Region {i}: ({x}, {y}, {w}, {h})")

    # Step 4: Find missing products
    print("\n[Step 4] Analyzing missing products...")

    h_img, w_img = current_image.shape[:2]
    adjusted_regions = [(rx + crop_x, ry + crop_y, rw, rh) for rx, ry, rw, rh in diff_regions]

    slots = calibration_data.get("slots", [])
    missing_products = []

    for slot in slots:
        bbox = slot.get("bbox", {})
        px, py = bbox.get("x", 0), bbox.get("y", 0)
        pw, ph = bbox.get("width", 0), bbox.get("height", 0)
        if pw <= 0 or ph <= 0:
            continue

        corners = np.float32(
            [[px, py], [px + pw, py], [px + pw, py + ph], [px, py + ph]]
        ).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(corners, H_full).reshape(-1, 2)
        except Exception:
            continue

        tx_min = max(0, int(transformed[:, 0].min()))
        ty_min = max(0, int(transformed[:, 1].min()))
        tx_max = min(w_img - 1, int(transformed[:, 0].max()))
        ty_max = min(h_img - 1, int(transformed[:, 1].max()))
        tw, th = tx_max - tx_min, ty_max - ty_min
        if tw <= 0 or th <= 0:
            continue

        max_iou = 0.0
        max_coverage = 0.0
        for region in adjusted_regions:
            rx, ry, rw, rh = region
            iou = compute_iou((tx_min, ty_min, tw, th), (rx, ry, rw, rh))
            max_iou = max(max_iou, iou)
            ix_min = max(tx_min, rx)
            iy_min = max(ty_min, ry)
            ix_max = min(tx_min + tw, rx + rw)
            iy_max = min(ty_min + th, ry + rh)
            if ix_min < ix_max and iy_min < iy_max:
                inter = (ix_max - ix_min) * (iy_max - iy_min)
                coverage = inter / (tw * th) if tw * th > 0 else 0
                max_coverage = max(max_coverage, coverage)

        is_missing = max_iou >= args.iou_threshold or max_coverage >= 0.6

        if is_missing:
            missing_products.append({
                "slot_id": slot.get("slot_id", ""),
                "sku": slot.get("expected_sku", "unknown"),
                "bbox": {"x": px, "y": py, "width": pw, "height": ph},
                "max_iou": round(max_iou, 3),
                "max_coverage": round(max_coverage, 3),
                "reason": f"IoU: {max_iou:.2f}, coverage: {max_coverage:.2f}",
            })

    print(f"  Found {len(missing_products)} missing products")
    for mp in missing_products:
        print(f"    - {mp['slot_id']}: {mp['sku']} (IoU: {mp['max_iou']}, coverage: {mp['max_coverage']})")

    # Step 5: Generate annotated results
    print("\n[Step 5] Generating annotated result...")

    annotated_crop = draw_boxes_on_image(cropped_current, missing_products, (0, 0))
    cv2.imwrite(str(run_dir / "07_annotated_current_crop.png"), annotated_crop)

    annotated_full = draw_boxes_on_image(current_image, missing_products, (0, 0))
    cv2.imwrite(str(run_dir / "08_annotated_current_full.png"), annotated_full)

    annotated_baseline = draw_boxes_on_image(baseline_image, missing_products, (0, 0))
    cv2.imwrite(str(run_dir / "09_annotated_baseline.png"), annotated_baseline)

    missing_json = {
        "shelf_id": shelf_id,
        "face": face,
        "products": missing_products,
        "count": len(missing_products),
    }
    (run_dir / "missing_products.json").write_text(
        json.dumps(missing_json, ensure_ascii=False, indent=2)
    )

    elapsed = time.perf_counter() - started
    finished_at = datetime.now().astimezone()

    report = {
        "created_at": created_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "total_seconds": round(elapsed, 2),
        "status": "success",
        "input": {
            "current_image": str(current_path),
            "baseline_image": str(baseline_path),
            "lab_distance_threshold": args.lab_distance_threshold,
            "iou_threshold": args.iou_threshold,
        },
        "match": match_info,
        "alignment": {
            "method": f"OpenCV {args.feature.upper()} + mutual Lowe ratio + USAC_MAGSAC",
            "cropped_size": {"width": cropped_baseline.shape[1], "height": cropped_baseline.shape[0]},
            "crop_offset_in_full_image": {"x": crop_x, "y": crop_y},
        },
        "diff_analysis": {
            "diff_regions_count": len(diff_regions),
            "regions": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in diff_regions],
            "method": "GaussianBlur + Lab distance + global lighting compensation + morphology",
        },
        "missing_products": missing_products,
        "missing_count": len(missing_products),
        "output_artifacts": {
            "current_source": "00_current_source.*",
            "baseline_full": "01_baseline_full.*",
            "aligned_baseline_crop": "02_aligned_baseline_crop.png",
            "current_overlap_crop": "03_current_overlap_crop.png",
            "valid_overlap_mask": "03b_valid_overlap_mask.png",
            "diff_heatmap": "04_diff_heatmap.png",
            "diff_binary": "05_diff_binary.png",
            "diff_regions_overlay": "06_diff_regions.png",
            "annotated_current_crop": "07_annotated_current_crop.png",
            "annotated_current_full": "08_annotated_current_full.png",
            "annotated_baseline": "09_annotated_baseline.png",
            "missing_products_json": "missing_products.json",
        },
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"RESULT: {len(missing_products)} missing products found")
    print(f"{'='*60}")
    print(f"Finished at: {finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}")
    print(f"Total: {elapsed:.2f}s")
    print(f"Run directory: {run_dir}")
    print(f"Report: {report_path}")
    print(f"\nAnnotated images:")
    print(f"  07_annotated_current_crop.png  - 标注后的裁剪当前图（主要查看）")
    print(f"  08_annotated_current_full.png  - 标注后的完整当前图")
    print(f"  09_annotated_baseline.png      - 标注后的基准图")

    return 0


if __name__ == "__main__":
    args = parse_args()
    current_path = args.current.expanduser().resolve()
    config = get_inspection_config()
    config.update({
        "max_image_side": args.max_image_side,
        "feature": args.feature,
        "max_features": args.max_features,
        "ratio_test": args.ratio_test,
        "reprojection_threshold": args.reprojection_threshold,
        "min_inliers": args.min_inliers,
        "min_inlier_ratio": args.min_inlier_ratio,
        "min_current_coverage": args.min_current_coverage,
        "lab_distance_threshold": args.lab_distance_threshold,
        "slot_change_ratio_threshold": args.slot_change_ratio_threshold,
    })
    if args.dino_confidence_threshold is not None:
        config["dino_confidence_threshold"] = args.dino_confidence_threshold
    if args.ambiguity_margin is not None:
        config["ambiguity_margin"] = args.ambiguity_margin
    if args.vlm_fallback:
        config["vlm_fallback"] = True
    run_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        report = run_inspection(current_path, config, run_dir if args.debug else None, debug=args.debug)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"Inspection failed: {error}") from error
    print(f"Inspected shelf {report['shelf_id']} face {report['face']}")
    print(f"Changed slots: {len(report['slots'])}")
    if args.debug:
        print(f"Report: {run_dir / 'result.json'}")
