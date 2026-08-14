"""Read-only RGB competition round for visibly misplaced products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from vision.local_qwen import detect_misplacement_response
from vision.rgbd_stockout_sku import encode_png, resolve_candidate_sku, sku_references


def _json_value(response: str) -> Any:
    value = response.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        return None
    return value


def parse_misplacement_response(response: str, width: int, height: int, minimum_confidence: float = 0.70) -> list[dict[str, Any]]:
    payload = _json_value(response)
    rows = payload.get("boxes") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bbox_2d = row.get("bbox_2d")
        if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
            x1, y1, x2, y2 = bbox_2d
            values = [x1, y1, x2 - x1 if isinstance(x1, int) and isinstance(x2, int) else None, y2 - y1 if isinstance(y1, int) and isinstance(y2, int) else None]
        else:
            values = [row.get(key) for key in ("x", "y", "width", "height")]
        confidence, reason = row.get("confidence"), row.get("reason")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not minimum_confidence <= confidence <= 1:
            continue
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 400:
            continue
        x, y, box_width, box_height = values
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > width or y + box_height > height:
            continue
        candidates.append({
            "box": {"x": x, "y": y, "width": box_width, "height": box_height},
            "confidence": float(confidence), "reason": reason.strip(),
        })
    return sorted(candidates, key=lambda item: -item["confidence"])[:2]


def run_rgb_misplacement(rgb: Any, db_path: str | Path, item_images_dir: str | Path, config: dict[str, Any],
                         output_dir: Path | None = None, debug: bool = True) -> dict[str, Any]:
    """Identify up to two visually obvious misplaced products without writes."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be BGR")
    if debug and output_dir is None:
        raise ValueError("output_dir is required when debug is enabled")
    source = encode_png(rgb)
    try:
        detector_response = detect_misplacement_response(source, config["qwen_model_dir"], int(config["qwen_detector_max_new_tokens"]))
    except Exception as error:
        detector_response = ""
        detector_error = f"{type(error).__name__}: {error}"
    else:
        detector_error = None
    detected = parse_misplacement_response(
        detector_response, rgb.shape[1], rgb.shape[0], float(config["minimum_confidence"])
    )
    references = sku_references(db_path, item_images_dir)
    if debug:
        output_dir.mkdir(parents=True, exist_ok=False)
        artifacts = {"result_overlay": "result_overlay.png", "result": "result.json", "input": "input.png", "detector_response": "detector_response.txt"}
        (output_dir / "input.png").write_bytes(source)
        (output_dir / "detector_response.txt").write_text(detector_response or detector_error or "unknown", encoding="utf-8")
    overlay = rgb.copy()
    candidates = []
    for index, detected_row in enumerate(detected, start=1):
        decision = resolve_candidate_sku(rgb, detected_row["box"], references, config)
        row = {
            **detected_row,
            "current_sku": decision["sku"],
            "source": decision["source"],
            "dino_matches": decision["dino_matches"],
        }
        candidates.append(row)
        box = detected_row["box"]
        x, y = box["x"], box["y"]
        cv2.rectangle(overlay, (x, y), (x + box["width"], y + box["height"]), (0, 0, 255), 4)
        cv2.putText(overlay, decision["sku"] or "unknown", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        if debug and "qwen_full_image" in decision:
            candidate_dir = output_dir / f"candidate_{index}"
            candidate_dir.mkdir()
            (candidate_dir / "full_image.png").write_bytes(decision["qwen_full_image"])
            (candidate_dir / "candidate_board.png").write_bytes(decision["qwen_candidate_board"])
            (candidate_dir / "raw_response.txt").write_text(decision["qwen_response"] or decision["qwen_error"] or "unknown", encoding="utf-8")
            artifacts.update({
                f"candidate_{index}_full_image": f"candidate_{index}/full_image.png",
                f"candidate_{index}_candidate_board": f"candidate_{index}/candidate_board.png",
                f"candidate_{index}_raw_response": f"candidate_{index}/raw_response.txt",
            })
    report: dict[str, Any] = {"candidates": candidates}
    if debug:
        (output_dir / "result_overlay.png").write_bytes(encode_png(overlay))
        report.update({"run_id": output_dir.name, "artifacts": artifacts})
        (output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
