"""SKU resolution for RGB-D rear-row stockout candidates."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont
import yaml

from shelf_database import ShelfDatabase
from vision.dino import reference_similarity_scores
from vision.local_qwen import choose_candidate_response


_CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def dino_is_decisive(ranked: list[tuple[str, float]], *, score: float = 0.80, margin: float = 0.05) -> bool:
    return bool(ranked) and ranked[0][1] >= score and (len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= margin)


def encode_png(image: Any) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode image")
    return encoded.tobytes()


def annotated_full_image(rgb: Any, box: dict[str, int]) -> bytes:
    image = rgb.copy()
    x, y, width, height = (box[key] for key in ("x", "y", "width", "height"))
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 5)
    cv2.putText(image, "?", (x + 6, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    return encode_png(image)


def candidate_board(top: list[tuple[str, float]], references: dict[str, Path]) -> bytes:
    tiles = []
    font = ImageFont.truetype(_CJK_FONT, 17) if _CJK_FONT.is_file() else ImageFont.load_default()
    for index, (sku, _score) in enumerate(top, start=1):
        with Image.open(references[sku]) as source:
            image = source.convert("RGB")
        image.thumbnail((220, 180), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (240, 230), "white")
        tile.paste(image, ((240 - image.width) // 2, 8))
        ImageDraw.Draw(tile).text((8, 198), f"{index}. {sku}", fill="black", font=font)
        tiles.append(tile)
    board = Image.new("RGB", (240 * len(tiles), 230), "white")
    for index, tile in enumerate(tiles):
        board.paste(tile, (index * 240, 0))
    output = BytesIO()
    board.save(output, format="PNG")
    return output.getvalue()


def sku_references(db_path: str | Path, item_images_dir: str | Path) -> dict[str, Path]:
    """Use the first saved phone-entry crop for each SKU as its DINO reference."""
    root = Path(item_images_dir)
    with ShelfDatabase(db_path) as db:
        slots = sorted(db.get_all_slots(), key=lambda slot: slot.slot_id)
    references: dict[str, Path] = {}
    for slot in slots:
        path = root / slot.slot_id / "0.png"
        if slot.expected_sku not in references and path.is_file():
            references[slot.expected_sku] = path
    return references


def rank_candidate_skus(rgb: Any, box: dict[str, int], references: dict[str, Path]) -> list[tuple[str, float]]:
    x, y, width, height = (box[key] for key in ("x", "y", "width", "height"))
    crop = rgb[y:y + height, x:x + width]
    if crop.size == 0 or not references:
        return []
    candidate = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with_images = [(sku, Image.open(path).convert("RGB")) for sku, path in references.items()]
    scores = reference_similarity_scores(candidate, [image for _, image in with_images])
    return sorted(((sku, float(score)) for (sku, _), score in zip(with_images, scores)), key=lambda item: (-item[1], item[0]))


def resolve_candidate_sku(
    rgb: Any,
    box: dict[str, int],
    references: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    ranked = rank_candidate_skus(rgb, box, references)
    top = ranked[:int(config["dino_top_k"])]
    matches = [{"sku": sku, "confidence": round(score, 4)} for sku, score in top]
    if dino_is_decisive(top, score=float(config["dino_accept_score"]), margin=float(config["dino_accept_margin"])):
        return {"sku": top[0][0], "source": "dino", "dino_matches": matches}
    if not top:
        return {"sku": None, "source": "unknown", "dino_matches": matches}
    full = annotated_full_image(rgb, box)
    board = candidate_board(top, references)
    try:
        choice, qwen_response = choose_candidate_response(full, board, len(top), config["qwen_model_dir"], int(config["qwen_max_new_tokens"]))
        qwen_error = None
    except Exception as error:
        choice = None
        qwen_response = ""
        qwen_error = f"{type(error).__name__}: {error}"
    return {
        "sku": top[choice - 1][0] if choice is not None else None,
        "source": "qwen" if choice is not None else "unknown",
        "dino_matches": matches,
        "qwen_full_image": full,
        "qwen_candidate_board": board,
        "qwen_response": qwen_response,
        "qwen_error": qwen_error,
    }


def _sample_files(sample_directory: Path) -> tuple[Path, Path, Path]:
    prefixes = sorted({path.name.removesuffix("_rgb.png") for path in sample_directory.glob("*_rgb.png")})
    for prefix in prefixes:
        rgb = sample_directory / f"{prefix}_rgb.png"
        depth = sample_directory / f"{prefix}_depth_raw.png"
        metadata = sample_directory / f"{prefix}_metadata.yaml"
        if rgb.is_file() and depth.is_file() and metadata.is_file():
            return rgb, depth, metadata
    raise FileNotFoundError("Sample requires matching *_rgb.png, *_depth_raw.png, and *_metadata.yaml files")


def run_rgbd_stockout(sample_directory: Path, db_path: str | Path, item_images_dir: str | Path,
                      config: dict[str, Any], output_dir: Path | None = None, debug: bool = True) -> dict[str, Any]:
    """Run one server-selected D435i sample without changing inventory."""
    from vision.rgbd_stockout import detect_stockout_candidates

    if debug and output_dir is None:
        raise ValueError("output_dir is required when debug is enabled")
    rgb_path, depth_path, metadata_path = _sample_files(sample_directory)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ValueError("Sample camera metadata is invalid YAML") from error
    detection = detect_stockout_candidates(rgb, depth, metadata, threshold_mm=float(config["setback_threshold_mm"]))
    references = sku_references(db_path, item_images_dir)
    if debug:
        output_dir.mkdir(parents=True, exist_ok=False)
        artifacts = {"result_overlay": "result_overlay.png", "result": "result.json"}
    candidates = []
    overlay = detection.overlay.copy()
    for candidate in detection.candidates:
        decision = resolve_candidate_sku(rgb, candidate.box, references, config)
        row = {
            "shelf_index": candidate.shelf_index,
            "group_index": candidate.group_index,
            "sku": decision["sku"],
            "box": candidate.box,
            "setback_mm": candidate.setback_mm,
            "source": decision["source"],
            "dino_matches": decision["dino_matches"],
        }
        candidates.append(row)
        x, y = candidate.box["x"], candidate.box["y"]
        label = decision["sku"] or "unknown"
        cv2.putText(overlay, label, (x, min(overlay.shape[0] - 10, y + candidate.box["height"] + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        if debug and "qwen_full_image" in decision:
            candidate_dir = output_dir / f"candidate_{candidate.shelf_index}_{candidate.group_index}"
            candidate_dir.mkdir()
            (candidate_dir / "full_image.png").write_bytes(decision["qwen_full_image"])
            (candidate_dir / "candidate_board.png").write_bytes(decision["qwen_candidate_board"])
            (candidate_dir / "raw_response.txt").write_text(decision["qwen_response"] or decision["qwen_error"] or "unknown", encoding="utf-8")
            prefix = candidate_dir.name
            artifacts[f"{prefix}_full_image"] = f"{prefix}/full_image.png"
            artifacts[f"{prefix}_candidate_board"] = f"{prefix}/candidate_board.png"
            artifacts[f"{prefix}_raw_response"] = f"{prefix}/raw_response.txt"
    report: dict[str, Any] = {
        "sample": sample_directory.name,
        "candidates": candidates,
        "skipped_shelves": detection.skipped_shelves,
    }
    if debug:
        (output_dir / "result_overlay.png").write_bytes(encode_png(overlay))
        report["run_id"] = output_dir.name
        report["artifacts"] = artifacts
        (output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
