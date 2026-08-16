#!/usr/bin/env python3
"""Local JSON API for the warehouse visual manager."""

import base64
import binascii
from io import BytesIO
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image

from vision.ark_grounding import build_prompt, extract_boxes, request_grounding_bytes
from vision.config import (
    get_inspection_config,
    get_rgb_misplacement_config,
    get_rgbd_stockout_config,
    get_sku_query_config,
    save_inspection_config,
    save_sku_query_config,
)
from vision.rgbd_stockout_sku import run_rgbd_stockout, sku_references
from vision.rgb_misplacement import run_rgb_misplacement
from vision.cv_restock_position import run_inspection
from vision.dino import preload_dino
from vision.owlv2 import preload_owlv2, run_owlv2_sku_query
from vision.image_stitch import rectify_stitched_image, run_image_stitch
from vision.vlm_sku_query import generate_grouped_owlv2_prompts, generate_owlv2_prompt, run_vlm_sku_query
from scene_geometry import delivery_table_spec
from shelf_database import ShelfDatabase
import calibration_manager
import robot_service


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SHELF_DB_PATH", os.path.join(BASE_DIR, "shelf_inventory.db"))
ITEM_IMAGES_DIR = os.path.join(BASE_DIR, "data", "item_images")
SKU_IMAGES_DIR = Path(BASE_DIR) / "data" / "sku_images"
SHELF_IMAGES_DIR = os.path.join(BASE_DIR, "data", "shelf_images")
VISION_OUTPUT_DIR = Path(BASE_DIR) / "vision" / "output" / "slot_inspection"
SKU_QUERY_OUTPUT_DIR = Path(BASE_DIR) / "vision" / "output" / "vlm_sku_query"
IMAGE_STITCH_OUTPUT_DIR = Path(BASE_DIR) / "vision" / "output" / "image_stitch"
RGBD_SAMPLE_ROOT = Path(BASE_DIR) / "test_pic" / "rgbd_stockout"
RGBD_OUTPUT_DIR = Path(BASE_DIR) / "vision" / "output" / "rgbd_stockout"
RGB_MISPLACEMENT_OUTPUT_DIR = Path(BASE_DIR) / "vision" / "output" / "rgb_misplacement"
JSON_SYNC_LOCK = threading.RLock()


def shelf_image_filename(face: int) -> str:
    if face == 0:
        return "-x_0.png"
    if face == 1:
        return "+x_0.png"
    raise ValueError("face must be 0 or 1")


def shelf_image_path(shelf_id: int, face: int) -> str:
    return os.path.join(SHELF_IMAGES_DIR, str(shelf_id), shelf_image_filename(face))


def sku_image_path(sku: str) -> Path:
    if not sku or sku in {".", ".."} or any(character in sku for character in ("/", "\\", "\x00")):
        raise ValueError("SKU name cannot contain a path separator")
    return SKU_IMAGES_DIR / f"{sku}.png"


def sku_reference_image(payload: dict) -> bytes | None:
    image_data = payload.get("reference_image_data")
    if image_data in (None, ""):
        return None
    if not isinstance(image_data, str) or not image_data.startswith("data:image/") or ";base64," not in image_data:
        raise ValueError("reference_image_data must be a base64-encoded image")
    try:
        image_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("reference_image_data is not valid base64") from error
    if not image_bytes or len(image_bytes) > 12 * 1024 * 1024:
        raise ValueError("reference_image_data must be between 1 byte and 12 MB")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            normalized = source.convert("RGB")
            output = BytesIO()
            normalized.save(output, format="PNG")
            return output.getvalue()
    except (OSError, ValueError) as error:
        raise ValueError("reference_image_data cannot be decoded") from error


def save_sku_reference_image(sku: str, image_bytes: bytes) -> str:
    target = sku_image_path(sku)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as staged:
        staged.write(image_bytes)
        staged_path = Path(staged.name)
    os.replace(staged_path, target)
    return f"data/sku_images/{sku}.png"


def normal_sku_slot_image(sku: str) -> Path | None:
    with ShelfDatabase(DB_PATH) as db:
        slots = sorted(
            (slot for slot in db.get_all_slots() if slot.expected_sku == sku and slot.actual_sku == sku),
            key=lambda slot: (-(slot.width_cm or 0) * (slot.height_cm or 0), slot.slot_id),
        )
    return next((path for slot in slots if (path := Path(ITEM_IMAGES_DIR) / slot.slot_id / "0.png").is_file()), None)


def sku_prompt_sources(sku: str) -> tuple[tuple[bytes, bytes] | None, str | None]:
    primary = sku_image_path(sku)
    if not primary.is_file():
        return None, "No SKU primary image"
    normal_image = normal_sku_slot_image(sku)
    if normal_image is None:
        return None, "No normal ID image"
    return (primary.read_bytes(), normal_image.read_bytes()), None


def read_number(payload, name, number_type=float, minimum=None):
    value = payload.get(name)
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        value = number_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def snapshot():
    """Return the dashboard payload in one consistent format."""
    with ShelfDatabase(DB_PATH) as db:
        shelf_types = [asdict(item) for item in db.get_all_shelf_types()]
        shelves = [asdict(item) for item in db.get_all_shelf_groups()]
        delivery_tables = [asdict(item) for item in db.get_all_delivery_tables()]
        skus = [asdict(item) for item in db.get_all_skus()]
        slots = db.get_all_slots_world()
        return {
            "stats": db.get_stats(),
            "shelf_types": shelf_types,
            "shelves": shelves,
            "shelf_images": [
                {"shelf_id": shelf["id"], "face": face}
                for shelf in shelves
                for face in (0, 1)
                if os.path.isfile(shelf_image_path(shelf["id"], face))
            ],
            "delivery_tables": delivery_tables,
            "delivery_table_spec": delivery_table_spec(),
            "skus": skus,
            "slots": slots,
        }


def sync_shelf_projection(shelf_id, bbox_by_slot=None):
    """Rebuild one shelf's JSON cache after SQLite has committed."""
    with JSON_SYNC_LOCK:
        robot_service.sync_shelf_projection(DB_PATH, shelf_id, bbox_by_slot)


def read_optional_dimension(payload, name):
    value = payload.get(name)
    return None if value in (None, "") else read_number(payload, name, float, 0)


def read_optional_bbox(payload):
    bbox = payload.get("bbox")
    if bbox is None:
        return None
    if not isinstance(bbox, dict):
        raise ValueError("bbox must be an object")
    return {
        "x": read_number(bbox, "x", int, 0),
        "y": read_number(bbox, "y", int, 0),
        "width": read_number(bbox, "width", int, 0),
        "height": read_number(bbox, "height", int, 0),
    }


def read_debug(payload):
    debug = payload.get("debug", False)
    if not isinstance(debug, bool):
        raise ValueError("debug must be a boolean")
    return debug


def rgbd_sample_directory(sample: str) -> Path:
    if not re.fullmatch(r"[\w\u4e00-\u9fff-]{1,80}", sample):
        raise ValueError("Invalid RGB-D sample name")
    root = RGBD_SAMPLE_ROOT.resolve()
    directory = (root / sample).resolve()
    if directory.parent != root or not directory.is_dir():
        raise FileNotFoundError("RGB-D sample not found")
    return directory


def list_rgbd_samples() -> list[str]:
    if not RGBD_SAMPLE_ROOT.is_dir():
        return []
    samples = []
    for directory in RGBD_SAMPLE_ROOT.iterdir():
        if not directory.is_dir():
            continue
        prefixes = {path.name.removesuffix("_rgb.png") for path in directory.glob("*_rgb.png")}
        if any((directory / f"{prefix}_depth_raw.png").is_file() and (directory / f"{prefix}_metadata.yaml").is_file() for prefix in prefixes):
            samples.append(directory.name)
    return sorted(samples)


def rgbd_run_identifier(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", run_id):
        raise ValueError("Invalid RGB-D run ID")
    return run_id


def inspection_config(payload):
    if not isinstance(payload, dict):
        raise ValueError("inspection config must be an object")
    config = get_inspection_config()
    for key in ("min_current_coverage", "analysis_center_ratio", "lab_distance_threshold", "slot_change_ratio_threshold", "dino_confidence_threshold", "ambiguity_margin", "vlm_top_k", "vlm_fallback"):
        if key in payload:
            config[key] = payload[key]
    try:
        config["min_current_coverage"] = float(config["min_current_coverage"])
        config["analysis_center_ratio"] = float(config["analysis_center_ratio"])
        config["lab_distance_threshold"] = float(config["lab_distance_threshold"])
        config["slot_change_ratio_threshold"] = float(config["slot_change_ratio_threshold"])
        config["dino_confidence_threshold"] = float(config["dino_confidence_threshold"])
        config["ambiguity_margin"] = float(config["ambiguity_margin"])
        config["vlm_top_k"] = int(config["vlm_top_k"])
    except (TypeError, ValueError) as error:
        raise ValueError("Inspection thresholds must be numeric") from error
    if not 0.01 <= config["min_current_coverage"] <= 0.3:
        raise ValueError("min_current_coverage must be between 0.01 and 0.3")
    if not 0.5 <= config["analysis_center_ratio"] <= 1:
        raise ValueError("analysis_center_ratio must be between 0.5 and 1")
    if config["lab_distance_threshold"] <= 0:
        raise ValueError("lab_distance_threshold must be positive")
    if not 0 < config["slot_change_ratio_threshold"] <= 1:
        raise ValueError("slot_change_ratio_threshold must be between 0 and 1")
    if not 0 <= config["dino_confidence_threshold"] <= 1:
        raise ValueError("dino_confidence_threshold must be between 0 and 1")
    if config["ambiguity_margin"] < 0:
        raise ValueError("ambiguity_margin must be at least 0")
    if not 1 <= config["vlm_top_k"] <= 9:
        raise ValueError("vlm_top_k must be between 1 and 9")
    for key in ("vlm_fallback",):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean")
    return config


def vision_run_directory(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid inspection run ID")
    return VISION_OUTPUT_DIR / run_id


def sku_query_run_directory(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid SKU query run ID")
    return SKU_QUERY_OUTPUT_DIR / run_id


def image_stitch_run_directory(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid image stitch run ID")
    return IMAGE_STITCH_OUTPUT_DIR / run_id


def sku_query_reference(query):
    """Resolve a submitted slot ID or SKU to one stable reference crop."""
    with ShelfDatabase(DB_PATH) as db:
        slot = db.get_slot_by_id(query)
        if slot is not None:
            sku_info = db.get_sku_info(slot.expected_sku)
            if sku_info is not None:
                reference_path = sku_image_path(sku_info.sku)
                if sku_info.reference_image_path and reference_path.is_file():
                    return sku_info.sku, None, reference_path
            candidates = [slot]
        elif (sku_info := db.get_sku_info(query)) is not None:
            reference_path = sku_image_path(sku_info.sku)
            if sku_info.reference_image_path and reference_path.is_file():
                return sku_info.sku, None, reference_path
            candidates = sorted(
                (item for item in db.get_all_slots_world() if item["expected_sku"] == query),
                key=lambda item: (item["actual_sku"] != query, item["slot_id"]),
            )
        else:
            raise ValueError("Query must be an existing slot ID or SKU")
    for candidate in candidates:
        slot_id = candidate.slot_id if hasattr(candidate, "slot_id") else candidate["slot_id"]
        expected_sku = candidate.expected_sku if hasattr(candidate, "expected_sku") else candidate["expected_sku"]
        reference_path = Path(ITEM_IMAGES_DIR) / slot_id / "0.png"
        if reference_path.is_file():
            return expected_sku, slot_id, reference_path
    raise ValueError("No reference image is available for this slot or SKU")


def sku_query_config(payload):
    if not isinstance(payload, dict):
        raise ValueError("SKU query config must be an object")
    config = get_sku_query_config()
    for key in ("max_boxes", "dino_fallback", "dino_confidence_threshold", "owlv2_score_threshold"):
        if key in payload:
            config[key] = payload[key]
    try:
        config["max_boxes"] = int(config["max_boxes"])
        config["dino_confidence_threshold"] = float(config["dino_confidence_threshold"])
        config["owlv2_score_threshold"] = float(config["owlv2_score_threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError("SKU query limits must be numeric") from error
    if not 1 <= config["max_boxes"] <= 20:
        raise ValueError("max_boxes must be between 1 and 20")
    if not 0 <= config["dino_confidence_threshold"] <= 1:
        raise ValueError("dino_confidence_threshold must be between 0 and 1")
    if not 0 <= config["owlv2_score_threshold"] <= 1:
        raise ValueError("owlv2_score_threshold must be between 0 and 1")
    if not isinstance(config["dino_fallback"], bool):
        raise ValueError("dino_fallback must be a boolean")
    return config


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "ShelfManager/0.1"

    def log_message(self, fmt, *args):
        print(f"[api] {self.address_string()} - {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self, maximum=64 * 1024):
        length = int(self.headers.get("Content-Length", "0"))
        if length > maximum:
            raise ValueError("Request body is too large")
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON") from error

    def send_file(self, filename, content_type):
        try:
            with open(filename, "rb") as image_file:
                body = image_file.read()
        except FileNotFoundError:
            self.send_json({"error": "Image not found"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def slot_id_for(slot):
        if isinstance(slot, dict):
            return slot["slot_id"]
        return slot.slot_id

    def remove_slot_images(self, slots):
        """Remove controlled per-instance image directories after database deletion."""
        for slot in slots:
            slot_id = self.slot_id_for(slot)
            image_dir = os.path.join(ITEM_IMAGES_DIR, slot_id)
            if os.path.isdir(image_dir):
                shutil.rmtree(image_dir, ignore_errors=True)

    @staticmethod
    def remove_shelf_images(shelf_id):
        """Remove a shelf's controlled source-photo directory with the shelf."""
        image_dir = os.path.join(SHELF_IMAGES_DIR, str(shelf_id))
        if os.path.isdir(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json({"ok": True})
        elif path == "/api/rgbd-stockout/samples":
            self.send_json({"samples": list_rgbd_samples()})
        elif path.startswith("/api/rgbd-stockout/runs/") and "/artifact/" in path:
            prefix, artifact_name = path.split("/artifact/", 1)
            run_id = rgbd_run_identifier(prefix[len("/api/rgbd-stockout/runs/"):].strip("/"))
            run_dir = RGBD_OUTPUT_DIR / run_id
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                self.send_json({"error": "RGB-D run not found"}, HTTPStatus.NOT_FOUND)
                return
            report = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_name = unquote(artifact_name)
            if artifact_name not in set(report.get("artifacts", {}).values()):
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            artifact = (run_dir / artifact_name).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file():
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = {
                ".json": "application/json; charset=utf-8",
                ".txt": "text/plain; charset=utf-8",
            }.get(artifact.suffix.lower(), "image/png")
            self.send_file(artifact, content_type)
        elif path.startswith("/api/rgb-misplacement/runs/") and "/artifact/" in path:
            prefix, artifact_name = path.split("/artifact/", 1)
            run_id = rgbd_run_identifier(prefix[len("/api/rgb-misplacement/runs/"):].strip("/"))
            run_dir = RGB_MISPLACEMENT_OUTPUT_DIR / run_id
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                self.send_json({"error": "RGB misplacement run not found"}, HTTPStatus.NOT_FOUND)
                return
            report = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_name = unquote(artifact_name)
            if artifact_name not in set(report.get("artifacts", {}).values()):
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            artifact = (run_dir / artifact_name).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file():
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = {".json": "application/json; charset=utf-8", ".txt": "text/plain; charset=utf-8"}.get(artifact.suffix.lower(), "image/png")
            self.send_file(artifact, content_type)
        elif path == "/api/state":
            self.send_json(snapshot())
        elif path == "/api/vision/config":
            self.send_json(inspection_config({}))
        elif path == "/api/sku-query/config":
            self.send_json(sku_query_config({}))
        elif path.startswith("/api/vision/runs/") and path.endswith("/result") and "/artifact/" not in path:
            run_id = path[len("/api/vision/runs/"):-len("/result")].strip("/")
            result_path = vision_run_directory(run_id) / "result.json"
            if not result_path.is_file():
                self.send_json({"error": "Inspection run not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(json.loads(result_path.read_text(encoding="utf-8")))
        elif path.startswith("/api/vision/runs/") and "/artifact/" in path:
            prefix, artifact_name = path.split("/artifact/", 1)
            run_id = prefix[len("/api/vision/runs/"):].strip("/")
            run_dir = vision_run_directory(run_id)
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                self.send_json({"error": "Inspection run not found"}, HTTPStatus.NOT_FOUND)
                return
            report = json.loads(result_path.read_text(encoding="utf-8"))
            if artifact_name not in set(report.get("artifacts", {}).values()):
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            artifact = (run_dir / artifact_name).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file():
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = "application/json; charset=utf-8" if artifact.suffix == ".json" else "image/png"
            self.send_file(artifact, content_type)
        elif path.startswith("/api/sku-query/runs/") and "/artifact/" in path:
            prefix, artifact_name = path.split("/artifact/", 1)
            run_id = prefix[len("/api/sku-query/runs/"):].strip("/")
            run_dir = sku_query_run_directory(run_id)
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                self.send_json({"error": "SKU query run not found"}, HTTPStatus.NOT_FOUND)
                return
            report = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_name = unquote(artifact_name)
            if artifact_name not in set(report.get("artifacts", {}).values()):
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            artifact = (run_dir / artifact_name).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file():
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = {
                ".json": "application/json; charset=utf-8",
                ".txt": "text/plain; charset=utf-8",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(artifact.suffix.lower(), "image/png")
            self.send_file(artifact, content_type)
        elif path.startswith("/api/image-stitch/runs/") and "/artifact/" in path:
            prefix, artifact_name = path.split("/artifact/", 1)
            run_id = prefix[len("/api/image-stitch/runs/"):].strip("/")
            run_dir = image_stitch_run_directory(run_id)
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                self.send_json({"error": "Image stitch run not found"}, HTTPStatus.NOT_FOUND)
                return
            report = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_name = unquote(artifact_name)
            if artifact_name not in set(report.get("artifacts", {}).values()):
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            artifact = (run_dir / artifact_name).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file():
                self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(artifact, "image/png")
        elif path == "/api/shortages":
            with ShelfDatabase(DB_PATH) as db:
                slots = [asdict(slot) for slot in db.get_shortage_slots()]
            self.send_json({"slots": slots})
        elif path == "/api/misplacements":
            with ShelfDatabase(DB_PATH) as db:
                slots = [asdict(slot) for slot in db.get_misplaced_slots()]
            self.send_json({"slots": slots})
        elif path.startswith("/api/shelves/") and path.endswith("/calibration"):
            shelf_id = int(path[len("/api/shelves/"):-len("/calibration")].strip("/"))
            with ShelfDatabase(DB_PATH) as db:
                calibration = db.get_shelf_calibration(shelf_id)
            if calibration is None:
                self.send_json({"error": "Shelf not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(calibration)
        elif path.startswith("/api/item-images/") and path.endswith("/0.png"):
            slot_id = unquote(path[len("/api/item-images/"):-len("/0.png")]).strip("/")
            with ShelfDatabase(DB_PATH) as db:
                try:
                    db.slot_id_str_to_tuple(slot_id)
                except ValueError:
                    self.send_json({"error": "Invalid slot ID"}, HTTPStatus.BAD_REQUEST)
                    return
            self.send_file(os.path.join(ITEM_IMAGES_DIR, slot_id, "0.png"), "image/png")
        elif path.startswith("/api/sku-images/"):
            sku = unquote(path[len("/api/sku-images/"):]).strip("/")
            with ShelfDatabase(DB_PATH) as db:
                info = db.get_sku_info(sku)
            if info is None or not info.reference_image_path:
                self.send_json({"error": "SKU image not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(sku_image_path(sku), "image/png")
        elif path.startswith("/api/shelf-images/") and path.endswith("/0.png"):
            parts = path.strip("/").split("/")
            if len(parts) != 5:
                self.send_json({"error": "Invalid shelf image path"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                shelf_id = int(parts[2])
                face = int(parts[3])
                filename = shelf_image_filename(face)
            except ValueError:
                self.send_json({"error": "Invalid shelf image path"}, HTTPStatus.BAD_REQUEST)
                return
            with ShelfDatabase(DB_PATH) as db:
                if db.get_shelf_group(shelf_id) is None:
                    self.send_json({"error": "Shelf not found"}, HTTPStatus.NOT_FOUND)
                    return
            self.send_file(os.path.join(SHELF_IMAGES_DIR, str(shelf_id), filename), "image/png")
        elif path.startswith("/api/skus/") and path.endswith("/world-positions"):
            sku = unquote(path[len("/api/skus/"):-len("/world-positions")]).strip("/")
            with ShelfDatabase(DB_PATH) as db:
                if not sku or db.get_sku_info(sku) is None:
                    self.send_json({"error": "SKU not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json({"sku": sku, "positions": db.find_sku_world_positions(sku)})
        elif path.startswith("/api/slots/") and path.endswith("/world-position"):
            slot_id = unquote(path[len("/api/slots/"):-len("/world-position")]).strip("/")
            with ShelfDatabase(DB_PATH) as db:
                try:
                    db.slot_id_str_to_tuple(slot_id)
                except ValueError:
                    self.send_json({"error": "Invalid slot ID"}, HTTPStatus.BAD_REQUEST)
                    return
                slot = db.get_slot_by_id(slot_id)
                position = None if slot is None else db.get_slot_world_pos(
                    slot.shelf_id, slot.face, slot.level, slot.y_cm, slot.height_cm
                )
                if slot is None or position is None:
                    self.send_json({"error": "Slot not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json({
                    "slot": {
                        **asdict(slot),
                        "frame": "map",
                        "world_x": position.x,
                        "world_y": position.y,
                        "world_z": position.z,
                    }
                })
        else:
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _import_manual_batch(self, payload):
        """Stage item crops and one shelf-face source photo before importing slots."""
        items = payload.get("items")
        new_skus = payload.get("new_skus", [])
        sku_prompts = payload.get("sku_prompts", [])
        sku_metadata = payload.get("sku_metadata", [])
        shelf_image = payload.get("shelf_image")
        layers_data = payload.get("layers", {})
        slots_calib = []
        if not isinstance(items, list) or not items:
            raise ValueError("items must be a non-empty list")
        if not isinstance(new_skus, list):
            raise ValueError("new_skus must be a list")
        if not isinstance(sku_prompts, list):
            raise ValueError("sku_prompts must be a list")
        if not isinstance(sku_metadata, list):
            raise ValueError("sku_metadata must be a list")

        batch_dir = os.path.join(ITEM_IMAGES_DIR, ".staging", uuid.uuid4().hex)
        staged_dirs = []
        staged_sku_images = []
        db_items = []
        promoted_dirs = []
        promoted_sku_images = []
        database_imported = False
        staged_shelf_image = None
        shelf_image_target = None
        shelf_image_backup = None
        shelf_image_promoted = False
        cal_shelf_id = None
        cal_face = None
        try:
            for item in items:
                shelf_id = read_number(item, "shelf_id", int, 1)
                face = read_number(item, "face", int, 0)
                level = read_number(item, "level", int, 0)
                y_cm = read_number(item, "y_cm", float, 0)
                width_cm = read_number(item, "width_cm", float, 0)
                height_cm = read_number(item, "height_cm", float, 0)
                expected_sku = str(item.get("expected_sku", "")).strip()
                actual_sku = item.get("actual_sku", expected_sku)
                if actual_sku is not None:
                    actual_sku = str(actual_sku).strip()
                if face not in (0, 1) or not expected_sku:
                    raise ValueError("Each imported slot needs a valid face and expected_sku")
                slot_id = ShelfDatabase.format_slot_id(shelf_id, face, level, y_cm)
                bbox = read_optional_bbox(item)
                if cal_shelf_id is None:
                    cal_shelf_id = shelf_id
                    cal_face = face
                if cal_shelf_id != shelf_id or cal_face != face:
                    raise ValueError("All imported slots must belong to one shelf face")
                calibrated_slot = {
                    "slot_id": slot_id,
                    "expected_sku": expected_sku,
                    "actual_sku": actual_sku,
                }
                if bbox is not None:
                    calibrated_slot["bbox"] = bbox
                slots_calib.append(calibrated_slot)

                image_data = str(item.get("image_png", ""))
                if not image_data.startswith("data:image/png;base64,"):
                    raise ValueError(f"{slot_id} is missing a PNG crop")
                try:
                    image_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ValueError(f"{slot_id} has invalid PNG data") from error
                if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"{slot_id} crop must be PNG")
                relative_dir = f"data/item_images/{slot_id}"
                staged_dir = os.path.join(batch_dir, slot_id)
                os.makedirs(staged_dir, exist_ok=False)
                with open(os.path.join(staged_dir, "0.png"), "wb") as crop_file:
                    crop_file.write(image_bytes)
                staged_dirs.append((staged_dir, os.path.join(ITEM_IMAGES_DIR, slot_id)))
                db_items.append({
                    "shelf_id": shelf_id, "face": face, "level": level, "y_cm": y_cm,
                    "expected_sku": expected_sku, "actual_sku": actual_sku,
                    "width_cm": width_cm, "height_cm": height_cm,
                    "image_dir": relative_dir,
                })

            prepared_sku_metadata = []
            for index, item in enumerate(sku_metadata):
                if not isinstance(item, dict):
                    raise ValueError("sku_metadata must contain objects")
                sku = str(item.get("sku", "")).strip()
                target = sku_image_path(sku)
                image_bytes = sku_reference_image(item)
                reference_image_path = ""
                if image_bytes is not None:
                    staged_image = os.path.join(batch_dir, f"sku-{index}.png")
                    with open(staged_image, "wb") as output:
                        output.write(image_bytes)
                    staged_sku_images.append((staged_image, target, None))
                    reference_image_path = f"data/sku_images/{sku}.png"
                prepared_sku_metadata.append({
                    "sku": sku,
                    "reference_image_path": reference_image_path,
                    "grasp_method": str(item.get("grasp_method", "夹爪")),
                })

            if shelf_image is not None:
                if not isinstance(shelf_image, dict):
                    raise ValueError("shelf_image must be an object")
                source_shelf_id = read_number(shelf_image, "shelf_id", int, 1)
                source_face = read_number(shelf_image, "face", int, 0)
                source_data = str(shelf_image.get("image_data", ""))
                if source_face not in (0, 1):
                    raise ValueError("shelf_image face must be 0 or 1")
                if any(item["shelf_id"] != source_shelf_id or item["face"] != source_face for item in db_items):
                    raise ValueError("shelf_image must match every imported item's shelf and face")
                if not source_data.startswith("data:image/") or ";base64," not in source_data:
                    raise ValueError("shelf_image image_data must be a base64-encoded image")
                try:
                    source_bytes = base64.b64decode(source_data.split(",", 1)[1], validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ValueError("shelf_image image_data is not valid base64") from error
                if not source_bytes or len(source_bytes) > 12 * 1024 * 1024:
                    raise ValueError("shelf_image must be between 1 byte and 12 MB")
                try:
                    with Image.open(BytesIO(source_bytes)) as source:
                        normalized_source = source.convert("RGB")
                        source_output = BytesIO()
                        normalized_source.save(source_output, format="PNG")
                except (OSError, ValueError) as error:
                    raise ValueError("shelf_image cannot be decoded") from error
                staged_shelf_image = os.path.join(batch_dir, "shelf_face.png")
                with open(staged_shelf_image, "wb") as output:
                    output.write(source_output.getvalue())
                shelf_image_target = shelf_image_path(source_shelf_id, source_face)

            os.makedirs(ITEM_IMAGES_DIR, exist_ok=True)
            with ShelfDatabase(DB_PATH) as db:
                if shelf_image is not None and db.get_shelf_group(source_shelf_id) is None:
                    raise ValueError("Shelf does not exist")
                for item, (_, target_dir) in zip(db_items, staged_dirs):
                    existing = db.get_slot(item["shelf_id"], item["face"], item["level"], item["y_cm"])
                    if existing is None and os.path.isdir(target_dir):
                        # Older versions did not remove an image directory when a
                        # slot was deleted. It is safe to repair that orphan here.
                        shutil.rmtree(target_dir, ignore_errors=True)
            for staged_dir, target_dir in staged_dirs:
                if os.path.exists(target_dir):
                    raise ValueError(f"Image directory already exists for {os.path.basename(target_dir)}")
            # The filesystem has no SQLite-style transaction. Promote prepared images
            # first, then remove them if the following database transaction rejects.
            for staged_dir, target_dir in staged_dirs:
                os.replace(staged_dir, target_dir)
                promoted_dirs.append(target_dir)
            for index, (staged_image, target, _backup) in enumerate(staged_sku_images):
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = None
                if target.is_file():
                    backup = Path(batch_dir) / f"sku-{index}.previous.png"
                    os.replace(target, backup)
                os.replace(staged_image, target)
                staged_sku_images[index] = (staged_image, target, backup)
                promoted_sku_images.append((target, backup))
            if staged_shelf_image and shelf_image_target:
                os.makedirs(os.path.dirname(shelf_image_target), exist_ok=True)
                if os.path.isfile(shelf_image_target):
                    shelf_image_backup = os.path.join(batch_dir, "shelf_face_previous.png")
                    os.replace(shelf_image_target, shelf_image_backup)
                os.replace(staged_shelf_image, shelf_image_target)
                shelf_image_promoted = True
            with ShelfDatabase(DB_PATH) as db:
                slot_ids = db.import_slots_batch(new_skus, db_items, sku_prompts, prepared_sku_metadata)
            database_imported = True

            if cal_shelf_id is not None and cal_face is not None:
                with JSON_SYNC_LOCK:
                    if isinstance(layers_data, dict) and str(cal_face) in layers_data:
                        face_layers = layers_data[str(cal_face)]
                        with ShelfDatabase(DB_PATH) as db:
                            shelf_group = db.get_shelf_group(cal_shelf_id)
                            shelf_name = shelf_group.name if shelf_group else f"{cal_shelf_id}号货架"
                        calibration_manager.save_calibration(
                            shelf_id=cal_shelf_id,
                            shelf_name=shelf_name,
                            face=cal_face,
                            layers=face_layers if isinstance(face_layers, dict) else {},
                            slots=slots_calib,
                        )
                    bbox_by_slot = {
                        slot["slot_id"]: slot["bbox"]
                        for slot in slots_calib
                        if "bbox" in slot
                    }
                    sync_shelf_projection(cal_shelf_id, bbox_by_slot)

            if shelf_image_backup and os.path.isfile(shelf_image_backup):
                os.remove(shelf_image_backup)
            for _target, backup in promoted_sku_images:
                if backup is not None and backup.is_file():
                    backup.unlink()
            self.send_json({"slot_ids": slot_ids, "state": snapshot()}, HTTPStatus.CREATED)
        except Exception:
            if not database_imported:
                for target_dir in promoted_dirs:
                    shutil.rmtree(target_dir, ignore_errors=True)
                for target, backup in promoted_sku_images:
                    target.unlink(missing_ok=True)
                    if backup is not None and backup.is_file():
                        os.replace(backup, target)
            if shelf_image_target and shelf_image_promoted:
                os.remove(shelf_image_target)
            if shelf_image_target and shelf_image_backup and os.path.isfile(shelf_image_backup):
                os.replace(shelf_image_backup, shelf_image_target)
            raise
        finally:
            if os.path.isdir(batch_dir):
                shutil.rmtree(batch_dir, ignore_errors=True)

    def _draft_owlv2_prompts(self, payload):
        requests = payload.get("requests")
        if not isinstance(requests, list) or not 1 <= len(requests) <= 50:
            raise ValueError("requests must contain between 1 and 50 items")
        drafts = []
        for item in requests:
            if not isinstance(item, dict):
                raise ValueError("requests must contain objects")
            sku = str(item.get("sku", "")).strip()
            slot_image = item.get("slot_image")
            if not sku or not isinstance(slot_image, str):
                raise ValueError("Each request needs a SKU and one normal slot image")
            sources = []
            sku_image = item.get("reference_image_data")
            if sku_image is None:
                with ShelfDatabase(DB_PATH) as db:
                    info = db.get_sku_info(sku)
                if info is not None and info.reference_image_path and sku_image_path(sku).is_file():
                    sku_image = "data:image/png;base64," + base64.b64encode(sku_image_path(sku).read_bytes()).decode()
            for image_data in (sku_image, slot_image):
                if image_data is None:
                    continue
                if not isinstance(image_data, str) or not image_data.startswith("data:image/") or ";base64," not in image_data:
                    raise ValueError("Prompt samples must be base64-encoded images")
                try:
                    image_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
                    with Image.open(BytesIO(image_bytes)) as image:
                        image.verify()
                except (ValueError, OSError, binascii.Error) as error:
                    raise ValueError("Prompt sample is not a valid image") from error
                sources.append(image_bytes)
            drafts.append({"sku": sku, "owlv2_prompt": generate_owlv2_prompt(sku, sources)})
        self.send_json({"drafts": drafts}, HTTPStatus.CREATED)

    def _import_sku_image_directory(self, payload):
        directory = Path(str(payload.get("directory", "")).strip()).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("directory must be an existing directory")
        imported, skipped = [], []
        for source in sorted(directory.glob("*.png")):
            sku = source.stem.strip()
            try:
                target = sku_image_path(sku)
                with Image.open(source) as image:
                    output = BytesIO()
                    image.convert("RGB").save(output, format="PNG")
                with ShelfDatabase(DB_PATH) as db:
                    current = db.get_sku_info(sku)
                    save_sku_reference_image(sku, output.getvalue())
                    if current is None:
                        db.register_sku(sku, reference_image_path=f"data/sku_images/{sku}.png")
                    else:
                        db.update_sku(sku, current.category, current.mesh_file, current.tex_file,
                                      current.owlv2_prompt, f"data/sku_images/{sku}.png", current.grasp_method)
                imported.append(sku)
            except (OSError, ValueError) as error:
                skipped.append({"file": source.name, "reason": str(error)})
        self.send_json({"imported": imported, "skipped": skipped, "state": snapshot()}, HTTPStatus.CREATED)

    def _generate_sku_prompts(self, payload):
        skus = payload.get("skus")
        if not isinstance(skus, list):
            raise ValueError("skus must be a list")
        names = list(dict.fromkeys(str(sku).strip() for sku in skus))
        if not 2 <= len(names) <= 8:
            raise ValueError("skus must contain between 2 and 8 unique items")
        complete, skipped = [], []
        for value in names:
            with ShelfDatabase(DB_PATH) as db:
                exists = db.get_sku_info(value) is not None
            if not exists:
                skipped.append({"sku": value, "reason": "SKU not found"})
                continue
            sources, reason = sku_prompt_sources(value)
            if reason:
                skipped.append({"sku": value, "reason": reason})
                continue
            primary, normal = sources
            complete.append((value, primary, normal))
        if len(complete) < 2:
            skipped.extend({"sku": value, "reason": "Need at least two complete SKUs"} for value, _primary, _normal in complete)
            self.send_json({"drafts": [], "skipped": skipped}, HTTPStatus.CREATED)
            return
        generated = generate_grouped_owlv2_prompts(complete)
        drafts = [{"sku": value, "owlv2_prompt": generated[value]} for value, _primary, _normal in complete]
        self.send_json({"drafts": drafts, "skipped": skipped}, HTTPStatus.CREATED)

    def _delete_skus(self, payload):
        skus = payload.get("skus")
        if not isinstance(skus, list):
            raise ValueError("skus must be a list")
        names = list(dict.fromkeys(str(sku).strip() for sku in skus))
        with ShelfDatabase(DB_PATH) as db:
            deleted = db.delete_skus_to_unknown(names)
        for sku in deleted:
            sku_image_path(sku).unlink(missing_ok=True)
        self.send_json({"deleted": deleted, "state": snapshot()})

    def _save_sku_batch(self, payload):
        rows = payload.get("skus")
        if not isinstance(rows, list) or not rows:
            raise ValueError("skus must be a non-empty list")
        names = [str(row.get("sku", "")).strip() for row in rows if isinstance(row, dict)]
        if len(names) != len(rows) or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("SKU names must be non-empty and unique")
        for row in rows:
            sku = str(row["sku"]).strip()
            original = str(row.get("original_sku", sku)).strip()
            image_bytes = sku_reference_image(row)
            with ShelfDatabase(DB_PATH) as db:
                current = db.get_sku_info(original)
                if current is None:
                    if original != sku:
                        raise ValueError(f"SKU {original} does not exist")
                    db.register_sku(sku, str(row.get("category", "")), str(row.get("mesh_file", "")),
                                    str(row.get("tex_file", "")), str(row.get("owlv2_prompt", "")),
                                    "", str(row.get("grasp_method", "夹爪")),
                                    str(row.get("qwen_grounding_prompt", "")))
                    current = db.get_sku_info(sku)
                elif original != sku:
                    old_path, new_path = sku_image_path(original), sku_image_path(sku)
                    moved = False
                    if old_path.is_file():
                        if new_path.exists():
                            raise ValueError(f"SKU image already exists for {sku}")
                        os.replace(old_path, new_path)
                        moved = True
                    try:
                        current = db.rename_sku(
                            original, sku,
                            f"data/sku_images/{sku}.png" if current.reference_image_path else "",
                        )
                    except Exception:
                        if moved:
                            os.replace(new_path, old_path)
                        raise
                reference_path = current.reference_image_path
                if image_bytes is not None:
                    reference_path = save_sku_reference_image(sku, image_bytes)
                db.update_sku(sku, str(row.get("category", current.category)),
                              str(row.get("mesh_file", current.mesh_file)), str(row.get("tex_file", current.tex_file)),
                              str(row.get("owlv2_prompt", current.owlv2_prompt)), reference_path,
                              str(row.get("grasp_method", current.grasp_method)),
                              str(row.get("qwen_grounding_prompt", current.qwen_grounding_prompt)))
        self.send_json({"state": snapshot()})

    def _ground_products(self, payload):
        """Run Ark grounding on a browser-provided photo without exposing the API key."""
        image_data = str(payload.get("image_data", ""))
        if not image_data.startswith("data:image/") or ";base64," not in image_data:
            raise ValueError("image_data must be a base64-encoded JPEG, PNG, or WebP image")
        prefix, encoded = image_data.split(",", 1)
        image_type = prefix.removeprefix("data:").removesuffix(";base64")
        if image_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Only JPEG, PNG, and WebP photos are supported")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("image_data is not valid base64") from error
        if not image_bytes or len(image_bytes) > 12 * 1024 * 1024:
            raise ValueError("Photo must be between 1 byte and 12 MB")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
        except (OSError, ValueError) as error:
            raise ValueError("Photo cannot be decoded") from error
        with ShelfDatabase(DB_PATH) as db:
            known_skus = [item.sku for item in db.get_all_skus()]
        prompt = build_prompt("每一层货架最前排", known_skus)
        try:
            response_text = request_grounding_bytes(image_bytes, image_type, "doubao-seed-2-1-pro-260628", prompt)
        except Exception as error:
            raise ValueError(f"Ark grounding failed: {error}") from error
        boxes = extract_boxes(response_text, width, height)
        self.send_json({
            "boxes": [
                {"label": box.label, "x": box.pixels[0], "y": box.pixels[1],
                 "width": box.pixels[2] - box.pixels[0], "height": box.pixels[3] - box.pixels[1]}
                for box in boxes
            ],
            "detected": len(boxes),
        })

    def _run_vision_inspection(self, payload):
        image_data = str(payload.get("image_data", ""))
        if not image_data.startswith("data:image/") or ";base64," not in image_data:
            raise ValueError("image_data must be a base64-encoded image")
        try:
            image_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
            current_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, binascii.Error) as error:
            raise ValueError("image_data is not a valid image") from error
        if not image_bytes or len(image_bytes) > 12 * 1024 * 1024 or current_image is None:
            raise ValueError("Photo must be between 1 byte and 12 MB")
        config = inspection_config(payload.get("config", {}))
        config["_sku_images"] = sku_references(DB_PATH, ITEM_IMAGES_DIR)
        debug = read_debug(payload)
        run_dir = vision_run_directory(uuid.uuid4().hex) if debug else None
        report = run_inspection(current_image, config, run_dir, debug=debug)
        self.send_json({"report": report}, HTTPStatus.CREATED)

    def _run_rgbd_stockout(self, payload):
        sample = str(payload.get("sample", "")).strip()
        directory = rgbd_sample_directory(sample)
        debug = read_debug(payload)
        output_dir = RGBD_OUTPUT_DIR / uuid.uuid4().hex if debug else None
        report = run_rgbd_stockout(
            directory, DB_PATH, ITEM_IMAGES_DIR, get_rgbd_stockout_config(), output_dir=output_dir, debug=debug
        )
        self.send_json({"report": report}, HTTPStatus.CREATED)

    def _run_rgb_misplacement(self, payload):
        image_data = str(payload.get("image_data", ""))
        if not image_data.startswith("data:image/") or ";base64," not in image_data:
            raise ValueError("image_data must be a base64-encoded JPEG, PNG, or WebP image")
        prefix, encoded = image_data.split(",", 1)
        image_type = prefix.removeprefix("data:").removesuffix(";base64")
        if image_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Only JPEG, PNG, and WebP photos are supported")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
            rgb = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, binascii.Error) as error:
            raise ValueError("image_data is not a valid image") from error
        if not image_bytes or len(image_bytes) > 12 * 1024 * 1024 or rgb is None:
            raise ValueError("Photo must be between 1 byte and 12 MB")
        debug = read_debug(payload)
        output_dir = RGB_MISPLACEMENT_OUTPUT_DIR / uuid.uuid4().hex if debug else None
        report = run_rgb_misplacement(
            rgb, DB_PATH, ITEM_IMAGES_DIR, get_rgb_misplacement_config(), output_dir=output_dir, debug=debug
        )
        self.send_json({"report": report}, HTTPStatus.CREATED)

    def _run_sku_query(self, payload):
        image_data = str(payload.get("image_data", ""))
        if not image_data.startswith("data:image/") or ";base64," not in image_data:
            raise ValueError("image_data must be a base64-encoded JPEG, PNG, or WebP image")
        prefix, encoded = image_data.split(",", 1)
        image_type = prefix.removeprefix("data:").removesuffix(";base64")
        extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(image_type)
        if extension is None:
            raise ValueError("Only JPEG, PNG, and WebP photos are supported")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
        except (ValueError, OSError, binascii.Error) as error:
            raise ValueError("image_data is not a valid image") from error
        if not image_bytes or len(image_bytes) > 12 * 1024 * 1024:
            raise ValueError("Photo must be between 1 byte and 12 MB")
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        provider = str(payload.get("provider", "ark")).strip()
        if provider not in {"ark", "siliconflow", "dashscope", "local"}:
            raise ValueError("provider must be ark, siliconflow, dashscope, or local")
        model = str(payload.get("model", "")).strip()
        if provider != "local" and (not model or len(model) > 200):
            raise ValueError("model must be between 1 and 200 characters")
        config = sku_query_config(payload.get("config", {}))
        debug = read_debug(payload)
        sku, reference_slot_id, reference_path = sku_query_reference(query)
        with ShelfDatabase(DB_PATH) as db:
            sku_info = db.get_sku_info(sku)
        if provider == "local" and (sku_info is None or not sku_info.owlv2_prompt):
            raise ValueError(f"SKU {sku} has no owlv2_prompt; review it before local querying")
        run_dir = sku_query_run_directory(uuid.uuid4().hex) if debug else None
        if debug:
            with tempfile.TemporaryDirectory(prefix="sku-query-") as temporary:
                shelf_path = Path(temporary) / f"shelf{extension}"
                shelf_path.write_bytes(image_bytes)
                if provider == "local":
                    report = run_owlv2_sku_query(
                        sku, sku_info.owlv2_prompt, reference_path, shelf_path, run_dir,
                        max_boxes=config["max_boxes"], owlv2_score_threshold=config["owlv2_score_threshold"],
                        dino_fallback=config["dino_fallback"],
                        dino_confidence_threshold=config["dino_confidence_threshold"], debug=True,
                    )
                else:
                    report = run_vlm_sku_query(
                        sku, reference_path, shelf_path, provider, model, run_dir,
                        max_boxes=config["max_boxes"], dino_fallback=config["dino_fallback"],
                        dino_confidence_threshold=config["dino_confidence_threshold"], debug=True,
                    )
        else:
            if provider == "local":
                report = run_owlv2_sku_query(
                    sku, sku_info.owlv2_prompt, reference_path, image_bytes,
                    max_boxes=config["max_boxes"], owlv2_score_threshold=config["owlv2_score_threshold"],
                    dino_fallback=config["dino_fallback"],
                    dino_confidence_threshold=config["dino_confidence_threshold"], debug=False,
                )
            else:
                report = run_vlm_sku_query(
                    sku, reference_path, image_bytes, provider, model,
                    max_boxes=config["max_boxes"], dino_fallback=config["dino_fallback"],
                    dino_confidence_threshold=config["dino_confidence_threshold"], debug=False,
                )
        report = {**report, "query": query, "reference_slot_id": reference_slot_id}
        if debug:
            (run_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.send_json({"report": report}, HTTPStatus.CREATED)

    def _run_image_stitch(self, payload):
        images = payload.get("images")
        if not isinstance(images, list) or not 2 <= len(images) <= 8:
            raise ValueError("images must contain between 2 and 8 photos")
        main_index = payload.get("main_index", 0)
        if not isinstance(main_index, int) or isinstance(main_index, bool):
            raise ValueError("main_index must be an integer")
        decoded: list[tuple[bytes, str]] = []
        total_bytes = 0
        for image_data in images:
            image_data = str(image_data)
            if not image_data.startswith("data:image/") or ";base64," not in image_data:
                raise ValueError("Every image must be a base64-encoded JPEG, PNG, or WebP photo")
            prefix, encoded = image_data.split(",", 1)
            image_type = prefix.removeprefix("data:").removesuffix(";base64")
            extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(image_type)
            if extension is None:
                raise ValueError("Only JPEG, PNG, and WebP photos are supported")
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
            except (ValueError, OSError, binascii.Error) as error:
                raise ValueError("One uploaded image is not valid") from error
            if not image_bytes or len(image_bytes) > 12 * 1024 * 1024:
                raise ValueError("Each photo must be between 1 byte and 12 MB")
            total_bytes += len(image_bytes)
            if total_bytes > 64 * 1024 * 1024:
                raise ValueError("Combined source photos must not exceed 64 MB")
            decoded.append((image_bytes, extension))
        run_dir = image_stitch_run_directory(uuid.uuid4().hex)
        with tempfile.TemporaryDirectory(prefix="image-stitch-") as temporary:
            paths = []
            for index, (image_bytes, extension) in enumerate(decoded):
                path = Path(temporary) / f"input_{index:02d}{extension}"
                path.write_bytes(image_bytes)
                paths.append(path)
            report = run_image_stitch(paths, run_dir, main_index)
        self.send_json({"report": report}, HTTPStatus.CREATED)

    def _rectify_image_stitch(self, run_id, payload):
        run_dir = image_stitch_run_directory(run_id)
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            self.send_json({"error": "Image stitch run not found"}, HTTPStatus.NOT_FOUND)
            return
        points = payload.get("points")
        if not isinstance(points, list):
            raise ValueError("points must be a list")
        report = rectify_stitched_image(run_dir, json.loads(result_path.read_text(encoding="utf-8")), points)
        self.send_json({"report": report})

    def _apply_vision_run(self, run_id, payload):
        requested = payload.get("slot_ids")
        if not isinstance(requested, list) or not requested or not all(isinstance(slot_id, str) for slot_id in requested):
            raise ValueError("slot_ids must be a non-empty list of slot IDs")
        if len(set(requested)) != len(requested):
            raise ValueError("slot_ids must not contain duplicates")
        report_path = vision_run_directory(run_id) / "result.json"
        if not report_path.is_file():
            self.send_json({"error": "Inspection run not found"}, HTTPStatus.NOT_FOUND)
            return
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = {
            row.get("slot_id"): row
            for row in report.get("slots", [])
            if isinstance(row, dict) and row.get("selected") is True
        }
        if any(slot_id not in rows for slot_id in requested):
            raise ValueError("Every applied slot must be a selected result from this run")
        changes = [(slot_id, rows[slot_id].get("actual_sku")) for slot_id in requested]
        with ShelfDatabase(DB_PATH) as db:
            changed_slots = db.set_actual_sku_batch(changes)
        for shelf_id in {slot.shelf_id for slot in changed_slots}:
            sync_shelf_projection(shelf_id)
        applied = set(report.get("applied_slot_ids", []))
        report["applied_slot_ids"] = sorted(applied | set(requested))
        for row in report.get("slots", []):
            if isinstance(row, dict) and row.get("slot_id") in requested:
                row["selected"] = False
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.send_json({"slots": [asdict(slot) for slot in changed_slots], "state": snapshot()})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json(96 * 1024 * 1024 if path == "/api/image-stitch" else 64 * 1024 * 1024 if path == "/api/skus/batch" else 20 * 1024 * 1024 if path in {"/api/imports/manual", "/api/grounding/products", "/api/vision/inspect", "/api/sku-query", "/api/sku-prompts/owlv2", "/api/rgb-misplacement/runs"} else 64 * 1024)
            if path == "/api/shelves":
                with ShelfDatabase(DB_PATH) as db:
                    shelf_id = db.add_shelf_group(
                        name=str(payload.get("name", "New shelf")),
                        world_x=float(payload.get("world_x", 0)),
                        world_y=float(payload.get("world_y", 0)),
                        yaw=float(payload.get("yaw", 0)),
                        shelf_type_id=payload.get("shelf_type_id"),
                    )
                self.send_json({"id": shelf_id, "state": snapshot()}, HTTPStatus.CREATED)
                return
            if path == "/api/delivery-tables":
                with ShelfDatabase(DB_PATH) as db:
                    table_id = db.add_delivery_table(
                        name=str(payload.get("name", "新交付桌")),
                        world_x=float(payload.get("world_x", 0)),
                        world_y=float(payload.get("world_y", 0)),
                        yaw=float(payload.get("yaw", 0)),
                    )
                self.send_json({"id": table_id, "state": snapshot()}, HTTPStatus.CREATED)
                return
            if path == "/api/skus":
                sku = str(payload.get("sku", "")).strip()
                if not sku:
                    raise ValueError("sku is required")
                reference_image = sku_reference_image(payload)
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_sku_info(sku) is not None:
                        self.send_json({"error": "SKU already exists"}, HTTPStatus.CONFLICT)
                        return
                    reference_image_path = ""
                    if reference_image is not None:
                        reference_image_path = save_sku_reference_image(sku, reference_image)
                    db.register_sku(sku, str(payload.get("category", "")),
                                    str(payload.get("mesh_file", "")), str(payload.get("tex_file", "")),
                                    str(payload.get("owlv2_prompt", "")), reference_image_path,
                                    str(payload.get("grasp_method", "夹爪")),
                                    str(payload.get("qwen_grounding_prompt", "")))
                self.send_json({"state": snapshot()}, HTTPStatus.CREATED)
                return
            if path == "/api/slots":
                shelf_id = read_number(payload, "shelf_id", int, 1)
                face = read_number(payload, "face", int, 0)
                level = read_number(payload, "level", int, 0)
                y_cm = read_number(payload, "y_cm", float, 0)
                expected_sku = str(payload.get("expected_sku", "")).strip()
                if not expected_sku:
                    raise ValueError("expected_sku is required")
                actual_sku = payload.get("actual_sku", expected_sku)
                if actual_sku is not None:
                    actual_sku = str(actual_sku).strip()
                bbox = read_optional_bbox(payload)
                with ShelfDatabase(DB_PATH) as db:
                    slot_id = db.create_slot(
                        shelf_id,
                        face,
                        level,
                        y_cm,
                        expected_sku,
                        actual_sku,
                        read_optional_dimension(payload, "width_cm"),
                        read_optional_dimension(payload, "height_cm"),
                        str(payload.get("image_dir", "")),
                    )
                    slot = db.get_slot_by_id(slot_id)
                sync_shelf_projection(
                    shelf_id, {slot_id: bbox} if bbox is not None else None
                )
                self.send_json(
                    {"slot": asdict(slot), "state": snapshot()}, HTTPStatus.CREATED
                )
                return
            if path.startswith("/api/slots/") and path.endswith("/take"):
                slot_id = unquote(path[len("/api/slots/"):-len("/take")]).strip("/")
                with ShelfDatabase(DB_PATH) as db:
                    slot = db.take_slot(slot_id)
                sync_shelf_projection(slot.shelf_id)
                self.send_json({"slot": asdict(slot), "state": snapshot()})
                return
            if path.startswith("/api/slots/") and path.endswith("/restock"):
                slot_id = unquote(path[len("/api/slots/"):-len("/restock")]).strip("/")
                with ShelfDatabase(DB_PATH) as db:
                    slot = db.restock_slot(slot_id)
                sync_shelf_projection(slot.shelf_id)
                self.send_json({"slot": asdict(slot), "state": snapshot()})
                return
            if path == "/api/imports/manual":
                self._import_manual_batch(payload)
                return
            if path == "/api/sku-prompts/owlv2":
                self._draft_owlv2_prompts(payload)
                return
            if path == "/api/skus/import-image-directory":
                self._import_sku_image_directory(payload)
                return
            if path == "/api/skus/prompts/generate":
                self._generate_sku_prompts(payload)
                return
            if path == "/api/skus/delete":
                self._delete_skus(payload)
                return
            if path == "/api/skus/batch":
                self._save_sku_batch(payload)
                return
            if path == "/api/grounding/products":
                self._ground_products(payload)
                return
            if path == "/api/vision/inspect":
                self._run_vision_inspection(payload)
                return
            if path == "/api/rgbd-stockout/runs":
                self._run_rgbd_stockout(payload)
                return
            if path == "/api/rgb-misplacement/runs":
                self._run_rgb_misplacement(payload)
                return
            if path == "/api/sku-query":
                self._run_sku_query(payload)
                return
            if path == "/api/image-stitch":
                self._run_image_stitch(payload)
                return
            if path.startswith("/api/image-stitch/runs/") and path.endswith("/rectify"):
                run_id = path[len("/api/image-stitch/runs/"):-len("/rectify")].strip("/")
                self._rectify_image_stitch(run_id, payload)
                return
            if path.startswith("/api/vision/runs/") and path.endswith("/apply"):
                run_id = path[len("/api/vision/runs/"):-len("/apply")].strip("/")
                self._apply_vision_run(run_id, payload)
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (OSError, json.JSONDecodeError) as error:
            self.send_json(
                {"error": f"Database updated but JSON sync failed: {error}", "state": snapshot()},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as error:
            self.send_json({"error": str(error)}, HTTPStatus.UNPROCESSABLE_ENTITY)

    def do_PUT(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if path == "/api/vision/config":
                self.send_json({"inspection": save_inspection_config(inspection_config(payload))})
                return
            if path == "/api/sku-query/config":
                self.send_json({"sku_query": save_sku_query_config(sku_query_config(payload))})
                return
            if path.startswith("/api/skus/"):
                sku = unquote(path[len("/api/skus/"):]).strip("/")
                if not sku:
                    raise ValueError("SKU is required")
                reference_image = sku_reference_image(payload)
                with ShelfDatabase(DB_PATH) as db:
                    current = db.get_sku_info(sku)
                    if current is None:
                        self.send_json({"error": "SKU not found"}, HTTPStatus.NOT_FOUND)
                        return
                    reference_image_path = current.reference_image_path
                    if reference_image is not None:
                        reference_image_path = save_sku_reference_image(sku, reference_image)
                    db.update_sku(
                        sku, str(payload.get("category", "")), str(payload.get("mesh_file", "")),
                        str(payload.get("tex_file", "")), str(payload.get("owlv2_prompt", "")),
                        reference_image_path, str(payload.get("grasp_method", current.grasp_method)),
                        str(payload.get("qwen_grounding_prompt", current.qwen_grounding_prompt)),
                    )
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/shelf-types/"):
                type_id = int(path.rsplit("/", 1)[1])
                with ShelfDatabase(DB_PATH) as db:
                    db.update_shelf_type(
                        type_id=type_id,
                        shelf_length=read_number(payload, "shelf_length", float, 0.000001),
                        shelf_width=read_number(payload, "shelf_width", float, 0.000001),
                        shelf_height=read_number(payload, "shelf_height", float, 0.000001),
                        num_levels=read_number(payload, "num_levels", int, 1),
                        bottom_clearance=read_number(payload, "bottom_clearance", float, 0),
                        level_spacing=read_number(payload, "level_spacing", float, 0.000001),
                        panel_thick=read_number(payload, "panel_thick", float, 0.000001),
                        back_thick=read_number(payload, "back_thick", float, 0.000001),
                        shelf_depth_normal=read_number(payload, "shelf_depth_normal", float, 0.000001),
                        shelf_depth_bottom=read_number(payload, "shelf_depth_bottom", float, 0.000001),
                    )
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/shelves/"):
                shelf_id = int(path.rsplit("/", 1)[1])
                allowed = {key: payload[key] for key in ("name", "world_x", "world_y", "yaw", "shelf_type_id") if key in payload}
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_shelf_group(shelf_id) is None:
                        self.send_json({"error": "Shelf not found"}, HTTPStatus.NOT_FOUND)
                        return
                    db.update_shelf_group(shelf_id, **allowed)
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/delivery-tables/"):
                table_id = int(path.rsplit("/", 1)[1])
                allowed = {key: payload[key] for key in ("name", "world_x", "world_y", "yaw") if key in payload}
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_delivery_table(table_id) is None:
                        self.send_json({"error": "Delivery table not found"}, HTTPStatus.NOT_FOUND)
                        return
                    db.update_delivery_table(table_id, **allowed)
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/slots/"):
                slot_id = unquote(path[len("/api/slots/"):]).strip("/")
                immutable = {"slot_id", "shelf_id", "face", "level", "y_cm", "previous"}
                if immutable & payload.keys():
                    raise ValueError("Slot location fields cannot be changed")
                bbox = read_optional_bbox(payload)
                changes = {
                    key: payload[key]
                    for key in ("expected_sku", "actual_sku", "image_dir")
                    if key in payload
                }
                for dimension in ("width_cm", "height_cm"):
                    if dimension in payload:
                        changes[dimension] = read_optional_dimension(payload, dimension)
                with ShelfDatabase(DB_PATH) as db:
                    slot = db.update_slot(slot_id, **changes)
                sync_shelf_projection(
                    slot.shelf_id, {slot_id: bbox} if bbox is not None else None
                )
                self.send_json({"slot": asdict(slot), "state": snapshot()})
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (OSError, json.JSONDecodeError) as error:
            self.send_json(
                {"error": f"Database updated but JSON sync failed: {error}", "state": snapshot()},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except (ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if path.startswith("/api/shelves/") and path.endswith("/inventory"):
                shelf_id = int(path.rsplit("/", 2)[1])
                scope = str(payload.get("scope", ""))
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_shelf_group(shelf_id) is None:
                        self.send_json({"error": "Shelf not found"}, HTTPStatus.NOT_FOUND)
                        return
                    shelf_slots = db.get_shelf_inventory(shelf_id)
                    if scope == "level":
                        face = read_number(payload, "face", int, 0)
                        level = read_number(payload, "level", int, 0)
                        if face not in (0, 1):
                            raise ValueError("face must be 0 or 1")
                        deleted_slots = [slot for slot in shelf_slots if slot.face == face and slot.level == level]
                        removed = db.clear_shelf_face_level(shelf_id, face, level)
                    elif scope == "face":
                        face = read_number(payload, "face", int, 0)
                        if face not in (0, 1):
                            raise ValueError("face must be 0 or 1")
                        deleted_slots = [slot for slot in shelf_slots if slot.face == face]
                        removed = db.clear_shelf_face(shelf_id, face)
                    elif scope == "all":
                        deleted_slots = shelf_slots
                        removed = db.clear_shelf(shelf_id)
                    else:
                        raise ValueError("scope must be level, face, or all")
                sync_shelf_projection(shelf_id)
                self.remove_slot_images(deleted_slots)
                self.send_json({"removed": removed, "state": snapshot()})
                return
            if path.startswith("/api/delivery-tables/"):
                table_id = int(path.rsplit("/", 1)[1])
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_delivery_table(table_id) is None:
                        self.send_json({"error": "Delivery table not found"}, HTTPStatus.NOT_FOUND)
                        return
                    db.remove_delivery_table(table_id)
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/shelves/"):
                shelf_id = int(path.rsplit("/", 1)[1])
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_shelf_group(shelf_id) is None:
                        self.send_json({"error": "Shelf not found"}, HTTPStatus.NOT_FOUND)
                        return
                    deleted_slots = db.get_shelf_inventory(shelf_id)
                    removed = db.remove_shelf_group(shelf_id)
                self.remove_slot_images(deleted_slots)
                self.remove_shelf_images(shelf_id)
                with JSON_SYNC_LOCK:
                    calibration_manager.remove_calibration(shelf_id)
                self.send_json({"removed": removed, "state": snapshot()})
                return
            if path.startswith("/api/slots/"):
                slot_id = unquote(path[len("/api/slots/"):]).strip("/")
                with ShelfDatabase(DB_PATH) as db:
                    deleted_slot = db.delete_slot(slot_id)
                if deleted_slot is None:
                    self.send_json({"error": "Slot not found"}, HTTPStatus.NOT_FOUND)
                    return
                sync_shelf_projection(deleted_slot.shelf_id)
                self.remove_slot_images([deleted_slot])
                self.send_json({"deleted": asdict(deleted_slot), "state": snapshot()})
                return
            if path.startswith("/api/skus/"):
                sku = unquote(path.rsplit("/", 1)[1])
                with ShelfDatabase(DB_PATH) as db:
                    info = db.get_sku_info(sku)
                    db.remove_sku_from_catalog(sku)
                if info is not None and info.reference_image_path:
                    sku_image_path(sku).unlink(missing_ok=True)
                self.send_json({"removed": sku, "state": snapshot()})
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (OSError, json.JSONDecodeError) as error:
            self.send_json(
                {"error": f"Database updated but JSON sync failed: {error}", "state": snapshot()},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except (ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main():
    port = int(os.environ.get("SHELF_API_PORT", "8000"))
    print("Loading DINOv2 model...")
    preload_dino()
    print("Loading OWLv2 model...")
    preload_owlv2()
    server = ThreadingHTTPServer(("127.0.0.1", port), ApiHandler)
    print(f"Warehouse API listening on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
