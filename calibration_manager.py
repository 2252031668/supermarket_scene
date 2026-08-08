import hashlib
import json
import os
import tempfile
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(BASE_DIR, "data", "shelf_calibration")
SHELF_IMAGES_DIR = os.path.join(BASE_DIR, "data", "shelf_images")


def _calibration_path(shelf_id: int) -> str:
    return os.path.join(CALIBRATION_DIR, f"{shelf_id}.json")


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's content."""
    if not os.path.isfile(file_path):
        return ""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def shelf_image_filename(face: int) -> str:
    if face == 0:
        return "-x_0.png"
    if face == 1:
        return "+x_0.png"
    raise ValueError("face must be 0 or 1")


def _shelf_image_path(shelf_id: int, face: int) -> str:
    return os.path.join(SHELF_IMAGES_DIR, str(shelf_id), shelf_image_filename(face))


def _load_for_update(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_calibration(
    shelf_id: int,
    shelf_name: str,
    face: int,
    layers: dict[str, Any],
    slots: list[dict[str, Any]],
) -> None:
    """Save one face's image calibration and fixed-slot projection."""
    path = _calibration_path(shelf_id)
    data = _load_for_update(path)
    data["schema_version"] = 2
    data["shelf_id"] = shelf_id
    data["shelf_name"] = shelf_name
    data.setdefault("faces", {})

    image_file = _shelf_image_path(shelf_id, face)
    image_hash = _compute_file_hash(image_file)

    previous_face = data["faces"].get(str(face), {})
    previous_slots = {
        slot["slot_id"]: slot
        for slot in previous_face.get("slots", [])
        if "slot_id" in slot
    }
    merged_slots = dict(previous_slots)
    for slot in slots:
        merged = dict(previous_slots.get(slot.get("slot_id"), {}))
        merged.update(slot)
        merged_slots[slot["slot_id"]] = merged
    merged_layers = dict(previous_face.get("layers", {}))
    merged_layers.update(layers)
    face_data = {
        "image_file": os.path.relpath(image_file, BASE_DIR) if os.path.isfile(image_file) else "",
        "image_hash": image_hash,
        "layers": merged_layers,
        "slots": list(merged_slots.values()),
    }
    data["faces"][str(face)] = face_data
    _write_json_atomic(path, data)


def get_calibration(shelf_id: int) -> dict[str, Any] | None:
    """Load calibration data for a shelf."""
    path = _calibration_path(shelf_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_calibrated_shelves() -> list[int]:
    """Return a list of shelf IDs that have calibration files."""
    shelves = []
    if not os.path.isdir(CALIBRATION_DIR):
        return shelves
    for filename in os.listdir(CALIBRATION_DIR):
        if filename.endswith(".json"):
            try:
                shelf_id = int(filename.replace(".json", ""))
                shelves.append(shelf_id)
            except ValueError:
                pass
    return shelves


def validate_calibration(shelf_id: int, face: int) -> bool:
    """Check if the calibration image hash matches the current shelf image."""
    cal_data = get_calibration(shelf_id)
    if not cal_data or "faces" not in cal_data:
        return False
    face_key = str(face)
    if face_key not in cal_data["faces"]:
        return False
    face_data = cal_data["faces"][face_key]
    expected_hash = face_data.get("image_hash", "")
    if not expected_hash:
        return False
    current_hash = _compute_file_hash(_shelf_image_path(shelf_id, face))
    return expected_hash == current_hash


def remove_calibration(shelf_id: int) -> None:
    """Delete the entire calibration file for a shelf."""
    path = _calibration_path(shelf_id)
    if os.path.isfile(path):
        os.remove(path)


def remove_face_calibration(shelf_id: int, face: int) -> None:
    """Remove calibration data for a specific face."""
    cal_data = get_calibration(shelf_id)
    if not cal_data or "faces" not in cal_data:
        return
    face_key = str(face)
    if face_key in cal_data["faces"]:
        del cal_data["faces"][face_key]
        _write_json_atomic(_calibration_path(shelf_id), cal_data)


def sync_shelf_slots(shelf_id: int, slots: list[dict[str, Any]]) -> None:
    """Project current SQLite slot state while preserving CV calibration data."""
    path = _calibration_path(shelf_id)
    data = _load_for_update(path)
    data["schema_version"] = 2
    data["shelf_id"] = shelf_id
    data.setdefault("shelf_name", f"{shelf_id}号货架")
    faces = data.setdefault("faces", {})

    grouped: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        face_key = str(slot["face"])
        old_face = faces.get(face_key, {})
        old_by_id = {item["slot_id"]: item for item in old_face.get("slots", [])}
        existing = old_by_id.get(slot["slot_id"], {})
        projected = {
            "slot_id": slot["slot_id"],
            "expected_sku": slot["expected_sku"],
            "actual_sku": slot.get("actual_sku"),
        }
        bbox = slot.get("bbox", existing.get("bbox"))
        if bbox is not None:
            projected["bbox"] = bbox
        grouped.setdefault(face_key, []).append(projected)

    for face_key in set(faces) | set(grouped):
        face_data = faces.setdefault(
            face_key,
            {"image_file": "", "image_hash": "", "layers": {}},
        )
        face_data["slots"] = grouped.get(face_key, [])
    _write_json_atomic(path, data)
