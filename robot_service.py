"""Typed, transport-independent operations used by the ROS2 adapter."""

from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import calibration_manager
from shelf_database import ShelfDatabase
from vision.config import get_inspection_config, get_sku_query_config
from vision.cv_restock_position import run_inspection
from vision.vlm_sku_query import run_vlm_sku_query


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(BASE_DIR / "shelf_inventory.db")
_UNSET = object()


@contextmanager
def _database(db_path_or_connection):
    if isinstance(db_path_or_connection, ShelfDatabase):
        yield db_path_or_connection
        return
    with ShelfDatabase(db_path_or_connection) as db:
        yield db


def sync_shelf_projection(db_path: str, shelf_id: int, bbox_by_slot: dict[str, Any] | None = None) -> None:
    """Rebuild the CV projection after a database mutation."""
    bbox_by_slot = bbox_by_slot or {}
    with _database(db_path) as db:
        slots = [asdict(slot) for slot in db.get_shelf_inventory(shelf_id)]
    for slot in slots:
        if slot["slot_id"] in bbox_by_slot:
            slot["bbox"] = bbox_by_slot[slot["slot_id"]]
    calibration_manager.sync_shelf_slots(shelf_id, slots)


def slot_dict(slot) -> dict[str, Any]:
    return asdict(slot)


def get_slot_pose(db_path: str | ShelfDatabase, slot_id: str) -> dict[str, Any]:
    with _database(db_path) as db:
        slot = db.get_slot_by_id(slot_id)
        if slot is None:
            raise ValueError(f"Slot {slot_id} does not exist")
        shelf = db.get_shelf_group(slot.shelf_id)
        position = db.get_slot_world_pos(
            slot.shelf_id, slot.face, slot.level, slot.y_cm, slot.height_cm
        )
    if shelf is None or position is None:
        raise ValueError(f"Shelf for slot {slot_id} does not exist")
    return {
        **slot_dict(slot),
        "frame_id": "map",
        "world_x": position.x,
        "world_y": position.y,
        "world_z": position.z,
        "shelf_yaw": shelf.yaw,
    }


def _slot_poses(db_path: str | ShelfDatabase, slots: list[Any]) -> list[dict[str, Any]]:
    with _database(db_path) as db:
        result = []
        for slot in slots:
            shelf = db.get_shelf_group(slot.shelf_id)
            position = db.get_slot_world_pos(
                slot.shelf_id, slot.face, slot.level, slot.y_cm, slot.height_cm
            )
            if shelf is None or position is None:
                raise ValueError(f"Shelf for slot {slot.slot_id} does not exist")
            result.append({
                **slot_dict(slot),
                "frame_id": "map",
                "world_x": position.x,
                "world_y": position.y,
                "world_z": position.z,
                "shelf_yaw": shelf.yaw,
            })
        return result


def get_sku_locations(db_path: str, sku: str) -> dict[str, Any]:
    sku = sku.strip()
    with _database(db_path) as db:
        if not sku or db.get_sku_info(sku) is None:
            raise ValueError(f"SKU {sku} does not exist")
        slots = [slot for slot in db.get_all_slots() if slot.actual_sku == sku]
        positions = _slot_poses(db, slots)
    return {"sku": sku, "count": len(positions), "positions": positions}


def list_shortages(db_path: str) -> list[dict[str, Any]]:
    with _database(db_path) as db:
        slots = db.get_shortage_slots()
    return _slot_poses(db_path, slots)


def list_misplacements(db_path: str) -> list[dict[str, Any]]:
    with _database(db_path) as db:
        slots = db.get_misplaced_slots()
    return _slot_poses(db_path, slots)


def take_slot(db_path: str, slot_id: str) -> dict[str, Any]:
    with _database(db_path) as db:
        slot = db.take_slot(slot_id)
    sync_shelf_projection(db_path, slot.shelf_id)
    return slot_dict(slot)


def restock_slot(db_path: str, slot_id: str) -> dict[str, Any]:
    with _database(db_path) as db:
        slot = db.restock_slot(slot_id)
    sync_shelf_projection(db_path, slot.shelf_id)
    return slot_dict(slot)


def update_slot(
    db_path: str,
    slot_id: str,
    *,
    expected_sku: Any = _UNSET,
    actual_sku: Any = _UNSET,
) -> dict[str, Any]:
    changes = {}
    if expected_sku is not _UNSET:
        changes["expected_sku"] = expected_sku
    if actual_sku is not _UNSET:
        changes["actual_sku"] = actual_sku
    with _database(db_path) as db:
        slot = db.update_slot(slot_id, **changes)
    sync_shelf_projection(db_path, slot.shelf_id)
    return slot_dict(slot)


def inspect_shelf(image, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run production inspection without creating debug artifacts."""
    return run_inspection(image, config or get_inspection_config(), debug=False)


def _sku_reference(db_path: str, query: str) -> tuple[str, str, Path]:
    with _database(db_path) as db:
        slot = db.get_slot_by_id(query)
        if slot is not None:
            candidates = [slot]
        elif db.get_sku_info(query) is not None:
            candidates = [
                item for item in db.get_all_slots()
                if item.expected_sku == query
            ]
        else:
            raise ValueError("Query must be an existing slot ID or SKU")
    for slot in candidates:
        path = BASE_DIR / "data" / "item_images" / slot.slot_id / "0.png"
        if path.is_file():
            return slot.expected_sku, slot.slot_id, path
    raise ValueError("No reference image is available for this slot or SKU")


def find_sku(
    db_path: str,
    image_bytes: bytes,
    query: str,
    provider: str,
    model: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sku, reference_slot_id, reference_path = _sku_reference(db_path, query.strip())
    options = config or get_sku_query_config()
    report = run_vlm_sku_query(
        sku,
        reference_path,
        image_bytes,
        provider,
        model,
        max_boxes=int(options["max_boxes"]),
        dino_fallback=bool(options["dino_fallback"]),
        dino_confidence_threshold=float(options["dino_confidence_threshold"]),
        debug=False,
    )
    return {
        **report,
        "query": query,
        "reference_slot_id": reference_slot_id,
    }
