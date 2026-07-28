#!/usr/bin/env python3
"""Local JSON API for the warehouse visual manager."""

import base64
import binascii
import json
import os
import shutil
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from shelf_database import ShelfDatabase


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SHELF_DB_PATH", os.path.join(BASE_DIR, "shelf_inventory.db"))
ITEM_IMAGES_DIR = os.path.join(BASE_DIR, "data", "item_images")


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
        skus = [asdict(item) for item in db.get_all_skus()]
        slots = db.get_all_slots_world()
        return {
            "stats": db.get_stats(),
            "shelf_types": shelf_types,
            "shelves": shelves,
            "skus": skus,
            "slots": slots,
        }


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
            return ShelfDatabase.format_slot_id(slot["shelf_id"], slot["face"], slot["level"], slot["y_cm"])
        return ShelfDatabase.format_slot_id(slot.shelf_id, slot.face, slot.level, slot.y_cm)

    def remove_slot_images(self, slots):
        """Remove controlled per-instance image directories after database deletion."""
        for slot in slots:
            slot_id = self.slot_id_for(slot)
            image_dir = os.path.join(ITEM_IMAGES_DIR, slot_id)
            if os.path.isdir(image_dir):
                shutil.rmtree(image_dir, ignore_errors=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json({"ok": True})
        elif path == "/api/state":
            self.send_json(snapshot())
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
                    shelf_id, face, level, y_cm = db.slot_id_str_to_tuple(slot_id)
                except ValueError:
                    self.send_json({"error": "Invalid slot ID"}, HTTPStatus.BAD_REQUEST)
                    return
                slot = db.get_slot(shelf_id, face, level, y_cm)
                position = db.get_slot_world_pos(shelf_id, face, level, y_cm, slot.height_cm if slot else None)
                if slot is None or position is None:
                    self.send_json({"error": "Slot not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json({
                    "slot": {
                        "slot_id_str": slot_id,
                        "shelf_id": shelf_id,
                        "face": face,
                        "level": level,
                        "y_cm": y_cm,
                        "sku": slot.sku,
                        "width_cm": slot.width_cm,
                        "height_cm": slot.height_cm,
                        "image_dir": slot.image_dir,
                        "frame": "map",
                        "world_x": position.x,
                        "world_y": position.y,
                        "world_z": position.z,
                    }
                })
        else:
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _import_manual_batch(self, payload):
        """Stage browser-produced PNG crops, then atomically insert their inventory records."""
        items = payload.get("items")
        new_skus = payload.get("new_skus", [])
        if not isinstance(items, list) or not items:
            raise ValueError("items must be a non-empty list")
        if not isinstance(new_skus, list):
            raise ValueError("new_skus must be a list")

        batch_dir = os.path.join(ITEM_IMAGES_DIR, ".staging", uuid.uuid4().hex)
        staged_dirs = []
        db_items = []
        promoted_dirs = []
        database_imported = False
        try:
            for item in items:
                shelf_id = read_number(item, "shelf_id", int, 1)
                face = read_number(item, "face", int, 0)
                level = read_number(item, "level", int, 0)
                y_cm = read_number(item, "y_cm", float, 0)
                width_cm = read_number(item, "width_cm", float, 0)
                height_cm = read_number(item, "height_cm", float, 0)
                sku = str(item.get("sku", "")).strip()
                if face not in (0, 1) or not sku:
                    raise ValueError("Each imported item needs a valid face and SKU")
                slot_id = ShelfDatabase.format_slot_id(shelf_id, face, level, y_cm)
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
                    "sku": sku, "width_cm": width_cm, "height_cm": height_cm,
                    "image_dir": relative_dir,
                })

            os.makedirs(ITEM_IMAGES_DIR, exist_ok=True)
            with ShelfDatabase(DB_PATH) as db:
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
            with ShelfDatabase(DB_PATH) as db:
                slot_ids = db.import_slots_batch(new_skus, db_items)
            database_imported = True
            self.send_json({"slot_ids": slot_ids, "state": snapshot()}, HTTPStatus.CREATED)
        except Exception:
            if not database_imported:
                for target_dir in promoted_dirs:
                    shutil.rmtree(target_dir, ignore_errors=True)
            raise
        finally:
            if os.path.isdir(batch_dir):
                shutil.rmtree(batch_dir, ignore_errors=True)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json(20 * 1024 * 1024 if path == "/api/imports/manual" else 64 * 1024)
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
            if path == "/api/skus":
                sku = str(payload.get("sku", "")).strip()
                if not sku:
                    raise ValueError("sku is required")
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_sku_info(sku) is not None:
                        self.send_json({"error": "SKU already exists"}, HTTPStatus.CONFLICT)
                        return
                    db.register_sku(sku, str(payload.get("category", "")),
                                    str(payload.get("mesh_file", "")), str(payload.get("tex_file", "")))
                self.send_json({"state": snapshot()}, HTTPStatus.CREATED)
                return
            if path == "/api/imports/manual":
                self._import_manual_batch(payload)
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_PUT(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
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
            if path == "/api/slots":
                shelf_id = read_number(payload, "shelf_id", int, 1)
                face = read_number(payload, "face", int, 0)
                level = read_number(payload, "level", int, 0)
                y_cm = read_number(payload, "y_cm", float, 0)
                width_cm = payload.get("width_cm")
                height_cm = payload.get("height_cm")
                width_cm = None if width_cm in (None, "") else read_number(payload, "width_cm", float, 0)
                height_cm = None if height_cm in (None, "") else read_number(payload, "height_cm", float, 0)
                sku = str(payload.get("sku", "")).strip()
                if face not in (0, 1):
                    raise ValueError("face must be 0 or 1")
                if not sku:
                    raise ValueError("sku is required")
                previous = payload.get("previous")
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_shelf_group(shelf_id) is None:
                        raise ValueError("Shelf does not exist")
                    if db.get_sku_info(sku) is None:
                        raise ValueError("SKU does not exist")
                    if previous:
                        db.remove_slot(int(previous["shelf_id"]), int(previous["face"]),
                                       int(previous["level"]), float(previous["y_cm"]))
                    db.set_slot(shelf_id, face, level, y_cm, sku, width_cm, height_cm,
                                str(payload.get("image_dir", "")))
                self.send_json({"state": snapshot()})
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
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
                self.remove_slot_images(deleted_slots)
                self.send_json({"removed": removed, "state": snapshot()})
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
                self.send_json({"removed": removed, "state": snapshot()})
                return
            if path == "/api/slots":
                with ShelfDatabase(DB_PATH) as db:
                    shelf_id = int(payload["shelf_id"])
                    face = int(payload["face"])
                    level = int(payload["level"])
                    y_cm = float(payload["y_cm"])
                    deleted_slot = db.get_slot(shelf_id, face, level, y_cm)
                    db.remove_slot(shelf_id, face, level, y_cm)
                if deleted_slot is not None:
                    self.remove_slot_images([deleted_slot])
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/skus/"):
                sku = path.rsplit("/", 1)[1]
                with ShelfDatabase(DB_PATH) as db:
                    deleted_slots = [slot for slot in db.get_all_slots_world() if slot["sku"] == sku]
                    db.remove_sku_from_catalog(sku)
                self.remove_slot_images(deleted_slots)
                self.send_json({"state": snapshot()})
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main():
    port = int(os.environ.get("SHELF_API_PORT", "8000"))
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
