#!/usr/bin/env python3
"""Local JSON API for the warehouse visual manager."""

import json
import os
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from shelf_database import ShelfDatabase


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SHELF_DB_PATH", os.path.join(BASE_DIR, "shelf_inventory.db"))


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

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("Request body is too large")
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON") from error

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json({"ok": True})
        elif path == "/api/state":
            self.send_json(snapshot())
        else:
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
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
                z_offset_cm = read_number(payload, "z_offset_cm", float)
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
                    db.set_slot(shelf_id, face, level, y_cm, z_offset_cm, sku)
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
                    if scope == "level":
                        face = read_number(payload, "face", int, 0)
                        level = read_number(payload, "level", int, 0)
                        if face not in (0, 1):
                            raise ValueError("face must be 0 or 1")
                        removed = db.clear_shelf_face_level(shelf_id, face, level)
                    elif scope == "face":
                        face = read_number(payload, "face", int, 0)
                        if face not in (0, 1):
                            raise ValueError("face must be 0 or 1")
                        removed = db.clear_shelf_face(shelf_id, face)
                    elif scope == "all":
                        removed = db.clear_shelf(shelf_id)
                    else:
                        raise ValueError("scope must be level, face, or all")
                self.send_json({"removed": removed, "state": snapshot()})
                return
            if path.startswith("/api/shelves/"):
                shelf_id = int(path.rsplit("/", 1)[1])
                with ShelfDatabase(DB_PATH) as db:
                    if db.get_shelf_group(shelf_id) is None:
                        self.send_json({"error": "Shelf not found"}, HTTPStatus.NOT_FOUND)
                        return
                    removed = db.remove_shelf_group(shelf_id)
                self.send_json({"removed": removed, "state": snapshot()})
                return
            if path == "/api/slots":
                with ShelfDatabase(DB_PATH) as db:
                    db.remove_slot(int(payload["shelf_id"]), int(payload["face"]),
                                   int(payload["level"]), float(payload["y_cm"]))
                self.send_json({"state": snapshot()})
                return
            if path.startswith("/api/skus/"):
                sku = path.rsplit("/", 1)[1]
                with ShelfDatabase(DB_PATH) as db:
                    db.remove_sku_from_catalog(sku)
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
