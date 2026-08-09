"""ROS2 typed service adapter for the supermarket scene manager."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _add_project_root() -> Path:
    configured = os.environ.get("SUPERMARKET_SCENE_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "robot_service.py").is_file():
            root = candidate.resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise RuntimeError(
        "Cannot find the supermarket_scene repository. "
        "Set SUPERMARKET_SCENE_ROOT to the repository root."
    )


PROJECT_ROOT = _add_project_root()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from supermarket_scene_interfaces.msg import DetectedBox, InspectionSlot, Slot, SlotPose
from supermarket_scene_interfaces.srv import (
    CreateDeliveryTable,
    CreateShelf,
    CreateSlot,
    FindSku,
    GetSkuLocations,
    GetSlotPose,
    InspectShelf,
    ListSlots,
    RestockSlot,
    TakeSlot,
    UpdateDeliveryTable,
    UpdateShelf,
    UpdateSlot,
)

import robot_service
from shelf_database import ShelfDatabase
from vision.config import get_inspection_config, get_sku_query_config
from vision.dino import preload_dino


class SupermarketSceneNode(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_scene_node")
        inspection = get_inspection_config()
        query = get_sku_query_config()
        self.declare_parameter("db_path", str(PROJECT_ROOT / "shelf_inventory.db"))
        self.declare_parameter("provider", "ark")
        self.declare_parameter("model", "")
        for key, value in inspection.items():
            self.declare_parameter(f"inspection.{key}", value)
        for key, value in query.items():
            self.declare_parameter(f"sku_query.{key}", value)

        self.create_service(InspectShelf, "inspect_shelf", self.inspect_shelf)
        self.create_service(FindSku, "find_sku", self.find_sku)
        self.create_service(GetSlotPose, "get_slot_pose", self.get_slot_pose)
        self.create_service(GetSkuLocations, "get_sku_locations", self.get_sku_locations)
        self.create_service(ListSlots, "list_shortages", self.list_shortages)
        self.create_service(ListSlots, "list_misplacements", self.list_misplacements)
        self.create_service(TakeSlot, "take_slot", self.take_slot)
        self.create_service(RestockSlot, "restock_slot", self.restock_slot)
        self.create_service(UpdateSlot, "update_slot", self.update_slot)
        self.create_service(CreateShelf, "create_shelf", self.create_shelf)
        self.create_service(UpdateShelf, "update_shelf", self.update_shelf)
        self.create_service(CreateDeliveryTable, "create_delivery_table", self.create_delivery_table)
        self.create_service(UpdateDeliveryTable, "update_delivery_table", self.update_delivery_table)
        self.create_service(CreateSlot, "create_slot", self.create_slot)

    @property
    def db_path(self) -> str:
        return str(self.get_parameter("db_path").value)

    def inspection_config(self) -> dict:
        return {
            key: self.get_parameter(f"inspection.{key}").value
            for key in get_inspection_config()
        }

    def sku_query_config(self) -> dict:
        return {
            key: self.get_parameter(f"sku_query.{key}").value
            for key in get_sku_query_config()
        }

    @staticmethod
    def image_bytes(message: CompressedImage) -> bytes:
        data = bytes(message.data)
        if not data:
            raise ValueError("image is empty")
        if cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
            raise ValueError("image cannot be decoded")
        return data

    @staticmethod
    def set_slot(message: Slot, value: dict) -> None:
        message.slot_id = value["slot_id"]
        message.shelf_id = value["shelf_id"]
        message.face = value["face"]
        message.level = value["level"]
        message.y_cm = value["y_cm"]
        message.expected_sku = value["expected_sku"]
        message.has_actual_sku = value["actual_sku"] is not None
        message.actual_sku = value["actual_sku"] or ""
        message.status = value["status"]

    @classmethod
    def set_pose(cls, message: SlotPose, value: dict) -> None:
        cls.set_slot(message.slot, value)
        message.frame_id = value["frame_id"]
        message.world_x = value["world_x"]
        message.world_y = value["world_y"]
        message.world_z = value["world_z"]
        message.shelf_yaw = value["shelf_yaw"]

    @staticmethod
    def fail(response, error: Exception):
        response.success = False
        response.error = str(error)
        return response

    def inspect_shelf(self, request, response):
        try:
            image = cv2.imdecode(
                np.frombuffer(self.image_bytes(request.image), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            report = robot_service.inspect_shelf(image, self.inspection_config())
            response.success = True
            response.error = ""
            response.shelf_id = report["shelf_id"]
            response.face = report["face"]
            for row in report["slots"]:
                item = InspectionSlot()
                self.set_slot(item.slot, row)
                item.source = row.get("source", "")
                item.reason = row.get("reason", "")
                item.has_confidence = row.get("confidence") is not None
                item.confidence = row.get("confidence") or 0.0
                bbox = row.get("bbox", {})
                item.x = bbox.get("x", 0)
                item.y = bbox.get("y", 0)
                item.width = bbox.get("width", 0)
                item.height = bbox.get("height", 0)
                response.anomalies.append(item)
            return response
        except Exception as error:
            return self.fail(response, error)

    def find_sku(self, request, response):
        try:
            report = robot_service.find_sku(
                self.db_path,
                self.image_bytes(request.image),
                request.query,
                str(self.get_parameter("provider").value),
                str(self.get_parameter("model").value),
                self.sku_query_config(),
            )
            response.success = True
            response.error = ""
            for key in ("sku", "reference_slot_id", "provider", "model", "request_id"):
                setattr(response, key, report.get(key) or "")
            response.request_seconds = report.get("request_seconds", 0.0)
            response.total_seconds = report.get("total_seconds", 0.0)
            for index, box in enumerate(report.get("detected_boxes", []), start=1):
                item = DetectedBox()
                item.index = index
                item.sku = box.get("label") or report["sku"]
                x1, y1, x2, y2 = box["pixels"]
                item.x, item.y = x1, y1
                item.width, item.height = x2 - x1, y2 - y1
                item.has_confidence = False
                response.boxes.append(item)
            return response
        except Exception as error:
            return self.fail(response, error)

    def get_slot_pose(self, request, response):
        try:
            self.set_pose(response.slot, robot_service.get_slot_pose(self.db_path, request.slot_id))
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def get_sku_locations(self, request, response):
        try:
            result = robot_service.get_sku_locations(self.db_path, request.sku)
            response.success, response.error = True, ""
            response.sku, response.count = result["sku"], result["count"]
            for value in result["positions"]:
                pose = SlotPose()
                self.set_pose(pose, value)
                response.positions.append(pose)
            return response
        except Exception as error:
            return self.fail(response, error)

    def _list_slots(self, request, response, loader):
        try:
            values = loader(self.db_path)
            response.success, response.error = True, ""
            response.count = len(values)
            for value in values:
                pose = SlotPose()
                self.set_pose(pose, value)
                response.slots.append(pose)
            return response
        except Exception as error:
            return self.fail(response, error)

    def list_shortages(self, request, response):
        return self._list_slots(request, response, robot_service.list_shortages)

    def list_misplacements(self, request, response):
        return self._list_slots(request, response, robot_service.list_misplacements)

    def take_slot(self, request, response):
        try:
            self.set_slot(response.slot, robot_service.take_slot(self.db_path, request.slot_id))
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def restock_slot(self, request, response):
        try:
            self.set_slot(response.slot, robot_service.restock_slot(self.db_path, request.slot_id))
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def update_slot(self, request, response):
        try:
            expected = request.expected_sku if request.update_expected_sku else robot_service._UNSET
            actual = robot_service._UNSET
            if request.update_actual_sku:
                actual = None if request.actual_sku_is_null else request.actual_sku
            value = robot_service.update_slot(
                self.db_path, request.slot_id, expected_sku=expected, actual_sku=actual
            )
            self.set_slot(response.slot, value)
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def create_shelf(self, request, response):
        try:
            with ShelfDatabase(self.db_path) as db:
                response.shelf_id = db.add_shelf_group(
                    request.name, request.world_x, request.world_y, request.yaw,
                    request.shelf_type_id if request.has_shelf_type_id else None,
                )
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def update_shelf(self, request, response):
        try:
            changes = {}
            if request.update_name:
                changes["name"] = request.name
            if request.update_pose:
                changes.update(world_x=request.world_x, world_y=request.world_y, yaw=request.yaw)
            if request.update_shelf_type_id and request.has_shelf_type_id:
                changes["shelf_type_id"] = request.shelf_type_id
            with ShelfDatabase(self.db_path) as db:
                if db.get_shelf_group(request.shelf_id) is None:
                    raise ValueError(f"Shelf {request.shelf_id} does not exist")
                db.update_shelf_group(request.shelf_id, **changes)
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def create_delivery_table(self, request, response):
        try:
            with ShelfDatabase(self.db_path) as db:
                response.table_id = db.add_delivery_table(
                    request.name, request.world_x, request.world_y, request.yaw
                )
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def update_delivery_table(self, request, response):
        try:
            changes = {}
            if request.update_name:
                changes["name"] = request.name
            if request.update_pose:
                changes.update(world_x=request.world_x, world_y=request.world_y, yaw=request.yaw)
            with ShelfDatabase(self.db_path) as db:
                if db.get_delivery_table(request.table_id) is None:
                    raise ValueError(f"Delivery table {request.table_id} does not exist")
                db.update_delivery_table(request.table_id, **changes)
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)

    def create_slot(self, request, response):
        try:
            actual = request.actual_sku if request.has_actual_sku else None
            with ShelfDatabase(self.db_path) as db:
                slot_id = db.create_slot(
                    request.shelf_id, request.face, request.level, request.y_cm,
                    request.expected_sku, actual,
                    request.width_cm or None, request.height_cm or None,
                )
                slot = db.get_slot_by_id(slot_id)
            robot_service.sync_shelf_projection(self.db_path, request.shelf_id)
            self.set_slot(response.slot, robot_service.slot_dict(slot))
            response.success, response.error = True, ""
            return response
        except Exception as error:
            return self.fail(response, error)


def main(args=None) -> None:
    rclpy.init(args=args)
    preload_dino()
    node = SupermarketSceneNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

