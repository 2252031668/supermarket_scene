#!/usr/bin/env python3
"""Capture one synchronized RGB/depth sample and camera calibration."""

from __future__ import annotations

from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class RgbDepthCapture(Node):
    """Save one approximately synchronized color/depth pair."""

    def __init__(self) -> None:
        super().__init__("rgb_depth_capture")
        self.declare_parameter("color_topic", "/head_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/head_camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/head_camera/color/camera_info")
        self.declare_parameter("output_dir", ".")
        self.declare_parameter("sample_name", "front_view")
        self.declare_parameter("max_depth_m", 2.0)
        self.declare_parameter("annotation", "")

        self._output_dir = Path(str(self.get_parameter("output_dir").value))
        self._sample_name = str(self.get_parameter("sample_name").value)
        self._max_depth_m = float(self.get_parameter("max_depth_m").value)
        self._annotation = str(self.get_parameter("annotation").value)
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._saved = False

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        color_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("color_topic").value), qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("depth_topic").value), qos_profile=qos_profile_sensor_data
        )
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=20, slop=0.1
        )
        synchronizer.registerCallback(self._image_callback)
        self._subscribers = (color_sub, depth_sub, synchronizer)
        self.get_logger().info("Waiting for synchronized RGB, depth, and camera info")

    @property
    def saved(self) -> bool:
        return self._saved

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _image_callback(self, color_msg: Image, depth_msg: Image) -> None:
        if self._saved or self._camera_info is None:
            return
        color = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        raw_depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":
            depth_mm = raw_depth.astype(np.uint16)
        elif depth_msg.encoding == "32FC1":
            depth_mm = np.nan_to_num(raw_depth * 1000.0, nan=0.0)
            depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            self.get_logger().error(f"Unsupported depth encoding: {depth_msg.encoding}")
            return
        self._save_sample(color, depth_mm, color_msg, depth_msg)
        self._saved = True
        self.get_logger().info(f"Saved sample to: {self._output_dir.resolve()}")

    def _save_sample(self, color: np.ndarray, depth_mm: np.ndarray, color_msg: Image, depth_msg: Image) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._output_dir / self._sample_name
        depth_color = self._colorize_depth(depth_mm)
        combined = np.hstack((color, depth_color))
        cv2.putText(combined, "RGB", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(combined, f"Depth 0-{self._max_depth_m:.1f} m", (color.shape[1] + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        paths = {
            "combined": Path(f"{prefix}.png"),
            "rgb": Path(f"{prefix}_rgb.png"),
            "depth_raw": Path(f"{prefix}_depth_raw.png"),
            "depth_color": Path(f"{prefix}_depth_color.png"),
            "metadata": Path(f"{prefix}_metadata.yaml"),
        }
        for key, image in (("combined", combined), ("rgb", color), ("depth_raw", depth_mm), ("depth_color", depth_color)):
            if not cv2.imwrite(str(paths[key]), image):
                raise OSError(f"Failed to write {paths[key]}")
        info = self._camera_info
        metadata = {
            "sample_name": self._sample_name,
            "annotation": self._annotation,
            "combined_file": paths["combined"].name,
            "color": {"file": paths["rgb"].name, "encoding": "bgr8", "width": int(color_msg.width), "height": int(color_msg.height), "stamp_sec": int(color_msg.header.stamp.sec), "stamp_nanosec": int(color_msg.header.stamp.nanosec)},
            "depth": {"file": paths["depth_raw"].name, "visualization_file": paths["depth_color"].name, "encoding": "16UC1", "unit": "millimeter", "visualization_range_m": [0.0, self._max_depth_m], "outside_visualization_range": "black", "width": int(depth_msg.width), "height": int(depth_msg.height), "stamp_sec": int(depth_msg.header.stamp.sec), "stamp_nanosec": int(depth_msg.header.stamp.nanosec)},
            "camera_info": {"frame_id": info.header.frame_id, "distortion_model": info.distortion_model, "d": [float(value) for value in info.d], "k": [float(value) for value in info.k], "r": [float(value) for value in info.r], "p": [float(value) for value in info.p]},
        }
        with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
            yaml.safe_dump(metadata, metadata_file, allow_unicode=True, sort_keys=False)

    def _colorize_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        max_depth_mm = self._max_depth_m * 1000.0
        valid = (depth_mm > 0) & (depth_mm <= max_depth_mm)
        depth_u8 = cv2.convertScaleAbs(np.clip(depth_mm, 0, max_depth_mm), alpha=255.0 / max_depth_mm)
        depth_color = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        return depth_color


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RgbDepthCapture()
    try:
        while rclpy.ok() and not node.saved:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
