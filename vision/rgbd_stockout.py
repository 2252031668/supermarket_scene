"""Callable adapter for the proven D435i rear-row stockout detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any

import numpy as np


SOURCE = Path(__file__).with_name("front_stockout_detector.py")
_DETECTOR: Any | None = None


@dataclass(frozen=True)
class StockoutCandidate:
    shelf_index: int
    group_index: int
    box: dict[str, int]
    setback_mm: float


@dataclass
class StockoutDetection:
    candidates: list[StockoutCandidate]
    skipped_shelves: list[dict[str, Any]]
    overlay: np.ndarray


def _detector() -> Any:
    global _DETECTOR
    if _DETECTOR is None:
        if not SOURCE.is_file():
            raise RuntimeError(f"RGB-D detector source is missing: {SOURCE}")
        spec = spec_from_file_location("openarmx_rgbd_detector", SOURCE)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load RGB-D detector source")
        _DETECTOR = module_from_spec(spec)
        sys.modules[spec.name] = _DETECTOR
        spec.loader.exec_module(_DETECTOR)
    return _DETECTOR


def _detect_assessments(
    rgb: np.ndarray, depth_mm: np.ndarray, camera_k: np.ndarray, threshold_mm: float
) -> tuple[list[Any], list[Any], np.ndarray]:
    detector = _detector()
    geometries, skipped = detector.detect_shelf_levels(rgb, depth_mm)
    all_assessments: list[Any] = []
    valid_shelves: list[tuple[int, Any]] = []
    group_offset = 0
    red_shelf = detector._red_shelf_mask(rgb)
    image_ys, image_xs = np.indices(depth_mm.shape)
    for shelf_index, geometry in enumerate(geometries, start=1):
        product_mask = detector.create_product_mask(rgb, depth_mm, geometry)
        for other_index, other in enumerate(geometries, start=1):
            if other_index == shelf_index:
                continue
            surface = other.surface_slope_px_per_pixel * image_xs + other.surface_intercept_px
            lower = other.lower_edge_slope_px_per_pixel * image_xs + other.lower_edge_intercept_px
            beam = red_shelf & (image_xs >= other.left_x) & (image_xs <= other.right_x) & (image_ys >= surface - 5) & (image_ys <= lower + 5)
            product_mask &= ~detector.cv2.dilate(beam.astype(np.uint8), np.ones((5, 9), np.uint8)).astype(bool)
        groups = detector.find_product_groups(product_mask, geometry)
        try:
            shelf_plane = detector.fit_shelf_front_plane(rgb, depth_mm, geometry, camera_k)
        except ValueError as error:
            skipped.append(detector.SkippedShelf(f"shelf_{shelf_index}", str(error), geometry.surface_y))
            continue
        all_assessments.extend(detector.assess_product_groups(
            depth_mm, product_mask, geometry, groups, shelf_plane, camera_k, threshold_mm,
            shelf_index=shelf_index, group_index_offset=group_offset,
        ))
        group_offset += len(groups)
        valid_shelves.append((shelf_index, geometry))
    if not valid_shelves:
        raise RuntimeError("No complete shelf level produced a stable 3D fit")
    return all_assessments, skipped, detector.draw_result(rgb, valid_shelves, all_assessments, threshold_mm)


def detect_stockout_candidates(
    rgb: np.ndarray, depth_mm: np.ndarray, metadata: dict[str, Any], *, threshold_mm: float = 60.0
) -> StockoutDetection:
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be BGR")
    if depth_mm is None or depth_mm.dtype != np.uint16:
        raise ValueError("Depth image must be uint16 millimetres")
    if rgb.shape[:2] != depth_mm.shape:
        raise ValueError("RGB and depth image dimensions do not match")
    try:
        camera_k = np.asarray(metadata["camera_info"]["k"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Camera metadata must contain a valid 3x3 intrinsic matrix") from error
    if camera_k.size != 9 or not np.isfinite(camera_k).all() or camera_k[0] <= 0 or camera_k[4] <= 0:
        raise ValueError("Camera metadata must contain a valid 3x3 intrinsic matrix")
    assessments, skipped, overlay = _detect_assessments(rgb, depth_mm, camera_k.reshape(3, 3), threshold_mm)
    return StockoutDetection(
        candidates=[
            StockoutCandidate(
                row.shelf_index, row.group_index,
                {"x": row.x_min, "y": row.y_min, "width": row.x_max - row.x_min + 1, "height": row.y_max - row.y_min + 1},
                row.setback_mm,
            )
            for row in assessments if row.status == "stockout_candidate"
        ],
        skipped_shelves=[asdict(row) for row in skipped],
        overlay=overlay,
    )
