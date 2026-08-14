#!/usr/bin/env python3
"""Explainable stockout detector for one or more visible RGB-D shelf levels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class ShelfGeometry:
    left_x: int
    right_x: int
    surface_y: int
    goods_top_y: int
    surface_slope_px_per_pixel: float
    surface_intercept_px: float
    lower_edge_slope_px_per_pixel: float
    lower_edge_intercept_px: float
    lower_edge_depth_jump_mm: float
    lower_edge_depth_support_ratio: float
    goods_height_px: int


@dataclass
class ShelfDepthModel:
    normal_x: float
    normal_y: float
    normal_z: float
    offset_mm: float
    sample_count: int
    residual_rms_mm: float


@dataclass
class ProductGroupAssessment:
    shelf_index: int
    group_index: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    status: str
    closest_item_setback_mm: float
    shelf_baseline_setback_mm: float
    setback_mm: float


@dataclass
class SkippedShelf:
    location: str
    reason: str
    approximate_y: int


def _odd(value: int) -> int:
    return max(3, value if value % 2 else value + 1)


def _red_shelf_mask(rgb: np.ndarray) -> np.ndarray:
    """Return red shelf-front pixels while rejecting weak reddish texture."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    return (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 100)
        & (hsv[:, :, 2] > 70)
    )


def _fit_line_ransac(
    points: np.ndarray,
    width: int,
    *,
    random_seed: int,
) -> tuple[float, float] | None:
    if len(points) < max(80, int(round(width * 0.15))):
        return None
    rng = np.random.default_rng(random_seed)
    best_mask = None
    best_count = 0
    for _ in range(500):
        pair = points[rng.choice(len(points), 2, replace=False)]
        dx = pair[1, 0] - pair[0, 0]
        if abs(dx) < width * 0.1:
            continue
        slope = (pair[1, 1] - pair[0, 1]) / dx
        if abs(slope) > 0.4:
            continue
        intercept = pair[0, 1] - slope * pair[0, 0]
        mask = np.abs(points[:, 1] - (slope * points[:, 0] + intercept)) < 3.0
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < width * 0.12:
        return None
    slope, intercept = np.polyfit(
        points[best_mask, 0], points[best_mask, 1], 1
    )
    return float(slope), float(intercept)


def _measure_lower_edge_depth_jump(
    depth_mm: np.ndarray,
    left_x: int,
    right_x: int,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    """Measure depth discontinuity support along a candidate lower edge."""
    height, width = depth_mm.shape
    jumps: list[float] = []
    for x in range(max(0, left_x), min(width, right_x + 1)):
        edge_y = int(round(slope * x + intercept))
        above = depth_mm[max(0, edge_y - 8):max(0, edge_y - 2), x]
        below = depth_mm[min(height, edge_y + 3):min(height, edge_y + 11), x]
        above = above[(above > 300) & (above < 2000)]
        below = below[(below > 300) & (below < 2000)]
        if above.size and below.size:
            jumps.append(abs(float(np.median(below) - np.median(above))))
    if not jumps:
        return 0.0, 0.0
    jump_array = np.asarray(jumps, dtype=np.float64)
    significant = jump_array[jump_array >= 20.0]
    if not significant.size:
        return 0.0, 0.0
    return (
        float(np.median(significant)),
        float(len(significant) / len(jump_array)),
    )


def _fit_shelf_line_slope(rgb: np.ndarray, fallback_y: int) -> float:
    """Estimate shelf roll from the long upper edge of the red shelf beam."""
    height, width = rgb.shape[:2]
    red = _red_shelf_mask(rgb)
    search_y0 = max(0, fallback_y - int(round(height * 0.04)))
    search_y1 = min(height, fallback_y + int(round(height * 0.12)))
    points: list[tuple[float, float]] = []
    for x in range(width):
        ys = np.flatnonzero(red[search_y0:search_y1, x])
        if ys.size:
            points.append((float(x), float(search_y0 + ys[0])))
    if len(points) < width * 0.25:
        return 0.0

    samples = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(11)
    best_mask = None
    best_count = 0
    for _ in range(500):
        pair = samples[rng.choice(len(samples), 2, replace=False)]
        dx = pair[1, 0] - pair[0, 0]
        if abs(dx) < width * 0.1:
            continue
        slope = (pair[1, 1] - pair[0, 1]) / dx
        if abs(slope) > 0.4:
            continue
        intercept = pair[0, 1] - slope * pair[0, 0]
        mask = np.abs(samples[:, 1] - (slope * samples[:, 0] + intercept)) < 3.0
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < width * 0.2:
        return 0.0
    slope, _ = np.polyfit(samples[best_mask, 0], samples[best_mask, 1], 1)
    return float(slope)


def detect_shelf_levels(
    rgb: np.ndarray,
    depth_mm: np.ndarray | None = None,
) -> tuple[list[ShelfGeometry], list[SkippedShelf]]:
    """Detect every complete red shelf front and record clipped levels."""
    height, width = rgb.shape[:2]
    red = _red_shelf_mask(rgb)
    closed = cv2.morphologyEx(
        red.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((_odd(max(5, height // 90)), _odd(max(15, width // 40))), np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    candidates: list[tuple[int, int, int, int, int]] = []
    skipped: list[SkippedShelf] = []
    for index in range(1, component_count):
        x, y, component_width, component_height, area = map(int, stats[index])
        if component_width < width * 0.4 or area < width * 8:
            continue
        touches_top = y <= 2
        reaches_bottom = y + component_height >= height - 2
        if touches_top or reaches_bottom:
            skipped.append(
                SkippedShelf(
                    location='top' if touches_top else 'bottom',
                    reason='shelf_front_clipped_by_image_boundary',
                    approximate_y=int(y if touches_top else y + component_height - 1),
                )
            )
            continue
        candidates.append((x, y, component_width, component_height, index))

    shelves: list[ShelfGeometry] = []
    goods_height = int(round(height * 0.27))
    for shelf_index, (x, y, component_width, component_height, label) in enumerate(
        sorted(candidates, key=lambda item: item[1]), start=1
    ):
        # For each image column, the first red pixel of the broad component is
        # the product-support edge. RANSAC tolerates labels and products that
        # locally occlude the beam.
        upper_points: list[tuple[float, float]] = []
        lower_points: list[tuple[float, float]] = []
        for column in range(x, x + component_width):
            rows = np.flatnonzero(labels[:, column] == label)
            if rows.size:
                upper_points.append((float(column), float(rows[0])))
                lower_points.append((float(column), float(rows[-1])))
        upper_fitted = _fit_line_ransac(
            np.asarray(upper_points, dtype=np.float64),
            width,
            random_seed=20 + shelf_index,
        )
        lower_fitted = _fit_line_ransac(
            np.asarray(lower_points, dtype=np.float64),
            width,
            random_seed=40 + shelf_index,
        )
        if upper_fitted is None or lower_fitted is None:
            skipped.append(
                SkippedShelf(
                    location='interior',
                    reason='shelf_front_edge_fit_failed',
                    approximate_y=int(y),
                )
            )
            continue
        slope, intercept = upper_fitted
        lower_slope, lower_intercept = lower_fitted
        center_y = int(round(slope * (width - 1) / 2.0 + intercept))
        if center_y - goods_height < 0:
            skipped.append(
                SkippedShelf(
                    location='top',
                    reason='product_band_clipped_by_image_boundary',
                    approximate_y=center_y,
                )
            )
            continue
        depth_jump_mm = 0.0
        depth_support_ratio = 0.0
        if depth_mm is not None:
            depth_jump_mm, depth_support_ratio = _measure_lower_edge_depth_jump(
                depth_mm,
                x,
                x + component_width - 1,
                lower_slope,
                lower_intercept,
            )
        shelves.append(
            ShelfGeometry(
                left_x=max(0, x),
                right_x=min(width - 1, x + component_width - 1),
                surface_y=center_y,
                goods_top_y=max(0, center_y - goods_height),
                surface_slope_px_per_pixel=slope,
                surface_intercept_px=intercept,
                lower_edge_slope_px_per_pixel=lower_slope,
                lower_edge_intercept_px=lower_intercept,
                lower_edge_depth_jump_mm=round(depth_jump_mm, 1),
                lower_edge_depth_support_ratio=round(depth_support_ratio, 3),
                goods_height_px=goods_height,
            )
        )

    if not shelves:
        # Preserve support for shelves whose front beam is not red.
        shelves = [detect_shelf_geometry(rgb)]

    # Products visible below the lowest complete shelf belong to another level,
    # but no millimetre setback can be measured without its front reference.
    lowest_surface = max(
        shelf.surface_slope_px_per_pixel * (width - 1) / 2.0
        + shelf.surface_intercept_px
        for shelf in shelves
    )
    if lowest_surface + goods_height < height:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        lower_edges = cv2.Canny(gray, 50, 150)
        lower_start = min(height, int(round(lowest_surface + 45)))
        if np.count_nonzero(lower_edges[lower_start:]) > width * 4:
            skipped.append(
                SkippedShelf(
                    location='bottom',
                    reason='products_visible_but_shelf_front_outside_image',
                    approximate_y=height - 1,
                )
            )
    return shelves, skipped


def detect_shelf_geometry(rgb: np.ndarray) -> ShelfGeometry:
    """Detect shelf uprights and the upper edge of the dominant shelf beam."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(60, width // 16),
        minLineLength=max(120, height // 4),
        maxLineGap=max(20, width // 40),
    )
    vertical_x: list[tuple[int, int]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = int(round(np.hypot(x2 - x1, y2 - y1)))
            if abs(angle - 90.0) <= 8.0:
                vertical_x.append(((x1 + x2) // 2, length))

    left_default = int(round(width * 0.09))
    right_default = int(round(width * 0.95))
    left_candidates = [item for item in vertical_x if item[0] < width * 0.35]
    right_candidates = [item for item in vertical_x if item[0] > width * 0.65]
    left_x = max(left_candidates, key=lambda item: item[1])[0] if left_candidates else left_default
    right_x = (
        max(right_candidates, key=lambda item: item[1])[0]
        if right_candidates
        else right_default
    )
    if right_x - left_x < width * 0.4:
        left_x, right_x = left_default, right_default

    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    row_score = np.mean(grad_y[:, left_x:right_x], axis=1)
    row_score -= 0.25 * np.mean(grad_x[:, left_x:right_x], axis=1)
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 9), 0).ravel()

    search_start = int(round(height * 0.45))
    search_end = int(round(height * 0.72))
    scores = row_score[search_start:search_end]
    peak_indices = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] >= scores[index - 1]
        and scores[index] >= scores[index + 1]
    ]
    if peak_indices:
        strongest = max(scores[index] for index in peak_indices)
        # A shelf beam produces several parallel edges. The product support
        # surface is the upper edge of that cluster, not its strongest edge.
        strong_peaks = [
            search_start + index
            for index in peak_indices
            if scores[index] >= strongest * 0.55
        ]
        surface_y = min(strong_peaks)
    else:
        surface_y = search_start + int(np.argmax(scores))

    goods_height = int(round(height * 0.27))
    surface_slope = _fit_shelf_line_slope(rgb, int(surface_y))
    surface_intercept = float(surface_y - surface_slope * (width - 1) / 2.0)
    return ShelfGeometry(
        left_x=int(left_x),
        right_x=int(right_x),
        surface_y=int(surface_y),
        goods_top_y=max(0, int(surface_y - goods_height)),
        surface_slope_px_per_pixel=surface_slope,
        surface_intercept_px=surface_intercept,
        lower_edge_slope_px_per_pixel=surface_slope,
        lower_edge_intercept_px=float(
            surface_intercept + max(58, int(round(height * 0.09)))
        ),
        lower_edge_depth_jump_mm=0.0,
        lower_edge_depth_support_ratio=0.0,
        goods_height_px=goods_height,
    )


def create_product_mask(
    rgb: np.ndarray, depth_mm: np.ndarray, geometry: ShelfGeometry
) -> np.ndarray:
    """Find colorful/bright/textured packaging above the shelf surface."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    colorful = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 145)
    bright = hsv[:, :, 2] > 190
    texture = cv2.Canny(gray, 50, 150)
    texture = cv2.dilate(texture, np.ones((5, 5), np.uint8)) > 0

    mask = colorful | bright | texture
    valid_depth = (depth_mm > 0) & (depth_mm <= 2000)
    mask &= valid_depth

    ys, xs = np.indices((height, width))
    surface_y = (
        geometry.surface_slope_px_per_pixel * xs
        + geometry.surface_intercept_px
    )
    roi = (
        (xs >= geometry.left_x)
        & (xs < geometry.right_x)
        & (ys >= surface_y - geometry.goods_height_px)
        & (ys < surface_y - 8)
    )
    mask &= roi
    kernel_size = _odd(max(5, width // 150))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)


def find_product_groups(
    product_mask: np.ndarray, geometry: ShelfGeometry
) -> list[tuple[int, int]]:
    """Return horizontal packaging groups separated by shelf-back gaps."""
    height, width = product_mask.shape
    occupancy = np.sum(product_mask, axis=0) / max(1, geometry.goods_height_px)
    smooth_width = _odd(max(9, width // 80))
    occupancy = cv2.GaussianBlur(
        occupancy.reshape(1, -1).astype(np.float32), (smooth_width, 1), 0
    ).ravel()
    active = occupancy > 0.08

    close_width = _odd(max(9, width // 100))
    active = cv2.morphologyEx(
        active.astype(np.uint8).reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, close_width), np.uint8),
    ).ravel().astype(bool)

    # A rear product looks narrower under a steep downward view. Four percent
    # keeps those groups while rejecting tiny shelf-edge fragments.
    min_group_width = max(35, int(round(width * 0.04)))
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for x, is_active in enumerate(np.append(active, False)):
        inside_shelf = geometry.left_x <= x < geometry.right_x
        if is_active and inside_shelf and start is None:
            start = x
        elif (not is_active or not inside_shelf) and start is not None:
            if x - start >= min_group_width:
                groups.append((start, x - 1))
            start = None
    return groups


def _depth_to_points(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    z = depth_mm[ys, xs].astype(np.float64)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.column_stack(
        ((xs - cx) * z / fx, (ys - cy) * z / fy, z)
    )


def _place_mask(
    local_mask: np.ndarray,
    image_shape: tuple[int, int],
    y_offset: int,
    x_offset: int,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=bool)
    height, width = local_mask.shape
    mask[y_offset:y_offset + height, x_offset:x_offset + width] = local_mask
    return mask


def fit_shelf_front_plane(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    geometry: ShelfGeometry,
    camera_matrix: np.ndarray,
    *,
    random_seed: int = 7,
) -> ShelfDepthModel:
    """Fit a 3D plane to the shelf front beam in camera coordinates."""
    height, width = depth_mm.shape
    red_shelf = _red_shelf_mask(rgb)
    # Only back-project pixels that visually belong to the red front beam.
    # A rectangular band can contain the horizontal support board when the
    # camera looks down, causing RANSAC to fit the wrong physical plane.
    ys, xs = np.indices((height, width))
    surface_y = (
        geometry.surface_slope_px_per_pixel * xs
        + geometry.surface_intercept_px
    )
    lower_edge_y = (
        geometry.lower_edge_slope_px_per_pixel * xs
        + geometry.lower_edge_intercept_px
    )
    valid = (
        red_shelf
        & (xs >= geometry.left_x)
        & (xs < geometry.right_x)
        & (ys >= surface_y + 2)
        & (ys <= lower_edge_y - 2)
        & (depth_mm > 300)
        & (depth_mm < 1500)
    )
    points = _depth_to_points(
        depth_mm,
        valid,
        camera_matrix,
    )
    if len(points) < 100:
        raise ValueError('Not enough valid shelf-front depth samples')
    rng = np.random.default_rng(random_seed)
    sample = points
    if len(sample) > 40000:
        sample = sample[rng.choice(len(sample), 40000, replace=False)]
    best_count = 0
    best_mask = None
    for _ in range(900):
        trio = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        # The front beam faces the camera. Reject shelf-board planes whose
        # normals point mainly along the camera Y axis under steep pitch.
        if normal[2] < 0.7:
            continue
        offset = -float(normal @ trio[0])
        mask = np.abs(sample @ normal + offset) < 4.0
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < 100:
        raise ValueError('Could not fit the shelf-front 3D plane')
    inliers = sample[best_mask]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    residual = np.abs(sample @ normal + offset)
    refined = sample[residual < 5.0]
    center = refined.mean(axis=0)
    _, _, vh = np.linalg.svd(refined - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    # Make positive signed distances point away from the shelf toward products.
    if normal[2] < 0:
        normal = -normal
        offset = -offset
    residual_rms = float(
        np.sqrt(np.mean(np.square(refined @ normal + offset)))
    )
    return ShelfDepthModel(
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
        normal_z=float(normal[2]),
        offset_mm=float(offset),
        sample_count=int(len(refined)),
        residual_rms_mm=residual_rms,
    )


def assess_product_groups(
    depth_mm: np.ndarray,
    product_mask: np.ndarray,
    geometry: ShelfGeometry,
    groups: list[tuple[int, int]],
    shelf_depth: ShelfDepthModel,
    camera_matrix: np.ndarray,
    setback_threshold_mm: float,
    *,
    shelf_index: int = 1,
    group_index_offset: int = 0,
) -> list[ProductGroupAssessment]:
    """Measure each group's closest stable product instance."""
    assessments: list[ProductGroupAssessment] = []
    normal = np.array(
        [
            shelf_depth.normal_x,
            shelf_depth.normal_y,
            shelf_depth.normal_z,
        ],
        dtype=np.float64,
    )
    for local_group_index, (x_min, x_max) in enumerate(groups, start=1):
        group_index = group_index_offset + local_group_index
        group_full_mask = np.zeros_like(depth_mm, dtype=bool)
        group_full_mask[:, x_min:x_max + 1] = product_mask[:, x_min:x_max + 1]
        values = depth_mm[group_full_mask]
        if values.size < 100:
            continue
        group_points = _depth_to_points(
            depth_mm, group_full_mask, camera_matrix
        )
        distances = group_points @ normal + shelf_depth.offset_mm
        closest_depth = float(np.percentile(distances, 10))
        distance_image = np.zeros_like(depth_mm, dtype=np.float32)
        ys, xs = np.nonzero(group_full_mask)
        distance_image[ys, xs] = distances.astype(np.float32)
        closest_mask = (
            group_full_mask
            & (distance_image >= closest_depth - 10.0)
            & (distance_image <= closest_depth + 35.0)
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            closest_mask.astype(np.uint8), connectivity=8
        )
        components = [
            stats[index]
            for index in range(1, component_count)
            if stats[index, cv2.CC_STAT_AREA] >= 30
        ]
        if not components:
            continue
        component = max(components, key=lambda item: item[cv2.CC_STAT_AREA])
        item_x = int(component[cv2.CC_STAT_LEFT])
        item_y = int(component[cv2.CC_STAT_TOP])
        item_width = int(component[cv2.CC_STAT_WIDTH])
        item_height = int(component[cv2.CC_STAT_HEIGHT])
        boundary_margin = max(3, depth_mm.shape[1] // 200)
        if (
            item_x <= boundary_margin
            or item_x + item_width >= depth_mm.shape[1] - boundary_margin
        ):
            continue
        assessments.append(
            ProductGroupAssessment(
                shelf_index=shelf_index,
                group_index=group_index,
                x_min=item_x,
                y_min=item_y,
                x_max=item_x + item_width - 1,
                y_max=item_y + item_height - 1,
                status='pending',
                closest_item_setback_mm=round(closest_depth, 1),
                shelf_baseline_setback_mm=0.0,
                setback_mm=0.0,
            )
        )
    if assessments:
        raw_setbacks = np.sort(np.asarray(
            [item.closest_item_setback_mm for item in assessments],
            dtype=np.float64,
        ))
        # The front-most half is a robust estimate of normal shelf stocking.
        # Rear/stockout groups cannot pull this baseline backward.
        front_count = max(1, (len(raw_setbacks) + 1) // 2)
        baseline = float(np.median(raw_setbacks[:front_count]))
        for assessment in assessments:
            relative_setback = (
                assessment.closest_item_setback_mm - baseline
            )
            assessment.shelf_baseline_setback_mm = round(baseline, 1)
            assessment.setback_mm = round(relative_setback, 1)
            assessment.status = (
                'stockout_candidate'
                if relative_setback >= setback_threshold_mm
                else 'normal'
            )
    return assessments


def draw_result(
    rgb: np.ndarray,
    shelves: list[tuple[int, ShelfGeometry]],
    assessments: list[ProductGroupAssessment],
    threshold_mm: float,
) -> np.ndarray:
    output = rgb.copy()
    height, _ = output.shape[:2]

    for shelf_index, geometry in shelves:
        surface_left = int(round(
            geometry.surface_slope_px_per_pixel * geometry.left_x
            + geometry.surface_intercept_px
        ))
        surface_right = int(round(
            geometry.surface_slope_px_per_pixel * geometry.right_x
            + geometry.surface_intercept_px
        ))
        lower_left = int(round(
            geometry.lower_edge_slope_px_per_pixel * geometry.left_x
            + geometry.lower_edge_intercept_px
        ))
        lower_right = int(round(
            geometry.lower_edge_slope_px_per_pixel * geometry.right_x
            + geometry.lower_edge_intercept_px
        ))
        cv2.line(
            output,
            (geometry.left_x, surface_left),
            (geometry.right_x, surface_right),
            (255, 255, 0),
            3,
        )
        cv2.line(
            output,
            (geometry.left_x, surface_left - geometry.goods_height_px),
            (geometry.right_x, surface_right - geometry.goods_height_px),
            (255, 180, 0),
            1,
        )
        cv2.line(
            output,
            (geometry.left_x, lower_left),
            (geometry.right_x, lower_right),
            (255, 0, 255),
            2,
        )
        cv2.putText(
            output,
            f'shelf {shelf_index}',
            (geometry.left_x + 8, max(24, surface_left - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    stockout_count = 0
    for assessment in assessments:
        is_stockout = assessment.status == 'stockout_candidate'
        color = (0, 0, 255) if is_stockout else (0, 200, 0)
        thickness = 3 if is_stockout else 1
        cv2.rectangle(
            output,
            (assessment.x_min, assessment.y_min),
            (assessment.x_max, assessment.y_max),
            color,
            thickness,
        )
        if is_stockout:
            stockout_count += 1
            label = (
                f'S{assessment.shelf_index} G{assessment.group_index} '
                f'STOCKOUT +{assessment.setback_mm:.0f}mm'
            )
        else:
            label = (
                f'S{assessment.shelf_index} G{assessment.group_index} normal'
            )
        cv2.putText(
            output,
            label,
            (
                assessment.x_min + 4,
                max(22, assessment.y_min - 8)
                if is_stockout else assessment.y_min + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 if is_stockout else 0.5,
            color,
            2 if is_stockout else 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        (
            f'stockout groups: {stockout_count} | '
            f'relative shelf setback >= {threshold_mm:.0f} mm'
        ),
        (18, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Detect stockout groups from the closest product-to-shelf setback '
            'for every complete shelf level in an RGB-D sample'
        )
    )
    parser.add_argument('--rgb', required=True, type=Path)
    parser.add_argument('--depth', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--threshold-mm', type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise SystemExit(f'Cannot read RGB image: {args.rgb}')
    if depth_mm is None or depth_mm.dtype != np.uint16:
        raise SystemExit(f'Depth must be a readable uint16 PNG: {args.depth}')
    if rgb.shape[:2] != depth_mm.shape:
        raise SystemExit('RGB and depth image dimensions do not match')
    if args.threshold_mm <= 0:
        raise SystemExit('--threshold-mm must be greater than zero')

    geometries, skipped_shelves = detect_shelf_levels(rgb, depth_mm)
    metadata_path = args.metadata or args.depth.with_name(
        args.depth.name.replace('_depth_raw.png', '_metadata.yaml')
    )
    if not metadata_path.exists():
        raise SystemExit(f'Camera metadata is required: {metadata_path}')
    with metadata_path.open('r', encoding='utf-8') as metadata_file:
        metadata = yaml.safe_load(metadata_file) or {}
    camera_k = np.asarray(metadata['camera_info']['k'], dtype=np.float64).reshape(3, 3)
    shelf_results: list[dict[str, object]] = []
    assessments: list[ProductGroupAssessment] = []
    group_index_offset = 0
    detected_group_count = 0
    valid_shelves: list[tuple[int, ShelfGeometry]] = []
    image_ys, image_xs = np.indices(depth_mm.shape)
    red_shelf = _red_shelf_mask(rgb)
    for shelf_index, geometry in enumerate(geometries, start=1):
        product_mask = create_product_mask(rgb, depth_mm, geometry)
        for other_index, other_geometry in enumerate(geometries, start=1):
            if other_index == shelf_index:
                continue
            other_surface_y = (
                other_geometry.surface_slope_px_per_pixel * image_xs
                + other_geometry.surface_intercept_px
            )
            other_lower_y = (
                other_geometry.lower_edge_slope_px_per_pixel * image_xs
                + other_geometry.lower_edge_intercept_px
            )
            other_beam = (
                red_shelf
                & (image_xs >= other_geometry.left_x)
                & (image_xs <= other_geometry.right_x)
                & (image_ys >= other_surface_y - 5)
                & (image_ys <= other_lower_y + 5)
            )
            other_beam = cv2.dilate(
                other_beam.astype(np.uint8),
                np.ones((5, 9), np.uint8),
            ).astype(bool)
            # Remove the actual RGB beam only. A broad rectangular exclusion
            # would erase short rear products under a downward view.
            product_mask &= ~other_beam
        groups = find_product_groups(product_mask, geometry)
        detected_group_count += len(groups)
        try:
            shelf_depth = fit_shelf_front_plane(
                rgb, depth_mm, geometry, camera_k
            )
        except ValueError as error:
            skipped_shelves.append(
                SkippedShelf(
                    location=f'shelf_{shelf_index}',
                    reason=str(error),
                    approximate_y=geometry.surface_y,
                )
            )
            continue
        shelf_assessments = assess_product_groups(
            depth_mm,
            product_mask,
            geometry,
            groups,
            shelf_depth,
            camera_k,
            setback_threshold_mm=args.threshold_mm,
            shelf_index=shelf_index,
            group_index_offset=group_index_offset,
        )
        group_index_offset += len(groups)
        assessments.extend(shelf_assessments)
        valid_shelves.append((shelf_index, geometry))
        shelf_results.append(
            {
                'shelf_index': shelf_index,
                'geometry': asdict(geometry),
                'shelf_front_depth_model': asdict(shelf_depth),
                'dynamic_baseline_setback_mm': (
                    shelf_assessments[0].shelf_baseline_setback_mm
                    if shelf_assessments else None
                ),
                'detected_group_count': len(groups),
                'product_groups': [
                    asdict(assessment) for assessment in shelf_assessments
                ],
            }
        )
    if not shelf_results:
        raise SystemExit('No complete shelf level produced a stable 3D fit')
    output = draw_result(
        rgb, valid_shelves, assessments, args.threshold_mm
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise SystemExit(f'Cannot write output image: {args.output}')

    report_path = args.report or args.output.with_suffix('.yaml')
    report = {
        'mode': 'multi_shelf_plane_3d',
        'relative_setback_threshold_mm': float(args.threshold_mm),
        'dynamic_baseline_method': 'median_of_frontmost_half_per_shelf',
        'shelves': shelf_results,
        'skipped_shelves': [
            asdict(skipped_shelf) for skipped_shelf in skipped_shelves
        ],
        'product_groups': [
            asdict(assessment) for assessment in assessments
        ],
        'limitations': [
            'The shelf-front plane must be visible enough for a stable 3D fit.',
            (
                'Each decision compares the closest RGB-D product instance '
                'with a robust normal-front baseline from the same shelf.'
            ),
            'Every shelf front is fitted independently; normals are not shared.',
            (
                'A clipped level is skipped when its own shelf front is not '
                'visible; another level plane is never reused for it.'
            ),
            'Errors in closest-product segmentation can affect the group status.',
        ],
    }
    with report_path.open('w', encoding='utf-8') as report_file:
        yaml.safe_dump(report, report_file, allow_unicode=True, sort_keys=False)

    print(f'Complete shelf levels: {len(shelf_results)}')
    for shelf_result in shelf_results:
        geometry = shelf_result['geometry']
        depth_model = shelf_result['shelf_front_depth_model']
        print(
            f"  shelf={shelf_result['shelf_index']}, "
            f"surface_y={geometry['surface_y']}, "
            f"plane_rms={depth_model['residual_rms_mm']:.2f} mm"
        )
    print(f'Skipped shelf levels: {len(skipped_shelves)}')
    for skipped_shelf in skipped_shelves:
        print(
            f'  location={skipped_shelf.location}, '
            f'reason={skipped_shelf.reason}'
        )
    print(f'Product groups: {detected_group_count}')
    stockout_groups = [
        item for item in assessments if item.status == 'stockout_candidate'
    ]
    print(f'Stockout groups: {len(stockout_groups)}')
    for assessment in assessments:
        print(
            f'  shelf={assessment.shelf_index}, '
            f'group={assessment.group_index}, '
            f'status={assessment.status}, '
            f'closest_item_setback={assessment.closest_item_setback_mm:.1f} mm, '
            f'setback={assessment.setback_mm:.1f} mm'
        )
    print(f'Wrote: {args.output}')
    print(f'Wrote: {report_path}')


if __name__ == '__main__':
    main()
