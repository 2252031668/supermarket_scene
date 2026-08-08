"""Build one frontal shelf mosaic from overlapping photographs in any order."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.reference_photo_align import (
    AlignmentError,
    create_feature_extractor,
    keep_strongest,
    mutual_ratio_matches,
    prepared_gray,
    resize_for_matching,
)


def pair_homography(source: np.ndarray, destination: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return the source-to-destination planar transform with conservative quality gates."""
    source_small, source_scale = resize_for_matching(source, 1800)
    destination_small, destination_scale = resize_for_matching(destination, 1800)
    extractor, norm = create_feature_extractor("sift", 7000)
    source_keypoints, source_descriptors = extractor.detectAndCompute(prepared_gray(source_small, True), None)
    destination_keypoints, destination_descriptors = extractor.detectAndCompute(prepared_gray(destination_small, True), None)
    source_keypoints, source_descriptors = keep_strongest(source_keypoints, source_descriptors, 7000)
    destination_keypoints, destination_descriptors = keep_strongest(destination_keypoints, destination_descriptors, 7000)
    matches = mutual_ratio_matches(source_descriptors, destination_descriptors, 0.72, norm)
    if len(matches) < 12:
        raise AlignmentError(f"Only {len(matches)} mutual SIFT matches; adjacent photos need more overlap")
    source_points = np.float32([source_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    destination_points = np.float32([destination_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.findHomography(
        source_points, destination_points, method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=4.0, maxIters=10000, confidence=0.999,
    )
    if matrix is None or inlier_mask is None:
        raise AlignmentError("USAC/MAGSAC could not align adjacent photos")
    inliers = int(inlier_mask.reshape(-1).sum())
    if inliers < 12 or inliers / len(matches) < 0.24:
        raise AlignmentError("Adjacent-photo alignment did not pass the inlier quality gate")
    return np.linalg.inv(destination_scale) @ matrix @ source_scale, {
        "matches": len(matches), "inliers": inliers, "inlier_ratio": round(inliers / len(matches), 4),
    }


def mosaic_bounds(images: list[np.ndarray], transforms: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    corners = []
    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]
        corners.append(cv2.perspectiveTransform(np.float32([[[0, 0], [width, 0], [width, height], [0, height]]]), transform))
    all_corners = np.concatenate(corners, axis=1).reshape(-1, 2)
    minimum = np.floor(all_corners.min(axis=0)).astype(int)
    maximum = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    return translation, int(maximum[0] - minimum[0]), int(maximum[1] - minimum[1])


def alignment_tree(images: list[np.ndarray], main_index: int) -> tuple[list[dict[str, Any]], list[int]]:
    if not 0 <= main_index < len(images):
        raise ValueError("main_index must identify an uploaded photo")
    parent = list(range(len(images)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = []
    for source_index in range(1, len(images)):
        for destination_index in range(source_index):
            try:
                matrix, metrics = pair_homography(images[source_index], images[destination_index])
            except AlignmentError:
                continue
            edges.append({
                "source_index": source_index,
                "destination_index": destination_index,
                "matrix": matrix,
                **metrics,
            })

    tree = []
    for edge in sorted(edges, key=lambda item: (item["inliers"], item["inlier_ratio"], item["matches"]), reverse=True):
        source_root = find(edge["source_index"])
        destination_root = find(edge["destination_index"])
        if source_root == destination_root:
            continue
        parent[source_root] = destination_root
        tree.append(edge)

    groups: dict[int, list[int]] = {}
    for index in range(len(images)):
        groups.setdefault(find(index), []).append(index)
    used_indices = groups[find(main_index)]
    if len(used_indices) < 2:
        raise AlignmentError("Selected main photo has no reliable alignment with another uploaded photo")
    return tree, sorted(used_indices)


def overlap_residual(reference: np.ndarray, reference_mask: np.ndarray,
                     candidate: np.ndarray, candidate_mask: np.ndarray) -> float:
    overlap = (reference_mask > 0) & (candidate_mask > 0)
    if int(overlap.sum()) < 500:
        return float("inf")
    difference = cv2.absdiff(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )
    return float(np.median(difference[overlap]))


def compose_stitch(images: list[np.ndarray], render_indices: list[int],
                   transforms_by_index: dict[int, np.ndarray], parent_by_index: dict[int, int | None],
                   translation: np.ndarray, width: int, height: int, main_index: int,
                   output_dir: Path) -> tuple[np.ndarray, list[int], list[dict[str, Any]], dict[str, str], list[int]]:
    warped_by_index: dict[int, np.ndarray] = {}
    masks_by_index: dict[int, np.ndarray] = {}
    for index in render_indices:
        matrix = translation @ transforms_by_index[index]
        image = images[index]
        warped_by_index[index] = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        masks_by_index[index] = cv2.warpPerspective(
            np.full(image.shape[:2], 255, dtype=np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST,
        )

    accepted = [main_index]
    rejected = []
    stitched = warped_by_index[main_index].copy()
    stitched_mask = masks_by_index[main_index].copy()
    warped_preview = stitched.copy()
    preview_mask = stitched_mask > 0
    seam_preview = np.zeros((height, width, 3), dtype=np.uint8)
    palette = ((79, 176, 222), (82, 184, 132), (116, 102, 229), (47, 178, 205), (192, 142, 58), (132, 91, 192))
    seam_preview[stitched_mask > 0] = palette[0]
    seam_overlap_pixels = []
    for index in render_indices[1:]:
        parent = parent_by_index[index]
        if parent not in accepted:
            rejected.append({"index": index, "reason": "Parent image was rejected"})
            continue
        residual = overlap_residual(
            warped_by_index[parent], masks_by_index[parent], warped_by_index[index], masks_by_index[index]
        )
        if residual > 52:
            rejected.append({"index": index, "reason": f"Overlap residual {residual:.1f} exceeds 52"})
            continue
        accepted.append(index)
        candidate_mask = masks_by_index[index]
        candidate = warped_by_index[index]
        fill = (candidate_mask > 0) & ~preview_mask
        warped_preview[fill] = candidate[fill]
        preview_mask |= candidate_mask > 0

        union = (stitched_mask > 0) | (candidate_mask > 0)
        y_values, x_values = np.where(union)
        x0, x1 = int(x_values.min()), int(x_values.max()) + 1
        y0, y1 = int(y_values.min()), int(y_values.max()) + 1
        base = stitched[y0:y1, x0:x1].copy()
        base_mask = stitched_mask[y0:y1, x0:x1].copy()
        candidate_crop = candidate[y0:y1, x0:x1].copy()
        candidate_mask_crop = candidate_mask[y0:y1, x0:x1].copy()
        compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN)
        compensator.feed([(0, 0), (0, 0)], [base, candidate_crop], [base_mask, candidate_mask_crop])
        compensator.apply(0, (0, 0), base, base_mask)
        compensator.apply(1, (0, 0), candidate_crop, candidate_mask_crop)

        crop_height, crop_width = base.shape[:2]
        seam_scale = min(1.0, 1200 / max(crop_width, crop_height))
        seam_width, seam_height = max(1, round(crop_width * seam_scale)), max(1, round(crop_height * seam_scale))
        small_images = [
            cv2.UMat(cv2.resize(image, (seam_width, seam_height), interpolation=cv2.INTER_AREA).astype(np.float32))
            for image in (base, candidate_crop)
        ]
        small_masks = [
            cv2.UMat(cv2.resize(mask, (seam_width, seam_height), interpolation=cv2.INTER_NEAREST))
            for mask in (base_mask, candidate_mask_crop)
        ]
        cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD").find(small_images, [(0, 0), (0, 0)], small_masks)
        base_seam_mask, candidate_seam_mask = [
            cv2.resize(mask.get(), (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)
            for mask in small_masks
        ]
        seam_overlap_pixels.append(int(((base_seam_mask > 0) & (candidate_seam_mask > 0)).sum()))
        seam_region = seam_preview[y0:y1, x0:x1]
        seam_region[candidate_seam_mask > 0] = palette[len(accepted) % len(palette)]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        base_blend_mask = cv2.bitwise_and(base_mask, cv2.dilate(base_seam_mask, kernel))
        candidate_blend_mask = cv2.bitwise_and(candidate_mask_crop, cv2.dilate(candidate_seam_mask, kernel))
        blender = cv2.detail_MultiBandBlender(0, 3)
        blender.prepare((0, 0, crop_width, crop_height))
        blender.feed(base.astype(np.int16), base_blend_mask, (0, 0))
        blender.feed(candidate_crop.astype(np.int16), candidate_blend_mask, (0, 0))
        blended, blended_mask = blender.blend(None, None)
        replace = blended_mask > 0
        target = stitched[y0:y1, x0:x1]
        target[replace] = np.clip(blended, 0, 255).astype(np.uint8)[replace]
        stitched_mask[y0:y1, x0:x1][replace] = 255
    warped_path = output_dir / "warped.png"
    cv2.imwrite(str(warped_path), warped_preview)
    seam_path = output_dir / "seams.png"
    cv2.imwrite(str(seam_path), seam_preview)
    return stitched, accepted, rejected, {
        "warped": warped_path.name,
        "seams": seam_path.name,
    }, seam_overlap_pixels


def run_image_stitch(image_paths: list[Path], output_dir: Path, main_index: int = 0) -> dict[str, Any]:
    if not 2 <= len(image_paths) <= 8:
        raise ValueError("Image stitching needs between 2 and 8 photos")
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(image is None for image in images):
        raise ValueError("One uploaded photo could not be decoded")
    images = [image for image in images if image is not None]
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, source_path in enumerate(image_paths):
        shutil.copy2(source_path, output_dir / f"input_{index:02d}{source_path.suffix.lower()}")

    tree, used_indices = alignment_tree(images, main_index)
    used_set = set(used_indices)
    pairs = [
        {key: value for key, value in edge.items() if key != "matrix"}
        for edge in tree
        if edge["source_index"] in used_set and edge["destination_index"] in used_set
    ]
    neighbors: dict[int, list[tuple[int, np.ndarray]]] = {index: [] for index in used_indices}
    for edge in tree:
        source_index, destination_index = edge["source_index"], edge["destination_index"]
        if source_index not in used_set or destination_index not in used_set:
            continue
        matrix = edge["matrix"]
        neighbors[source_index].append((destination_index, np.linalg.inv(matrix)))
        neighbors[destination_index].append((source_index, matrix))
    root = main_index
    transforms_by_index = {root: np.eye(3, dtype=np.float64)}
    parent_by_index: dict[int, int | None] = {root: None}
    pending = [root]
    render_indices = [root]
    while pending:
        current = pending.pop()
        for neighbor, neighbor_to_current in neighbors[current]:
            if neighbor in transforms_by_index:
                continue
            transforms_by_index[neighbor] = transforms_by_index[current] @ neighbor_to_current
            parent_by_index[neighbor] = current
            pending.append(neighbor)
            render_indices.append(neighbor)
    used_images = [images[index] for index in render_indices]
    transforms = [transforms_by_index[index] for index in render_indices]
    translation, width, height = mosaic_bounds(used_images, transforms)
    if width > 12000 or height > 12000 or width * height > 80_000_000:
        raise ValueError("Stitched image would be too large; upload fewer or smaller photos")

    stitched, accepted, rejected, debug_artifacts, seam_overlap_pixels = compose_stitch(
        images, render_indices, transforms_by_index, parent_by_index, translation, width, height, main_index, output_dir
    )
    stitched_path = output_dir / "stitched.png"
    cv2.imwrite(str(stitched_path), stitched)
    accepted_set = set(accepted)
    report = {
        "run_id": output_dir.name,
        "width": width,
        "height": height,
        "main_index": main_index,
        "pairs": [
            pair for pair in pairs
            if pair["source_index"] in accepted_set and pair["destination_index"] in accepted_set
        ],
        "used_indices": sorted(accepted),
        "skipped": [
            {"index": index, "reason": "No reliable alignment with selected photo group"}
            for index in range(len(images)) if index not in used_set
        ] + rejected,
        "rendering": {
            "seam_method": "graphcut_colorgrad",
            "blend_bands": 3,
            "rejected_indices": [item["index"] for item in rejected],
            "seam_overlap_pixels": seam_overlap_pixels,
        },
        "artifacts": {"stitched": stitched_path.name, "final_image": stitched_path.name, **debug_artifacts},
    }
    (output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def rectify_stitched_image(run_dir: Path, report: dict[str, Any], points: list[list[float]]) -> dict[str, Any]:
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("points must contain four image coordinates")
    source_name = report.get("artifacts", {}).get("final_image")
    source_path = (run_dir / str(source_name)).resolve()
    if not source_path.is_relative_to(run_dir.resolve()) or not source_path.is_file():
        raise ValueError("Stitch result image is not available")
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Stitch result image could not be decoded")
    source = np.float32(points)
    top = max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3]))
    side = max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1]))
    width, height = max(1, int(round(top))), max(1, int(round(side)))
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    rectified = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (width, height))
    output_path = run_dir / "rectified.png"
    cv2.imwrite(str(output_path), rectified)
    report["width"], report["height"] = width, height
    report["artifacts"]["rectified"] = output_path.name
    report["artifacts"]["final_image"] = output_path.name
    (run_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
