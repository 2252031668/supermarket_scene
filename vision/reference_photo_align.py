#!/usr/bin/env python3
"""Align a full shelf reference photo to a current partial shelf photo.

The output reference image is warped into the current photo's pixels.  This is
the registration stage for later VLM comparison or conventional difference
analysis; it does not decide whether a product is abnormal or missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np


VISION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = VISION_DIR / "output" / "reference_photo_align"


@dataclass(frozen=True)
class MatchMetrics:
    baseline_keypoints: int
    current_keypoints: int
    ratio_matches: int
    mutual_matches: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error_px: float
    current_inlier_coverage: float
    valid_overlap_ratio: float


class AlignmentError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp a full shelf reference image into the pixels and view of a current partial shelf photo."
    )
    parser.add_argument("--baseline", required=True, type=Path, help="Complete, correct shelf-face reference photo")
    parser.add_argument("--current", required=True, type=Path, help="Current partial shelf photo to align against")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-image-side", type=int, default=1800, help="Maximum side length used only for feature matching")
    parser.add_argument("--feature", choices=("sift", "orb"), default="sift", help="Local feature extractor and descriptor")
    parser.add_argument("--dim-output", type=Path, help="Deep Image Matching result directory containing features.h5 and matches.h5")
    parser.add_argument("--dim-baseline-name", help="Baseline image name used inside Deep Image Matching, such as baseline.jpg")
    parser.add_argument("--dim-current-name", help="Current image name used inside Deep Image Matching, such as current.png")
    parser.add_argument("--max-features", type=int, default=7000, help="Maximum retained features per image")
    parser.add_argument("--ratio-test", type=float, default=0.72, help="Lowe ratio threshold for descriptor matching")
    parser.add_argument("--reprojection-threshold", type=float, default=4.0, help="USAC/MAGSAC homography inlier threshold in matching-image pixels")
    parser.add_argument("--min-inliers", type=int, default=18)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.24)
    parser.add_argument("--min-current-coverage", type=float, default=0.05, help="Minimum convex-hull area of inlier points in the current image")
    parser.add_argument("--min-valid-overlap", type=float, default=0.45, help="Minimum fraction of current pixels covered by the warped reference")
    parser.add_argument("--no-clahe", action="store_true", help="Disable local contrast normalisation before feature extraction")
    parser.add_argument("--ecc-refine", action="store_true", help="Refine an accepted homography with ECC image alignment")
    parser.add_argument("--ecc-iterations", type=int, default=80)
    return parser.parse_args()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AlignmentError(f"Cannot read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def resize_for_matching(image: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, np.eye(3, dtype=np.float64)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)


def prepared_gray(image: np.ndarray, use_clahe: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not use_clahe:
        return gray
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def create_feature_extractor(name: str, max_features: int) -> tuple[cv2.Feature2D, int]:
    if name == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING


def keep_strongest(keypoints: list[cv2.KeyPoint], descriptors: np.ndarray | None,
                   maximum: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if descriptors is None or len(keypoints) <= maximum:
        return keypoints, descriptors
    indexes = sorted(range(len(keypoints)), key=lambda index: keypoints[index].response, reverse=True)[:maximum]
    return [keypoints[index] for index in indexes], descriptors[indexes]


def ratio_matches(descriptors_a: np.ndarray | None, descriptors_b: np.ndarray | None, ratio: float,
                  norm: int) -> list[cv2.DMatch]:
    if descriptors_a is None or descriptors_b is None:
        return []
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    return [first for pair in matches if len(pair) == 2 for first, second in [pair] if first.distance < ratio * second.distance]


def mutual_ratio_matches(descriptors_base: np.ndarray | None, descriptors_current: np.ndarray | None,
                         ratio: float, norm: int) -> list[cv2.DMatch]:
    forward = ratio_matches(descriptors_base, descriptors_current, ratio, norm)
    reverse = ratio_matches(descriptors_current, descriptors_base, ratio, norm)
    reverse_pairs = {(match.trainIdx, match.queryIdx) for match in reverse}
    return [match for match in forward if (match.queryIdx, match.trainIdx) in reverse_pairs]


def reprojection_error(source: np.ndarray, destination: np.ndarray, matrix: np.ndarray) -> float:
    projected = cv2.perspectiveTransform(source, matrix)
    errors = np.linalg.norm(projected.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
    return float(np.median(errors))


def load_dim_matches(output_dir: Path, baseline_name: str, current_name: str) -> tuple[
        list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], np.ndarray, np.ndarray]:
    """Load Deep Image Matching's geometrically verified pair matches."""
    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    if not features_path.is_file() or not matches_path.is_file():
        raise AlignmentError(f"Deep Image Matching output must contain features.h5 and matches.h5: {output_dir}")
    try:
        with h5py.File(features_path, "r") as features, h5py.File(matches_path, "r") as matches_file:
            baseline_points = np.asarray(features[baseline_name]["keypoints"], dtype=np.float32)
            current_points = np.asarray(features[current_name]["keypoints"], dtype=np.float32)
            pairs = np.asarray(matches_file[baseline_name][current_name], dtype=np.int64)
    except (KeyError, OSError) as error:
        raise AlignmentError(
            f"Cannot load Deep Image Matching pair {baseline_name!r} -> {current_name!r} from {output_dir}: {error}"
        ) from error
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 4:
        raise AlignmentError("Deep Image Matching output contains fewer than four pair matches")
    if pairs.min() < 0 or pairs[:, 0].max() >= len(baseline_points) or pairs[:, 1].max() >= len(current_points):
        raise AlignmentError("Deep Image Matching pair indices are outside the feature arrays")
    baseline_keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in baseline_points]
    current_keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in current_points]
    matches = [cv2.DMatch(int(query), int(train), 0.0) for query, train in pairs]
    source = baseline_points[pairs[:, 0]].reshape(-1, 1, 2)
    destination = current_points[pairs[:, 1]].reshape(-1, 1, 2)
    return baseline_keypoints, current_keypoints, matches, source, destination


def refine_with_ecc(baseline: np.ndarray, current: np.ndarray, matrix: np.ndarray,
                    source_inliers: np.ndarray, destination_inliers: np.ndarray,
                    baseline_scale: np.ndarray, current_scale: np.ndarray,
                    iterations: int) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Make a conservative local correction after feature-based registration."""
    initial = cv2.warpPerspective(baseline, matrix, (current.shape[1], current.shape[0]), flags=cv2.INTER_LINEAR)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    initial_gray = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    initial_error = reprojection_error(source_inliers, destination_inliers, current_scale @ matrix @ np.linalg.inv(baseline_scale))
    try:
        score, correction = cv2.findTransformECC(
            current_gray, initial_gray, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, 1e-6), None, 5,
        )
    except cv2.error as error:
        return matrix, {"attempted": True, "accepted": False, "reason": str(error), "initial_reprojection_error_px": initial_error}
    refined = np.linalg.inv(correction.astype(np.float64)) @ matrix
    refined_error = reprojection_error(source_inliers, destination_inliers, current_scale @ refined @ np.linalg.inv(baseline_scale))
    if not np.isfinite(refined).all() or refined_error > initial_error * 1.25:
        return matrix, {"attempted": True, "accepted": False, "reason": "feature_reprojection_error_worsened", "ecc_score": float(score), "initial_reprojection_error_px": initial_error, "refined_reprojection_error_px": refined_error}
    return refined, {"attempted": True, "accepted": True, "ecc_score": float(score), "initial_reprojection_error_px": initial_error, "refined_reprojection_error_px": refined_error}


def coverage(points: np.ndarray, image_shape: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull)) / (image_shape[0] * image_shape[1])


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest axis-aligned rectangle made only of valid mask pixels."""
    valid = mask > 0
    height, width = valid.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)
    for y, row in enumerate(valid):
        histogram = np.where(row, histogram + 1, 0)
        stack: list[int] = []
        for x in range(width + 1):
            current_height = 0 if x == width else int(histogram[x])
            while stack and int(histogram[stack[-1]]) >= current_height:
                column = stack.pop()
                rectangle_height = int(histogram[column])
                left = stack[-1] + 1 if stack else 0
                rectangle_width = x - left
                area = rectangle_height * rectangle_width
                if area > best_area:
                    best_area = area
                    best = (left, y - rectangle_height + 1, rectangle_width, rectangle_height)
            stack.append(x)
    if best_area == 0:
        raise AlignmentError("No rectangular valid overlap remains after warping the baseline")
    return best


def draw_matches(base: np.ndarray, current: np.ndarray, base_keypoints: list[cv2.KeyPoint],
                 current_keypoints: list[cv2.KeyPoint], matches: list[cv2.DMatch], mask: np.ndarray | None,
                 output: Path) -> None:
    match_mask = None if mask is None else mask.astype(np.uint8).reshape(-1).tolist()
    rendered = cv2.drawMatches(
        base, base_keypoints, current, current_keypoints, matches, None,
        matchesMask=match_mask,
        matchColor=(28, 190, 80), singlePointColor=(100, 130, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output), rendered)


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    current_path = args.current.expanduser().resolve()
    if not baseline_path.is_file() or not current_path.is_file():
        missing = baseline_path if not baseline_path.is_file() else current_path
        raise SystemExit(f"Image not found: {missing}")
    if args.max_image_side < 400 or args.max_features < 100 or args.ecc_iterations < 1 or not 0 < args.ratio_test < 1:
        raise SystemExit("--max-image-side must be >= 400; --max-features and --ecc-iterations >= 1; --ratio-test must be in (0, 1)")
    if args.reprojection_threshold <= 0 or args.min_inliers < 4 or not 0 < args.min_inlier_ratio <= 1:
        raise SystemExit("Use a positive reprojection threshold, at least four inliers, and an inlier ratio in (0, 1]")

    created_at = datetime.now().astimezone()
    run_dir = args.output_dir.expanduser().resolve() / created_at.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline = read_bgr(baseline_path)
    current = read_bgr(current_path)
    baseline_copy = run_dir / f"00_baseline_source{baseline_path.suffix.lower()}"
    current_copy = run_dir / f"01_current_source{current_path.suffix.lower()}"
    shutil.copy2(baseline_path, baseline_copy)
    shutil.copy2(current_path, current_copy)

    print(f"Baseline: {baseline_path} ({baseline.shape[1]}x{baseline.shape[0]})")
    print(f"Current: {current_path} ({current.shape[1]}x{current.shape[0]})")
    dim_output = args.dim_output.expanduser().resolve() if args.dim_output else None
    if dim_output:
        baseline_match, current_match = baseline, current
        baseline_scale = current_scale = np.eye(3, dtype=np.float64)
        baseline_name = args.dim_baseline_name or baseline_path.name
        current_name = args.dim_current_name or current_path.name
        phase = time.perf_counter()
        base_keypoints, current_keypoints, matches, source_points, destination_points = load_dim_matches(
            dim_output, baseline_name, current_name,
        )
        feature_seconds = time.perf_counter() - phase
        tentative = matches
        match_label = f"Deep Image Matching tiled SIFT matches from {dim_output}"
        print(f"DIM matching size: {baseline_match.shape[1]}x{baseline_match.shape[0]} -> {current_match.shape[1]}x{current_match.shape[0]}")
    else:
        baseline_match, baseline_scale = resize_for_matching(baseline, args.max_image_side)
        current_match, current_scale = resize_for_matching(current, args.max_image_side)
        print(f"{args.feature.upper()} matching size: {baseline_match.shape[1]}x{baseline_match.shape[0]} -> {current_match.shape[1]}x{current_match.shape[0]}")
        phase = time.perf_counter()
        extractor, descriptor_norm = create_feature_extractor(args.feature, args.max_features)
        base_keypoints, base_descriptors = extractor.detectAndCompute(prepared_gray(baseline_match, not args.no_clahe), None)
        current_keypoints, current_descriptors = extractor.detectAndCompute(prepared_gray(current_match, not args.no_clahe), None)
        base_keypoints, base_descriptors = keep_strongest(base_keypoints, base_descriptors, args.max_features)
        current_keypoints, current_descriptors = keep_strongest(current_keypoints, current_descriptors, args.max_features)
        feature_seconds = time.perf_counter() - phase
        tentative = ratio_matches(base_descriptors, current_descriptors, args.ratio_test, descriptor_norm)
        matches = mutual_ratio_matches(base_descriptors, current_descriptors, args.ratio_test, descriptor_norm)
        source_points = np.float32([base_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        destination_points = np.float32([current_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
        match_label = f"OpenCV {args.feature.upper()} + mutual Lowe ratio test"
    draw_matches(baseline_match, current_match, base_keypoints, current_keypoints, tentative, None, run_dir / "02_ratio_matches.png")
    if len(matches) < 4:
        raise AlignmentError(f"Only {len(matches)} matches; at least four are required")
    phase = time.perf_counter()
    matrix_match, inlier_mask = cv2.findHomography(
        source_points, destination_points, method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=args.reprojection_threshold, maxIters=10000, confidence=0.999,
    )
    homography_seconds = time.perf_counter() - phase
    if matrix_match is None or inlier_mask is None:
        raise AlignmentError("USAC/MAGSAC could not estimate a homography")
    inliers = inlier_mask.reshape(-1).astype(bool)
    draw_matches(baseline_match, current_match, base_keypoints, current_keypoints, matches, inliers, run_dir / "03_homography_inliers.png")
    source_inliers, destination_inliers = source_points[inliers], destination_points[inliers]
    reprojection = reprojection_error(source_inliers, destination_inliers, matrix_match)
    metrics_without_overlap = {
        "baseline_keypoints": len(base_keypoints), "current_keypoints": len(current_keypoints),
        "ratio_matches": len(tentative), "mutual_matches": len(matches), "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()), "median_reprojection_error_px": reprojection,
        "current_inlier_coverage": coverage(destination_inliers.reshape(-1, 2), current_match.shape[:2]),
    }
    failures = []
    if metrics_without_overlap["inliers"] < args.min_inliers:
        failures.append(f"inliers {metrics_without_overlap['inliers']} < {args.min_inliers}")
    if metrics_without_overlap["inlier_ratio"] < args.min_inlier_ratio:
        failures.append(f"inlier ratio {metrics_without_overlap['inlier_ratio']:.3f} < {args.min_inlier_ratio:.3f}")
    if metrics_without_overlap["current_inlier_coverage"] < args.min_current_coverage:
        failures.append(f"current inlier coverage {metrics_without_overlap['current_inlier_coverage']:.3f} < {args.min_current_coverage:.3f}")
    if failures:
        raise AlignmentError("Homography rejected: " + "; ".join(failures))

    matrix_original = np.linalg.inv(current_scale) @ matrix_match @ baseline_scale
    ecc = {"attempted": False, "accepted": False}
    if args.ecc_refine:
        cv2.imwrite(str(run_dir / "04_aligned_baseline_initial.png"), cv2.warpPerspective(baseline, matrix_original, (current.shape[1], current.shape[0]), flags=cv2.INTER_LINEAR))
        matrix_original, ecc = refine_with_ecc(
            baseline, current, matrix_original, source_inliers, destination_inliers,
            baseline_scale, current_scale, args.ecc_iterations,
        )
    aligned = cv2.warpPerspective(baseline, matrix_original, (current.shape[1], current.shape[0]), flags=cv2.INTER_LINEAR)
    source_mask = np.full(baseline.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(source_mask, matrix_original, (current.shape[1], current.shape[0]), flags=cv2.INTER_NEAREST)
    valid_overlap = float(np.count_nonzero(valid_mask)) / valid_mask.size
    if valid_overlap < args.min_valid_overlap:
        raise AlignmentError(f"Warped reference covers only {valid_overlap:.3f} of the current image")
    overlay = current.copy()
    blended = cv2.addWeighted(current, 0.5, aligned, 0.5, 0)
    overlay[valid_mask > 0] = blended[valid_mask > 0]
    difference = cv2.absdiff(current, aligned)
    difference[valid_mask == 0] = 0
    crop_x, crop_y, crop_width, crop_height = largest_valid_rectangle(valid_mask)
    crop_rows = slice(crop_y, crop_y + crop_height)
    crop_columns = slice(crop_x, crop_x + crop_width)
    current_crop = current[crop_rows, crop_columns]
    aligned_crop = aligned[crop_rows, crop_columns]
    valid_mask_crop = valid_mask[crop_rows, crop_columns]
    overlay_crop = overlay[crop_rows, crop_columns]
    difference_crop = difference[crop_rows, crop_columns]
    cv2.imwrite(str(run_dir / "04_aligned_baseline.png"), aligned)
    cv2.imwrite(str(run_dir / "05_valid_overlap_mask.png"), valid_mask)
    cv2.imwrite(str(run_dir / "06_alignment_overlay.png"), overlay)
    cv2.imwrite(str(run_dir / "07_absolute_difference.png"), difference)
    cv2.imwrite(str(run_dir / "08_overlap_crop_current.png"), current_crop)
    cv2.imwrite(str(run_dir / "09_overlap_crop_aligned_baseline.png"), aligned_crop)
    cv2.imwrite(str(run_dir / "10_overlap_crop_valid_mask.png"), valid_mask_crop)
    cv2.imwrite(str(run_dir / "11_overlap_crop_overlay.png"), overlay_crop)
    cv2.imwrite(str(run_dir / "12_overlap_crop_absolute_difference.png"), difference_crop)
    metrics = MatchMetrics(**metrics_without_overlap, valid_overlap_ratio=valid_overlap)
    report = {
        "success": True, "created_at": created_at.isoformat(timespec="seconds"),
        "baseline": str(baseline_path), "current": str(current_path),
        "matcher": f"{match_label} + USAC_MAGSAC homography",
        "matching_image_sizes": {"baseline": [baseline_match.shape[1], baseline_match.shape[0]], "current": [current_match.shape[1], current_match.shape[0]]},
        "homography_baseline_to_current": matrix_original.tolist(), "metrics": asdict(metrics), "ecc_refinement": ecc,
        "overlap_crop_xywh": [crop_x, crop_y, crop_width, crop_height],
        "timings_seconds": {"feature_extraction": feature_seconds, "homography": homography_seconds, "total": time.perf_counter() - started},
        "artifacts": {"baseline_source": baseline_copy.name, "current_source": current_copy.name, "ratio_matches": "02_ratio_matches.png", "inlier_matches": "03_homography_inliers.png", "aligned_baseline": "04_aligned_baseline.png", "valid_mask": "05_valid_overlap_mask.png", "overlay": "06_alignment_overlay.png", "absolute_difference": "07_absolute_difference.png", "crop_current": "08_overlap_crop_current.png", "crop_aligned_baseline": "09_overlap_crop_aligned_baseline.png", "crop_valid_mask": "10_overlap_crop_valid_mask.png", "crop_overlay": "11_overlap_crop_overlay.png", "crop_absolute_difference": "12_overlap_crop_absolute_difference.png", **({"initial_aligned_baseline": "04_aligned_baseline_initial.png"} if args.ecc_refine else {})},
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Keypoints: baseline={metrics.baseline_keypoints}, current={metrics.current_keypoints}")
    print(f"Matches: ratio={metrics.ratio_matches}, mutual={metrics.mutual_matches}, inliers={metrics.inliers} ({metrics.inlier_ratio:.1%})")
    print(f"Median reprojection error: {metrics.median_reprojection_error_px:.2f}px; current coverage: {metrics.current_inlier_coverage:.1%}; overlap: {metrics.valid_overlap_ratio:.1%}")
    print(f"Full-valid overlap crop: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}")
    if ecc["attempted"]:
        if ecc["accepted"]:
            print(f"ECC refinement accepted: {ecc['initial_reprojection_error_px']:.2f}px -> {ecc['refined_reprojection_error_px']:.2f}px")
        else:
            print(f"ECC refinement rejected: {ecc['reason']}")
    print(f"Elapsed: {report['timings_seconds']['total']:.2f}s")
    print(f"Aligned baseline: {run_dir / '04_aligned_baseline.png'}")
    print(f"Overlay: {run_dir / '06_alignment_overlay.png'}")


if __name__ == "__main__":
    try:
        main()
    except AlignmentError as error:
        raise SystemExit(f"Reference alignment failed: {error}") from error
