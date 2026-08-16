#!/usr/bin/env python3
"""Replace four imported shelf photos with matching high-resolution sources."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CALIBRATION_DIR = DATA_DIR / "shelf_calibration"
SHELF_IMAGES_DIR = DATA_DIR / "shelf_images"
ITEM_IMAGES_DIR = DATA_DIR / "item_images"
PAIRS = ((1, 0, "1.jpg"), (1, 1, "2.jpg"), (2, 0, "3.jpg"), (2, 1, "4.jpg"))


@dataclass
class FaceMigration:
    shelf_id: int
    face: int
    filename: str
    low_path: Path
    high_path: Path
    shelf_path: Path
    calibration: dict
    homography: np.ndarray
    high_image: np.ndarray
    metrics: dict


def shelf_filename(face: int) -> str:
    return "-x_0.png" if face == 0 else "+x_0.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def transform_bbox(box: dict, homography: np.ndarray, width: int, height: int) -> dict:
    x, y = int(box["x"]), int(box["y"])
    right, bottom = x + int(box["width"]), y + int(box["height"])
    corners = np.float32([[[x, y], [right, y], [right, bottom], [x, bottom]]])
    mapped = cv2.perspectiveTransform(corners, homography)[0]
    left = max(0, math.floor(float(mapped[:, 0].min())))
    top = max(0, math.floor(float(mapped[:, 1].min())))
    right = min(width, math.ceil(float(mapped[:, 0].max())))
    bottom = min(height, math.ceil(float(mapped[:, 1].max())))
    if right <= left or bottom <= top:
        raise ValueError(f"Transformed bbox is empty: {box}")
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def transform_point(point: dict, homography: np.ndarray, width: int, height: int) -> dict:
    source = np.float32([[[point["x"], point["y"]]]])
    x, y = cv2.perspectiveTransform(source, homography)[0, 0]
    return {"x": int(np.clip(round(x), 0, width - 1)), "y": int(np.clip(round(y), 0, height - 1))}


def find_homography(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, dict]:
    sift = cv2.SIFT_create()
    low_points, low_desc = sift.detectAndCompute(low, None)
    high_points, high_desc = sift.detectAndCompute(high, None)
    if low_desc is None or high_desc is None:
        raise ValueError("Could not extract SIFT features")
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(low_desc, high_desc, k=2)
    good = [first for first, second in raw if first.distance < 0.72 * second.distance]
    if len(good) < 50:
        raise ValueError(f"Only {len(good)} ratio-test matches")
    source = np.float32([low_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    target = np.float32([high_points[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or mask is None:
        raise ValueError("Could not estimate a homography")
    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    ratio = count / len(good)
    if count < 50 or ratio < 0.8:
        raise ValueError(f"Weak image match: {count}/{len(good)} inliers")
    projected = cv2.perspectiveTransform(source[inliers], homography)
    error = np.linalg.norm(projected[:, 0] - target[inliers, 0], axis=1)
    median_error = float(np.median(error))
    if median_error > 1.5:
        raise ValueError(f"High reprojection error: {median_error:.3f}px")
    return homography, {
        "ratio_matches": len(good),
        "inliers": count,
        "inlier_ratio": round(ratio, 4),
        "median_reprojection_error_px": round(median_error, 4),
        "max_reprojection_error_px": round(float(np.max(error)), 4),
    }


def preflight(low_dir: Path, high_dir: Path) -> list[FaceMigration]:
    migrations = []
    for shelf_id, face, filename in PAIRS:
        low_path, high_path = low_dir / filename, high_dir / filename
        shelf_path = SHELF_IMAGES_DIR / str(shelf_id) / shelf_filename(face)
        calibration_path = CALIBRATION_DIR / f"{shelf_id}.json"
        if not low_path.is_file() or not high_path.is_file() or not shelf_path.is_file() or not calibration_path.is_file():
            raise ValueError(f"Missing required file for shelf {shelf_id} face {face}")
        low_color = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
        high_color = cv2.imread(str(high_path), cv2.IMREAD_COLOR)
        current_color = cv2.imread(str(shelf_path), cv2.IMREAD_COLOR)
        if low_color is None or high_color is None or current_color is None:
            raise ValueError(f"Could not decode required image for shelf {shelf_id} face {face}")
        if not np.array_equal(low_color, current_color):
            raise ValueError(f"Current shelf image is not the expected low-resolution source: {shelf_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        face_data = calibration.get("faces", {}).get(str(face))
        if not isinstance(face_data, dict):
            raise ValueError(f"Missing calibration face {face} for shelf {shelf_id}")
        homography, metrics = find_homography(
            cv2.cvtColor(low_color, cv2.COLOR_BGR2GRAY), cv2.cvtColor(high_color, cv2.COLOR_BGR2GRAY)
        )
        width, height = high_color.shape[1], high_color.shape[0]
        transformed = [transform_bbox(slot["bbox"], homography, width, height) for slot in face_data.get("slots", []) if "bbox" in slot]
        if len(transformed) != len(face_data.get("slots", [])):
            raise ValueError(f"Shelf {shelf_id} face {face} has slots without bbox")
        crop_count = 0
        for slot in face_data["slots"]:
            crop_path = ITEM_IMAGES_DIR / slot["slot_id"] / "0.png"
            if not crop_path.is_file():
                raise ValueError(f"Missing ID crop: {crop_path}")
            crop_count += 1
        metrics.update({
            "low_size": [low_color.shape[1], low_color.shape[0]],
            "high_size": [width, height],
            "slot_count": len(face_data["slots"]),
            "id_crops_to_replace": crop_count,
        })
        migrations.append(FaceMigration(
            shelf_id, face, filename, low_path, high_path, shelf_path, calibration,
            homography, high_color, metrics,
        ))
    return migrations


def stage_migration(migrations: list[FaceMigration], staging: Path) -> tuple[list[tuple[Path, Path]], dict]:
    staged: list[tuple[Path, Path]] = []
    calibrations: dict[int, dict] = {}
    manifest = {"faces": []}
    for migration in migrations:
        width, height = migration.high_image.shape[1], migration.high_image.shape[0]
        staged_shelf = staging / "shelf_images" / str(migration.shelf_id) / shelf_filename(migration.face)
        staged_shelf.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(migration.high_path) as source:
            source.convert("RGB").save(staged_shelf, format="PNG")
        staged.append((staged_shelf, migration.shelf_path))

        calibration = calibrations.setdefault(migration.shelf_id, copy.deepcopy(migration.calibration))
        face_data = calibration["faces"][str(migration.face)]
        face_data["image_file"] = str(migration.shelf_path.relative_to(ROOT))
        face_data["image_hash"] = sha256(staged_shelf)
        face_data["layers"] = {
            level: [transform_point(point, migration.homography, width, height) for point in points]
            for level, points in face_data.get("layers", {}).items()
        }
        for slot in face_data["slots"]:
            slot["bbox"] = transform_bbox(slot["bbox"], migration.homography, width, height)
            staged_crop = staging / "item_images" / slot["slot_id"] / "0.png"
            staged_crop.parent.mkdir(parents=True, exist_ok=True)
            box = slot["bbox"]
            crop = migration.high_image[box["y"]:box["y"] + box["height"], box["x"]:box["x"] + box["width"]]
            if crop.size == 0 or not cv2.imwrite(str(staged_crop), crop):
                raise ValueError(f"Could not write high-resolution crop for {slot['slot_id']}")
            staged.append((staged_crop, ITEM_IMAGES_DIR / slot["slot_id"] / "0.png"))
        manifest["faces"].append({
            "shelf_id": migration.shelf_id,
            "face": migration.face,
            "source": str(migration.high_path),
            "metrics": migration.metrics,
        })
    for shelf_id, calibration in calibrations.items():
        staged_calibration = staging / "shelf_calibration" / f"{shelf_id}.json"
        staged_calibration.parent.mkdir(parents=True, exist_ok=True)
        staged_calibration.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.append((staged_calibration, CALIBRATION_DIR / f"{shelf_id}.json"))
    manifest["id_crops_replaced"] = sum(item.metrics["id_crops_to_replace"] for item in migrations)
    return staged, manifest


def apply_migration(migrations: list[FaceMigration]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = DATA_DIR / "highres_migration_backups" / stamp
    with tempfile.TemporaryDirectory(prefix=".highres-migration-", dir=DATA_DIR) as temporary:
        staged, manifest = stage_migration(migrations, Path(temporary))
        targets = [target for _source, target in staged]
        if len(targets) != len(set(targets)):
            raise ValueError("Migration would replace a target more than once")
        for target in targets:
            backup_target = backup / target.relative_to(DATA_DIR)
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_target)
        (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            for source, target in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        except Exception:
            for target in targets:
                backup_target = backup / target.relative_to(DATA_DIR)
                if backup_target.is_file():
                    shutil.copy2(backup_target, target)
            raise
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-dir", required=True, type=Path)
    parser.add_argument("--high-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Create a backup and replace generated assets")
    args = parser.parse_args()
    migrations = preflight(args.low_dir.resolve(), args.high_dir.resolve())
    report = {"faces": [item.metrics | {"shelf_id": item.shelf_id, "face": item.face, "source": item.filename} for item in migrations]}
    report["id_crops_to_replace"] = sum(item.metrics["id_crops_to_replace"] for item in migrations)
    if args.apply:
        report["backup"] = str(apply_migration(migrations))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
