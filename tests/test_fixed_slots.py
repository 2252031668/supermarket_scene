import json
import base64
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from PIL import Image
import api_server
import calibration_manager
from mujoco import generate_scene_from_database
from vision import config as vision_config

from shelf_database import (
    DEFAULT_BACK_THICK,
    DEFAULT_BOTTOM_CLEARANCE,
    DEFAULT_LEVEL_SPACING,
    DEFAULT_PANEL_THICK,
    DEFAULT_SHELF_DEPTH_BOTTOM,
    DEFAULT_SHELF_DEPTH_NORMAL,
    DEFAULT_SHELF_HEIGHT,
    DEFAULT_SHELF_LENGTH,
    DEFAULT_SHELF_WIDTH,
    ShelfDatabase,
)


class FixedSlotTests(unittest.TestCase):
    def setUp(self):
        self.db = ShelfDatabase(":memory:")
        type_id = self.db.add_shelf_type(
            "test",
            DEFAULT_SHELF_LENGTH,
            DEFAULT_SHELF_WIDTH,
            DEFAULT_SHELF_HEIGHT,
            5,
            DEFAULT_BOTTOM_CLEARANCE,
            DEFAULT_LEVEL_SPACING,
            DEFAULT_PANEL_THICK,
            DEFAULT_BACK_THICK,
            DEFAULT_SHELF_DEPTH_NORMAL,
            DEFAULT_SHELF_DEPTH_BOTTOM,
        )
        self.shelf_id = self.db.add_shelf_group("Shelf", shelf_type_id=type_id)
        self.db.register_skus_batch(
            [{"sku": "cola"}, {"sku": "sprite"}, {"sku": "tea"}]
        )

    def tearDown(self):
        self.db.close()

    def test_slot_id_is_stable_and_status_is_derived(self):
        slot_id = self.db.create_slot(
            self.shelf_id, 0, 2, 43, "sprite", "sprite"
        )
        slot = self.db.get_slot_by_id(slot_id)
        self.assertEqual(slot.slot_id, "1-0-2-43")
        self.assertEqual(slot.status, "正常")

        self.db.set_actual_sku(slot.slot_id, None)
        self.assertEqual(self.db.get_slot_by_id(slot.slot_id).status, "缺货")

        self.db.set_actual_sku(slot.slot_id, "cola")
        self.assertEqual(self.db.get_slot_by_id(slot.slot_id).status, "摆放错误")

    def test_take_and_restock_keep_slot_id(self):
        slot_id = self.db.create_slot(
            self.shelf_id, 0, 2, 50, "sprite", "sprite"
        )
        self.db.take_slot(slot_id)
        self.assertIsNone(self.db.get_slot_by_id(slot_id).actual_sku)

        self.db.restock_slot(slot_id)
        slot = self.db.get_slot_by_id(slot_id)
        self.assertEqual(slot.actual_sku, "sprite")
        self.assertEqual(slot.slot_id, "1-0-2-50")

    def test_robot_service_returns_shelf_yaw_and_face(self):
        from robot_service import get_slot_pose

        self.db.update_shelf_group(self.shelf_id, world_x=1.5, world_y=2.5, yaw=0.7)
        slot_id = self.db.create_slot(self.shelf_id, 1, 2, 43, "sprite", "sprite")
        pose = get_slot_pose(self.db, slot_id)

        self.assertEqual(pose["slot_id"], slot_id)
        self.assertEqual(pose["face"], 1)
        self.assertEqual(pose["shelf_yaw"], 0.7)
        self.assertEqual(pose["frame_id"], "map")

    def test_robot_service_take_updates_projection_and_status(self):
        from unittest.mock import patch
        from robot_service import take_slot

        slot_id = self.db.create_slot(self.shelf_id, 0, 2, 43, "sprite", "sprite")
        with patch("robot_service.sync_shelf_projection") as sync:
            slot = take_slot(self.db, slot_id)

        self.assertIsNone(slot["actual_sku"])
        self.assertEqual(slot["status"], "缺货")
        sync.assert_called_once_with(self.db, self.shelf_id)

    def test_slot_location_is_not_updated(self):
        slot_id = self.db.create_slot(self.shelf_id, 0, 2, 65, "tea", "tea")
        with self.assertRaises(ValueError):
            self.db.update_slot(slot_id, shelf_id=self.shelf_id, face=1)

    def test_missing_slots_remain_queryable_without_actual_product(self):
        slot_id = self.db.create_slot(self.shelf_id, 0, 2, 80, "tea", None)
        slot = self.db.get_slot_by_id(slot_id)
        world_rows = self.db.get_all_slots_world()

        self.assertEqual(slot.status, "缺货")
        self.assertEqual(world_rows[0]["slot_id"], slot_id)
        self.assertIsNone(world_rows[0]["actual_sku"])

    def test_scene_uses_actual_sku_and_skips_shortages(self):
        self.db.create_slot(self.shelf_id, 0, 2, 20, "sprite", "sprite")
        self.db.create_slot(self.shelf_id, 0, 2, 40, "sprite", None)
        self.db.create_slot(self.shelf_id, 0, 2, 60, "sprite", "cola")
        worldbody = ET.Element("worldbody")

        with patch.object(generate_scene_from_database, "place_mesh_product") as place:
            count = generate_scene_from_database.place_products_from_database(
                worldbody, self.db
            )

        self.assertEqual(count, 2)
        self.assertEqual([call.args[1] for call in place.call_args_list], ["sprite", "cola"])

    def test_batch_import_keeps_empty_and_misplaced_states(self):
        slot_ids = self.db.import_slots_batch(
            [],
            [
                {
                    "shelf_id": self.shelf_id,
                    "face": 0,
                    "level": 2,
                    "y_cm": 30,
                    "expected_sku": "sprite",
                    "actual_sku": None,
                },
                {
                    "shelf_id": self.shelf_id,
                    "face": 0,
                    "level": 2,
                    "y_cm": 50,
                    "expected_sku": "sprite",
                    "actual_sku": "cola",
                },
            ],
        )

        self.assertEqual(slot_ids, ["1-0-2-30", "1-0-2-50"])
        self.assertEqual(self.db.get_slot_by_id(slot_ids[0]).status, "缺货")
        self.assertEqual(self.db.get_slot_by_id(slot_ids[1]).status, "摆放错误")

    def test_sku_owlv2_prompt_is_editable_and_batch_import_is_atomic(self):
        self.db.register_sku("cola", owlv2_prompt="a red cola bottle")
        self.assertEqual(self.db.get_sku_info("cola").owlv2_prompt, "a red cola bottle")

        self.db.import_slots_batch(
            [{"sku": "water"}],
            [{
                "shelf_id": self.shelf_id, "face": 0, "level": 2, "y_cm": 30,
                "expected_sku": "water", "actual_sku": "water",
            }],
            [{"sku": "water", "owlv2_prompt": "a green mineral water bottle"}],
            [{"sku": "water", "reference_image_path": "data/sku_images/water.png", "grasp_method": "吸盘"}],
        )
        self.assertEqual(self.db.get_sku_info("water").owlv2_prompt, "a green mineral water bottle")
        self.assertEqual(self.db.get_sku_info("water").reference_image_path, "data/sku_images/water.png")
        self.assertEqual(self.db.get_sku_info("water").grasp_method, "吸盘")

        self.db.update_sku("water", "drink", "water.xml", "water.png", "a clear water bottle")
        self.assertEqual(self.db.get_sku_info("water").owlv2_prompt, "a clear water bottle")

    def test_rename_sku_moves_all_slot_references(self):
        first = self.db.create_slot(self.shelf_id, 0, 2, 20, "cola", "cola")
        second = self.db.create_slot(self.shelf_id, 0, 2, 40, "sprite", "cola")

        self.db.rename_sku("cola", "orange-cola")

        self.assertIsNone(self.db.get_sku_info("cola"))
        self.assertEqual(self.db.get_slot_by_id(first).expected_sku, "orange-cola")
        self.assertEqual(self.db.get_slot_by_id(first).actual_sku, "orange-cola")
        self.assertEqual(self.db.get_slot_by_id(second).actual_sku, "orange-cola")

    def test_delete_skus_replaces_references_with_reserved_unknown(self):
        slot_id = self.db.create_slot(self.shelf_id, 0, 2, 20, "cola", "cola")

        deleted = self.db.delete_skus_to_unknown(["cola"])

        slot = self.db.get_slot_by_id(slot_id)
        self.assertEqual(deleted, ["cola"])
        self.assertEqual((slot.expected_sku, slot.actual_sku), ("unknown", "unknown"))
        self.assertIsNotNone(self.db.get_sku_info("unknown"))
        with self.assertRaises(ValueError):
            self.db.delete_skus_to_unknown(["unknown"])

    def test_sku_owlv2_prompt_survives_database_reopen(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "inventory.db")
            with ShelfDatabase(path) as db:
                db.register_sku("water", owlv2_prompt="a clear mineral water bottle")
            with ShelfDatabase(path) as db:
                self.assertEqual(db.get_sku_info("water").owlv2_prompt, "a clear mineral water bottle")

    def test_qwen_grounding_prompt_persists_through_update_and_rename(self):
        self.db.register_sku("water", qwen_grounding_prompt="透明瓶身，蓝色瓶盖，正面有白色品牌大字")
        self.assertEqual(self.db.get_sku_info("water").qwen_grounding_prompt, "透明瓶身，蓝色瓶盖，正面有白色品牌大字")

        self.db.update_sku("water", "drink", "", "", "a water bottle")
        self.assertEqual(self.db.get_sku_info("water").qwen_grounding_prompt, "透明瓶身，蓝色瓶盖，正面有白色品牌大字")

        self.db.update_sku("water", "drink", "", "", "a water bottle", qwen_grounding_prompt="蓝色瓶盖与白色品牌字样")
        self.db.rename_sku("water", "blue-water")
        self.assertEqual(self.db.get_sku_info("blue-water").qwen_grounding_prompt, "蓝色瓶盖与白色品牌字样")

    def test_sku_reference_image_and_grasp_method(self):
        self.db.register_sku("cola", reference_image_path="data/sku_images/cola.png", grasp_method="吸盘")
        sku = self.db.get_sku_info("cola")
        self.assertEqual(sku.reference_image_path, "data/sku_images/cola.png")
        self.assertEqual(sku.grasp_method, "吸盘")
        self.assertEqual(self.db.get_sku_info("tea").grasp_method, "夹爪")
        with self.assertRaisesRegex(ValueError, "grasp_method"):
            self.db.register_sku("water", grasp_method="手拿")

    def test_inspection_decision_without_vlm_turns_low_confidence_into_shortage(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="cola",
            score=0.60,
            second_score=0.51,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=False,
            vlm_result=None,
        )

        self.assertIsNone(result["actual_sku"])
        self.assertEqual(result["reason"], "low_confidence")

    def test_inspection_decision_uses_ark_result_when_fallback_is_enabled(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="cola",
            score=0.60,
            second_score=0.51,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=True,
            vlm_result={"kind": "sku", "sku": "cola"},
        )

        self.assertEqual(result["actual_sku"], "cola")
        self.assertEqual(result["source"], "ark")

    def test_failed_ark_result_becomes_shortage(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="cola",
            score=0.60,
            second_score=0.51,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=True,
            vlm_result={"kind": "unresolved"},
        )

        self.assertIsNone(result["actual_sku"])

    def test_high_confidence_dino_result_keeps_detected_sku(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="cola",
            score=0.88,
            second_score=0.42,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=False,
            vlm_result=None,
        )

        self.assertEqual(result["actual_sku"], "cola")
        self.assertEqual(result["status"], "摆放错误")
        self.assertEqual(result["source"], "dino")

    def test_ambiguous_high_score_uses_top_match_without_vlm(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="cola",
            score=0.88,
            second_score=0.84,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=False,
            vlm_result=None,
        )

        self.assertEqual(result["actual_sku"], "cola")
        self.assertEqual(result["status"], "摆放错误")
        self.assertEqual(result["reason"], "dino_match")

    def test_ambiguous_high_score_uses_ark_when_fallback_is_enabled(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="cola",
            score=0.88,
            second_score=0.84,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=True,
            vlm_result={"kind": "sku", "sku": "sprite"},
        )

        self.assertEqual(result["actual_sku"], "sprite")
        self.assertEqual(result["source"], "ark")

    def test_high_confidence_expected_sku_ignores_ambiguity(self):
        from vision.cv_restock_position import classify_candidate

        result = classify_candidate(
            expected_sku="tea",
            top_sku="tea",
            score=0.88,
            second_score=0.84,
            confidence_threshold=0.72,
            ambiguity_margin=0.05,
            vlm_fallback=False,
            vlm_result=None,
        )

        self.assertEqual(result["actual_sku"], "tea")
        self.assertEqual(result["status"], "正常")
        self.assertEqual(result["reason"], "dino_expected_match")

    def test_dino_box_filter_keeps_highest_scores_within_limit(self):
        from vision.ark_grounding import BoundingBox
        from vision.vlm_sku_query import select_dino_boxes

        boxes = [
            BoundingBox(index=1, label="sprite", normalized=(0, 0, 100, 100), pixels=(0, 0, 10, 10)),
            BoundingBox(index=2, label="sprite", normalized=(100, 0, 200, 100), pixels=(10, 0, 20, 10)),
            BoundingBox(index=3, label="sprite", normalized=(200, 0, 300, 100), pixels=(20, 0, 30, 10)),
        ]

        matches = select_dino_boxes(boxes, [0.82, 0.91, 0.79], confidence_threshold=0.8, max_results=2)

        self.assertEqual([box.index for box in matches], [2, 1])

    def test_owlv2_candidate_filter_uses_current_transformers_processor_api(self):
        import torch
        from vision.owlv2 import owlv2_candidates

        model_output = object()
        testcase = self

        class Processor:
            def __call__(self, **_kwargs):
                testcase.assertTrue(_kwargs["padding"] == "max_length")
                testcase.assertTrue(_kwargs["truncation"])
                testcase.assertEqual(_kwargs["max_length"], 16)
                return {"pixel_values": torch.zeros((1, 3, 4, 4))}

            def post_process_grounded_object_detection(self, outputs, threshold, target_sizes):
                testcase.assertIs(outputs, model_output)
                testcase.assertEqual(threshold, 0.5)
                testcase.assertEqual(target_sizes, [(20, 40)])
                return [{"boxes": torch.tensor([[1, 2, 11, 18]], dtype=torch.float32), "scores": torch.tensor([0.9])}]

        class Model:
            def __call__(self, **_kwargs):
                return model_output

        image = BytesIO()
        Image.new("RGB", (40, 20), "white").save(image, format="PNG")
        processor = Processor()
        with patch("vision.owlv2.get_owlv2_runtime", return_value=(processor, Model(), "cpu")):
            boxes, scores = owlv2_candidates(image.getvalue(), "water", "a water bottle", 0.5, 1)
        self.assertEqual(boxes[0].pixels, (1, 2, 11, 18))
        self.assertAlmostEqual(scores[0], 0.9)

    def test_owlv2_production_mode_returns_scores_without_artifacts(self):
        from vision.ark_grounding import BoundingBox
        from vision.owlv2 import run_owlv2_sku_query

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.png"
            Image.new("RGB", (10, 10), "green").save(target)
            shelf = BytesIO()
            Image.new("RGB", (20, 10), "white").save(shelf, format="PNG")
            candidate = BoundingBox(1, "water", (0, 0, 500, 999), (0, 0, 10, 10))
            with patch("vision.owlv2.owlv2_candidates", return_value=([candidate], [0.9])):
                report = run_owlv2_sku_query(
                    "water", "a water bottle", target, shelf.getvalue(), root / "run", debug=False,
                )
            self.assertEqual(report["owlv2_scores"], [{"index": 1, "confidence": 0.9}])
            self.assertFalse((root / "run").exists())

    def test_owlv2_dino_fallback_ranks_all_owlv2_candidates_before_top_n(self):
        from vision.ark_grounding import BoundingBox
        from vision.owlv2 import run_owlv2_sku_query

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.png"
            Image.new("RGB", (10, 10), "green").save(target)
            shelf = BytesIO()
            Image.new("RGB", (20, 10), "white").save(shelf, format="PNG")
            candidates = [
                BoundingBox(index=1, label="water", normalized=(0, 0, 200, 999), pixels=(0, 0, 4, 10)),
                BoundingBox(index=2, label="water", normalized=(200, 0, 400, 999), pixels=(4, 0, 8, 10)),
                BoundingBox(index=3, label="water", normalized=(400, 0, 600, 999), pixels=(8, 0, 12, 10)),
            ]
            with patch("vision.owlv2.owlv2_candidates", return_value=(candidates, [0.91, 0.88, 0.84])) as detector:
                with patch("vision.owlv2.dino_box_scores", return_value=[0.71, 0.96, 0.82]) as dino:
                    report = run_owlv2_sku_query(
                        "water", "a green water bottle", target, shelf.getvalue(),
                        max_boxes=1, owlv2_score_threshold=0.8, dino_fallback=True,
                        dino_confidence_threshold=0.72, debug=False,
                    )

            self.assertEqual(detector.call_args.args[-1], None)
            dino.assert_called_once_with(target, shelf.getvalue(), candidates)
            self.assertEqual([box["index"] for box in report["detected_boxes"]], [2])

    def test_dino_runtime_is_loaded_once_and_reused(self):
        from vision import dino

        previous = dino._DINO_RUNTIME
        dino._DINO_RUNTIME = None
        runtime = (object(), object(), "cpu")
        try:
            with patch.object(dino, "load_dino_runtime", return_value=runtime) as loader:
                self.assertIs(dino.get_dino_runtime(), runtime)
                self.assertIs(dino.preload_dino(), runtime)
                loader.assert_called_once()
        finally:
            dino._DINO_RUNTIME = previous

    def test_image_stitch_builds_a_mosaic_from_overlapping_photos(self):
        from PIL import ImageDraw
        from vision.image_stitch import run_image_stitch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shelf = Image.new("RGB", (420, 180), "white")
            draw = ImageDraw.Draw(shelf)
            for x in range(20, 401, 38):
                for y in range(20, 161, 35):
                    draw.rectangle((x, y, x + 13, y + 13), fill=((x * 3) % 255, (y * 7) % 255, (x + y) % 255))
                    draw.line((x, y, x + 13, y + 13), fill="black", width=2)
            left_path, right_path = directory / "left.png", directory / "right.png"
            shelf.crop((0, 0, 270, 180)).save(left_path)
            shelf.crop((150, 0, 420, 180)).save(right_path)

            report = run_image_stitch([left_path, right_path], directory / "output")

            self.assertTrue((directory / "output" / "stitched.png").is_file())
            self.assertGreater(report["width"], 350)
            self.assertGreaterEqual(report["height"], 180)
            self.assertLess(report["height"], 190)

    def test_image_stitch_writes_graphcut_seam_debug_image(self):
        from PIL import ImageDraw
        from vision.image_stitch import run_image_stitch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shelf = Image.new("RGB", (420, 180), "white")
            draw = ImageDraw.Draw(shelf)
            for x in range(20, 401, 38):
                for y in range(20, 161, 35):
                    draw.rectangle((x, y, x + 13, y + 13), fill=((x * 3) % 255, (y * 7) % 255, (x + y) % 255))
                    draw.line((x, y, x + 13, y + 13), fill="black", width=2)
            left_path, right_path = directory / "left.png", directory / "right.png"
            shelf.crop((0, 0, 270, 180)).save(left_path)
            shelf.crop((150, 0, 420, 180)).save(right_path)

            report = run_image_stitch([left_path, right_path], directory / "output")

            self.assertEqual(report["rendering"]["seam_method"], "graphcut_colorgrad")
            self.assertEqual(report["rendering"]["blend_bands"], 3)
            self.assertTrue((directory / "output" / "seams.png").is_file())

    def test_image_stitch_graphcut_assigns_both_sources_in_a_full_overlap(self):
        import numpy as np
        from vision.image_stitch import run_image_stitch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left_path, right_path = directory / "left.png", directory / "right.png"
            Image.new("RGB", (120, 80), (220, 30, 30)).save(left_path)
            Image.new("RGB", (120, 80), (30, 30, 220)).save(right_path)
            metrics = {"matches": 30, "inliers": 24, "inlier_ratio": 0.8}
            with patch("vision.image_stitch.pair_homography", return_value=(np.eye(3), metrics)):
                report = run_image_stitch([left_path, right_path], directory / "output")
            self.assertEqual(report["rendering"]["seam_overlap_pixels"], [0])

    def test_image_stitch_uses_the_largest_unordered_match_group(self):
        import numpy as np
        from vision.reference_photo_align import AlignmentError
        from vision.image_stitch import run_image_stitch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for index in range(4):
                path = directory / f"input-{index}.png"
                Image.new("RGB", (40, 20), (index + 1, 0, 0)).save(path)
                paths.append(path)
            metrics = {"matches": 30, "inliers": 24, "inlier_ratio": 0.8}

            def pair(source, destination):
                source_index = int(source[0, 0, 2]) - 1
                destination_index = int(destination[0, 0, 2]) - 1
                if (source_index, destination_index) in {(1, 0), (3, 1)}:
                    return np.eye(3), metrics
                raise AlignmentError("no shared shelf area")

            with patch(
                "vision.image_stitch.pair_homography",
                side_effect=pair,
            ):
                report = run_image_stitch(paths, directory / "output")

            self.assertEqual(report["used_indices"], [0, 1, 3])
            self.assertEqual(
                report["pairs"],
                [
                    {"source_index": 1, "destination_index": 0, **metrics},
                    {"source_index": 3, "destination_index": 1, **metrics},
                ],
            )
            self.assertEqual(
                report["skipped"],
                [{"index": 2, "reason": "No reliable alignment with selected photo group"}],
            )

    def test_image_stitch_uses_selected_main_plane(self):
        import numpy as np
        from vision.reference_photo_align import AlignmentError
        from vision.image_stitch import run_image_stitch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for index in range(5):
                path = directory / f"input-{index}.png"
                Image.new("RGB", (40, 20), (index + 1, 0, 0)).save(path)
                paths.append(path)
            metrics = {"matches": 30, "inliers": 24, "inlier_ratio": 0.8}

            def pair(source, destination):
                source_index = int(source[0, 0, 2]) - 1
                destination_index = int(destination[0, 0, 2]) - 1
                if (source_index, destination_index) in {(1, 0), (3, 1), (4, 2)}:
                    return np.eye(3), metrics
                raise AlignmentError("no shared shelf area")

            with patch("vision.image_stitch.pair_homography", side_effect=pair):
                report = run_image_stitch(paths, directory / "output", main_index=2)

            self.assertEqual(report["main_index"], 2)
            self.assertEqual(report["used_indices"], [2, 4])
            self.assertEqual([item["index"] for item in report["skipped"]], [0, 1, 3])

    def test_lab_difference_ignores_uniform_lighting_shift(self):
        import cv2
        import numpy as np
        from vision.cv_restock_position import compute_difference_mask

        baseline = np.full((40, 40, 3), (50, 80, 110), dtype=np.uint8)
        current = cv2.convertScaleAbs(baseline, alpha=1.08, beta=12)

        _, mask = compute_difference_mask(baseline, current, 12.0)

        self.assertEqual(cv2.countNonZero(mask), 0)

    def test_lab_difference_marks_local_packaging_change(self):
        import cv2
        import numpy as np
        from vision.cv_restock_position import compute_difference_mask

        baseline = np.full((80, 80, 3), (50, 80, 110), dtype=np.uint8)
        current = baseline.copy()
        cv2.rectangle(current, (20, 20), (59, 59), (220, 30, 20), -1)

        _, mask = compute_difference_mask(baseline, current, 12.0)

        self.assertGreater(cv2.countNonZero(mask[20:60, 20:60]), 1200)
        self.assertEqual(cv2.countNonZero(mask[:15, :15]), 0)

    def test_inspection_report_uses_stable_slot_ids(self):
        import cv2
        import numpy as np
        from vision import cv_restock_position

        with tempfile.TemporaryDirectory() as temporary:
            temp_path = Path(temporary)
            current_path = temp_path / "current.png"
            current = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.rectangle(current, (0, 10), (9, 39), (255, 255, 255), -1)
            cv2.rectangle(current, (10, 10), (39, 39), (255, 255, 255), -1)
            cv2.imwrite(str(current_path), current)
            baseline = np.zeros((100, 100, 3), dtype=np.uint8)
            calibration = {
                "slots": [
                    {
                        "slot_id": "1-0-2-20",
                        "expected_sku": "tea",
                        "bbox": {"x": 0, "y": 10, "width": 10, "height": 30},
                    },
                    {
                        "slot_id": "1-0-2-43",
                        "expected_sku": "tea",
                        "bbox": {"x": 10, "y": 10, "width": 30, "height": 30},
                    },
                    {
                        "slot_id": "1-0-2-65",
                        "expected_sku": "cola",
                        "bbox": {"x": 60, "y": 10, "width": 30, "height": 30},
                    },
                ]
            }
            best_match = {
                "shelf_id": 1,
                "face": 0,
                "baseline_image": baseline,
                "baseline_path": str(current_path),
                "homography": np.eye(3),
                "inliers": 20,
                "inlier_ratio": 0.8,
                "coverage": 0.5,
                "calibration_data": calibration,
            }
            config = {
                "analysis_center_ratio": 0.8,
                "dino_confidence_threshold": 0.72,
                "ambiguity_margin": 0.05,
                "vlm_fallback": False,
                "vlm_top_k": 4,
            }
            with patch.object(cv_restock_position, "find_best_match", return_value=best_match), \
                    patch.object(cv_restock_position, "rank_slot_crop", return_value=[("cola", 0.88)]):
                report = cv_restock_position.run_inspection(
                    current_path, config, temp_path / "run"
                )

            self.assertEqual(report["run_id"], "run")
            self.assertEqual(report["shelf_id"], 1)
            self.assertEqual(report["face"], 0)
            self.assertEqual(len(report["slots"]), 1)
            self.assertEqual(report["analysis"]["skipped_edge_slots"], 1)
            self.assertEqual(report["analysis"]["roi"], {"x": 10, "y": 10, "width": 80, "height": 80})
            changed = report["slots"][0]
            self.assertEqual(changed["slot_id"], "1-0-2-43")
            self.assertEqual(changed["actual_sku"], "cola")
            self.assertEqual(changed["status"], "摆放错误")
            self.assertTrue(changed["selected"])
            self.assertGreaterEqual(changed["difference_ratio"], 0.15)
            self.assertTrue((temp_path / "run" / "result.json").is_file())
            self.assertEqual(
                report["artifacts"],
                {
                    "result": "result.json",
                    "result_overlay": "result_overlay.png",
                    "aligned_reference": "aligned_reference.png",
                    "current_overlap": "current_overlap.png",
                    "difference": "difference.png",
                    "candidate_boxes": "candidate_boxes.png",
                },
            )
            self.assertTrue((temp_path / "run" / "result_overlay.png").is_file())

    def test_inspection_production_mode_creates_no_artifacts(self):
        import cv2
        import numpy as np
        from vision import cv_restock_position

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            current_path = directory / "current.png"
            current = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.rectangle(current, (20, 20), (49, 49), (255, 255, 255), -1)
            cv2.imwrite(str(current_path), current)
            best_match = {
                "shelf_id": 1,
                "face": 0,
                "baseline_image": np.zeros((100, 100, 3), dtype=np.uint8),
                "homography": np.eye(3),
                "inliers": 20,
                "inlier_ratio": 0.8,
                "coverage": 0.5,
                "calibration_data": {
                    "slots": [{
                        "slot_id": "1-0-2-43",
                        "expected_sku": "tea",
                        "bbox": {"x": 20, "y": 20, "width": 30, "height": 30},
                    }]
                },
            }
            config = {
                "analysis_center_ratio": 1,
                "lab_distance_threshold": 12,
                "slot_change_ratio_threshold": 0.15,
                "dino_confidence_threshold": 0.72,
                "ambiguity_margin": 0.05,
                "vlm_fallback": False,
            }
            with patch.object(cv_restock_position, "find_best_match", return_value=best_match), \
                    patch.object(cv_restock_position, "rank_slot_crop", return_value=[("cola", 0.88)]):
                report = cv_restock_position.run_inspection(current_path, config, debug=False)

            self.assertNotIn("run_id", report)
            self.assertNotIn("artifacts", report)
            self.assertEqual(report["slots"][0]["slot_id"], "1-0-2-43")
            self.assertEqual(list(directory.iterdir()), [current_path])

    def test_sku_query_production_mode_creates_no_artifacts(self):
        from vision.vlm_sku_query import ProviderResponse, run_vlm_sku_query

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target_path = directory / "target.png"
            shelf_path = directory / "shelf.png"
            Image.new("RGB", (20, 20), "green").save(target_path)
            Image.new("RGB", (100, 50), "white").save(shelf_path)
            response = ProviderResponse(
                provider="ark", model="test-model", content="<bbox>10 20 30 40</bbox>",
                request_id="request", usage=None, request_seconds=0.1,
            )
            with patch("vision.vlm_sku_query.request_ark", return_value=response):
                report = run_vlm_sku_query(
                    "sprite", target_path, shelf_path, "ark", "test-model",
                    output_dir=None, debug=False,
                )

            self.assertNotIn("run_id", report)
            self.assertNotIn("artifacts", report)
            self.assertNotIn("raw_response", report)
            self.assertEqual(report["detected_boxes"][0]["label"], "sprite")
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["shelf.png", "target.png"])

    def test_owlv2_prompt_generation_uses_visual_description_not_pinyin(self):
        from vision.vlm_sku_query import ProviderResponse, generate_owlv2_prompt

        sample = BytesIO()
        Image.new("RGB", (20, 20), "green").save(sample, format="PNG")
        response = ProviderResponse(
            provider="ark", model="test-model", content="green plastic tea bottle",
            request_id="request", usage=None, request_seconds=0.1,
        )
        with patch("vision.vlm_sku_query.request_ark", return_value=response) as request:
            prompt = generate_owlv2_prompt("醒目", [sample.getvalue()])

        self.assertEqual(prompt, "green plastic tea bottle")
        instruction = request.call_args.args[0][0]["text"]
        self.assertIn("不要把它当作检索词", instruction)
        self.assertIn("不要输出 Xingmu", instruction)
        self.assertIn("自由文本对象描述", instruction)

    def test_grouped_owlv2_prompt_generation(self):
        from vision.vlm_sku_query import ProviderResponse, generate_grouped_owlv2_prompts

        blue_primary, blue_slot, orange_primary, orange_slot = (BytesIO() for _ in range(4))
        Image.new("RGB", (20, 40), "blue").save(blue_primary, format="PNG")
        Image.new("RGB", (40, 20), "navy").save(blue_slot, format="PNG")
        Image.new("RGB", (20, 40), "orange").save(orange_primary, format="PNG")
        Image.new("RGB", (40, 20), "red").save(orange_slot, format="PNG")
        response = ProviderResponse(
            provider="ark", model="test-model",
            content=(
                '{"items":[{"index":1,"prompt":"blue berry sports drink bottle"},'
                '{"index":2,"prompt":"orange mango sports drink bottle"}]}'
            ),
            request_id="request", usage=None, request_seconds=0.1,
        )

        with patch("vision.vlm_sku_query.request_ark", return_value=response) as request:
            prompts = generate_grouped_owlv2_prompts([
                ("脉动蓝莓", blue_primary.getvalue(), blue_slot.getvalue()),
                ("脉动芒果", orange_primary.getvalue(), orange_slot.getvalue()),
            ])

        self.assertEqual(prompts, {
            "脉动蓝莓": "blue berry sports drink bottle",
            "脉动芒果": "orange mango sports drink bottle",
        })
        request.assert_called_once()
        content = request.call_args.args[0]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 1)
        self.assertIn("JSON", content[0]["text"])
        self.assertIn("比较", content[0]["text"])
        self.assertIn("不要翻译中文 SKU", content[0]["text"])

    def test_grouped_owlv2_prompt_generation_rejects_incomplete_json(self):
        from vision.vlm_sku_query import ProviderResponse, generate_grouped_owlv2_prompts

        sample = BytesIO()
        Image.new("RGB", (20, 20), "blue").save(sample, format="PNG")
        response = ProviderResponse(
            provider="ark", model="test-model",
            content='{"items":[{"index":1,"prompt":"blue drink bottle"}]}',
            request_id="request", usage=None, request_seconds=0.1,
        )

        with patch("vision.vlm_sku_query.request_ark", return_value=response):
            with self.assertRaises(RuntimeError):
                generate_grouped_owlv2_prompts([
                    ("蓝莓", sample.getvalue(), sample.getvalue()),
                    ("芒果", sample.getvalue(), sample.getvalue()),
                ])

    def test_grouped_owlv2_prompt_generation_rejects_boolean_index(self):
        from vision.vlm_sku_query import ProviderResponse, generate_grouped_owlv2_prompts

        sample = BytesIO()
        Image.new("RGB", (20, 20), "blue").save(sample, format="PNG")
        response = ProviderResponse(
            provider="ark", model="test-model",
            content=(
                '{"items":[{"index":true,"prompt":"blue drink bottle"},'
                '{"index":2,"prompt":"green drink bottle"}]}'
            ),
            request_id="request", usage=None, request_seconds=0.1,
        )

        with patch("vision.vlm_sku_query.request_ark", return_value=response):
            with self.assertRaises(RuntimeError):
                generate_grouped_owlv2_prompts([
                    ("蓝莓", sample.getvalue(), sample.getvalue()),
                    ("芒果", sample.getvalue(), sample.getvalue()),
                ])


class CalibrationProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.calibration_dir = Path(self.temp_dir.name)
        self.path = self.calibration_dir / "1.json"
        self.original = {
            "schema_version": 2,
            "shelf_id": 1,
            "shelf_name": "Shelf",
            "faces": {
                "0": {
                    "image_file": "face.png",
                    "image_hash": "sha256:test",
                    "layers": {"2": [{"x": 1, "y": 2}]},
                    "slots": [
                        {
                            "slot_id": "1-0-2-43",
                            "expected_sku": "sprite",
                            "actual_sku": "sprite",
                            "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                        }
                    ],
                }
            },
        }
        self.path.write_text(
            json.dumps(self.original, ensure_ascii=False), encoding="utf-8"
        )
        self.calibration_patch = patch.object(
            calibration_manager, "CALIBRATION_DIR", self.temp_dir.name
        )
        self.calibration_patch.start()

    def tearDown(self):
        self.calibration_patch.stop()
        self.temp_dir.cleanup()

    def test_slot_projection_preserves_calibration_and_bbox(self):
        calibration_manager.sync_shelf_slots(
            1,
            [
                {
                    "slot_id": "1-0-2-43",
                    "face": 0,
                    "expected_sku": "sprite",
                    "actual_sku": None,
                }
            ],
        )

        data = json.loads(self.path.read_text(encoding="utf-8"))
        face = data["faces"]["0"]
        self.assertEqual(face["layers"], self.original["faces"]["0"]["layers"])
        self.assertEqual(face["slots"][0]["bbox"], self.original["faces"]["0"]["slots"][0]["bbox"])
        self.assertIsNone(face["slots"][0]["actual_sku"])

    def test_failed_replace_leaves_original_json_unchanged(self):
        original_bytes = self.path.read_bytes()
        with patch("calibration_manager.os.replace", side_effect=OSError("blocked")):
            with self.assertRaises(OSError):
                calibration_manager.sync_shelf_slots(1, [])

        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(
            [path.name for path in self.calibration_dir.iterdir()], ["1.json"]
        )

    def test_save_calibration_merges_existing_layers_and_slots(self):
        calibration_manager.save_calibration(
            shelf_id=1,
            shelf_name="Shelf",
            face=0,
            layers={"3": [{"x": 3, "y": 4}]},
            slots=[
                {
                    "slot_id": "1-0-2-43",
                    "expected_sku": "sprite",
                    "actual_sku": None,
                }
            ],
        )

        data = json.loads(self.path.read_text(encoding="utf-8"))
        face = data["faces"]["0"]
        self.assertIn("2", face["layers"])
        self.assertIn("3", face["layers"])
        self.assertEqual(face["slots"][0]["bbox"], {"x": 10, "y": 20, "width": 30, "height": 40})


class FixedSlotApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "inventory.db")
        with ShelfDatabase(self.db_path) as db:
            type_id = db.add_shelf_type(
                "test",
                DEFAULT_SHELF_LENGTH,
                DEFAULT_SHELF_WIDTH,
                DEFAULT_SHELF_HEIGHT,
                5,
                DEFAULT_BOTTOM_CLEARANCE,
                DEFAULT_LEVEL_SPACING,
                DEFAULT_PANEL_THICK,
                DEFAULT_BACK_THICK,
                DEFAULT_SHELF_DEPTH_NORMAL,
                DEFAULT_SHELF_DEPTH_BOTTOM,
            )
            db.add_shelf_group("Shelf", shelf_type_id=type_id)
            db.register_skus_batch([{"sku": "cola"}, {"sku": "sprite"}])

        self.db_patch = patch.object(api_server, "DB_PATH", self.db_path)
        self.calibration_patch = patch.object(
            calibration_manager, "CALIBRATION_DIR", self.temp_dir.name
        )
        self.vision_output_patch = patch.object(
            api_server, "VISION_OUTPUT_DIR", Path(self.temp_dir.name) / "runs", create=True
        )
        self.item_images_patch = patch.object(
            api_server, "ITEM_IMAGES_DIR", str(Path(self.temp_dir.name) / "item_images")
        )
        self.sku_images_patch = patch.object(
            api_server, "SKU_IMAGES_DIR", Path(self.temp_dir.name) / "sku_images"
        )
        self.sku_query_output_patch = patch.object(
            api_server, "SKU_QUERY_OUTPUT_DIR", Path(self.temp_dir.name) / "sku-query-runs", create=True
        )
        self.image_stitch_output_patch = patch.object(
            api_server, "IMAGE_STITCH_OUTPUT_DIR", Path(self.temp_dir.name) / "image-stitch-runs", create=True
        )
        self.config_path_patch = patch.object(
            vision_config, "LOCAL_CONFIG_PATH", Path(self.temp_dir.name) / "config.local.yaml"
        )
        self.db_patch.start()
        self.calibration_patch.start()
        self.vision_output_patch.start()
        self.item_images_patch.start()
        self.sku_images_patch.start()
        self.sku_query_output_patch.start()
        self.image_stitch_output_patch.start()
        self.config_path_patch.start()
        vision_config.local_config.cache_clear()
        vision_config.local_api_keys.cache_clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), api_server.ApiHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        vision_config.local_config.cache_clear()
        vision_config.local_api_keys.cache_clear()
        self.config_path_patch.stop()
        self.vision_output_patch.stop()
        self.item_images_patch.stop()
        self.sku_images_patch.stop()
        self.sku_query_output_patch.stop()
        self.image_stitch_output_patch.stop()
        self.calibration_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def request(self, path, method="GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)

    def test_fixed_slot_api_lifecycle(self):
        status, created = self.request(
            "/api/slots",
            "POST",
            {
                "shelf_id": 1,
                "face": 0,
                "level": 2,
                "y_cm": 43,
                "expected_sku": "sprite",
                "actual_sku": "sprite",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["slot"]["slot_id"], "1-0-2-43")
        self.assertEqual(created["slot"]["status"], "正常")

        _, changed = self.request(
            "/api/slots/1-0-2-43", "PUT", {"actual_sku": "cola"}
        )
        self.assertEqual(changed["slot"]["status"], "摆放错误")
        _, misplaced = self.request("/api/misplacements")
        self.assertEqual([item["slot_id"] for item in misplaced["slots"]], ["1-0-2-43"])

        _, taken = self.request("/api/slots/1-0-2-43/take", "POST", {})
        self.assertEqual(taken["slot"]["status"], "缺货")
        _, shortages = self.request("/api/shortages")
        self.assertEqual([item["slot_id"] for item in shortages["slots"]], ["1-0-2-43"])

        _, restocked = self.request("/api/slots/1-0-2-43/restock", "POST", {})
        self.assertEqual(restocked["slot"]["status"], "正常")
        _, world = self.request("/api/slots/1-0-2-43/world-position")
        self.assertEqual(world["slot"]["slot_id"], "1-0-2-43")

    def test_sku_reference_prefers_a_normal_slot_image(self):
        for y_cm, actual_sku in ((10, "cola"), (20, "sprite")):
            self.request(
                "/api/slots",
                "POST",
                {
                    "shelf_id": 1,
                    "face": 0,
                    "level": 2,
                    "y_cm": y_cm,
                    "expected_sku": "sprite",
                    "actual_sku": actual_sku,
                },
            )
            reference_dir = Path(api_server.ITEM_IMAGES_DIR) / f"1-0-2-{y_cm}"
            reference_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), "green").save(reference_dir / "0.png")

        sku, slot_id, _reference = api_server.sku_query_reference("sprite")

        self.assertEqual(sku, "sprite")
        self.assertEqual(slot_id, "1-0-2-20")

    def test_sku_image_upload_is_stored_and_served(self):
        output = BytesIO()
        Image.new("RGB", (2, 2), "red").save(output, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        status, response = self.request("/api/skus", "POST", {
            "sku": "tea", "reference_image_data": image_data, "grasp_method": "吸盘",
        })
        self.assertEqual(status, 201)
        sku = next(item for item in response["state"]["skus"] if item["sku"] == "tea")
        self.assertEqual(sku["reference_image_path"], "data/sku_images/tea.png")
        self.assertEqual(sku["grasp_method"], "吸盘")
        reference_sku, reference_slot_id, reference_path = api_server.sku_query_reference("tea")
        self.assertEqual((reference_sku, reference_slot_id), ("tea", None))
        self.assertEqual(reference_path, api_server.SKU_IMAGES_DIR / "tea.png")
        with urllib.request.urlopen(self.base_url + "/api/sku-images/tea") as image_response:
            self.assertEqual(Image.open(BytesIO(image_response.read())).convert("RGB").getpixel((0, 0)), (255, 0, 0))

    def test_sku_folder_import_creates_and_replaces_sku_images(self):
        directory = Path(self.temp_dir.name) / "sku-import"
        directory.mkdir()
        Image.new("RGB", (2, 2), "blue").save(directory / "tea.png")

        status, body = self.request("/api/skus/import-image-directory", "POST", {"directory": str(directory)})

        self.assertEqual(status, 201)
        tea = next(item for item in body["state"]["skus"] if item["sku"] == "tea")
        self.assertEqual(tea["grasp_method"], "夹爪")
        self.assertEqual(Image.open(api_server.SKU_IMAGES_DIR / "tea.png").convert("RGB").getpixel((0, 0)), (0, 0, 255))

    def _create_grouped_prompt_sources(self):
        api_server.SKU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for y_cm, sku, color in ((10, "cola", "red"), (20, "sprite", "green")):
            self.request("/api/slots", "POST", {
                "shelf_id": 1, "face": 0, "level": 2, "y_cm": y_cm,
                "expected_sku": sku, "actual_sku": sku,
            })
            item_dir = Path(api_server.ITEM_IMAGES_DIR) / f"1-0-2-{y_cm}"
            item_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), color).save(item_dir / "0.png")
            Image.new("RGB", (20, 20), color).save(api_server.SKU_IMAGES_DIR / f"{sku}.png")

    def test_grouped_sku_prompt_generation(self):
        self._create_grouped_prompt_sources()
        generated = {"cola": "red cola bottle", "sprite": "green lemon soda bottle"}

        with patch("api_server.generate_grouped_owlv2_prompts", return_value=generated, create=True) as grouped, \
                patch("api_server.generate_owlv2_prompt", return_value="unused"):
            status, body = self.request(
                "/api/skus/prompts/generate", "POST", {"skus": ["cola", "sprite"]}
            )

        self.assertEqual(status, 201)
        self.assertEqual(body["drafts"], [
            {"sku": "cola", "owlv2_prompt": "red cola bottle"},
            {"sku": "sprite", "owlv2_prompt": "green lemon soda bottle"},
        ])
        grouped.assert_called_once()
        self.assertEqual(len(grouped.call_args.args[0]), 2)
        self.assertTrue(all(len(item) == 3 for item in grouped.call_args.args[0]))

    def test_grouped_sku_prompt_generation_skips_incomplete_items(self):
        self._create_grouped_prompt_sources()
        (api_server.SKU_IMAGES_DIR / "sprite.png").unlink()

        with patch("api_server.generate_grouped_owlv2_prompts", create=True) as grouped, \
                patch("api_server.generate_owlv2_prompt", return_value="unused"):
            status, body = self.request(
                "/api/skus/prompts/generate", "POST", {"skus": ["cola", "sprite"]}
            )

        self.assertEqual(status, 201)
        self.assertEqual(body["drafts"], [])
        self.assertIn({"sku": "sprite", "reason": "No SKU primary image"}, body["skipped"])
        self.assertIn({"sku": "cola", "reason": "Need at least two complete SKUs"}, body["skipped"])
        grouped.assert_not_called()

    def test_grouped_sku_prompt_generation_rejects_more_than_eight(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(
                "/api/skus/prompts/generate", "POST", {"skus": [f"sku-{index}" for index in range(9)]}
            )

        self.assertEqual(error.exception.code, 400)
        self.assertIn("between 2 and 8", error.exception.read().decode())

    def test_batch_delete_replaces_slot_skus_with_unknown(self):
        self.request("/api/slots", "POST", {
            "shelf_id": 1, "face": 0, "level": 2, "y_cm": 30,
            "expected_sku": "cola", "actual_sku": "cola",
        })

        status, body = self.request("/api/skus/delete", "POST", {"skus": ["cola"]})

        self.assertEqual(status, 200)
        self.assertEqual(body["state"]["slots"][0]["expected_sku"], "unknown")
        self.assertEqual(body["state"]["slots"][0]["actual_sku"], "unknown")

    def test_sku_batch_save_updates_qwen_grounding_prompt(self):
        status, body = self.request("/api/skus/batch", "POST", {
            "skus": [{
                "sku": "cola", "original_sku": "cola", "category": "", "mesh_file": "", "tex_file": "",
                "owlv2_prompt": "a cola bottle", "qwen_grounding_prompt": "深色瓶身，红色标签，白色品牌字样",
                "reference_image_path": "", "grasp_method": "夹爪",
            }],
        })

        self.assertEqual(status, 200)
        cola = next(item for item in body["state"]["skus"] if item["sku"] == "cola")
        self.assertEqual(cola["qwen_grounding_prompt"], "深色瓶身，红色标签，白色品牌字样")

    def test_vision_run_is_read_only_until_selected_result_is_applied(self):
        self.request(
            "/api/slots",
            "POST",
            {
                "shelf_id": 1,
                "face": 0,
                "level": 2,
                "y_cm": 43,
                "expected_sku": "sprite",
                "actual_sku": "sprite",
            },
        )
        _, config = self.request("/api/vision/config")
        self.assertNotIn("api_keys", config)
        _, saved_config = self.request(
            "/api/vision/config", "PUT", {
                "min_current_coverage": 0.03,
                "vlm_fallback": True,
                "vlm_top_k": 6,
            }
        )
        self.assertEqual(saved_config["inspection"]["min_current_coverage"], 0.03)
        self.assertTrue(saved_config["inspection"]["vlm_fallback"])
        self.assertEqual(saved_config["inspection"]["vlm_top_k"], 6)

        report = {
            "run_id": "test-run",
            "shelf_id": 1,
            "face": 0,
            "slots": [
                {
                    "slot_id": "1-0-2-43",
                    "expected_sku": "sprite",
                    "actual_sku": None,
                    "status": "缺货",
                    "source": "dino",
                    "confidence": 0.9,
                    "reason": "dino_match",
                    "selected": True,
                }
            ],
            "artifacts": {"result": "result.json"},
        }

        def fake_run(_current_image, _config, output_dir, *, debug):
            self.assertTrue(debug)
            output_dir.mkdir(parents=True)
            result = {**report, "run_id": output_dir.name}
            (output_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
            return result

        image_bytes = BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()
        with patch.object(api_server, "run_inspection", side_effect=fake_run):
            _, response = self.request(
                "/api/vision/inspect", "POST", {"image_data": image_data, "debug": True}
            )
        run_id = response["report"]["run_id"]
        self.assertTrue(run_id)
        _, stored_report = self.request(f"/api/vision/runs/{run_id}/result")
        self.assertEqual(stored_report["run_id"], run_id)
        _, stored_report_artifact = self.request(
            f"/api/vision/runs/{run_id}/artifact/result.json"
        )
        self.assertEqual(stored_report_artifact["run_id"], run_id)

        with ShelfDatabase(self.db_path) as db:
            self.assertEqual(db.get_slot_by_id("1-0-2-43").actual_sku, "sprite")

        _, applied = self.request(
            f"/api/vision/runs/{run_id}/apply",
            "POST",
            {"slot_ids": ["1-0-2-43"]},
        )
        self.assertEqual(applied["slots"][0]["status"], "缺货")
        _, applied_report = self.request(f"/api/vision/runs/{run_id}/result")
        self.assertFalse(applied_report["slots"][0]["selected"])
        with ShelfDatabase(self.db_path) as db:
            self.assertIsNone(db.get_slot_by_id("1-0-2-43").actual_sku)
        calibration = json.loads((Path(self.temp_dir.name) / "1.json").read_text())
        self.assertIsNone(calibration["faces"]["0"]["slots"][0]["actual_sku"])

    def test_vision_production_request_does_not_create_a_run(self):
        image_bytes = BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()

        def fake_run(current_image, _config, output_dir, *, debug):
            self.assertFalse(debug)
            self.assertIsNone(output_dir)
            self.assertEqual(current_image.shape[:2], (1, 1))
            return {"shelf_id": 1, "face": 0, "slots": []}

        with patch.object(api_server, "run_inspection", side_effect=fake_run):
            status, response = self.request("/api/vision/inspect", "POST", {"image_data": image_data})

        self.assertEqual(status, 201)
        self.assertNotIn("run_id", response["report"])
        self.assertFalse(Path(api_server.VISION_OUTPUT_DIR).exists())

    def test_vision_runtime_error_returns_json(self):
        image_bytes = BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()

        with patch.object(api_server, "run_inspection", side_effect=RuntimeError("No matching calibrated shelf face found")):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.request("/api/vision/inspect", "POST", {"image_data": image_data})

        self.assertEqual(raised.exception.code, 422)
        self.assertEqual(json.load(raised.exception)["error"], "No matching calibrated shelf face found")

    def test_sku_query_uses_slot_reference_without_changing_inventory(self):
        _, config = self.request("/api/sku-query/config")
        self.assertEqual(config["max_boxes"], 1)
        _, saved_config = self.request(
            "/api/sku-query/config",
            "PUT",
            {"max_boxes": 2, "dino_fallback": True, "dino_confidence_threshold": 0.81},
        )
        self.assertEqual(saved_config["sku_query"]["max_boxes"], 2)
        self.assertTrue(saved_config["sku_query"]["dino_fallback"])

        _, created = self.request(
            "/api/slots",
            "POST",
            {
                "shelf_id": 1,
                "face": 0,
                "level": 2,
                "y_cm": 43,
                "expected_sku": "sprite",
                "actual_sku": "sprite",
            },
        )
        slot_id = created["slot"]["slot_id"]
        reference_dir = Path(api_server.ITEM_IMAGES_DIR) / slot_id
        reference_dir.mkdir(parents=True)
        Image.new("RGB", (2, 2), "green").save(reference_dir / "0.png")

        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()

        def fake_run(sku, target_path, shelf_path, provider, model, output_dir, **options):
            self.assertEqual(sku, "sprite")
            self.assertEqual(target_path, reference_dir / "0.png")
            self.assertTrue(shelf_path.is_file())
            self.assertEqual(provider, "ark")
            self.assertEqual(model, "test-model")
            self.assertEqual(options["max_boxes"], 2)
            self.assertTrue(options["dino_fallback"])
            self.assertEqual(options["dino_confidence_threshold"], 0.81)
            self.assertTrue(options["debug"])
            output_dir.mkdir(parents=True)
            return {
                "run_id": output_dir.name,
                "sku": sku,
                "provider": provider,
                "model": model,
                "request_seconds": 1.2,
                "total_seconds": 1.3,
                "raw_response": "<bbox>1 2 3 4</bbox>",
                "detected_boxes": [],
                "artifacts": {"annotated_matches": "ark/matches.png"},
            }

        with patch.object(api_server, "run_vlm_sku_query", side_effect=fake_run, create=True):
            status, response = self.request(
                "/api/sku-query",
                "POST",
                {"image_data": image_data, "query": slot_id, "provider": "ark", "model": "test-model", "debug": True},
            )

        self.assertEqual(status, 201)
        self.assertEqual(response["report"]["reference_slot_id"], slot_id)
        self.assertEqual(response["report"]["sku"], "sprite")
        with ShelfDatabase(self.db_path) as db:
            self.assertEqual(db.get_slot_by_id(slot_id).actual_sku, "sprite")

    def test_sku_query_production_request_does_not_create_a_run(self):
        slot_id = self.request(
            "/api/slots", "POST", {
                "shelf_id": 1, "face": 0, "level": 2, "y_cm": 43,
                "expected_sku": "sprite", "actual_sku": "sprite",
            },
        )[1]["slot"]["slot_id"]
        reference_dir = Path(api_server.ITEM_IMAGES_DIR) / slot_id
        reference_dir.mkdir(parents=True)
        Image.new("RGB", (2, 2), "green").save(reference_dir / "0.png")
        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()

        def fake_run(sku, target_path, shelf_source, provider, model, output_dir=None, **options):
            self.assertEqual(sku, "sprite")
            self.assertEqual(target_path, reference_dir / "0.png")
            self.assertIsInstance(shelf_source, bytes)
            self.assertIsNone(output_dir)
            self.assertFalse(options["debug"])
            return {"sku": sku, "provider": provider, "model": model, "detected_boxes": []}

        with patch.object(api_server, "run_vlm_sku_query", side_effect=fake_run):
            status, response = self.request(
                "/api/sku-query",
                "POST",
                {"image_data": image_data, "query": slot_id, "provider": "ark", "model": "test-model"},
            )

        self.assertEqual(status, 201)
        self.assertNotIn("run_id", response["report"])
        self.assertFalse(Path(api_server.SKU_QUERY_OUTPUT_DIR).exists())

    def test_local_sku_query_uses_reviewed_prompt(self):
        with ShelfDatabase(self.db_path) as db:
            db.update_sku("sprite", "", "", "", "a green lemon lime soda bottle")
        slot_id = self.request(
            "/api/slots", "POST", {
                "shelf_id": 1, "face": 0, "level": 2, "y_cm": 50,
                "expected_sku": "sprite", "actual_sku": "sprite",
            },
        )[1]["slot"]["slot_id"]
        reference_dir = Path(api_server.ITEM_IMAGES_DIR) / slot_id
        reference_dir.mkdir(parents=True)
        Image.new("RGB", (2, 2), "green").save(reference_dir / "0.png")
        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()

        def fake_run(sku, prompt, target_path, shelf_source, output_dir=None, **options):
            self.assertEqual(sku, "sprite")
            self.assertEqual(prompt, "a green lemon lime soda bottle")
            self.assertIsInstance(shelf_source, bytes)
            self.assertIsNone(output_dir)
            self.assertEqual(options["owlv2_score_threshold"], 0.19)
            self.assertFalse(options["debug"])
            return {"sku": sku, "provider": "local", "model": "google/owlv2-large-patch14-ensemble", "detected_boxes": []}

        with patch.object(api_server, "run_owlv2_sku_query", side_effect=fake_run):
            status, response = self.request(
                "/api/sku-query", "POST", {
                    "image_data": image_data, "query": slot_id, "provider": "local",
                    "config": {"owlv2_score_threshold": 0.19},
                },
            )
        self.assertEqual(status, 201)
        self.assertEqual(response["report"]["provider"], "local")

    def test_image_stitch_returns_a_read_only_result(self):
        image_bytes = BytesIO()
        Image.new("RGB", (8, 6), "white").save(image_bytes, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()

        def fake_stitch(image_paths, output_dir, main_index):
            self.assertEqual(len(image_paths), 2)
            self.assertEqual(main_index, 1)
            output_dir.mkdir(parents=True)
            Image.new("RGB", (12, 6), "white").save(output_dir / "stitched.png")
            return {
                "run_id": output_dir.name,
                "width": 12,
                "height": 6,
                "main_index": main_index,
                "used_indices": [0, 1],
                "skipped": [],
                "artifacts": {"stitched": "stitched.png", "final_image": "stitched.png"},
            }

        with patch.object(api_server, "run_image_stitch", side_effect=fake_stitch, create=True):
            status, response = self.request(
                "/api/image-stitch", "POST", {"images": [image_data, image_data], "main_index": 1}
            )

        self.assertEqual(status, 201)
        self.assertEqual(response["report"]["width"], 12)
        self.assertEqual(response["report"]["main_index"], 1)
        self.assertEqual(response["report"]["artifacts"]["final_image"], "stitched.png")


if __name__ == "__main__":
    unittest.main()
