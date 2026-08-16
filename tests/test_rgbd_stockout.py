import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
from PIL import Image
from shelf_database import ShelfDatabase


class RgbdDetectorTests(unittest.TestCase):
    def test_detector_source_is_project_local(self):
        import vision.rgbd_stockout as stockout

        self.assertEqual(stockout.SOURCE, Path(stockout.__file__).with_name("front_stockout_detector.py"))

    def test_detect_keeps_only_stockout_candidates(self):
        from vision.rgbd_stockout import StockoutCandidate, detect_stockout_candidates

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.full((8, 8), 700, dtype=np.uint16)
        metadata = {"camera_info": {"k": [1, 0, 4, 0, 1, 4, 0, 0, 1]}}
        stockout = type("Assessment", (), {
            "status": "stockout_candidate", "shelf_index": 2, "group_index": 6,
            "x_min": 1, "y_min": 2, "x_max": 4, "y_max": 6, "setback_mm": 111.5,
        })()
        normal = type("Assessment", (), {
            "status": "normal", "shelf_index": 2, "group_index": 7,
            "x_min": 5, "y_min": 2, "x_max": 6, "y_max": 6, "setback_mm": 2.0,
        })()

        with patch("vision.rgbd_stockout._detect_assessments", return_value=([stockout, normal], [], rgb)):
            result = detect_stockout_candidates(rgb, depth, metadata, threshold_mm=60)

        self.assertEqual(result.candidates, [
            StockoutCandidate(2, 6, {"x": 1, "y": 2, "width": 4, "height": 5}, 111.5),
        ])


class RgbdDecisionTests(unittest.TestCase):
    def test_sku_references_prefer_sku_image_then_slot_crop(self):
        from vision.rgbd_stockout_sku import sku_references

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item_images = root / "data" / "item_images"
            sku_image = root / "data" / "sku_images" / "cola.png"
            sku_image.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2), "red").save(sku_image)
            with ShelfDatabase(root / "inventory.db") as db:
                db.register_sku("cola", reference_image_path="data/sku_images/cola.png")
                db.register_sku("tea")
                shelf_id = db.add_shelf_group("test")
                slot_id = db.create_slot(shelf_id, 0, 0, 10, "tea", "tea")
            fallback = item_images / slot_id / "0.png"
            fallback.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2), "green").save(fallback)

            references = sku_references(root / "inventory.db", item_images)
            self.assertEqual(references["cola"], sku_image)
            self.assertEqual(references["tea"], fallback)

    def test_dino_is_decisive_only_for_high_clear_winner(self):
        from vision.rgbd_stockout_sku import dino_is_decisive

        self.assertTrue(dino_is_decisive([("cola", 0.80), ("tea", 0.75)]))
        self.assertFalse(dino_is_decisive([("cola", 0.79), ("tea", 0.20)]))
        self.assertFalse(dino_is_decisive([("cola", 0.95), ("tea", 0.91)]))

    def test_qwen_choice_accepts_only_presented_numbers_or_unknown(self):
        from vision.local_qwen import parse_candidate_choice

        self.assertEqual(parse_candidate_choice('{"choice": 2}', 3), 2)
        self.assertEqual(parse_candidate_choice('{"choice": "2"}', 3), 2)
        self.assertIsNone(parse_candidate_choice("unknown", 3))
        self.assertIsNone(parse_candidate_choice('{"choice": 4}', 3))
        self.assertIsNone(parse_candidate_choice('{"sku": "invented"}', 3))

    def test_qwen_template_disables_thinking(self):
        from vision.local_qwen import render_chat_prompt

        processor = Mock()
        render_chat_prompt(processor, [{"role": "user", "content": []}])

        self.assertFalse(processor.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_full_qwen_image_marks_only_its_candidate_box(self):
        from vision.rgbd_stockout_sku import annotated_full_image
        import cv2

        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        image = cv2.imdecode(np.frombuffer(annotated_full_image(rgb, {"x": 8, "y": 9, "width": 12, "height": 14}), dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertEqual(tuple(image[9, 8]), (0, 0, 255))
        self.assertEqual(tuple(image[0, 0]), (0, 0, 0))


class RgbdApiTests(unittest.TestCase):
    def test_default_sample_root_is_project_relative(self):
        import api_server

        self.assertEqual(api_server.RGBD_SAMPLE_ROOT, Path(api_server.BASE_DIR) / "test_pic" / "rgbd_stockout")

    def test_list_samples_returns_only_complete_directories(self):
        import api_server

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good"
            good.mkdir()
            for suffix in ("_rgb.png", "_depth_raw.png", "_metadata.yaml"):
                (good / f"good{suffix}").write_bytes(b"x")
            (root / "incomplete").mkdir()
            with patch.object(api_server, "RGBD_SAMPLE_ROOT", root):
                self.assertEqual(api_server.list_rgbd_samples(), ["good"])

    def test_run_endpoint_passes_server_sample_directory(self):
        import api_server

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "test"
            sample.mkdir()
            for suffix in ("_rgb.png", "_depth_raw.png", "_metadata.yaml"):
                (sample / f"test{suffix}").write_bytes(b"x")
            handler = object.__new__(api_server.ApiHandler)
            handler.send_json = unittest.mock.Mock()
            with patch.object(api_server, "RGBD_SAMPLE_ROOT", root), patch.object(api_server, "run_rgbd_stockout", return_value={"sample": "test", "candidates": []}) as run:
                handler._run_rgbd_stockout({"sample": "test", "debug": False})
            handler.send_json.assert_called_once_with({"report": {"sample": "test", "candidates": []}}, 201)
            self.assertEqual(run.call_args.args[0], sample.resolve())
            self.assertFalse(run.call_args.kwargs["debug"])


if __name__ == "__main__":
    unittest.main()
