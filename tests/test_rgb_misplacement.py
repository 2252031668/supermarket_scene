import unittest
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


class RgbMisplacementTests(unittest.TestCase):
    def test_response_filters_invalid_low_confidence_and_keeps_best_two(self):
        from vision.rgb_misplacement import parse_misplacement_response

        response = '''{
          "boxes": [
            {"x": 4, "y": 6, "width": 20, "height": 30, "confidence": 0.82, "reason": "与同层相邻包装明显不同"},
            {"x": 2, "y": 3, "width": 20, "height": 30, "confidence": 0.96, "reason": "破坏连续商品序列"},
            {"x": 9, "y": 2, "width": 20, "height": 30, "confidence": 0.75, "reason": "品类不同"},
            {"x": 1, "y": 1, "width": 10, "height": 10, "confidence": 0.69, "reason": "低置信"},
            {"x": 95, "y": 1, "width": 10, "height": 10, "confidence": 0.99, "reason": "越界"}
          ]
        }'''

        result = parse_misplacement_response(response, 100, 80)

        self.assertEqual(result, [
            {"box": {"x": 2, "y": 3, "width": 20, "height": 30}, "confidence": 0.96, "reason": "破坏连续商品序列"},
            {"box": {"x": 4, "y": 6, "width": 20, "height": 30}, "confidence": 0.82, "reason": "与同层相邻包装明显不同"},
        ])

    def test_response_accepts_empty_anomaly_list_and_rejects_invalid_json(self):
        from vision.rgb_misplacement import parse_misplacement_response

        self.assertEqual(parse_misplacement_response('{"boxes": []}', 100, 80), [])
        self.assertEqual(parse_misplacement_response('not json', 100, 80), [])

    def test_response_accepts_qwen_bbox_array_format(self):
        from vision.rgb_misplacement import parse_misplacement_response

        response = '''```json
        [{"bbox_2d": [20, 10, 50, 40], "width": 30, "height": 30, "confidence": 0.9, "reason": "商品与相邻包装不同"}]
        ```'''

        self.assertEqual(parse_misplacement_response(response, 100, 80), [{
            "box": {"x": 20, "y": 10, "width": 30, "height": 30},
            "confidence": 0.9,
            "reason": "商品与相邻包装不同",
        }])

    def test_run_resolves_each_valid_anomaly_as_current_sku(self):
        from vision.rgb_misplacement import run_rgb_misplacement

        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        response = '{"boxes":[{"x":4,"y":6,"width":20,"height":30,"confidence":0.82,"reason":"不同"}]}'
        config = {"minimum_confidence": 0.70, "qwen_model_dir": "unused", "qwen_detector_max_new_tokens": 48}
        with tempfile.TemporaryDirectory() as temporary, \
             patch("vision.rgb_misplacement.detect_misplacement_response", return_value=response), \
             patch("vision.rgb_misplacement.sku_references", return_value={"cola": Path("cola.png")}), \
             patch("vision.rgb_misplacement.resolve_candidate_sku", return_value={"sku": "cola", "source": "dino", "dino_matches": [{"sku": "cola", "confidence": 0.9}]}) as resolve:
            report = run_rgb_misplacement(rgb, "unused.db", "unused", config, Path(temporary) / "run")

        self.assertEqual(report["candidates"][0]["current_sku"], "cola")
        self.assertNotIn("expected_sku", report["candidates"][0])
        self.assertEqual(resolve.call_args.args[1], {"x": 4, "y": 6, "width": 20, "height": 30})


class RgbMisplacementApiTests(unittest.TestCase):
    def test_rgb_misplacement_is_allowed_the_image_request_limit(self):
        import api_server

        handler = object.__new__(api_server.ApiHandler)
        handler.path = "/api/rgb-misplacement/runs"
        handler.read_json = unittest.mock.Mock(return_value={})
        handler._run_rgb_misplacement = unittest.mock.Mock()

        handler.do_POST()

        handler.read_json.assert_called_once_with(20 * 1024 * 1024)

    def test_endpoint_decodes_upload_and_runs_read_only_module(self):
        import api_server

        image = Image.new("RGB", (8, 6), "white")
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        payload = {"image_data": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode(), "debug": False}
        handler = object.__new__(api_server.ApiHandler)
        handler.send_json = unittest.mock.Mock()
        with patch.object(api_server, "run_rgb_misplacement", return_value={"candidates": []}) as run:
            handler._run_rgb_misplacement(payload)

        handler.send_json.assert_called_once_with({"report": {"candidates": []}}, 201)
        self.assertEqual(run.call_args.args[0].shape, (6, 8, 3))
        self.assertFalse(run.call_args.kwargs["debug"])


if __name__ == "__main__":
    unittest.main()
