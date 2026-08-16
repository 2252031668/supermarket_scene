import unittest

import numpy as np


class HighResolutionMigrationTests(unittest.TestCase):
    def test_transform_bbox_projects_corners_and_clips_to_image(self):
        from tools.migrate_highres_shelf_images import transform_bbox

        result = transform_bbox(
            {"x": 10, "y": 20, "width": 30, "height": 40},
            np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
            width=70,
            height=100,
        )

        self.assertEqual(result, {"x": 20, "y": 40, "width": 50, "height": 60})
