import unittest
import numpy as np
import cv2
from tel_vision import (
    VisionConfig,
    GeometryProjector,
    RigidBodyTarget,
    AntiOverexposureColorFilter,
    HUDRenderer,
    HttpSender,
    VisionPipeline
)
from tel_vision.detector import DetectedObject

class TestTELVision(unittest.TestCase):

    def test_config_loading(self):
        config = VisionConfig.from_yaml("config.yaml")
        self.assertIsNotNone(config)
        self.assertEqual(len(config.geometry.row_ratios), 4)
        self.assertEqual(config.color_filter.min_red_pixel_count, 50)
        self.assertTrue(config.http_sender.enabled)

    def test_geometry_projection(self):
        config = VisionConfig().geometry
        projector = GeometryProjector(config)

        # 3 mock TEL markers: Center (cx=300), Left (cx=100), Right (cx=500)
        mock_tels = [
            DetectedObject(cx=300, cy=100, w=60, h=40, box=(270, 80, 330, 120), conf=0.95, cls_id=0, cls_name="TEL"),
            DetectedObject(cx=100, cy=100, w=60, h=40, box=(70, 80, 130, 120), conf=0.92, cls_id=0, cls_name="TEL"),
            DetectedObject(cx=500, cy=100, w=60, h=40, box=(470, 80, 530, 120), conf=0.90, cls_id=0, cls_name="TEL"),
        ]

        frame_shape = (480, 640, 3)
        target = projector.project(mock_tels, frame_shape)

        self.assertTrue(target.acquired)
        self.assertEqual(len(target.tels), 3)
        # Should be sorted Left -> Center -> Right
        self.assertEqual(target.tels[0].cx, 100)
        self.assertEqual(target.tels[1].cx, 300)
        self.assertEqual(target.tels[2].cx, 500)

        # 12 holes projected (3 columns x 4 rows)
        self.assertEqual(len(target.all_holes), 12)
        self.assertEqual(len(target.row1_holes), 3)

        # Row 1 hole for Column 0 (Left TEL)
        # ratio = 2.0, h = 40, down_y = 1.0 => cy + 2.0*40 = 100 + 80 = 180
        hx, hy = target.row1_holes[0]
        self.assertEqual(hx, 100)
        self.assertEqual(hy, 180)

        # Dynamic mask shape
        self.assertEqual(target.dynamic_mask.shape, (480, 640))
        # Mask at hole center should be white (255)
        self.assertEqual(target.dynamic_mask[180, 100], 255)

    def test_color_filter_detection(self):
        config = VisionConfig()
        filter_module = AntiOverexposureColorFilter(config.color_filter, roi_radius=14)

        # Create a blank 480x640 frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Construct target with 3 Row 1 holes at (100, 180), (300, 180), (500, 180)
        target = RigidBodyTarget(
            acquired=True,
            row1_holes={0: (100, 180), 1: (300, 180), 2: (500, 180)},
            dynamic_mask=np.zeros((480, 640), dtype=np.uint8)
        )

        # Paint an overexposed bright red LED light at Center hole (300, 180)
        # Low saturation, high brightness: BGR = (100, 100, 255) -> Red hue in HSV
        cv2.circle(frame, (300, 180), 8, (100, 100, 255), -1)

        result = filter_module.evaluate_target(frame, target)

        # Center hole (column index 1) should be active
        self.assertEqual(result.active_column, 1)
        self.assertEqual(result.active_column_name, "center")
        self.assertGreaterEqual(result.red_scores[1], config.color_filter.min_red_pixel_count)
        self.assertEqual(result.red_scores[0], 0)
        self.assertEqual(result.red_scores[2], 0)

    def test_http_payload_generation(self):
        config = VisionConfig().http_sender
        config.enabled = False  # Avoid real HTTP calls in unit test
        sender = HttpSender(config)

        # Test queueing telemetry without error
        sender.send_telemetry(
            active_target="center",
            target_detected=True,
            fire_control_state="READY",
            target_coords=(300, 180),
            column_index=1
        )
        sender.stop()

if __name__ == "__main__":
    unittest.main()
