"""Centered native-pixel geometry, readable scaled overlays and live output."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

import frame_coordinates
from frame_coordinates import annotate_frame, coordinate_system, frame_position
import iphone_yolo
from scene_panel import ScenePanel, CLASS_NAMES
from test_iphone_yolo import SimulatedCapture, authorized_system


def result(bounds):
    box = SimpleNamespace(xyxy=np.array([bounds]), cls=np.array(0), conf=np.array(0.9))
    return SimpleNamespace(boxes=[box], names={0: "cup"})


class GeometryTests(unittest.TestCase):
    def test_center_and_all_four_quadrants(self):
        cases = [([300, 220, 340, 260], {"x": 0, "y": 0}),
                 ([80, 50, 120, 90], {"x": -220, "y": 170}),
                 ([500, 50, 540, 90], {"x": 200, "y": 170}),
                 ([80, 350, 120, 390], {"x": -220, "y": -130}),
                 ([500, 350, 540, 390], {"x": 200, "y": -130})]
        for box, expected in cases:
            with self.subTest(box=box):
                self.assertEqual(frame_position(box, 640, 480), expected)

    def test_uses_box_center_not_corner_and_keeps_native_resolution(self):
        self.assertEqual(frame_position([960, 540, 1160, 640], 1920, 1080), {"x": 100, "y": -50})
        self.assertEqual(coordinate_system(1920, 1080), {
            "origin": [960, 540], "unit": "px", "x_positive": "right",
            "y_positive": "up", "position_reference": "bounding_box_center"})

    def test_odd_resolution_origin_and_frame_edges(self):
        self.assertEqual(frame_position([300, 220, 341, 261], 641, 481), {"x": 0, "y": 0})
        self.assertEqual(frame_position([0, 0, 0, 0], 641, 481), {"x": -320.5, "y": 240.5})
        self.assertEqual(frame_position([641, 481, 641, 481], 641, 481), {"x": 320.5, "y": -240.5})


class OverlayTests(unittest.TestCase):
    def test_axes_are_drawn_without_detections_or_modifying_raw_input(self):
        raw = np.zeros((480, 640, 3), np.uint8)
        output = annotate_frame(raw, [])
        self.assertFalse(np.any(raw))
        self.assertTrue(np.any(output[240, 10:600]))
        self.assertTrue(np.any(output[20:450, 320]))
        self.assertTrue(np.any(output[:, 420]))  # +100 native pixel grid line

    def test_scaled_overlay_marks_box_center_but_displays_native_coordinates(self):
        raw = np.zeros((360, 640, 3), np.uint8)
        item = {"id": 1, "class": "person", "xyxy": [960, 540, 1160, 640],
                "position": {"x": 100, "y": -50}}
        with patch.object(cv2, "drawMarker", wraps=cv2.drawMarker) as marker, \
                patch.object(frame_coordinates, "draw_label", wraps=frame_coordinates.draw_label) as label:
            annotate_frame(raw, [item], native_size=(1920, 1080))
        self.assertEqual(marker.call_args.args[1], (353, 197))
        self.assertTrue(any("(+100.0, -50.0)" in call.args[1] for call in label.call_args_list))

    def test_coordinate_labels_can_avoid_nearby_labels(self):
        raw = np.zeros((240, 320, 3), np.uint8)
        occupied = []
        for text in ["#1 person", "#2 person"]:
            frame_coordinates.draw_label(raw, text, 40, 60, (255, 255, 255), occupied=occupied)
        left, top, right, bottom = occupied[0]
        other_left, other_top, other_right, other_bottom = occupied[1]
        self.assertTrue(right <= other_left or other_right <= left or bottom <= other_top or other_bottom <= top)

    def test_panel_can_page_crowded_scenes_and_handle_custom_model_classes(self):
        panel = ScenePanel()
        items = [{"id": i + 1, "class": "custom-object", "xyxy": [10, 20, 80, 100],
                  "position": {"x": -115, "y": 60}} for i in range(18)]
        frame = np.zeros((240, 320, 3), np.uint8)
        first = panel.render(frame, items, page=0)
        second = panel.render(frame, items, page=1)
        self.assertEqual(first.shape, (760, 1280, 3))
        self.assertTrue(np.any(first[:, 840:] != second[:, 840:]))
        self.assertEqual(CLASS_NAMES["person"], "人")
        self.assertEqual(CLASS_NAMES["toothbrush"], "牙刷")


class LiveOutputTests(unittest.TestCase):
    def test_moving_detection_updates_coordinates_on_each_frame_without_waiting(self):
        capture = SimulatedCapture(np.zeros((480, 640, 3), np.uint8))
        model = MagicMock()
        model.names = {0: "cup"}
        model.predict.side_effect = [[result([80, 50, 120, 90])],
                                     [result([300, 220, 340, 260])],
                                     [result([500, 350, 540, 390])]]
        output = io.StringIO()
        with patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(iphone_yolo, "load_model", return_value=(model, "cpu")), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(iphone_yolo.main(["--headless", "--frames", "3"]), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["detections"][0]["position"] for row in rows],
                         [{"x": -220, "y": 170}, {"x": 0, "y": 0}, {"x": 200, "y": -130}])
        self.assertTrue(all(row["detections"][0]["id"] == 1 for row in rows))
        self.assertTrue(capture.released.is_set())


if __name__ == "__main__":
    unittest.main()
