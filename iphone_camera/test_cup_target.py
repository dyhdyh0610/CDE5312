"""Cup precedence, ROI boundaries, multiple targets and loss handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from iphone_yolo import detect_frame

from cup_target import cup_target, prioritize_cups


def item(number, label, cx, cy, confidence=0.8, half=20):
    return {"id": number, "class": label, "confidence": confidence,
            "xyxy": [cx - half, cy - half, cx + half, cy + half],
            "position": {"x": cx - 500, "y": 250 - cy}}


class CupTargetTests(unittest.TestCase):
    def test_inference_filters_to_named_cup_class_and_discards_others(self):
        model = MagicMock()
        model.names = {0: "person", 7: "cup", 9: "laptop"}
        boxes = [SimpleNamespace(xyxy=np.array([[100, 100, 150, 180]]),
                                 cls=np.array(label), conf=np.array(0.9)) for label in [9, 7, 0]]
        model.predict.return_value = [SimpleNamespace(boxes=boxes, names=model.names)]
        _, detections = detect_frame(model, "cpu", np.zeros((240, 320, 3), np.uint8), 0.35, False)
        self.assertEqual(model.predict.call_args.kwargs["classes"], [7])
        self.assertEqual([d["class"] for d in detections], ["cup"])

    def test_model_without_cup_fails_clearly(self):
        model = MagicMock()
        model.names = {0: "person"}
        with self.assertRaisesRegex(RuntimeError, "不包含 cup"):
            detect_frame(model, "cpu", np.zeros((24, 32, 3), np.uint8), 0.35, False)
        model.predict.assert_not_called()

    def test_cup_wins_over_more_confident_centered_other_objects(self):
        items = [item(1, "laptop", 500, 250, 0.99), item(2, "cup", 100, 100)]
        state = cup_target(items, 1000, 500)
        self.assertEqual(state["target"]["id"], 2)
        self.assertEqual(state["status"], "outside")
        self.assertEqual([i["id"] for i in prioritize_cups(items)], [2, 1])
        self.assertEqual(items[0]["id"], 1)

    def test_center_inside_is_not_enough_when_box_extends_beyond_roi(self):
        state = cup_target([item(1, "cup", 960, 540, half=400)], 1920, 1080)
        self.assertFalse(state["inside"])
        self.assertEqual(state["status"], "too_large")
        self.assertFalse(any(state["directions"].values()))
        self.assertEqual(state["roi_xyxy"], [710, 165, 1210, 915])

    def test_touching_each_boundary_counts_as_inside(self):
        for x, y in [(730, 540), (1190, 540), (960, 185), (960, 895)]:
            with self.subTest(x=x, y=y):
                self.assertTrue(cup_target([item(1, "cup", x, y)], 1920, 1080)["inside"])
        for x, y in [(729, 540), (1191, 540), (960, 184), (960, 896)]:
            self.assertFalse(cup_target([item(1, "cup", x, y)], 1920, 1080)["inside"])

    def test_camera_directions_follow_box_edge_not_just_center(self):
        for x, y, expected in [(729, 540, {"left"}), (1191, 540, {"right"}),
                               (960, 184, {"up"}), (960, 896, {"down"}),
                               (1191, 184, {"up", "right"})]:
            with self.subTest(x=x, y=y):
                state = cup_target([item(1, "cup", x, y)], 1920, 1080)
                self.assertEqual({key for key, active in state["directions"].items() if active}, expected)
                self.assertEqual(state["status"], "outside")

    def test_full_containment_stops_directions(self):
        cup = item(1, "cup", 960, 540)
        cup["xyxy"] = [710, 165, 1210, 915]
        state = cup_target([cup], 1920, 1080)
        self.assertTrue(state["inside"])
        self.assertEqual(state["criterion"], "entire_cup_box_inside_roi")
        self.assertFalse(any(state["directions"].values()))

    def test_small_frame_clips_roi_to_image_and_keeps_native_pixels(self):
        self.assertEqual(cup_target([], 320, 240)["roi_xyxy"], [0, 0, 320, 240])
        self.assertEqual(cup_target([], 1280, 720)["roi_xyxy"], [390, 0, 890, 720])

    def test_nearest_cup_selected_without_following_other_classes(self):
        items = [item(1, "cup", 100, 100), item(2, "person", 500, 250),
                 item(3, "cup", 510, 250)]
        state = cup_target(items, 1000, 500)
        self.assertEqual(state["target"]["id"], 3)
        self.assertEqual(state["cup_count"], 2)

    def test_no_cup_does_not_fall_back_or_retain_stale_target(self):
        cup_target([item(1, "cup", 500, 250)], 1000, 500)
        for items in [[], [item(1, "bottle", 500, 250)], [item(1, "person", 500, 250)]]:
            state = cup_target(items, 1000, 500)
            self.assertEqual(state["status"], "no_cup")
            self.assertIsNone(state["target"])
            self.assertFalse(state["inside"])
            self.assertFalse(any(state["directions"].values()))
        self.assertEqual(cup_target(None, 1000, 500)["status"], "preview")


if __name__ == "__main__":
    unittest.main()
