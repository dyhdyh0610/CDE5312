import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import unittest

import cv2
import numpy as np

from sony_guidance import GuidanceController, GuidancePanel, StabilityGate, movement_guidance
from sony_targets import TREE_LABELS, TargetDetector
from sony_wifi_yolo import ROOT, annotate_frame, main
from test_sony_wifi_yolo import packet


def item(track_id="base:1", kind="person", x=100, y=50, number=1):
    cx, cy = 320 + x, 240 - y
    return {"track_id": track_id, "class": kind, "confidence": 0.9,
            "xyxy": [cx - 30, cy - 40, cx + 30, cy + 40], "position": {"x": x, "y": y}, "number": number}


def stabilize(controller, frame):
    ready = [controller.observe(frame, tick / 10) for tick in (0, 3, 6, 9, 12, 15)]
    return ready


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    def test_exactly_one_point_five_seconds_before_first_detection(self):
        controller = GuidanceController()
        ready = stabilize(controller, self.frame)
        self.assertEqual(ready, [False, False, False, False, False, True])
        controller.update_detections([item()])
        self.assertFalse(controller.observe(self.frame, 1.6))
        self.assertEqual(controller.phase, "choose")

    def test_scene_change_restarts_countdown_and_clears_old_choices(self):
        controller = GuidanceController()
        stabilize(controller, self.frame)
        controller.update_detections([item()])
        changed = self.frame.copy()
        changed[100:350, 100:400] = 220
        self.assertFalse(controller.observe(changed, 1.6))
        self.assertEqual(controller.phase, "stabilizing")
        self.assertEqual(controller.candidates, [])
        self.assertFalse(controller.select(1))
        for tick in (1.9, 2.2, 2.5, 2.8):
            self.assertFalse(controller.observe(changed, tick))
        self.assertTrue(controller.observe(changed, 3.1))

    def test_unobserved_time_does_not_count_as_stability(self):
        gate = StabilityGate()
        self.assertFalse(gate.observe(self.frame, 0))
        self.assertFalse(gate.observe(self.frame, 2))
        self.assertEqual(gate.elapsed, 0)

    def test_slow_initial_detection_does_not_erase_valid_choices(self):
        controller = GuidanceController()
        stabilize(controller, self.frame)
        controller.update_detections([item()], completed_at=4.0)
        self.assertFalse(controller.observe(self.frame, 4.1))
        self.assertEqual(controller.phase, "choose")
        changed = self.frame.copy()
        changed[100:350, 100:400] = 220
        controller.observe(changed, 4.2)
        self.assertEqual(controller.phase, "stabilizing")
        self.assertEqual(controller.candidates, [])

    def test_minor_sensor_noise_and_exposure_are_tolerated(self):
        gate = StabilityGate()
        generator = np.random.default_rng(7)
        gate.observe(self.frame, 0)
        for tick in (0.3, 0.6, 0.9, 1.2, 1.5):
            noise = generator.integers(-2, 3, self.frame.shape, dtype=np.int16)
            frame = np.clip(self.frame.astype(np.int16) + noise + 3, 0, 255).astype(np.uint8)
            ready = gate.observe(frame, tick)
        self.assertTrue(ready)

    def test_slow_camera_pan_cannot_accumulate_a_false_stable_window(self):
        gate = StabilityGate()
        generator = np.random.default_rng(19)
        frame = generator.integers(0, 256, (96, 160, 3), dtype=np.uint8)
        flags = [gate.observe(np.roll(frame, step, axis=1), step * 0.1) for step in range(24)]
        self.assertFalse(any(flags))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.controller = GuidanceController()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stabilize(self.controller, self.frame)

    def test_only_the_three_requested_categories_are_candidates(self):
        self.controller.update_detections([item(), item("tree:2", "tree"), item("base:3", "cup"),
                                           item("base:4", "bottle"), item("base:5", "potted plant")])
        state = self.controller.snapshot(640, 480)
        self.assertEqual(state["counts"], {"person": 1, "tree": 1, "cup": 1})
        self.assertEqual([candidate["number"] for candidate in state["candidates"]], [1, 2, 3])
        self.assertEqual(state["phase"], "choose")
        self.assertIsNone(state["guidance"])

    def test_empty_scene_waits_for_new_scene_or_explicit_reset(self):
        self.controller.update_detections([])
        state = self.controller.snapshot(640, 480)
        self.assertEqual(state["phase"], "empty")
        self.assertIn("未检测到", state["message"])
        self.assertFalse(self.controller.select(1))
        self.assertFalse(self.controller.observe(self.frame, 1.7))

    def test_selection_does_not_issue_directions_from_the_old_choice_frame(self):
        self.controller.update_detections([item()])
        self.assertFalse(self.controller.select(9))
        self.assertTrue(self.controller.select(1))
        state = self.controller.snapshot(640, 480)
        self.assertIsNone(state["target"])
        self.assertIsNone(state["guidance"])

    def test_selected_object_remains_locked_when_same_class_detections_reorder(self):
        self.controller.update_detections([item("base:1", x=-100), item("base:2", x=100)])
        self.controller.select(2)
        changed = np.full_like(self.frame, 255)
        self.assertTrue(self.controller.observe(changed, 1.51))
        self.controller.update_detections([item("base:2", x=30), item("base:1", x=-50)])
        state = self.controller.snapshot(640, 480)
        self.assertEqual(state["selected_track_id"], "base:2")
        self.assertEqual(state["target"]["position"]["x"], 30)
        self.assertIn("向右", state["guidance"]["horizontal"])

    def test_lost_target_never_falls_back_to_another_person_or_stale_guidance(self):
        self.controller.update_detections([item()])
        self.controller.select(1)
        self.controller.update_detections([item(x=0, y=0)])
        self.assertTrue(self.controller.snapshot(640, 480)["guidance"]["centered"])
        self.controller.update_detections([item("base:2", x=0, y=0)])
        state = self.controller.snapshot(640, 480)
        self.assertEqual(state["phase"], "lost")
        self.assertEqual(state["selected_track_id"], "base:1")
        self.assertIsNone(state["guidance"])
        self.assertEqual(self.controller.display_items(), [])
        self.controller.update_detections([item("base:1", x=-100)])
        state = self.controller.snapshot(640, 480)
        self.assertEqual(state["phase"], "tracking")
        self.assertIn("向左", state["guidance"]["horizontal"])

    def test_reset_requires_another_stable_window_and_explicit_selection(self):
        self.controller.update_detections([item()])
        self.controller.select(1)
        self.controller.reset()
        self.assertFalse(self.controller.observe(self.frame, 10))
        self.assertIsNone(self.controller.selected_id)
        self.assertEqual(self.controller.phase, "stabilizing")


class DirectionTests(unittest.TestCase):
    def test_pan_and_tilt_directions_in_all_quadrants(self):
        for x, y, horizontal, vertical in [(100, 80, "向右", "向上"), (-100, 80, "向左", "向上"),
                                            (-100, -80, "向左", "向下"), (100, -80, "向右", "向下")]:
            with self.subTest(x=x, y=y):
                guidance = movement_guidance({"x": x, "y": y}, 640, 480)
                self.assertIn(horizontal, guidance["horizontal"])
                self.assertIn(vertical, guidance["vertical"])
                self.assertFalse(guidance["centered"])

    def test_distance_is_image_plane_pixels_and_magnitude_uses_frame_fraction(self):
        result = movement_guidance({"x": 300, "y": 400}, 1000, 1000)
        self.assertEqual(result["distance_px"], 500)
        self.assertIn("较大幅度", result["horizontal"])
        self.assertIn("30.0%", result["horizontal"])
        self.assertIn("小幅", movement_guidance({"x": 50, "y": 0}, 1000, 1000)["horizontal"])
        self.assertIn("适度", movement_guidance({"x": 150, "y": 0}, 1000, 1000)["horizontal"])

    def test_center_tolerance_and_hysteresis(self):
        self.assertTrue(movement_guidance({"x": 0, "y": 0}, 640, 480)["centered"])
        self.assertTrue(movement_guidance({"x": 21, "y": 0}, 640, 480, was_centered=True)["centered"])
        self.assertFalse(movement_guidance({"x": 21, "y": 0}, 640, 480)["centered"])
        self.assertFalse(movement_guidance({"x": 30, "y": 0}, 640, 480, was_centered=True)["centered"])


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.panel = GuidancePanel()
        self.controller = GuidanceController()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_chinese_waiting_choice_tracking_and_lost_screens_render(self):
        for phase in ("stabilizing", "choose", "tracking", "lost"):
            if phase == "choose":
                stabilize(self.controller, self.frame)
                self.controller.update_detections([item(), item("tree:2", "tree"), item("base:3", "cup")])
            elif phase == "tracking":
                self.controller.select(1)
                self.controller.update_detections([item()])
            elif phase == "lost":
                self.controller.update_detections([])
            state = self.controller.snapshot(640, 480)
            canvas = self.panel.render(annotate_frame(self.frame, self.controller.display_items()), state)
            self.assertEqual(canvas.shape, (760, 1280, 3))
            self.assertEqual(canvas.dtype, np.uint8)

    def test_clicking_a_choice_row_selects_the_correct_candidate(self):
        stabilize(self.controller, self.frame)
        self.controller.update_detections([item("base:1", x=-100), item("base:2", x=100)])
        self.panel.render(self.frame, self.controller.snapshot(640, 480))
        left, top, right, bottom, number, _ = self.panel.hit_regions[1]
        self.panel.mouse(cv2.EVENT_LBUTTONDOWN, int((left + right) / 2), int((top + bottom) / 2), 0, None)
        self.assertEqual(self.panel.take_selection(), number)
        self.assertIsNone(self.panel.take_selection())

    def test_page_contains_clickable_entries_beyond_the_first_seven_objects(self):
        stabilize(self.controller, self.frame)
        self.controller.update_detections([item(f"base:{i}", x=i * 10) for i in range(10)])
        self.panel.page = 1
        self.panel.render(self.frame, self.controller.snapshot(640, 480))
        rows = [region for region in self.panel.hit_regions if region[-1] == -1]
        self.assertEqual([region[-2] for region in rows], [8, 9, 10])


class RealModelTests(unittest.TestCase):
    def test_real_tree_vocabulary_excludes_non_tree_homonyms(self):
        self.assertIn("eucalyptus tree", TREE_LABELS)
        self.assertNotIn("potted plant", TREE_LABELS)
        self.assertNotIn("cat tree", TREE_LABELS)
        self.assertNotIn("shoe tree", TREE_LABELS)
        self.assertNotIn("tree frog", TREE_LABELS)

    def test_cached_models_track_person_tree_and_cup_with_persistent_ids(self):
        detector = TargetDetector(ROOT / "yolo26n.pt", ROOT / "yoloe-26n-seg-pf.pt", "cpu")
        for source, expected in [(ROOT / "installation_check/bus.jpg", "person"),
                                 (ROOT / "installation_check/tree_sample.jpg", "tree"),
                                 (ROOT / "installation_check/cup_sample.jpg", "cup")]:
            self.assertTrue(source.is_file(), f"Prepare the documented local validation image: {source}")
            frame = cv2.imread(str(source))
            detector.reset()
            first = detector.track(frame, expected)[0][1]
            second = detector.track(frame, expected)[0][1]
            self.assertGreater(len(first.boxes), 0)
            self.assertIsNotNone(first.boxes.id)
            self.assertTrue(set(first.boxes.id.int().tolist()) & set(second.boxes.id.int().tolist()))
            if expected == "tree":
                self.assertTrue(all(first.names[int(box.cls.item())] in TREE_LABELS for box in first.boxes))

    def test_loopback_liveview_waits_then_locks_and_reports_current_guidance(self):
        image = (ROOT / "installation_check/bus.jpg").read_bytes()
        calls = []

        class CameraEmulator(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                method = request["method"]
                calls.append(method)
                if method == "getAvailableApiList":
                    result = [["startLiveview"]] if "startRecMode" in calls else [["startRecMode"]]
                elif method == "startLiveview":
                    result = [f"http://127.0.0.1:{self.server.server_port}/liveview"]
                else:
                    result = [0]
                body = json.dumps({"id": request["id"], "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(packet(image))
                        self.wfile.flush()
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), CameraEmulator)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                status = main(["--api-url", f"http://127.0.0.1:{server.server_port}/sony/camera",
                               "--headless", "--frames", "24", "--device", "cpu", "--target-number", "1"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        self.assertEqual(status, 0)
        records = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{"frame"')]
        self.assertEqual(len(records), 24)
        self.assertEqual(records[0]["shooting"]["phase"], "stabilizing")
        self.assertEqual(records[0]["detections"], [])
        tracked = [record for record in records if record["shooting"]["guidance"] is not None]
        self.assertGreater(len(tracked), 0)
        locked_ids = {record["shooting"]["selected_track_id"] for record in tracked}
        self.assertEqual(len(locked_ids), 1)
        for record in tracked:
            self.assertEqual(record["shooting"]["selected_class"], "person")
            self.assertTrue(all(candidate["class"] == "person" for candidate in record["detections"]))
            self.assertIn("向左", record["shooting"]["guidance"]["horizontal"])
        self.assertEqual(calls[-2:], ["stopLiveview", "stopRecMode"])


if __name__ == "__main__":
    unittest.main()
