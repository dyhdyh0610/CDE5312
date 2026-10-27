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
from sony_targets import CUP_LABELS, TREE_LABELS, TargetDetector, target_class
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
        generator = np.random.default_rng(42)
        texture = generator.integers(20, 230, (24, 32, 3), dtype=np.uint8)
        self.textured_frame = cv2.resize(texture, (640, 480), interpolation=cv2.INTER_NEAREST)

    def handheld_frame(self, dx=0, dy=0, angle=0, width=640, height=480):
        frame = cv2.resize(self.textured_frame, (width, height))
        transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        transform[:, 2] += (dx, dy)
        return cv2.warpAffine(frame, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    def test_handheld_translation_noise_and_exposure_do_not_restart_countdown(self):
        for width, height in ((640, 480), (1280, 720)):
            with self.subTest(size=(width, height)):
                controller = GuidanceController()
                generator = np.random.default_rng(17)
                for step, (x, y) in enumerate(((0, 0), (0.02, 0.015), (-0.02, -0.015),
                                             (0.015, -0.01), (-0.015, 0.01), (0.02, 0.015))):
                    frame = self.handheld_frame(x * width, y * height, width=width, height=height)
                    if step:
                        noise = generator.integers(-2, 3, frame.shape, dtype=np.int16)
                        frame = np.clip(frame.astype(np.int16) + noise + 3, 0, 255).astype(np.uint8)
                    ready = controller.observe(frame, step * 0.3)
                    self.assertEqual(ready, step == 5)
                self.assertAlmostEqual(controller.gate.elapsed, 1.5)

    def test_small_handheld_rotations_preserve_countdown_and_object_choices(self):
        controller = GuidanceController()
        for step, angle in enumerate((0, 0.6, -0.6, 0.4, -0.5, 0.5)):
            ready = controller.observe(self.handheld_frame(angle=angle), step * 0.3)
            self.assertEqual(ready, step == 5)
        controller.update_detections([item()])
        self.assertFalse(controller.observe(self.handheld_frame(dx=-9, dy=5, angle=-0.6), 1.6))
        self.assertEqual(controller.phase, "choose")
        self.assertEqual(len(controller.candidates), 1)

    def test_large_camera_movement_still_clears_object_choices(self):
        controller = GuidanceController()
        stabilize(controller, self.textured_frame)
        controller.update_detections([item()])
        self.assertFalse(controller.observe(self.handheld_frame(dx=40, dy=25), 1.6))
        self.assertEqual(controller.phase, "stabilizing")
        self.assertEqual(controller.gate.elapsed, 0)
        self.assertEqual(controller.candidates, [])

    def test_slow_textured_camera_pan_is_compared_to_the_original_anchor(self):
        gate = StabilityGate()
        flags = [gate.observe(self.handheld_frame(dx=step * 2), step * 0.1) for step in range(24)]
        self.assertFalse(any(flags))

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

    def test_stable_scene_refreshes_missing_categories_before_selection(self):
        self.controller.update_detections([item()], completed_at=1.5)
        self.assertFalse(self.controller.observe(self.frame, 1.8))
        self.assertTrue(self.controller.observe(self.frame, 2.3))
        self.controller.update_detections([item(), item("tree:2", "tree"), item("base:3", "cup")],
                                         completed_at=2.3)
        self.assertEqual(self.controller.snapshot(640, 480)["counts"], {"person": 1, "tree": 1, "cup": 1})
        self.assertTrue(self.controller.select(3))

    def test_refresh_keeps_existing_numbers_when_a_new_person_appears(self):
        self.controller.update_detections([item(), item("base:3", "cup")])
        self.controller.update_detections([item("base:2", x=-150), item(), item("base:3", "cup")])
        self.assertTrue(self.controller.select(2))
        self.assertEqual(self.controller.selected_id, "base:3")

    def test_cup_detected_by_both_models_is_only_one_selectable_object(self):
        self.controller.update_detections([item("base:3", "cup")])
        self.controller.update_detections([item("scene:4", "cup"), item("base:3", "cup")])
        state = self.controller.snapshot(640, 480)
        self.assertEqual(state["counts"]["cup"], 1)
        self.assertEqual(state["candidates"][0]["track_id"], "base:3")

    def test_two_distinct_cups_are_not_merged(self):
        self.controller.update_detections([item("base:3", "cup", x=-100), item("scene:4", "cup", x=100)])
        self.assertEqual(self.controller.snapshot(640, 480)["counts"]["cup"], 2)

    def test_selected_cup_is_not_replaced_by_the_other_models_duplicate(self):
        self.controller.update_detections([item("base:3", "cup")])
        self.controller.select(1)
        other_model = dict(item("scene:4", "cup"), confidence=0.99)
        self.controller.update_detections([other_model, item("base:3", "cup")])
        self.assertEqual(self.controller.snapshot(640, 480)["target"]["track_id"], "base:3")
        self.controller.update_detections([other_model])
        self.assertEqual(self.controller.phase, "lost")
        self.assertIsNone(self.controller.snapshot(640, 480)["guidance"])

    def test_empty_stable_scene_retries_without_moving_the_camera(self):
        self.controller.update_detections([], completed_at=1.5)
        self.assertFalse(self.controller.observe(self.frame, 1.8))
        self.assertTrue(self.controller.observe(self.frame, 2.3))
        self.controller.update_detections([item("base:3", "cup")])
        self.assertTrue(self.controller.select(1))

    def test_empty_scene_does_not_rescan_each_video_frame(self):
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
    def mixed_preview(self, width=640, height=284, quality=90):
        sources = [cv2.imread(str(ROOT / "installation_check" / name))
                   for name in ("bus.jpg", "tree_sample.jpg", "cup_sample.jpg")]
        mixed = cv2.hconcat([cv2.resize(frame, (480, 640)) for frame in sources])
        # Recompress a JPEG source, as in the reproducible preview probe.
        _, jpeg = cv2.imencode(".jpg", mixed)
        mixed = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        frame = cv2.resize(mixed, (width, height))
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    def test_mixed_low_resolution_previews_include_people_trees_and_cups(self):
        detector = TargetDetector(ROOT / "yolo26n.pt", ROOT / "yoloe-26n-seg-pf.pt", "cpu")
        for width, height, quality in ((640, 284, 90), (480, 212, 40)):
            with self.subTest(size=(width, height), quality=quality):
                frame = self.mixed_preview(width, height, quality)
                detector.reset()
                labels = set()
                for source, result in detector.track(frame):
                    for box in result.boxes:
                        if box.id is not None:
                            label = result.names[int(box.cls)]
                            labels.add(target_class(label))
                self.assertTrue({"person", "tree", "cup"}.issubset(labels), labels)

    def test_low_confidence_cup_and_tree_can_start_real_tracks(self):
        import torch
        from ultralytics.engine.results import Boxes

        detector = TargetDetector(ROOT / "yolo26n.pt", ROOT / "yoloe-26n-seg-pf.pt", "cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for model, class_id in ((detector.base, detector.base_classes["cup"]),
                                (detector.trees, detector.tree_classes[0])):
            with self.subTest(class_id=class_id):
                boxes = Boxes(torch.tensor([[100, 100, 200, 250, 0.22, class_id]]), frame.shape[:2])
                tracks = model.predictor.trackers[0].update(boxes.numpy(), frame)
                self.assertEqual(len(tracks), 1)

    def test_cup_variant_labels_are_normalized_without_unrelated_objects(self):
        for label in CUP_LABELS:
            self.assertEqual(target_class(label), "cup")
        for label in ("cupcake", "trophy cup", "measuring cup", "bottle", "potted plant", "cat tree"):
            self.assertIsNone(target_class(label))

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

    def run_liveview(self, images, frames=24, target_number=None):
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
                    for step in range(120):
                        image = images[min(step, len(images) - 1)]
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
                args = ["--api-url", f"http://127.0.0.1:{server.server_port}/sony/camera",
                        "--headless", "--frames", str(frames), "--device", "cpu"]
                if target_number is not None:
                    args.extend(["--target-number", str(target_number)])
                status = main(args)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        self.assertEqual(status, 0)
        records = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{"frame"')]
        self.assertEqual(calls[-2:], ["stopLiveview", "stopRecMode"])
        return records

    def test_loopback_liveview_waits_then_locks_and_reports_current_guidance(self):
        image = (ROOT / "installation_check/bus.jpg").read_bytes()
        records = self.run_liveview([image], target_number=1)
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

    def test_loopback_keeps_scanning_until_people_trees_and_cups_are_available(self):
        clear = self.mixed_preview()
        weak = cv2.resize(cv2.resize(clear, (320, 142)), (640, 284))
        _, weak_jpeg = cv2.imencode(".jpg", weak, [cv2.IMWRITE_JPEG_QUALITY, 40])
        _, clear_jpeg = cv2.imencode(".jpg", clear, [cv2.IMWRITE_JPEG_QUALITY, 90])
        records = self.run_liveview([weak_jpeg.tobytes()] * 22 + [clear_jpeg.tobytes()], frames=48)
        choices = [record for record in records if record["shooting"]["phase"] == "choose"]
        self.assertTrue(choices)
        self.assertFalse(all(choices[0]["shooting"]["counts"].values()))
        complete = [record for record in choices if all(record["shooting"]["counts"].values())]
        self.assertTrue(complete, [record["shooting"]["counts"] for record in choices])
        first_choice = records.index(choices[0])
        self.assertTrue(all(record["shooting"]["phase"] == "choose" for record in records[first_choice:]))
        numbers = {}
        for record in choices:
            for candidate in record["shooting"]["candidates"]:
                expected = numbers.setdefault(candidate["track_id"], candidate["number"])
                self.assertEqual(candidate["number"], expected)
        self.assertTrue(all(record["shooting"]["selected_track_id"] is None for record in records))
        self.assertTrue(all(record["shooting"]["guidance"] is None for record in records))
        self.assertEqual(complete[-1]["shooting"]["counts"]["cup"], 1)


if __name__ == "__main__":
    unittest.main()
