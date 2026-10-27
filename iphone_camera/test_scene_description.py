"""Motion timing, asynchronous cancellation, local vision protocol and GUI flow."""

from contextlib import redirect_stderr, redirect_stdout
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import iphone_yolo
import scene_description
from scene_description import Cancelled, OllamaCaptioner, SceneController, SceneWorker, StabilityGate
from scene_panel import ScenePanel
from test_iphone_yolo import SimulatedCapture, authorized_system


def frame(value=0):
    return np.full((192, 256, 3), value, dtype=np.uint8)


def textured_frame():
    pixels = np.random.default_rng(42).integers(15, 240, (48, 64, 3), dtype=np.uint8)
    return cv2.resize(pixels, (768, 576), interpolation=cv2.INTER_NEAREST)


def jittered_frame(image, dx=0, dy=0, angle=0):
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    transform[:, 2] += (dx, dy)
    return cv2.warpAffine(image, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Background worker did not reach expected state")


class StabilityTests(unittest.TestCase):
    def test_requires_two_continuous_seconds(self):
        gate = StabilityGate()
        for now in [0, 0.5, 1, 1.5, 1.99]:
            self.assertFalse(gate.update(frame(), now).ready)
        self.assertTrue(gate.update(frame(), 2).ready)

    def test_motion_resets_timer_and_expires_old_description_generation(self):
        gate = StabilityGate()
        for now in [0, 0.5, 1, 1.5, 2]:
            before = gate.update(frame(), now)
        changed = gate.update(frame(100), 2.1)
        self.assertFalse(changed.ready)
        self.assertEqual(changed.generation, before.generation)
        for now in [2.5, 2.89]:
            self.assertFalse(gate.update(frame(100), now).ready)
        changed = gate.update(frame(100), 2.9)
        self.assertGreater(changed.generation, before.generation)
        self.assertEqual(changed.reset_reason, "scene_change")
        for now in [3.4, 3.9, 4.4, 4.89]:
            self.assertFalse(gate.update(frame(100), now).ready)
        self.assertTrue(gate.update(frame(100), 4.9).ready)

    def test_small_camera_noise_and_exposure_drift_are_tolerated(self):
        gate = StabilityGate()
        gate.update(frame(40), 0)
        for index in range(1, 6):
            state = gate.update(frame(40 + index * 5), index * 0.25)
        self.assertEqual(state.generation, 1)
        self.assertFalse(state.ready)
        gate = StabilityGate()
        for index in range(9):
            state = gate.update(frame(40 + index % 4), index * 0.25)
        self.assertTrue(state.ready)

    def test_unobserved_gap_does_not_count_as_stability(self):
        gate = StabilityGate()
        gate.update(frame(), 0)
        state = gate.update(frame(), 4)
        self.assertFalse(state.ready)
        self.assertEqual(state.stable_for, 0)

    def test_sustained_large_local_object_change_resets_stability(self):
        gate = StabilityGate()
        gate.update(frame(), 0)
        changed = frame()
        changed[50:125, 60:160] = 200
        for now in [0.5, 0.9]:
            self.assertEqual(gate.update(changed, now).generation, 1)
        self.assertGreater(gate.update(changed, 1.3).generation, 1)


class JitterToleranceTests(unittest.TestCase):
    def test_micro_translation_rotation_and_sensor_noise_keep_two_second_window(self):
        image = textured_frame()
        random = np.random.default_rng(123)
        gate = StabilityGate()
        for index in range(21):
            moved = jittered_frame(image, 5 * np.sin(index * 1.7), 4 * np.sin(index * 0.9),
                                   0.6 * np.sin(index * 1.1))
            noise = random.normal(0, 1.5, moved.shape)
            moved = np.clip(moved.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            state = gate.update(moved, index * 0.1)
            self.assertEqual(state.generation, 1, f"Minor jitter reset at frame {index}")
        self.assertTrue(state.ready)
        self.assertAlmostEqual(state.stable_for, 2)

    def test_continued_jitter_submits_once_and_preserves_description_generation(self):
        image = textured_frame()
        worker = MagicMock()
        controller = SceneController(worker)
        for index in range(41):
            moved = jittered_frame(image, 5 * np.sin(index), 4 * np.sin(index * 0.7),
                                   0.5 * np.sin(index * 0.8))
            controller.update(moved, index * 0.1)
        self.assertEqual(worker.submit.call_count, 1)
        self.assertEqual({call.args[0] for call in worker.invalidate.call_args_list}, {1})

    def test_slow_pan_still_resets_against_fixed_anchor(self):
        image = textured_frame()
        gate = StabilityGate()
        for index in range(21):
            state = gate.update(jittered_frame(image, dx=index * 6), index * 0.1)
        self.assertGreater(state.generation, 1)
        self.assertFalse(state.ready)

    def test_sustained_large_camera_translation_or_rotation_still_resets(self):
        image = textured_frame()
        for kwargs in [{"dx": 120}, {"angle": 12}]:
            with self.subTest(**kwargs):
                gate = StabilityGate()
                for now in [0, 0.5, 1, 1.5, 2]:
                    gate.update(image, now)
                state = gate.update(jittered_frame(image, **kwargs), 2.1)
                self.assertEqual(state.generation, 1)
                for now in [2.5, 2.9]:
                    state = gate.update(jittered_frame(image, **kwargs), now)
                self.assertGreater(state.generation, 1)
                self.assertFalse(state.ready)

    def test_real_object_change_is_not_hidden_by_camera_compensation(self):
        image = textured_frame()
        gate = StabilityGate()
        for now in [0, 0.5, 1, 1.5, 2]:
            gate.update(image, now)
        changed = image.copy()
        changed[140:410, 180:490] = 0
        state = gate.update(jittered_frame(changed, dx=4, dy=-3, angle=0.4), 2.1)
        self.assertEqual(state.generation, 1)
        for now in [2.5, 2.9]:
            state = gate.update(jittered_frame(changed, dx=4, dy=-3, angle=0.4), now)
        self.assertGreater(state.generation, 1)
        self.assertFalse(state.ready)

    def test_one_borderline_outlier_does_not_clear_countdown(self):
        gate = StabilityGate()
        for index in range(19):
            gate.update(frame(), index * 0.1)
        outlier = frame()
        outlier[60:120, 80:160] = 200
        suspect = gate.update(outlier, 1.9)
        self.assertEqual(suspect.generation, 1)
        self.assertFalse(suspect.ready)
        restored = gate.update(frame(), 2)
        self.assertEqual(restored.generation, 1)
        self.assertTrue(restored.ready)

    def test_persistent_borderline_content_change_still_resets(self):
        gate = StabilityGate()
        for index in range(18):
            gate.update(frame(), index * 0.1)
        changed = frame()
        changed[60:120, 80:160] = 200
        for now in [1.8, 1.9, 2.01, 2.5]:
            state = gate.update(changed, now)
            self.assertFalse(state.ready)
            self.assertEqual(state.generation, 1)
        state = gate.update(changed, 2.61)
        self.assertGreater(state.generation, 1)
        self.assertFalse(state.ready)


class RelaxedStabilityTests(unittest.TestCase):
    def test_wider_handheld_jitter_keeps_two_second_window(self):
        image = textured_frame()
        gate = StabilityGate()
        for index in range(21):
            moved = jittered_frame(image, 50 * np.sin(index * 0.8), 25 * np.sin(index * 0.9),
                                   4 * np.sin(index * 0.6))
            state = gate.update(moved, index * 0.1)
            self.assertEqual(state.generation, 1, f"Handheld jitter reset at frame {index}")
        self.assertTrue(state.ready)

    def test_alignment_failure_does_not_turn_minor_jitter_into_scene_change(self):
        image = textured_frame()
        gate = StabilityGate()
        with patch.object(cv2, "findTransformECC", side_effect=cv2.error("no convergence")):
            for index in range(21):
                moved = jittered_frame(image, 4 * np.sin(index), 3 * np.sin(index * 0.7),
                                       0.8 * np.sin(index * 0.8))
                state = gate.update(moved, index * 0.1)
                self.assertEqual(state.generation, 1)
        self.assertTrue(state.ready)

    def test_short_observation_gap_pauses_and_preserves_progress(self):
        gate = StabilityGate()
        for now in [0, 0.4, 0.8]:
            gate.update(frame(), now)
        resumed = gate.update(frame(), 2.4)
        self.assertEqual(resumed.generation, 1)
        self.assertAlmostEqual(resumed.stable_for, 0.8)
        self.assertFalse(resumed.ready)
        for now in [2.8, 3.2, 3.6]:
            resumed = gate.update(frame(), now)
        self.assertTrue(resumed.ready)

    def test_gap_longer_than_three_seconds_still_resets(self):
        gate = StabilityGate()
        gate.update(frame(), 0)
        gate.update(frame(), 0.5)
        resumed = gate.update(frame(), 3.6)
        self.assertEqual(resumed.generation, 2)
        self.assertEqual(resumed.stable_for, 0)

    def test_single_large_visual_outlier_does_not_clear_countdown(self):
        gate = StabilityGate()
        for index in range(19):
            gate.update(frame(), index * 0.1)
        suspect = gate.update(frame(255), 1.9)
        self.assertEqual(suspect.generation, 1)
        self.assertFalse(suspect.ready)
        restored = gate.update(frame(), 2)
        self.assertEqual(restored.generation, 1)
        self.assertTrue(restored.ready)

    def test_small_exposure_and_local_pixel_changes_are_tolerated(self):
        local = frame()
        local[50:100, 80:140] = 200
        for changed in [frame(20), local]:
            with self.subTest(local_change=bool(changed[0, 0, 0] == 0)):
                gate = StabilityGate()
                gate.update(frame(), 0)
                for index in range(1, 21):
                    state = gate.update(changed, index * 0.1)
                    self.assertEqual(state.generation, 1)
                self.assertTrue(state.ready)

    def test_even_large_change_requires_eight_hundred_ms_confirmation(self):
        gate = StabilityGate()
        for now in [0, 0.4, 0.8]:
            gate.update(frame(), now)
        for now in [1, 1.3, 1.7]:
            state = gate.update(frame(100), now)
            self.assertEqual(state.generation, 1)
            self.assertFalse(state.ready)
        confirmed = gate.update(frame(100), 1.8)
        self.assertEqual(confirmed.generation, 2)
        self.assertEqual(confirmed.stable_for, 0)


class ControllerTests(unittest.TestCase):
    def test_only_one_description_per_stable_scene_and_motion_clears_it(self):
        worker = MagicMock()
        controller = SceneController(worker)
        for now in [0, 0.5, 1, 1.5, 1.99]:
            controller.update(frame(), now)
        worker.submit.assert_not_called()
        for now in [2, 2.2, 2.5]:
            controller.update(frame(), now)
        self.assertEqual(worker.submit.call_count, 1)
        first_generation = worker.submit.call_args.args[0]
        controller.update(frame(100), 2.6)
        self.assertEqual(worker.invalidate.call_args.args[0], first_generation)
        for now in [3, 3.4]:
            controller.update(frame(100), now)
        self.assertGreater(worker.invalidate.call_args.args[0], first_generation)
        for now in [3.9, 4.4, 4.9, 5.4]:
            controller.update(frame(100), now)
        self.assertEqual(worker.submit.call_count, 2)

    def test_retry_requires_a_new_stable_interval(self):
        worker = MagicMock()
        controller = SceneController(worker)
        for now in [0, 0.5, 1, 1.5, 2]:
            controller.update(frame(), now)
        controller.retry()
        controller.update(frame(), 2.1)
        self.assertEqual(worker.submit.call_count, 1)
        for now in [2.6, 3.1, 3.6, 4.1]:
            controller.update(frame(), now)
        self.assertEqual(worker.submit.call_count, 2)


class WorkerTests(unittest.TestCase):
    def test_backend_ignoring_cancellation_still_cannot_publish_old_scene(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        class Backend:
            def describe(self, image, on_text, cancel):
                entered.set()
                release.wait(1)
                on_text("过期的纸盒描述")
                finished.set()
                return "过期的纸盒描述"

        worker = SceneWorker(Backend())
        try:
            worker.invalidate(1)
            worker.submit(1, frame())
            self.assertTrue(entered.wait(1))
            worker.invalidate(2)
            release.set()
            self.assertTrue(finished.wait(1))
            self.assertEqual(worker.snapshot().generation, 2)
            self.assertEqual(worker.snapshot().text, "")
            self.assertEqual(worker.snapshot().state, "waiting")
        finally:
            release.set()
            worker.close()

    def test_motion_cancels_stream_and_never_publishes_stale_text(self):
        entered = threading.Event()
        old_cancelled = threading.Event()

        class Backend:
            def describe(self, image, on_text, cancel):
                if image[0, 0, 0] == 0:
                    on_text("旧场景")
                    entered.set()
                    if cancel.wait(1):
                        old_cancelled.set()
                        raise Cancelled()
                    return "过期描述"
                on_text("新场景")
                return "近景右侧是电脑。"

        worker = SceneWorker(Backend())
        try:
            worker.invalidate(1)
            worker.submit(1, frame())
            self.assertTrue(entered.wait(1))
            worker.invalidate(2)
            self.assertEqual(worker.snapshot().text, "")
            worker.submit(2, frame(100))
            wait_until(lambda: worker.snapshot().state == "ready")
            self.assertTrue(old_cancelled.is_set())
            self.assertEqual(worker.snapshot().generation, 2)
            self.assertEqual(worker.snapshot().text, "近景右侧是电脑。")
        finally:
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def test_backend_failure_is_visible_and_does_not_stop_camera(self):
        backend = MagicMock()
        backend.describe.side_effect = RuntimeError("Ollama 已停止")
        worker = SceneWorker(backend)
        try:
            worker.invalidate(1)
            worker.submit(1, frame())
            wait_until(lambda: worker.snapshot().state == "error")
            self.assertIn("Ollama", worker.snapshot().error)
        finally:
            worker.close()


class ProtocolTests(unittest.TestCase):
    def response_session(self, events):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [json.dumps(event).encode() for event in events]
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value = response
        return session

    def test_sends_real_jpeg_only_to_loopback_and_streams_content(self):
        session = self.response_session([
            {"message": {"thinking": "hidden", "content": "近景"}},
            {"message": {"content": "有纸盒。"}},
            {"done": True, "done_reason": "stop"},
        ])
        updates = []
        with patch.object(OllamaCaptioner, "session", return_value=session):
            self.assertEqual(OllamaCaptioner().describe(frame(), updates.append), "近景有纸盒。")
        self.assertIn("近景", updates)
        call = session.post.call_args
        self.assertEqual(call.args[0], "http://127.0.0.1:11434/api/chat")
        self.assertFalse(call.kwargs["allow_redirects"])
        body = call.kwargs["json"]
        jpeg = base64.b64decode(body["messages"][1]["images"][0])
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
        self.assertFalse(body["think"])
        self.assertTrue(body["stream"])
        real_session = OllamaCaptioner.session()
        self.assertFalse(real_session.trust_env)
        real_session.close()

    def test_truncated_stream_is_not_accepted_as_completed_description(self):
        session = self.response_session([{"message": {"content": "一半"}}])
        with patch.object(OllamaCaptioner, "session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "完整描述"):
                OllamaCaptioner().describe(frame())

    def test_rejects_cloud_alias_and_remote_or_text_only_model(self):
        with self.assertRaisesRegex(ValueError, "本地模型"):
            OllamaCaptioner("qwen:cloud")
        for data in [{"remote_host": "https://example.com", "capabilities": ["vision"]},
                     {"capabilities": ["completion"]}]:
            session = self.response_session([])
            session.post.return_value.json.return_value = data
            with patch.object(OllamaCaptioner, "session", return_value=session):
                with self.assertRaises(RuntimeError):
                    OllamaCaptioner().check()


class IntegrationTests(unittest.TestCase):
    def test_coordinates_do_not_call_ollama_or_start_stability_worker(self):
        capture = SimulatedCapture(frame())
        with patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(iphone_yolo, "load_model", return_value=(None, None)), \
                patch.object(scene_description, "SceneWorker") as scenes, \
                patch("requests.sessions.Session.request") as request, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(iphone_yolo.main(["--headless", "--frames", "1"]), 0)
        scenes.assert_not_called()
        request.assert_not_called()
        self.assertTrue(capture.released.is_set())

    def test_real_camera_reader_keeps_observing_while_ui_is_busy(self):
        capture = SimulatedCapture(frame())
        backend = MagicMock()
        backend.describe.return_value = "近景左侧有纸盒。"
        worker = SceneWorker(backend)
        controller = SceneController(worker, seconds=0.15)
        reader = iphone_yolo.CameraFrames(capture)
        reader.observer = controller.update
        reader.start()
        try:
            # No get() calls: observation must be independent of foreground inference.
            wait_until(lambda: worker.snapshot().state == "ready")
            self.assertEqual(backend.describe.call_count, 1)
            self.assertGreaterEqual(controller.as_dict()["stable_for"], 0.15)
        finally:
            reader.close()
            worker.close()
        self.assertTrue(capture.released.is_set())

    def test_headless_cli_includes_coordinates_and_releases_camera(self):
        capture = SimulatedCapture(frame())
        output = io.StringIO()
        with patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(iphone_yolo, "load_model", return_value=(None, None)), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(iphone_yolo.main(["--headless", "--frames", "3"]), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["coordinate_system"]["origin"] == [128, 96] for row in rows))
        self.assertTrue(all("scene" not in row for row in rows))
        self.assertTrue(capture.released.is_set())

    def test_still_image_verifies_side_panel_without_opening_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jpg"
            destination = Path(temporary) / "panel.jpg"
            cv2.imwrite(str(source), frame(80))
            with patch.object(iphone_yolo, "MacCameras") as cameras, \
                    patch.object(iphone_yolo, "load_model", return_value=(None, None)), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(iphone_yolo.main(["--image", str(source), "--headless",
                                                  "--output", str(destination)]), 0)
            cameras.assert_not_called()
            self.assertEqual(cv2.imread(str(destination)).shape, (760, 1280, 3))

    def test_live_output_flag_cannot_silently_save_camera_images(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            iphone_yolo.main(["--output", "/tmp/camera.jpg"])

    def test_chinese_coordinate_panel_renders_empty_and_preview_states(self):
        panel = ScenePanel()
        for detections in (None, []):
            image = panel.render(frame(), detections)
            self.assertEqual(image.shape, (760, 1280, 3))
            self.assertGreater(np.count_nonzero(image[:, 840:]), 0)


if __name__ == "__main__":
    unittest.main()
