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
        self.assertGreater(changed.generation, before.generation)
        for now in [2.6, 3.1, 3.6, 4.09]:
            self.assertFalse(gate.update(frame(100), now).ready)
        self.assertTrue(gate.update(frame(100), 4.1).ready)

    def test_small_camera_noise_is_tolerated_but_slow_drift_is_not(self):
        gate = StabilityGate()
        gate.update(frame(40), 0)
        for index in range(1, 6):
            state = gate.update(frame(40 + index * 5), index * 0.25)
        self.assertGreater(state.generation, 1)
        self.assertFalse(state.ready)
        gate = StabilityGate()
        for index in range(9):
            state = gate.update(frame(40 + index % 4), index * 0.25)
        self.assertTrue(state.ready)

    def test_unobserved_gap_does_not_count_as_stability(self):
        gate = StabilityGate()
        gate.update(frame(), 0)
        state = gate.update(frame(), 3)
        self.assertFalse(state.ready)
        self.assertEqual(state.stable_for, 0)

    def test_local_object_change_resets_stability(self):
        gate = StabilityGate()
        gate.update(frame(), 0)
        changed = frame()
        changed[50:100, 80:140] = 200
        self.assertGreater(gate.update(changed, 0.5).generation, 1)


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
        self.assertGreater(worker.invalidate.call_args.args[0], first_generation)
        for now in [3, 3.5, 4, 4.6, 4.7]:
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
    def test_missing_scene_service_fails_before_opening_camera(self):
        backend = MagicMock()
        backend.check.side_effect = RuntimeError("无法连接本机 Ollama")
        with patch.object(iphone_yolo, "OllamaCaptioner", return_value=backend), \
                patch.object(iphone_yolo, "MacCameras") as cameras, \
                patch.object(cv2, "VideoCapture") as capture:
            with self.assertRaisesRegex(RuntimeError, "Ollama"):
                iphone_yolo.main(["--headless"])
        cameras.assert_not_called()
        capture.assert_not_called()

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

    def test_headless_cli_includes_scene_state_and_releases_camera(self):
        capture = SimulatedCapture(frame())
        backend = MagicMock()
        backend.describe.return_value = "近景左侧有纸盒。"
        output = io.StringIO()
        with patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(iphone_yolo, "OllamaCaptioner", return_value=backend), \
                patch.object(iphone_yolo, "load_model", return_value=(None, None)), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(iphone_yolo.main(["--headless", "--frames", "35", "--stable-seconds", "0.15"]), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(any(row["scene"]["state"] == "ready" for row in rows))
        self.assertTrue(capture.released.is_set())

    def test_still_image_verifies_side_panel_without_opening_camera(self):
        backend = MagicMock()
        backend.describe.return_value = "近景左侧有纸盒，右侧是电脑。"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jpg"
            destination = Path(temporary) / "panel.jpg"
            cv2.imwrite(str(source), frame(80))
            with patch.object(iphone_yolo, "MacCameras") as cameras, \
                    patch.object(iphone_yolo, "OllamaCaptioner", return_value=backend), \
                    patch.object(iphone_yolo, "load_model", return_value=(None, None)), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(iphone_yolo.main(["--image", str(source), "--headless",
                                                  "--output", str(destination)]), 0)
            cameras.assert_not_called()
            self.assertEqual(cv2.imread(str(destination)).shape, (760, 1280, 3))

    def test_live_output_flag_cannot_silently_save_camera_images(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            iphone_yolo.main(["--output", "/tmp/camera.jpg"])

    def test_chinese_panel_can_render_waiting_streaming_and_error_states(self):
        panel = ScenePanel()
        for state, text in [("waiting", ""), ("describing", "纸盒右侧是电脑。"),
                            ("ready", "中间两个人面前各有电脑。"), ("error", "")]:
            scene = {"state": state, "text": text, "error": "连接失败",
                     "required_seconds": 2, "stable_for": 1}
            image = panel.render(frame(), scene, model="test")
            self.assertEqual(image.shape, (760, 1280, 3))
            self.assertGreater(np.count_nonzero(image[:, 840:]), 0)


if __name__ == "__main__":
    unittest.main()
