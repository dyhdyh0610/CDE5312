"""Selection, capture lifecycle and local-model integration without opening hardware."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import iphone_yolo
from mac_cameras import Camera, select_iphone

PHONE = Camera(0, "Alice 的相机", "phone-1", "iPhone18,3", "AVCaptureDeviceTypeExternal", True)
MAC = Camera(1, "Mac Camera", "mac-1", "Mac Camera", "AVCaptureDeviceTypeBuiltInWideAngleCamera")


class SimulatedCapture:
    def __init__(self, frame=None):
        self.frame = np.zeros((24, 32, 3), dtype=np.uint8) if frame is None else frame
        self.released = threading.Event()
        self.open_calls = []

    def open(self, index, backend):
        self.open_calls.append((index, backend))
        return True

    def read(self):
        self.released.wait(0.01)
        return True, self.frame.copy()

    def release(self):
        self.released.set()


def authorized_system(cameras=None):
    system = MagicMock()
    system.authorization.return_value = 3
    system.list.return_value = [PHONE, MAC] if cameras is None else cameras
    return system


class SelectionTests(unittest.TestCase):
    def test_automatic_selection_excludes_builtin_and_desk_view(self):
        desk = replace(PHONE, unique_id="desk", device_type="AVCaptureDeviceTypeDeskViewCamera")
        self.assertEqual(select_iphone([MAC, desk, PHONE]), PHONE)

    def test_builtin_is_not_a_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "没有找到"):
            select_iphone([MAC])
        with self.assertRaisesRegex(RuntimeError, "没有找到"):
            select_iphone([PHONE, MAC], MAC.unique_id)

    def test_multiple_phones_require_explicit_selection(self):
        second = replace(PHONE, index=2, name="Alice 的相机 2", unique_id="phone-2")
        with self.assertRaisesRegex(RuntimeError, "多台"):
            select_iphone([PHONE, second])
        self.assertEqual(select_iphone([PHONE, second], PHONE.name), PHONE)
        self.assertEqual(select_iphone([PHONE, second], second.unique_id), second)
        with self.assertRaisesRegex(RuntimeError, "多台"):
            select_iphone([PHONE, second], "Alice")


class CaptureTests(unittest.TestCase):
    def test_permission_denial_does_not_open_camera(self):
        system = authorized_system()
        system.authorization.return_value = 2
        backend = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "访问被拒绝"):
            iphone_yolo.open_capture(backend, PHONE, system)
        backend.VideoCapture.assert_not_called()

    def test_index_is_resolved_again_before_opening(self):
        moved = replace(PHONE, index=1)
        capture = SimulatedCapture()
        backend = MagicMock(CAP_AVFOUNDATION=cv2.CAP_AVFOUNDATION)
        backend.VideoCapture.return_value = capture
        system = authorized_system([replace(MAC, index=0), moved])
        self.assertIs(iphone_yolo.open_capture(backend, PHONE, system), capture)
        self.assertEqual(capture.open_calls, [(1, cv2.CAP_AVFOUNDATION)])
        capture.release()

    def test_topology_change_during_open_releases_camera(self):
        capture = SimulatedCapture()
        backend = MagicMock(CAP_AVFOUNDATION=cv2.CAP_AVFOUNDATION)
        backend.VideoCapture.return_value = capture
        system = authorized_system()
        system.list.side_effect = [[PHONE, MAC], [replace(MAC, index=0)]]
        with self.assertRaisesRegex(RuntimeError, "列表发生变化"):
            iphone_yolo.open_capture(backend, PHONE, system)
        self.assertTrue(capture.released.is_set())

    def test_initial_permission_request_retries_with_new_device_index(self):
        capture = MagicMock()
        capture.open.side_effect = [False, True]
        backend = MagicMock(CAP_AVFOUNDATION=cv2.CAP_AVFOUNDATION)
        backend.VideoCapture.return_value = capture
        system = authorized_system()
        system.authorization.return_value = 0
        system.wait_for_authorization.return_value = 3
        updated = [replace(MAC, index=0), replace(PHONE, index=1)]
        system.list.side_effect = [[PHONE, MAC], updated, updated]
        with redirect_stderr(io.StringIO()):
            self.assertIs(iphone_yolo.open_capture(backend, PHONE, system), capture)
        self.assertEqual([call.args[0] for call in capture.open.call_args_list], [0, 1])

    def test_disconnect_produces_error_and_releases_camera(self):
        capture = SimulatedCapture()
        capture.read = lambda: (False, None)
        reader = iphone_yolo.CameraFrames(capture, timeout=0.08)
        reader.start()
        try:
            with self.assertRaises((RuntimeError, TimeoutError)):
                reader.get(0)
        finally:
            reader.close()
        self.assertTrue(capture.released.is_set())
        self.assertFalse(reader.thread.is_alive())

    def test_finite_preview_reads_actual_shapes_and_releases(self):
        capture = SimulatedCapture()
        output = io.StringIO()
        with patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(iphone_yolo, "load_model") as model_loader, \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(iphone_yolo.main(["--preview", "--headless", "--frames", "3"]), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["frame"] for row in rows], [1, 2, 3])
        self.assertTrue(all((row["width"], row["height"]) == (32, 24) for row in rows))
        self.assertTrue(all(row["detections"] is None for row in rows))
        model_loader.assert_not_called()
        self.assertTrue(capture.released.is_set())

    def test_model_failure_still_releases_camera(self):
        capture = SimulatedCapture()
        with patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "本地模型不存在"):
                iphone_yolo.main(["--no-scene", "--model", "/missing/local/model.pt", "--headless", "--frames", "1"])
        self.assertTrue(capture.released.is_set())


class ModelIntegrationTests(unittest.TestCase):
    def test_local_yolo_detects_bus_from_simulated_camera(self):
        image_path = iphone_yolo.PROJECT / "installation_check" / "bus.jpg"
        model_path = iphone_yolo.PROJECT / "yolo26n.pt"
        if not image_path.is_file() or not model_path.is_file():
            self.skipTest("Project-local image/model not available; no network downloads in tests.")
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image)
        capture = SimulatedCapture(image)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(iphone_yolo, "HERE", Path(temporary)), \
                patch.dict("os.environ"), \
                patch.object(iphone_yolo, "MacCameras", return_value=authorized_system()), \
                patch.object(cv2, "VideoCapture", return_value=capture), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(iphone_yolo.main(["--all-objects", "--no-scene", "--headless", "--frames", "2", "--device", "cpu"]), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(any(item["class"] == "bus" for item in row["detections"]) for row in rows))
        self.assertTrue(capture.released.is_set())


if __name__ == "__main__":
    unittest.main()
