import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from sony_guidance import movement_guidance
from sony_haptics import FRAME_TIMEOUT, SerialHaptics, ZERO, directions_for_state, resolve_port
import test_sony_guidance as guidance_tests


def state(x=100, y=50):
    return {"phase": "tracking", "target": {"position": {"x": x, "y": y}},
            "guidance": movement_guidance({"x": x, "y": y}, 640, 480)}


class Esp32Peer:
    """Real OS serial endpoint; never opens a physical USB device."""

    def __init__(self, mask=15):
        self.master, self.slave = pty.openpty()
        self.port = os.ttyname(self.slave)
        self.mask = mask
        self.frames = []
        self.commands = []
        self.fail = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=2)
        os.close(self.master)
        os.close(self.slave)

    def run(self):
        buffer = b""
        while not self.done.is_set():
            if not select.select([self.master], [], [], 0.05)[0]:
                continue
            buffer += os.read(self.master, 4096)
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                command = raw.decode().strip()
                self.commands.append(command)
                if command == "PING":
                    os.write(self.master, f"READY HAPTIC1 {self.mask}\n".encode())
                elif command.startswith("D1,"):
                    _, sequence, *directions = command.split(",")
                    self.frames.append(tuple(bool(int(v)) for v in directions))
                    if not self.fail:
                        os.write(self.master, f"ACK {sequence}\n".encode())


class DirectionTests(unittest.TestCase):
    def test_four_directions_and_diagonal_follow_native_coordinate_signs(self):
        for x, y, expected in ((100, 0, (False, True, False, False)), (-100, 0, (False, False, False, True)),
                               (0, 100, (True, False, False, False)), (0, -100, (False, False, True, False)),
                               (200, 30, (True, True, False, False))):
            self.assertEqual(directions_for_state(state(x, y), 640, 480), expected)

    def test_center_hysteresis_and_nontracking_states_are_silent(self):
        for phase in ("stabilizing", "detecting", "choose", "empty", "lost"):
            self.assertEqual(directions_for_state(dict(state(), phase=phase), 640, 480), ZERO)
        self.assertEqual(directions_for_state(state(5, 5), 640, 480), ZERO)
        hysteresis = state(21, 0)
        hysteresis["guidance"] = movement_guidance({"x": 21, "y": 0}, 640, 480, was_centered=True)
        self.assertEqual(directions_for_state(hysteresis, 640, 480), ZERO)
        self.assertEqual(directions_for_state(dict(state(), target=None), 640, 480), ZERO)
        self.assertEqual(directions_for_state(state(float("nan"), 0), 640, 480), ZERO)

    def test_direction_is_independent_of_preview_size_and_offset_magnitude(self):
        first = state(100, 50)
        second = state(200, 100)
        self.assertEqual(directions_for_state(first, 640, 480), directions_for_state(second, 1280, 960))
        self.assertEqual(directions_for_state(first, 640, 480), directions_for_state(second, 640, 480))


class SerialTests(unittest.TestCase):
    def test_acknowledged_coordinates_and_shutdown_use_real_serial_bytes(self):
        with Esp32Peer() as board:
            connection = SerialHaptics(board.port)
            try:
                self.assertEqual(board.frames[-1], ZERO)
                output = connection.update(state(), 640, 480, observed_at=time.monotonic())
                self.assertEqual(output["directions"], {"up": True, "right": True, "down": False, "left": False})
                self.assertEqual(board.frames[-1], (True, True, False, False))
            finally:
                self.assertTrue(connection.close())
            self.assertEqual(board.frames[-1], ZERO)

    def test_slow_inference_cannot_restart_stale_direction(self):
        with Esp32Peer() as board:
            connection = SerialHaptics(board.port)
            try:
                connection.update(state(), 640, 480, observed_at=time.monotonic())
                output = connection.update(state(), 640, 480,
                                           observed_at=time.monotonic() - FRAME_TIMEOUT - 0.01)
                self.assertFalse(output["fresh"])
                self.assertEqual(board.frames[-1], ZERO)
            finally:
                connection.close()

    def test_heartbeat_is_driven_by_fresh_updates_and_direction_changes_are_immediate(self):
        with Esp32Peer() as board:
            connection = SerialHaptics(board.port)
            try:
                connection.send_directions((False, True, False, False))
                count = len(board.frames)
                connection.send_directions((False, True, False, False))
                self.assertEqual(len(board.frames), count)
                time.sleep(0.22)
                connection.send_directions((False, True, False, False))
                self.assertEqual(len(board.frames), count + 1)
                connection.send_directions((False, False, False, True))
                self.assertEqual(board.frames[-1], (False, False, False, True))
            finally:
                connection.close()

    def test_missing_module_or_missing_ack_is_not_silently_accepted(self):
        with Esp32Peer(mask=7) as board:
            with self.assertRaisesRegex(RuntimeError, "未全部就绪"):
                SerialHaptics(board.port)
        with Esp32Peer() as board:
            connection = SerialHaptics(board.port)
            board.fail = True
            try:
                with self.assertRaisesRegex(RuntimeError, "未确认"):
                    connection.send_directions((True, False, False, False))
            finally:
                self.assertFalse(connection.close())

    def test_auto_port_selection_refuses_ambiguity_without_opening_ports(self):
        for ports in ([], [{"device": "a", "usb_candidate": True}, {"device": "b", "usb_candidate": True}]):
            with patch("sony_haptics.list_ports", return_value=ports):
                with self.assertRaises(RuntimeError):
                    resolve_port("auto")


class FirmwareTests(unittest.TestCase):
    def test_actual_sketch_protocol_timeout_errors_and_manual_mode(self):
        root = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / "firmware_test")
            result = subprocess.run(["clang++", "-std=c++17", "-I", str(root / "tests/firmware_stubs"),
                                     str(root / "tests/test_sony_haptics_firmware.cpp"), "-o", executable],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr[:4000])
            result = subprocess.run([executable], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr[:4000])


class CameraIntegrationTests(unittest.TestCase):
    def test_sony_jpeg_yolo_selection_coordinates_to_usb_and_stop(self):
        case = guidance_tests.RealModelTests()
        image = (guidance_tests.ROOT / "installation_check/bus.jpg").read_bytes()
        with Esp32Peer() as board:
            records = case.run_liveview([image], target_number=1, extra_args=("--haptics-port", board.port))
            tracked = [r for r in records if r["shooting"]["guidance"] is not None]
            self.assertTrue(tracked)
            self.assertTrue(any(any(directions) for directions in board.frames))
            for record in tracked:
                expected = directions_for_state(record["shooting"], record["image_size"]["width"],
                                            record["image_size"]["height"])
                actual = record["shooting"]["haptics"]["directions"]
                if record["shooting"]["haptics"]["fresh"]:
                    self.assertEqual(tuple(actual[name] for name in ("up", "right", "down", "left")), expected)
            self.assertEqual(board.frames[0], ZERO)
            self.assertEqual(board.frames[-1], ZERO)


if __name__ == "__main__":
    unittest.main()
