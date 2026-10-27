import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

import cv2
import numpy as np

from sony_wifi_yolo import (
    ROOT, annotate_frame, camera_from_description, draw_label, frame_position, jpeg_frames, main,
)


def packet(payload, kind=1, padding=0):
    header = b"\xff" + bytes([kind]) + b"\x00\x01" + (1234).to_bytes(4, "big")
    header += b"\x24\x35\x68\x79" + len(payload).to_bytes(3, "big") + bytes([padding])
    return header + bytes(120) + payload + bytes(padding)


class FragmentedStream(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3) if size >= 0 else 3)


class FrameCoordinateTests(unittest.TestCase):
    def test_center_is_origin(self):
        self.assertEqual(frame_position([300, 220, 340, 260], 640, 480), {"x": 0, "y": 0})

    def test_all_four_quadrants_use_cartesian_signs(self):
        for box, expected in [
            ([400, 100, 440, 140], {"x": 100, "y": 120}),
            ([200, 100, 240, 140], {"x": -100, "y": 120}),
            ([200, 340, 240, 380], {"x": -100, "y": -120}),
            ([400, 340, 440, 380], {"x": 100, "y": -120}),
        ]:
            with self.subTest(box=box):
                self.assertEqual(frame_position(box, 640, 480), expected)

    def test_position_uses_box_center_and_original_frame_resolution(self):
        # A box with a top-left corner on the origin is not itself centered there.
        self.assertEqual(
            frame_position([960, 540, 1160, 640], 1920, 1080), {"x": 100, "y": -50},
        )

    def test_odd_frame_dimensions_preserve_half_pixel_origin(self):
        self.assertEqual(frame_position([300, 220, 341, 261], 641, 481), {"x": 0, "y": 0})
        self.assertEqual(frame_position([0, 0, 0, 0], 641, 481), {"x": -320.5, "y": 240.5})

    def test_axes_cover_empty_frames_without_modifying_the_input(self):
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = annotate_frame(original, [])
        self.assertEqual(rendered.shape, original.shape)
        self.assertEqual(rendered.dtype, original.dtype)
        self.assertFalse(original.any())
        self.assertTrue(rendered[240, 5].any())
        self.assertTrue(rendered[470, 320].any())

    def test_detection_center_marker_and_labels_render(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        item = {"class": "person", "confidence": 0.9, "xyxy": [40, 60, 80, 100],
                "position": {"x": -100, "y": 40}}
        rendered = annotate_frame(image, [item])
        # The marker is on the object center, not on the top-left of its box.
        self.assertTrue(rendered[80, 60, 1] > rendered[80, 60, 0])
        self.assertGreater(np.count_nonzero(rendered[35:60, 40:200]), 0)

    def test_adjacent_objects_get_nonoverlapping_coordinate_labels(self):
        image = np.zeros((240, 640, 3), dtype=np.uint8)
        occupied = []
        draw_label(image, "person 0.90 (-100.0, +40.0) px", 40, 100, occupied=occupied)
        draw_label(image, "person 0.90 (+100.0, +40.0) px", 160, 100, occupied=occupied)
        left1, top1, right1, bottom1 = occupied[0]
        left2, top2, right2, bottom2 = occupied[1]
        self.assertTrue(right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)
        for left, top, right, bottom in occupied:
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, 640)
            self.assertLessEqual(bottom, 240)


class SonyProtocolTests(unittest.TestCase):
    def test_jpeg_survives_fragmented_reads_and_padding(self):
        jpeg = b"\xff\xd8example\xff\xd9"
        frames = jpeg_frames(FragmentedStream(packet(jpeg, padding=7)))
        self.assertEqual(next(frames), jpeg)

    def test_frame_metadata_is_skipped(self):
        frames = jpeg_frames(io.BytesIO(packet(b"metadata", kind=2) + packet(b"jpeg")))
        self.assertEqual(next(frames), b"jpeg")

    def test_truncated_payload_is_reported(self):
        frames = jpeg_frames(io.BytesIO(packet(b"jpeg")[:-1]))
        with self.assertRaises(EOFError):
            next(frames)

    def test_bad_protocol_marker_is_rejected(self):
        invalid = bytearray(packet(b"jpeg"))
        invalid[8] = 0
        with self.assertRaises(RuntimeError):
            next(jpeg_frames(io.BytesIO(invalid)))

    def test_namespaced_camera_description(self):
        xml = b'''<root xmlns="urn:schemas-upnp-org:device-1-0" xmlns:s="urn:schemas-sony-com:av">
        <device><friendlyName>ILCE-6000</friendlyName><s:X_ScalarWebAPI_ServiceList>
        <s:X_ScalarWebAPI_Service><s:X_ScalarWebAPI_ServiceType>camera</s:X_ScalarWebAPI_ServiceType>
        <s:X_ScalarWebAPI_ActionList_URL>http://camera.local:8080/sony</s:X_ScalarWebAPI_ActionList_URL>
        </s:X_ScalarWebAPI_Service></s:X_ScalarWebAPI_ServiceList></device></root>'''
        self.assertEqual(camera_from_description(xml), ("ILCE-6000", "http://camera.local:8080/sony/camera"))

    def test_loopback_camera_reaches_yolo_and_cleans_up(self):
        calls = []
        image = (ROOT / "installation_check" / "bus.jpg").read_bytes()

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
                body = packet(b"metadata", kind=2) + packet(image, padding=5)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), CameraEmulator)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                status = main([
                    "--api-url", f"http://127.0.0.1:{server.server_port}/sony/camera",
                    "--headless", "--frames", "1", "--device", "cpu",
                    "--coordinates-only",
                ])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        self.assertEqual(status, 0)
        detections = next(json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{"frame"'))
        labels = [item["class"] for item in detections["detections"]]
        self.assertIn("person", labels)
        self.assertIn("bus", labels)
        source_height, source_width = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        self.assertEqual(detections["image_size"], {"width": source_width, "height": source_height})
        self.assertEqual(detections["coordinate_system"], {
            "unit": "pixel", "origin": [source_width / 2, source_height / 2],
            "x_positive": "right", "y_positive": "up", "position_reference": "bounding_box_center",
        })
        for item in detections["detections"]:
            x1, y1, x2, y2 = item["xyxy"]
            self.assertEqual(item["position"], {
                "x": round((x1 + x2 - source_width) / 2, 1),
                "y": round((source_height - y1 - y2) / 2, 1),
            })
            self.assertLessEqual(abs(item["position"]["x"]), source_width / 2)
            self.assertLessEqual(abs(item["position"]["y"]), source_height / 2)
        self.assertEqual(calls, ["getAvailableApiList", "startRecMode", "getAvailableApiList", "startLiveview", "stopLiveview", "stopRecMode"])
        (ROOT / "installation_check" / "wifi_bridge_check.json").write_text(
            json.dumps({"source": "local camera emulator, not physical alpha6000", "api_calls": calls,
                        "result": detections}, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
