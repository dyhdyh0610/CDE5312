"""iPhone YOLO detections and live centered image-plane pixel coordinates."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

from mac_cameras import MacCameras, select_iphone
from frame_coordinates import annotate_frame, coordinate_system, detection_items
from cup_target import cup_target, prioritize_cups

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WINDOW = "iPhone Camera - YOLO"
PERMISSION_HELP = (
    "请在 macOS 系统设置 → 隐私与安全 → 摄像头中允许实际启动程序的应用"
    "（例如 Terminal/终端或 Codex），再重新启动程序。"
)


def open_capture(cv2, camera, system):
    status = system.authorization()
    if status in (1, 2):
        raise RuntimeError(f"摄像头访问被拒绝或受到系统限制。{PERMISSION_HELP}")
    capture = cv2.VideoCapture()
    try:
        # Resolve by identity again immediately before using OpenCV's numeric index.
        devices = system.list()
        camera = select_iphone(devices, camera.unique_id)
        opened = capture.open(camera.index, cv2.CAP_AVFOUNDATION)
        if not opened and status == 0:
            print("请在系统弹窗中允许摄像头访问；等待最多 30 秒。", file=sys.stderr, flush=True)
            status = system.wait_for_authorization()
            if status == 3:
                # The device index may change while the permission prompt is open.
                devices = system.list()
                camera = select_iphone(devices, camera.unique_id)
                opened = capture.open(camera.index, cv2.CAP_AVFOUNDATION)
            else:
                raise RuntimeError(f"尚未获得摄像头权限。{PERMISSION_HELP}")
        if not opened:
            raise RuntimeError(
                f"无法打开 {camera.name}。请检查手机锁屏、连续互通和 USB 信任状态，"
                "退出使用摄像头的其他应用后重试。"
            )
        if [(item.index, item.unique_id) for item in devices] != [
            (item.index, item.unique_id) for item in system.list()
        ]:
            raise RuntimeError("打开期间摄像头列表发生变化；已停止取流，请重新启动程序。")
        return capture
    except BaseException:
        capture.release()
        raise


class CameraFrames:
    """Read continuously, retain the newest frame, and release on the reader thread."""

    def __init__(self, capture, timeout=12):
        self.capture = capture
        self.timeout = timeout
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.sequence = 0
        self.frame = None
        self.error = None
        self.observer = None
        self.thread = threading.Thread(target=self._read, daemon=True, name="iphone-camera")

    def start(self):
        self.thread.start()

    def _read(self):
        last_frame = time.monotonic()
        observed_at = float("-inf")
        try:
            while not self.stopped.is_set():
                ok, frame = self.capture.read()
                if ok and frame is not None and frame.size:
                    last_frame = time.monotonic()
                    # Observe raw camera frames independently of YOLO/model latency.
                    if self.observer is not None and last_frame - observed_at >= 0.1:
                        self.observer(frame, last_frame)
                        observed_at = last_frame
                    with self.condition:
                        self.frame = frame
                        self.sequence += 1
                        self.condition.notify_all()
                else:
                    if time.monotonic() - last_frame >= self.timeout:
                        raise TimeoutError("iPhone 没有送来画面，可能已解锁、暂停或断开连接。")
                    self.stopped.wait(0.05)
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.capture.release()

    def get(self, previous_sequence):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > previous_sequence or self.error is not None or self.stopped.is_set(),
                timeout=self.timeout,
            )
            if self.sequence > previous_sequence:
                return self.sequence, self.frame
            if self.error is not None:
                raise RuntimeError(f"读取 iPhone 失败：{self.error}") from self.error
            if not ready:
                raise TimeoutError(f"{self.timeout:g} 秒内没有新画面，请检查手机状态并重新启动。")
            raise RuntimeError("iPhone 读取已停止。")

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            print("摄像头驱动尚未返回；程序退出后系统会释放此进程的连接。", file=sys.stderr)


def load_model(args):
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise RuntimeError(f"本地模型不存在：{model_path}。请用 --model 指定已有模型。")
    os.environ["YOLO_CONFIG_DIR"] = str(HERE / ".yolo-config")
    # Keep headless stdout as JSON; library startup output goes to stderr.
    with redirect_stdout(sys.stderr):
        import torch
        from ultralytics import YOLO
        device = args.device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前环境无法使用 MPS，请使用 --device cpu。")
        model = YOLO(str(model_path))
    return model, device


def detect_frame(model, device, frame, confidence, annotate, cups_only=True):
    if model is None:
        return annotate_frame(frame, []) if annotate else frame, None
    options = {}
    if cups_only:
        names = model.names
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        classes = [index for index, name in entries if name == "cup"]
        if not classes:
            raise RuntimeError("当前模型不包含 cup（杯子）类别，请使用支持杯子的模型。")
        options["classes"] = classes
    with redirect_stdout(sys.stderr):
        result = model.predict(frame, device=device, conf=confidence, imgsz=640, verbose=False,
                               **options)[0]
    height, width = frame.shape[:2]
    detections = detection_items(result, width, height)
    if cups_only:
        detections = [item for item in detections if item["class"] == "cup"]
    detections = prioritize_cups(detections)
    return annotate_frame(frame, detections) if annotate else frame, detections


def process_image(args, panel):
    """Explicit still-image verification, without opening any camera."""
    import cv2
    frame = cv2.imread(str(Path(args.image).expanduser()))
    if frame is None:
        raise RuntimeError(f"无法读取图片：{args.image}")
    height, width = frame.shape[:2]
    model, device = (None, None) if args.preview else load_model(args)
    _, detections = detect_frame(model, device, frame, args.confidence, False,
                                 cups_only=not args.all_objects)
    rendered = panel.render(frame, detections) if panel is not None else None
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), rendered):
            raise RuntimeError(f"无法保存预览：{destination}")
    print(json.dumps({"source": "image", "width": width, "height": height,
                      "image_size": {"width": width, "height": height},
                      "coordinate_system": coordinate_system(width, height),
                      "detections": detections,
                      "cup_guidance": cup_target(detections, width, height)}, ensure_ascii=False), flush=True)
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, rendered)
        while cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) & 0xFF in (ord("q"), ord("Q"), 27):
                break
        cv2.destroyAllWindows()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="iPhone YOLO 实时物体坐标；画面中心为原点，X 向右、Y 向上，单位像素。")
    parser.add_argument("--version", action="version", version="iPhone Camera V1.2.0")
    parser.add_argument("--list", action="store_true", help="列出设备和权限状态，不启动摄像头")
    parser.add_argument("--camera", help="指定 iPhone 的名称、名称片段或完整 unique_id")
    parser.add_argument("--preview", action="store_true", help="只预览，不加载 YOLO")
    parser.add_argument("--model", default=str(PROJECT / "yolo26n.pt"), help="已有本地模型路径")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--all-objects", action="store_true", help="恢复显示所有类别；默认只检测杯子")
    parser.add_argument("--headless", action="store_true", help="不显示窗口，每帧输出 JSON")
    parser.add_argument("--frames", type=int, default=0, help="处理指定帧数后退出；0 表示持续运行")
    parser.add_argument("--timeout", type=float, default=12, help="等待新画面的最长秒数")
    # Existing launch commands may include this flag. V1.2 always uses YOLO only.
    parser.add_argument("--no-scene", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--font", help="用于中文侧栏的字体文件")
    parser.add_argument("--image", help="分析本地静态图片，不打开相机（用于验证）")
    parser.add_argument("--output", help="仅用于 --image：保存画面和文字侧栏的预览图片")
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1 or args.frames < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("confidence 必须在 0 到 1 之间，frames 必须大于等于 0，timeout 必须为有限正数。")
    if args.output and not args.image:
        parser.error("--output 仅用于 --image 静态图片验证。")
    if args.list:
        system = MacCameras()
        cameras = system.list()
        status = system.authorization()
        print(json.dumps({
            "permission": {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(status, "unknown"),
            "cameras": [{**asdict(camera), "is_iphone": camera.is_iphone} for camera in cameras],
        }, ensure_ascii=False, indent=2))
        return 0
    panel = None
    if not args.headless or args.output:
        from scene_panel import ScenePanel
        panel = ScenePanel(args.font)
    if args.image:
        return process_image(args, panel)
    system = MacCameras()
    cameras = system.list()
    camera = select_iphone(cameras, args.camera)
    import cv2
    capture = open_capture(cv2, camera, system)
    reader = CameraFrames(capture, args.timeout)
    sequence = count = page = 0
    started = False
    try:
        reader.start()
        started = True
        model, device = (None, None) if args.preview else load_model(args)
        print(f"连接：{camera.name}；模式：{'仅预览' if model is None else 'YOLO / ' + device}",
              file=sys.stderr, flush=True)
        print("按 Q、Esc 或 Ctrl+C 退出。", file=sys.stderr, flush=True)
        displayed_at = time.monotonic()
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            if panel is not None:
                cv2.resizeWindow(WINDOW, 1280, 760)
        while True:
            sequence, frame = reader.get(sequence)
            height, width = frame.shape[:2]
            _, detections = detect_frame(model, device, frame, args.confidence, False,
                                         cups_only=not args.all_objects)
            guidance = cup_target(detections, width, height)
            count += 1
            if args.headless:
                row = {"frame": count, "source_sequence": sequence, "width": width,
                       "height": height, "image_size": {"width": width, "height": height},
                       "coordinate_system": coordinate_system(width, height), "detections": detections,
                       "cup_guidance": guidance}
                print(json.dumps(row, ensure_ascii=False), flush=True)
            else:
                now = time.monotonic()
                fps = 1 / max(now - displayed_at, 0.001)
                displayed_at = now
                pages = max(1, (len(detections or []) + panel.PAGE_SIZE - 1) // panel.PAGE_SIZE)
                page = min(page, pages - 1)
                cv2.imshow(WINDOW, panel.render(frame, detections, fps, page))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("n"), ord("N")):
                    page = min(page + 1, pages - 1)
                elif key in (ord("p"), ord("P")):
                    page = max(0, page - 1)
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.frames and count >= args.frames:
                break
    finally:
        reader.observer = None
        if started:
            reader.close()
        else:
            capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已停止。", file=sys.stderr)
        sys.exit(0)
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
