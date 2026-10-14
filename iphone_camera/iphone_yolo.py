"""Independent iPhone Continuity Camera preview and local YOLO detection on Mac."""

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
from scene_description import DEFAULT_SCENE_MODEL, OllamaCaptioner, SceneController, SceneWorker

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


def detect_frame(model, device, frame, confidence, annotate):
    if model is None:
        return frame, None
    with redirect_stdout(sys.stderr):
        result = model.predict(frame, device=device, conf=confidence, imgsz=640, verbose=False)[0]
    detections = [
        {"class": result.names[int(box.cls.item())], "confidence": round(float(box.conf.item()), 3),
         "xyxy": box.xyxy[0].tolist()} for box in result.boxes
    ]
    return result.plot() if annotate else frame, detections


def describe_image(args, captioner, panel):
    """Explicit still-image verification, without opening any camera."""
    import cv2
    frame = cv2.imread(str(Path(args.image).expanduser()))
    if frame is None:
        raise RuntimeError(f"无法读取图片：{args.image}")
    height, width = frame.shape[:2]
    model, device = (None, None) if args.preview else load_model(args)
    rendered, detections = detect_frame(model, device, frame, args.confidence, True)
    scene = None
    if captioner is not None:
        started = time.monotonic()
        text = captioner.describe(frame)
        scene = {"state": "ready", "text": text, "error": "", "generation": 1,
                 "stable_for": args.stable_seconds, "required_seconds": args.stable_seconds,
                 "elapsed": round(time.monotonic() - started, 2)}
        if panel is not None:
            rendered = panel.render(rendered, scene, model=args.scene_model)
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), rendered):
            raise RuntimeError(f"无法保存预览：{destination}")
    print(json.dumps({"source": "image", "width": width, "height": height,
                      "detections": detections, "scene": scene}, ensure_ascii=False), flush=True)
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, rendered)
        while cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) & 0xFF in (ord("q"), ord("Q"), 27):
                break
        cv2.destroyAllWindows()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="iPhone YOLO 和本地中文空间描述；画面稳定 2 秒后分析。")
    parser.add_argument("--version", action="version", version="iPhone Camera V1.1.0")
    parser.add_argument("--list", action="store_true", help="列出设备和权限状态，不启动摄像头")
    parser.add_argument("--camera", help="指定 iPhone 的名称、名称片段或完整 unique_id")
    parser.add_argument("--preview", action="store_true", help="只预览，不加载 YOLO")
    parser.add_argument("--model", default=str(PROJECT / "yolo26n.pt"), help="已有本地模型路径")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--headless", action="store_true", help="不显示窗口，每帧输出 JSON")
    parser.add_argument("--frames", type=int, default=0, help="处理指定帧数后退出；0 表示持续运行")
    parser.add_argument("--timeout", type=float, default=12, help="等待新画面的最长秒数")
    parser.add_argument("--no-scene", action="store_true", help="仅运行 V1.0 的 YOLO 检测功能")
    parser.add_argument("--scene-model", default=DEFAULT_SCENE_MODEL, help="Ollama 已安装的本地视觉模型")
    parser.add_argument("--stable-seconds", type=float, default=2.0, help="画面保持稳定的秒数，默认 2")
    parser.add_argument("--motion-threshold", type=float, default=0.025,
                        help="发生明显像素变化的面积比例阈值，默认 0.025；越小越敏感")
    parser.add_argument("--check-scene", action="store_true", help="检查本地视觉模型，不打开相机")
    parser.add_argument("--font", help="用于中文侧栏的字体文件")
    parser.add_argument("--image", help="分析本地静态图片，不打开相机（用于验证）")
    parser.add_argument("--output", help="仅用于 --image：保存画面和文字侧栏的预览图片")
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1 or args.frames < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("confidence 必须在 0 到 1 之间，frames 必须大于等于 0，timeout 必须为有限正数。")
    if not math.isfinite(args.stable_seconds) or args.stable_seconds <= 0 or not 0 < args.motion_threshold < 1:
        parser.error("stable-seconds 必须为有限正数，motion-threshold 必须在 0 到 1 之间。")
    if args.output and not args.image:
        parser.error("--output 仅用于 --image 静态图片验证。")
    if args.check_scene:
        print(json.dumps(OllamaCaptioner(args.scene_model).check(), ensure_ascii=False, indent=2))
        return 0

    if args.list:
        system = MacCameras()
        cameras = system.list()
        status = system.authorization()
        print(json.dumps({
            "permission": {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(status, "unknown"),
            "cameras": [{**asdict(camera), "is_iphone": camera.is_iphone} for camera in cameras],
        }, ensure_ascii=False, indent=2))
        return 0
    captioner = panel = None
    if not args.no_scene and not args.preview:
        captioner = OllamaCaptioner(args.scene_model)
        captioner.check()
        if not args.headless or args.output:
            from scene_panel import ScenePanel
            panel = ScenePanel(args.font)
    if args.image:
        return describe_image(args, captioner, panel)
    system = MacCameras()
    cameras = system.list()
    camera = select_iphone(cameras, args.camera)
    import cv2
    capture = open_capture(cv2, camera, system)
    reader = CameraFrames(capture, args.timeout)
    sequence = count = 0
    started = False
    worker = controller = None
    try:
        reader.start()
        started = True
        model, device = (None, None) if args.preview else load_model(args)
        if captioner is not None:
            worker = SceneWorker(captioner)
            controller = SceneController(worker, args.stable_seconds, args.motion_threshold)
            reader.observer = controller.update
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
            frame, detections = detect_frame(model, device, frame, args.confidence, not args.headless)
            count += 1
            if args.headless:
                row = {"frame": count, "source_sequence": sequence, "width": width,
                       "height": height, "detections": detections}
                if controller is not None:
                    row["scene"] = controller.as_dict()
                print(json.dumps(row, ensure_ascii=False), flush=True)
            else:
                now = time.monotonic()
                fps = 1 / max(now - displayed_at, 0.001)
                displayed_at = now
                if panel is not None:
                    frame = panel.render(frame, controller.as_dict(), fps, args.scene_model)
                else:
                    cv2.putText(frame, f"iPhone | {fps:.1f} FPS | Q: quit", (12, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("r"), ord("R")) and controller is not None:
                    controller.retry()
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
        if worker is not None:
            worker.close()
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
