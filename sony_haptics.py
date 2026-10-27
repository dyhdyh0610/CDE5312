"""Fresh Sony target coordinates -> acknowledged ESP32 USB vibration commands."""

import argparse
import json
import math
import sys
import time


DIRECTIONS = ("up", "right", "down", "left")
NAMES = ("上", "右", "下", "左")
ZERO = (False, False, False, False)
FRAME_TIMEOUT = 0.75  # Must match FRAME_TIMEOUT_MS in the ESP32 sketch.
USB_VIDS = {0x303A, 0x10C4, 0x1A86, 0x0403, 0x2341}


def directions_for_state(state, width, height, tolerance=0.03):
    """Use the same native coordinates and centering decision as text guidance."""
    if (not state or state.get("phase") != "tracking" or not state.get("target")
            or not state.get("guidance") or state["guidance"].get("centered", True)):
        return ZERO
    x, y = (state["target"]["position"][axis] for axis in ("x", "y"))
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in (x, y)):
        return ZERO

    horizontal = abs(x) > max(8, width * tolerance)
    vertical = abs(y) > max(8, height * tolerance)
    return (y > 0 and vertical, x > 0 and horizontal, y < 0 and vertical, x < 0 and horizontal)


def list_ports():
    try:
        from serial.tools import list_ports as serial_ports
    except ImportError as error:
        raise RuntimeError("缺少 pyserial，请在项目虚拟环境安装 pyserial==3.5。") from error
    return [{"device": p.device, "description": p.description,
             "usb_candidate": p.vid in USB_VIDS} for p in serial_ports.comports()]


def resolve_port(port):
    if port != "auto":
        return port
    candidates = [p["device"] for p in list_ports() if p["usb_candidate"]]
    if len(candidates) != 1:
        detail = "没有找到 USB 控制板" if not candidates else "找到多个 USB 控制板"
        raise RuntimeError(f"{detail}；请先连接 ESP32，用 --list-haptics-ports 查看，再用 --haptics-port 指定。")
    return candidates[0]


class SerialHaptics:
    HEARTBEAT_INTERVAL = 0.20
    ACK_TIMEOUT = 0.35

    def __init__(self, port="auto"):
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("缺少 pyserial，请安装 pyserial==3.5。") from error
        self.device = resolve_port(port)
        self.sequence = 0
        self.last_directions = None
        self.last_sent = float("-inf")
        self.buffer = bytearray()
        self.serial = serial.Serial(port=None, baudrate=115200, timeout=0.04, write_timeout=0.20,
                                    exclusive=True)
        # Avoid deliberately asserting the ESP32 boot/reset lines. Opening a USB
        # bridge may still reset the board, so the handshake allows startup time.
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.port = self.device
        try:
            self.serial.open()
            self._handshake()
            self.send_directions(ZERO, force=True)
        except (OSError, RuntimeError) as error:
            self.serial.close()
            raise RuntimeError(f"振动控制板连接失败（{self.device}）：{error}") from error

    def _write(self, data):
        try:
            if self.serial.write(data) != len(data):
                raise RuntimeError("串口指令未完整发送")
        except OSError as error:
            raise RuntimeError(f"振动串口写入失败：{error}") from error

    def _line(self, deadline):
        while time.monotonic() < deadline:
            try:
                character = self.serial.read(1)
            except OSError as error:
                raise RuntimeError(f"振动串口读取失败：{error}") from error
            if character == b"\n":
                line = self.buffer.decode("ascii", errors="replace").strip()
                self.buffer.clear()
                return line
            if character:
                self.buffer.extend(character)
                if len(self.buffer) > 256:
                    raise RuntimeError("控制板响应过长，请确认已上传本项目固件")
        return None

    def _handshake(self):
        deadline = time.monotonic() + 4.0
        next_ping = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() >= next_ping:
                self._write(b"PING\n")
                next_ping = time.monotonic() + 0.25
            line = self._line(min(deadline, next_ping))
            if line and line.startswith("READY HAPTIC1 "):
                if line != "READY HAPTIC1 15":
                    raise RuntimeError(f"四个 DRV2605 未全部就绪（{line}），请检查通道 0/4/2/3")
                return
            if line and line.startswith("ERR "):
                raise RuntimeError(f"ESP32：{line}")
        raise RuntimeError("未收到 HAPTIC1 握手，请先上传 arduino/sony_haptics/sony_haptics.ino")

    def send_directions(self, directions, force=False):
        directions = tuple(directions)
        if len(directions) != 4 or any(type(v) is not bool for v in directions):
            raise ValueError("四个方向必须分别指定振动或停止")
        if (directions[0] and directions[2]) or (directions[1] and directions[3]):
            raise ValueError("相反方向不能同时振动")
        now = time.monotonic()
        if not force and directions == self.last_directions and now - self.last_sent < self.HEARTBEAT_INTERVAL:
            return
        self.sequence = self.sequence % 0xFFFFFFFF + 1
        data = f"D1,{self.sequence},{','.join(str(int(v)) for v in directions)}\n".encode("ascii")
        self._write(data)
        deadline = time.monotonic() + self.ACK_TIMEOUT
        while time.monotonic() < deadline:
            line = self._line(deadline)
            if line == f"ACK {self.sequence}":
                self.last_directions = directions
                self.last_sent = time.monotonic()
                return
            if line and line.startswith("ERR "):
                raise RuntimeError(f"ESP32 振动控制失败：{line}")
        raise RuntimeError("ESP32 未确认振动指令；请检查 USB 连接和固件")

    def update(self, state, width, height, tolerance=0.03, observed_at=None):
        age = float("inf") if observed_at is None else time.monotonic() - observed_at
        fresh = 0 <= age < FRAME_TIMEOUT
        directions = directions_for_state(state, width, height, tolerance) if fresh else ZERO
        self.send_directions(directions)
        text = " / ".join(name for name, enabled in zip(NAMES, directions) if enabled) or "停止"
        if not fresh:
            text = "画面更新超时，已停止"
        return {"directions": dict(zip(DIRECTIONS, directions)), "text": text, "fresh": fresh}

    def close(self):
        if not self.serial.is_open:
            return True
        stopped = True
        try:
            self.send_directions(ZERO, force=True)
        except RuntimeError:
            stopped = False
            try:
                self._write(b"0\n")
            except RuntimeError:
                pass
        finally:
            self.serial.close()
        return stopped


def main(argv=None):
    parser = argparse.ArgumentParser(description="ESP32 四向振动连接与安装位置测试")
    parser.add_argument("--list", action="store_true", help="只枚举串口，不打开设备")
    parser.add_argument("--port", default="auto", help="ESP32 串口或 auto")
    parser.add_argument("--test", choices=(*DIRECTIONS, "all"), help="每个指定方向短振 0.5 秒")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(list_ports(), ensure_ascii=False, indent=2))
        return 0
    if args.test is None:
        parser.error("请指定 --list 或 --test")
    haptics = SerialHaptics(args.port)
    try:
        for direction in DIRECTIONS if args.test == "all" else (args.test,):
            directions = tuple(name == direction for name in DIRECTIONS)
            print(f"测试：{NAMES[DIRECTIONS.index(direction)]}", flush=True)
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                haptics.send_directions(directions)
                time.sleep(0.05)
            haptics.send_directions(ZERO, force=True)
            time.sleep(0.3)
    finally:
        if not haptics.close():
            print("未收到停止确认，请检查 ESP32 和模块供电。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
