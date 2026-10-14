"""Enumerate macOS cameras in the same order as OpenCV's AVFoundation backend.

Uses the system Objective-C runtime, so PyObjC and extra packages are unnecessary.
Only object, integer, boolean and double messages are used; no struct ABI is needed.
"""

import ctypes
from dataclasses import dataclass
import sys
import time


@dataclass(frozen=True)
class Camera:
    index: int
    name: str
    unique_id: str
    model: str
    device_type: str
    continuity: bool = False

    @property
    def is_iphone(self):
        return "DeskView" not in self.device_type and (
            self.model.lower().startswith("iphone") or self.continuity
            or "ContinuityCamera" in self.device_type
        )


class MacCameras:
    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("此程序使用 macOS 连续互通相机，需要在 Mac 上运行。")
        self.av = ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.address = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value
        self.device_class = self.objc.objc_getClass(b"AVCaptureDevice")
        self.video = ctypes.c_void_p.in_dll(self.av, "AVMediaTypeVideo").value

    def send(self, obj, selector, result=ctypes.c_void_p, types=(), values=()):
        function = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *types)(self.address)
        return function(obj, self.objc.sel_registerName(selector.encode()), *values)

    def string(self, obj):
        value = self.send(obj, "UTF8String", ctypes.c_char_p) if obj else None
        return value.decode("utf-8") if value else ""

    def authorization(self):
        # AVAuthorizationStatus: not determined=0, restricted=1, denied=2, authorized=3.
        return self.send(self.device_class, "authorizationStatusForMediaType:",
                         ctypes.c_long, (ctypes.c_void_p,), (self.video,))

    def wait_for_authorization(self, timeout=30):
        """Pump the main run loop after OpenCV has requested camera permission."""
        pool = self.send(self.objc.objc_getClass(b"NSAutoreleasePool"), "new")
        try:
            run_loop = self.send(self.objc.objc_getClass(b"NSRunLoop"), "mainRunLoop")
            date_class = self.objc.objc_getClass(b"NSDate")
            deadline = time.monotonic() + timeout
            while self.authorization() == 0 and time.monotonic() < deadline:
                date = self.send(date_class, "dateWithTimeIntervalSinceNow:",
                                 types=(ctypes.c_double,), values=(0.1,))
                self.send(run_loop, "runUntilDate:", None, (ctypes.c_void_p,), (date,))
            return self.authorization()
        finally:
            self.send(pool, "drain", None)

    def list(self):
        pool = self.send(self.objc.objc_getClass(b"NSAutoreleasePool"), "new")
        try:
            muxed = ctypes.c_void_p.in_dll(self.av, "AVMediaTypeMuxed").value
            video_devices = self.send(self.device_class, "devicesWithMediaType:",
                                      types=(ctypes.c_void_p,), values=(self.video,))
            muxed_devices = self.send(self.device_class, "devicesWithMediaType:",
                                      types=(ctypes.c_void_p,), values=(muxed,))
            devices = self.send(video_devices, "arrayByAddingObjectsFromArray:",
                                types=(ctypes.c_void_p,), values=(muxed_devices,))
            # Use NSString's native comparison, exactly as OpenCV does for uniqueID.
            key = self.send(self.objc.objc_getClass(b"NSString"), "stringWithUTF8String:",
                            types=(ctypes.c_char_p,), values=(b"uniqueID",))
            descriptor = self.send(self.objc.objc_getClass(b"NSSortDescriptor"),
                                   "sortDescriptorWithKey:ascending:",
                                   types=(ctypes.c_void_p, ctypes.c_bool), values=(key, True))
            descriptors = self.send(self.objc.objc_getClass(b"NSArray"), "arrayWithObject:",
                                    types=(ctypes.c_void_p,), values=(descriptor,))
            devices = self.send(devices, "sortedArrayUsingDescriptors:",
                                types=(ctypes.c_void_p,), values=(descriptors,))
            result = []
            for index in range(self.send(devices, "count", ctypes.c_ulong)):
                device = self.send(devices, "objectAtIndex:", types=(ctypes.c_ulong,), values=(index,))
                selector = self.objc.sel_registerName(b"isContinuityCamera")
                has_continuity = self.send(device, "respondsToSelector:", ctypes.c_bool,
                                           (ctypes.c_void_p,), (selector,))
                result.append(Camera(
                    index, self.string(self.send(device, "localizedName")),
                    self.string(self.send(device, "uniqueID")),
                    self.string(self.send(device, "modelID")),
                    self.string(self.send(device, "deviceType")),
                    self.send(device, "isContinuityCamera", ctypes.c_bool) if has_continuity else False,
                ))
            return result
        finally:
            self.send(pool, "drain", None)


def select_iphone(cameras, selector=None):
    phones = [camera for camera in cameras if camera.is_iphone]
    if selector:
        exact = [camera for camera in phones if camera.unique_id == selector or camera.name == selector]
        phones = exact or [camera for camera in phones if selector.casefold() in camera.name.casefold()]
    if not phones:
        raise RuntimeError(
            "没有找到匹配的 iPhone 摄像头。请开启连续互通相机、连接并信任 Mac、"
            "锁屏固定手机，然后用 --list 查看设备。"
        )
    if len(phones) != 1:
        raise RuntimeError("发现多台 iPhone，请用 --list 查看，再用 --camera 指定名称或 unique_id。")
    return phones[0]
