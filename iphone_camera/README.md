# 独立 iPhone 摄像头程序

链路：iPhone 后置相机 → macOS 连续互通相机 → OpenCV AVFoundation → 本地 YOLO → 预览窗口。

所有新文件位于本目录。程序不导入 Sony 程序，复用上一级项目的 `.venv` 和 `yolo26n.pt`，无需再安装 Python 包。
自动选择 iPhone；找不到 iPhone 或同时发现多台时会明确报错，不会自动切换到 Mac 内置摄像头。

## 第一次准备

1. 使用 iPhone 17 和 Mac。苹果要求 iOS 16 或更新版本、macOS 13 或更新版本。
2. 两台设备登录同一 Apple 账户，该账户已开启双重认证。
3. iPhone：设置 → 通用 → 隔空播放与连续互通（或隔空播放与接力）→ 开启连续互通相机。
4. 两台设备开启 Wi-Fi 和蓝牙；关闭 iPhone 个人热点、Mac 互联网共享。首次建议使用可传输数据的 USB-C 线直连 Mac。
5. 解锁 iPhone，按提示选择“信任”Mac；如 Mac 询问是否允许连接配件，也选择允许。
6. 手机横放固定在 Mac 附近，后置镜头朝向要识别的物体，然后锁屏。运行期间解锁、接听电话或点“暂停”可能中断画面。
7. 退出正在使用相机的 FaceTime、会议软件等应用，再启动本程序。
8. 首次使用时，在 macOS 摄像头访问弹窗中允许访问。如被拒绝，前往系统设置 → 隐私与安全 → 摄像头，允许实际启动本程序的应用（通常是 Terminal/终端；从 Codex 启动时可能是 Codex）。

苹果说明：[连续互通相机](https://support.apple.com/zh-cn/102546)、[摄像头访问权限](https://support.apple.com/zh-cn/guide/mac-help/mchlf6d108da/mac)。

## 操作

双击本目录的 `启动iPhone识别.command`，默认显示带 YOLO 检测框的实时画面。按 Q、Esc 或 Ctrl+C 退出。

终端方式，先进入本目录：

```bash
cd "/path/to/CDE5312/iphone_camera"

# 第一步：确认系统列出了 iPhone；本操作不会打开相机
../.venv/bin/python iphone_yolo.py --list

# 第二步：只预览，先确认视频链路
../.venv/bin/python iphone_yolo.py --preview

# 第三步：启动实时识别
../.venv/bin/python iphone_yolo.py

# 多台 iPhone 时，使用 --list 返回的名称或完整 unique_id
../.venv/bin/python iphone_yolo.py --camera '你的 iPhone 名称'

# 处理 10 帧后退出，JSON 输出到终端，不显示窗口
../.venv/bin/python iphone_yolo.py --headless --frames 10

# 如 MPS 推理失败，可明确使用 CPU
../.venv/bin/python iphone_yolo.py --device cpu
```

USB 和无线方式都由 macOS 连续互通相机提供；不需要输入手机 IP 或 Sony API 地址。
程序持续读取相机但只保留最新帧，避免推理速度较慢时积压画面。
分辨率和帧率使用相机后端实际提供的值；headless 输出中的 width/height 是真实收到的尺寸。
YOLO 预训练模型检测常见目标，不负责自然语言描述，也不保证识别任意物体。
正常识别不会上传画面，不自动保存照片或录像。

## 排查

| 现象 | 处理 |
| --- | --- |
| --list 没有 iPhone | 检查账户、连续互通开关、Wi-Fi/蓝牙、USB 信任和锁屏状态；先用 FaceTime 的“视频”菜单确认系统能选择 iPhone |
| 多台 iPhone | 用 --camera 指定本次要用的名称或 unique_id |
| 摄像头访问被拒绝 | 在 macOS 摄像头权限页允许启动程序的应用，再重启程序；无需重置所有应用权限 |
| 已列出 iPhone 但无法打开或没有画面 | 退出其他相机应用，重新锁屏并插拔 USB，再启动 --preview |
| 解锁、来电或断线后停止 | 恢复手机锁屏及连接，然后重新运行；程序不会悄悄切换到内置相机 |
| 有画面但没有检测框 | 检查画面中是否有模型支持的常见目标；必要时调整 --confidence，例如 0.25 |

## 验证范围

测试命令：

```bash
../.venv/bin/python -m unittest -v test_iphone_yolo
```

设备枚举可以用实际 Mac 验证。自动选择、多设备歧义、权限拒绝、断流清理和有限帧模式用测试验证。
创建时已在当前 Mac 验证 iPhone 设备枚举，11 项测试全部通过，包含使用项目中的公交车图片作为模拟相机输入并运行真实 YOLO 模型、检查公交车检测结果。
测试中的视频来源为模拟摄像头；实际 iPhone 取流、系统授权弹窗及长时间运行仍需手机连接后验证。
OpenCV 设备索引通过 AVFoundation 的 uniqueID 排序匹配，依据当前项目 OpenCV 5.0 的实现；更新 OpenCV 后需重新核实。

实现依据：[OpenCV AVFoundation 后端](https://github.com/opencv/opencv/blob/5.x/modules/videoio/src/cap_avfoundation_mac.mm)。
