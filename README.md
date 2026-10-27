# 本地相机识别与拍摄引导

在 Apple Silicon Mac 上连接 Sony Wi-Fi 相机或 iPhone 连续互通相机，在本地完成物体检测、画面坐标显示和拍摄引导。Sony 可通过 USB 连接 ESP32，提供上、右、下、左四方向振动。

| 功能 | 入口 | 说明 |
| --- | --- | --- |
| Sony 识别与文字引导 | `启动相机识别.command` | 稳定取景后选择人、树或水杯，再持续跟踪并提示调整方向 |
| Sony 与 ESP32 振动引导 | `启动相机振动引导.command` | 将已选目标的方向传给四个振动模块 |
| 四向振动独立测试 | `测试四向振动.command` | 不连接相机，依次验证四个模块 |
| iPhone 实时坐标 | `iphone_camera/启动iPhone识别.command` | YOLO 检测框与坐标表逐帧更新 |

坐标使用原始画面的检测框中心，画面中心为 `(0, 0)`，X 向右、Y 向上，单位为像素。坐标不是现实距离。旧版视觉场景描述模块保留在 `iphone_camera/`，当前 iPhone 默认模式不调用 Ollama。

## 首次安装

本项目已在 macOS、Apple Silicon、Python 3.11 环境验证。先将仓库下载到本地，再在项目根目录运行：

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-installed.txt
```

`requirements-installed.txt` 是已验证环境的依赖快照。模型文件不随代码上传；请在连接相机热点前联网准备：

```bash
./.venv/bin/python -c 'from ultralytics import YOLO, YOLOE; YOLO("yolo26n.pt"); YOLOE("yoloe-26n-seg-pf.pt")'
```

两个 `.pt` 文件应位于项目根目录。完成准备后，正常相机识别在本地运行。ESP32 需另外上传本仓库中的 Arduino 固件。

连接方法与参数：

- [Sony 安装与取景引导](INSTALLATION.md)
- [ESP32 接线、固件与振动引导](HAPTICS.md)
- [iPhone 连接与使用](iphone_camera/README.md)
- [历史版本变更](iphone_camera/CHANGELOG.md)
- [当前状态与验证限制](docs/HANDOFF.md)

文档中的 `/path/to/CDE5312` 是占位路径，使用时替换为实际下载目录。

## 验证

```bash
./.venv/bin/python -m unittest -v test_sony_wifi_yolo test_sony_guidance test_sony_haptics
./.venv/bin/python -m unittest discover -s iphone_camera -v
```

2026-10-27 本地验证：Sony／振动引导 59 项、iPhone 52 项测试通过。固件测试使用 `clang++` 编译实际草图与测试替身；串口测试使用虚拟串口。实体相机、ESP32 上传和实际振动效果仍需按各自使用文档检查。

完整集成测试依赖本地模型及 `installation_check/` 中的 `bus.jpg`、`tree_sample.jpg`、`cup_sample.jpg`。这些图片和生成结果不会上传，因此仅下载仓库不能直接重现全部集成测试。iPhone 实图测试在缺少本地素材时跳过，Sony 相应测试需要先准备素材。无需图像素材的部分测试可单独执行：

```bash
./.venv/bin/python -m unittest -v test_sony_wifi_yolo.FrameCoordinateTests test_sony_guidance.StabilityTests test_sony_haptics.DirectionTests test_sony_haptics.SerialTests test_sony_haptics.FirmwareTests
```

## 数据与版本

- 上传内容包括全部应用源码、Arduino 固件、测试、启动脚本、依赖清单和使用文档。
- 照片、录像、识别输出、模型、虚拟环境、日志、凭据及本机配置由 `.gitignore` 排除。
- 文档中的个人用户名与本机绝对路径已脱敏；发布历史也执行相同清理。
- GitHub 使用独立脱敏发布历史，保留历史分支及 V1.0、V1.1、V1.2 标签；由于内容脱敏，提交哈希与原本地历史不同。GitHub 的 `main` 指向本次完整代码。
- 不把实际相机地址、Wi-Fi 密码、设备唯一标识或串口枚举结果提交到仓库。示例中的回环地址、SSDP 组播地址及 Sony 通用热点地址属于协议或示例值。
