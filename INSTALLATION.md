# α6000 无线 YOLO：电脑端安装

程序和依赖位于当前项目，不需要 Imaging Edge。Python 解释器使用 `.venv/bin/python`。

## 实机运行

1. 相机：关闭飞行模式，打开 MENU → 应用程序 → 应用程序列表 → 智能遥控 / Smart Remote Embedded。
2. Mac：连接相机显示的 DIRECT-… Wi-Fi。macOS 若询问本地网络访问权限，请允许运行程序访问。
3. 双击 `启动相机识别.command`。程序自动发现相机，并显示坐标轴、网格和带中心坐标的 YOLO 检测框。
4. 按 Q、Esc 或 Ctrl+C 退出。退出时程序请求停止取景，并结束由本程序启动的遥控拍摄状态。

相机热点通常不提供互联网；依赖与模型已经预先安装，识别不需要在线服务。

## 取景框二维坐标

坐标单位为原始取景画面的像素，原点位于画面几何中心 `(宽度 / 2, 高度 / 2)`。
X 向右为正，Y 向上为正；每个物体的位置使用检测框中心，标签显示 `类别 置信度 (X, Y) px`。
画面网格间隔为 100 像素，十字标记为物体中心。仅预览模式也显示网格和坐标轴。

例如，640 × 480 的画面中，检测框中心位于图像像素 `(420, 190)` 时，标签坐标为 `(+100, +50)`。
公式为 `X = (x1 + x2) / 2 - 宽度 / 2`，`Y = 高度 / 2 - (y1 + y2) / 2`。
计算使用原始帧大小，不使用模型内部的 640 像素推理输入大小，也不表示现实世界的距离或物体间关系。

`--headless` 每帧输出 JSON，保留原来的 `class`、`confidence`、`xyxy`，新增每个检测的
`position: {"x": 100.0, "y": 50.0}`，以及帧级 `image_size` 和 `coordinate_system`。
其中 `coordinate_system.origin` 使用原图左上角为原点的像素坐标描述新原点的位置；
`position` 则使用画面中心为原点的坐标。坐标显示精度为 0.1 像素，不代表识别精度。

## 排查连接

```bash
cd "/path/to/CDE5312"

# 只发现相机，不启动取景
./.venv/bin/python sony_wifi_yolo.py --discover

# 先验证画面，再加入识别
./.venv/bin/python sony_wifi_yolo.py --preview

# 如需使用 CPU
./.venv/bin/python sony_wifi_yolo.py --device cpu
```

如果 SSDP 自动发现失败，可以使用 `--api-url` 指定从设备信息获得的完整 camera API 地址。
部分 α6000 的常见地址是 `http://192.168.122.1:8080/sony/camera`，但实际地址必须核实。
网络流断开时程序会明确报错并退出，重新连接后再次启动即可。

## 已安装组件及验证

- YOLO 模型：项目根目录的 `yolo26n.pt`。
- 当前完整依赖版本：`requirements-installed.txt`。
- 安装清单与版本：`installation_check/installation_report.json`。
- 静态图片识别结果：`installation_check/detected_bus.jpg`。
- 模型推理结果：`installation_check/model_check.json`。
- Sony 协议测试：`test_sony_wifi_yolo.py`。

运行测试：`./.venv/bin/python -m unittest -v test_sony_wifi_yolo`。

实机连接、相机内置遥控应用、实际预览分辨率和长期稳定性仍需在相机连接后验证。
预训练模型检测 COCO 常见目标，不包含对整体景物的自然语言描述。

## 参考

- Ultralytics 官方安装与模型说明：https://docs.ultralytics.com/quickstart/
- α6000 Smart Remote 接口说明：https://github.com/micolous/gst-plugins-sonyalpha
- Sony Liveview 帧结构：https://github.com/micolous/gst-plugins-sonyalpha/blob/master/FRAMING_FORMAT.md
