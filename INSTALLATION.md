# α6000 无线拍摄引导：电脑端安装

程序和依赖位于当前项目，不需要 Imaging Edge。Python 解释器使用 `.venv/bin/python`。

## 实机运行

1. 相机：关闭飞行模式，打开 MENU → 应用程序 → 应用程序列表 → 智能遥控 / Smart Remote Embedded。
2. Mac：连接相机显示的 DIRECT-… Wi-Fi。macOS 若询问本地网络访问权限，请允许运行程序访问。
3. 双击 `启动相机识别.command`。程序自动发现相机，显示取景画面、坐标轴和中文提示面板。
4. 大致保持取景 1.5 秒，允许轻微手持抖动。程序只列出人、树、水杯；每个具体对象都有独立编号。
   选择前会定期刷新列表，首次漏检时可继续保持取景，等待补充识别。
5. 点击取景框中的对象或右侧条目，锁定想拍摄的对象。对象较多时用 N/P 翻页。
6. 此后可以移动相机。程序持续跟踪已选择的对象，逐帧显示距中心的像素偏移和中文调整提示。
7. 目标居中后显示“可以拍照”。程序不自动触发快门。按 R 清除选择，重新开始稳定计时。
8. 按 Q、Esc 或 Ctrl+C 退出。退出时程序请求停止取景，并结束由本程序启动的遥控拍摄状态。

相机热点通常不提供互联网；依赖与模型已经预先安装，识别不需要在线服务。

## 三类目标与引导规则

- 人、水杯使用现有 `yolo26n.pt` 的 `person` 和 `cup` 类，其他 COCO 类别不作为候选对象。
- 水杯还使用现有 YOLOE 模型的 `cup`、`coffee cup`、`paper cup`、`insulated cup`、`mug`、`teacup`
  补检，统一显示为“水杯”。不把蛋糕、奖杯、量杯或瓶子归入水杯。
  同一水杯被两个模型同时检出时合并重叠框，优先保留已有编号；锁定后仍只跟随选定的跟踪编号。
- 树使用本地 `yoloe-26n-seg-pf.pt` 的树和真实树种类别，例如 `tree`、`oak tree`、`eucalyptus tree`。
  这些结果统一显示为“树”，保留 `source_class` 供诊断；不把盆栽、猫爬架、鞋楦或树蛙当作树。
- 人的检测阈值保持 0.35，水杯、树分别使用 0.20，避免相机预览中的弱检测被统一的 0.35 阈值过滤。
  可分别用 `--confidence`、`--cup-confidence`、`--tree-confidence` 调整，取值 0 到 1；
  `--coordinates-only` 仍只使用 `--confidence`。跟踪器的新目标接收阈值同步下调，
  避免已通过检测的对象又被跟踪阶段丢掉。阈值降低提高检出机会，也可能增加误报。
- 稳定等待仍为 1.5 秒，已放宽手持抖动容差。运动估计可靠时，允许相对计时起始参考帧
  水平、垂直各不超过画面宽、高的 2.5% 位移，并在比较前对齐；例如 640 × 480 画面约为
  水平 16、垂直 12 像素。轻微转动、图像噪声和曝光波动由放宽后的图像差异阈值容忍。
  这些位移相对固定参考帧计算，不能逐帧累积，因此持续平移相机仍会重新计时。
  对齐只用于稳定性判断，不移动显示画面，也不改变目标识别、坐标或后续引导。
  在 160 × 96 灰度比较图上，平均差异阈值从 4 提高到 12；变化像素阈值从灰度差 20
  提高到 30，允许的变化比例从 4% 提高到 12%。纹理不足时回退到未对齐的图像差异比较。
  选择前明显位移或场景变化会清除旧候选并重新计时；识别计算本身不重新触发等待。
- 达到稳定条件后，每次识别完成约 0.75 秒后刷新三类候选，直到用户选择，包括空场景自动重试。
  刷新时保持持续跟踪和已有编号，不重置跟踪器；新增对象分配新编号。场景改变或按 R 时才重新编号。
- 锁定后不再等待 1.5 秒，使用 BoT-SORT 的持续跟踪编号更新对象位置。
  目标丢失时清除方向提示；只接受同一跟踪编号重新出现，或由用户按 R 重新选择。
  遮挡、快速移动、同类目标交叉仍可能影响跟踪身份，不能保证物理身份永不混淆。
- 对象在画面右侧时提示向右转动相机，在上方时提示向上抬起镜头，其余方向同理。
  提示“小幅 / 适度 / 较大幅度”按偏移占整幅宽度或高度的比例分级，阈值分别为 10% 和 25%。
  同时显示具体像素偏移与百分比，不推断未经标定的转动角度或现实距离。
- 默认居中容差为宽、高各 3%，最低 8 像素；进入居中状态后使用 1.3 倍退出容差，减少文字抖动。
  `--center-tolerance` 可调整比例，`--stable-seconds` 可调整首次选择前的稳定时长。

新增跟踪依赖和树模型已在当前电脑准备。其他电脑应在连接相机 Wi-Fi 前完成：

```bash
cd "/path/to/CDE5312"
./.venv/bin/python -m pip install scipy==1.17.1 lap==0.5.13
./.venv/bin/python -c 'from ultralytics import YOLOE; YOLOE("yoloe-26n-seg-pf.pt")'
```

模型为 prompt-free 版本，取景时不需要文本编码器、Ollama 或外部服务。
程序启动前检查本地模型和中文字体，缺少时明确报错，不在相机热点下自动下载。

## 取景框二维坐标

坐标单位为原始取景画面的像素，原点位于画面几何中心 `(宽度 / 2, 高度 / 2)`。
X 向右为正，Y 向上为正；每个物体的位置使用检测框中心，标签显示 `类别 置信度 (X, Y) px`。
画面网格间隔为 100 像素，十字标记为物体中心。仅预览模式也显示网格和坐标轴。

例如，640 × 480 的画面中，检测框中心位于图像像素 `(420, 190)` 时，标签坐标为 `(+100, +50)`。
公式为 `X = (x1 + x2) / 2 - 宽度 / 2`，`Y = 高度 / 2 - (y1 + y2) / 2`。
计算使用原始帧大小，不使用模型内部的 640 像素推理输入大小，也不表示现实世界的距离或物体间关系。

默认拍摄引导模式的 `--headless` 输出还包含 `shooting`：当前阶段、三类对象计数、候选对象、
锁定编号、最新目标及中文提示。可用 `--target-number N` 明确预先选择稳定识别后的候选编号，
供无窗口集成使用。等待阶段不运行用户画面的目标检测；画面稳定后定期重扫，空场景也会自动重试。
跟踪来源前缀为 `base:` 或 `scene:`；YOLOE 的杯型、树种保留 `source_class`，规范类别仍为 `cup`、`tree`。

`--coordinates-only` 保留旧版逐帧全类别坐标模式，不运行选择或引导流程。
此模式的 `--headless` 每帧输出 JSON，保留原来的 `class`、`confidence`、`xyxy`，每个检测包含
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
- 树木及杯型补检模型：项目根目录的 `yoloe-26n-seg-pf.pt`，只筛选真实树木和上述饮水杯类别。
- 当前完整依赖版本：`requirements-installed.txt`。
- 安装清单与版本：`installation_check/installation_report.json`。
- 静态图片识别结果：`installation_check/detected_bus.jpg`。
- 模型推理结果：`installation_check/model_check.json`。
- Sony 协议测试：`test_sony_wifi_yolo.py`。
- 拍摄流程、中文界面、真实三类模型和跟踪测试：`test_sony_guidance.py`。

运行测试：`./.venv/bin/python -m unittest -v test_sony_wifi_yolo test_sony_guidance`。
模型集成测试使用本机 `installation_check/` 中的 `bus.jpg`、`tree_sample.jpg`、`cup_sample.jpg`；
它们是本地校验资源，不随代码提交。

实机连接、相机内置遥控应用、实际预览分辨率和长期稳定性仍需在相机连接后验证。
预训练模型检测 COCO 常见目标，不包含对整体景物的自然语言描述。

## 参考

- Ultralytics 官方安装与模型说明：https://docs.ultralytics.com/quickstart/
- YOLOE 模型与内置类别说明：https://docs.ultralytics.com/models/yoloe/
- α6000 Smart Remote 接口说明：https://github.com/micolous/gst-plugins-sonyalpha
- Sony Liveview 帧结构：https://github.com/micolous/gst-plugins-sonyalpha/blob/master/FRAMING_FORMAT.md
