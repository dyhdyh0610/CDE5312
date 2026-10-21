# iPhone 实时物体坐标 · V1.2

链路：iPhone 连续互通相机 → 本地 YOLO 检测 → 检测框中心坐标 → 视频和坐标表逐帧更新。
新版只列出当前 YOLO 检测框内的物体，不再生成前后、远近等相对位置描述。
坐标直接由同一帧的检测框计算，不等待画面稳定 2 秒，也不调用 Ollama。
旧版场景描述和手持稳定调试代码仍保留，但不参与当前识别链路。

## 坐标约定

- 以原始取景画面的中心为 `(0, 0)`，X 向右为正、Y 向上为正。
- 单位为原始画面的像素 `px`，物体位置取检测框中心，保留一位小数。
- 视频窗口缩放、黑边、右侧文字面板和窗口位置都不会改变坐标。
- 原图为 `W × H`、检测框为 `[x1, y1, x2, y2]` 时：
  `X = (x1 + x2) / 2 - W / 2`，`Y = H / 2 - (y1 + y2) / 2`。
- 例如在 1920 × 1080 画面中，中心为 `(1060, 590)` 的物体显示为 `(+100.0, -50.0)`。
- 画面叠加网格、X/Y 轴、原点、检测框及中心十字；右侧表格列出同一帧的编号、物体、X 和 Y。
- 编号仅对应当前帧，便于核对标记和表格，不表示跨帧身份跟踪。
- 坐标是图像位置，不代表现实世界距离。识别范围取决于当前 YOLO 模型；没有检测框的物体不列入表格。

## 启动与操作

沿用原有 iPhone 连接和摄像头权限设置，复用上一级 `.venv` 和 `yolo26n.pt`。
连接并信任 Mac，开启连续互通相机，固定并锁定手机，然后双击本目录的 `启动iPhone识别.command`。
无需开启 Ollama，无需安装额外依赖。

- Q / Esc：退出；Ctrl+C：从终端停止。
- N / P：右侧表格下一页 / 上一页（每页 15 个，画面仍标记全部检测框）。
- `--preview`：只预览坐标轴，不运行识别。
- `--headless`：输出每帧 JSON，含 `image_size`、`coordinate_system` 以及物体的 `id/class/confidence/xyxy/position`。
- 原有 `--no-scene` 参数仍可接受；新版默认就不调用场景描述模型。

```bash
cd "/path/to/CDE5312/iphone_camera"
../.venv/bin/python iphone_yolo.py --list
../.venv/bin/python iphone_yolo.py

# 只验证视频和坐标轴
../.venv/bin/python iphone_yolo.py --preview

# 无窗口运行 80 帧；按需选择 CPU 或 MPS
../.venv/bin/python iphone_yolo.py --headless --frames 80

# 静态图片验证，不打开摄像头；只有显式 --output 才保存预览
../.venv/bin/python iphone_yolo.py --image ../installation_check/bus.jpg --headless --output output/iphone_coordinates_preview.jpg
```

实时性来自单次 YOLO 推理后立即计算坐标，以及相机线程只保留最新帧，避免积压旧画面。
正常相机模式不保存图像或录像，所有推理在电脑本地进行。

## 测试和版本

测试：`../.venv/bin/python -m unittest discover -s . -v`。
当前版本见 `VERSION`，变更见 `CHANGELOG.md`，实机验证范围和性能记录见上级 `docs/HANDOFF.md`。
V1.0 和 V1.1 标签保留旧版程序；切换版本前先保存工作区中的修改。

<details>
<summary>V1.1 场景描述与手持稳定调试记录（非当前运行模式）</summary>

# iPhone 摄像头识别与画面描述 · V1.1

链路：iPhone 后置相机 → macOS 连续互通相机 → 原始画面稳定检测 + YOLO → 左侧视频 / 右侧中文描述。

程序不导入 Sony 程序，复用上一级项目的 `.venv` 和 `yolo26n.pt`。
画面描述使用本机 Ollama 的 `qwen3.5:9b` 视觉模型；当前电脑已经安装此模型，无需新下载或新增 Python 包。
自动选择 iPhone；找不到 iPhone 或同时发现多台时会明确报错，不会自动切换到 Mac 内置摄像头。

## 画面描述

- 左侧保留实时视频和 YOLO 检测框；右侧显示中文画面内容。
- 镜头和场景基本稳定满 **2 秒** 后，提交一张原始画面给本地视觉模型。
- 描述主要物体的左、中、右位置，以及近景、中景、背景和有把握的前后关系。
- 文字逐步显示。2 秒是开始分析的等待时间，不是保证完成描述的时间。
  此电脑在参考照片上的完整描述耗时约 18 秒；首次加载和场景复杂度会影响耗时。
- 同一稳定场景只分析一次，文字保持显示；场景明显变化会清空旧描述、取消旧请求，重新等待 2 秒。
- 按 **R** 清空并重新等待稳定、生成描述；按 **Q / Esc** 退出。
- 远近来自遮挡和透视等图像线索，是视觉推断，不是实际距离测量；模型可能漏认或误判。
  预训练 YOLO 不包含纸盒等所有类别；中文描述直接看原图，不受 YOLO 类别列表限制。
- 图像只在内存中编码并提交至 `127.0.0.1:11434`，禁用代理及重定向，拒绝云端模型。
  正常相机模式不保存照片、录像或描述日志。

稳定检测在相机读取线程以约 10 Hz 观察未经标注的原始画面，避免推理和检测框变化影响计时。
默认将缩小、降噪后发生明显变化的像素面积超过 2.5% 视为场景变化，并对固定场景参考比较，防止慢速移动累积。
若长时间提示等待稳定，可先固定手机；按需通过 `--motion-threshold` 调整敏感度。

## 第一次准备

1. 使用 iPhone 17 和 Mac。苹果要求 iOS 16 或更新版本、macOS 13 或更新版本。
2. 两台设备登录同一 Apple 账户，该账户已开启双重认证。
3. iPhone：设置 → 通用 → 隔空播放与连续互通（或隔空播放与接力）→ 开启连续互通相机。
4. 两台设备开启 Wi-Fi 和蓝牙；关闭 iPhone 个人热点、Mac 互联网共享。首次建议使用可传输数据的 USB-C 线直连 Mac。
5. 解锁 iPhone，按提示选择“信任”Mac；如 Mac 询问是否允许连接配件，也选择允许。
6. 手机横放固定在 Mac 附近，后置镜头朝向要识别的物体，然后锁屏。运行期间解锁、接听电话或点“暂停”可能中断画面。
7. 退出正在使用相机的 FaceTime、会议软件等应用，再启动本程序。
8. 首次使用时，在 macOS 摄像头访问弹窗中允许访问。如被拒绝，前往系统设置 → 隐私与安全 → 摄像头，允许实际启动本程序的应用（通常是 Terminal/终端；从 Codex 启动时可能是 Codex）。
9. 打开电脑上的 Ollama 应用；描述功能需要本机 Ollama 服务运行。
   换电脑时若没有模型，可执行 `ollama pull qwen3.5:9b`，首次需要下载模型。

苹果说明：[连续互通相机](https://support.apple.com/zh-cn/102546)、[摄像头访问权限](https://support.apple.com/zh-cn/guide/mac-help/mchlf6d108da/mac)。

## 操作

双击本目录的 `启动iPhone识别.command`，默认打开视频和中文描述侧栏。
固定镜头并保持场景基本稳定 2 秒，侧栏会进入“正在理解”，随后逐步显示文字。

终端方式，先进入本目录：

```bash
cd "/path/to/CDE5312/iphone_camera"

# 第一步：确认系统列出了 iPhone；本操作不会打开相机
../.venv/bin/python iphone_yolo.py --list

# 第二步：检查本机 Ollama 和视觉模型；本操作不会打开相机
../.venv/bin/python iphone_yolo.py --check-scene

# 第三步：只预览，先确认视频链路（不调用视觉模型）
../.venv/bin/python iphone_yolo.py --preview

# 第四步：退出纯预览后，启动 YOLO + 中文空间描述
../.venv/bin/python iphone_yolo.py

# 仅运行原版 YOLO 功能，不需要 Ollama
../.venv/bin/python iphone_yolo.py --no-scene

# 可选：改用当前电脑也已安装的较小模型
../.venv/bin/python iphone_yolo.py --scene-model qwen3.5:4b

# 多台 iPhone 时，使用 --list 返回的名称或完整 unique_id
../.venv/bin/python iphone_yolo.py --camera '你的 iPhone 名称'

# 处理 10 帧后退出，JSON 输出到终端，不显示窗口
../.venv/bin/python iphone_yolo.py --headless --frames 10

# 如 MPS 推理失败，可明确使用 CPU
../.venv/bin/python iphone_yolo.py --device cpu

# 静态照片验证，不打开摄像头；此模式直接分析照片，跳过稳定计时
../.venv/bin/python iphone_yolo.py --image '/完整路径/照片.jpg' --headless --output output/preview.jpg
```

USB 和无线方式都由 macOS 连续互通相机提供；不需要输入手机 IP 或 Sony API 地址。
程序持续读取相机但只保留最新帧，避免推理速度较慢时积压画面。
分辨率和帧率使用相机后端实际提供的值；headless 输出中的 width/height 是真实收到的尺寸。
YOLO 预训练模型负责常见目标检测，本地视觉模型负责自然语言描述；两者都可能漏认或误判。
headless 模式的每帧 JSON 包含 `scene.state/text/stable_for`。有限帧测试若在 2 秒前退出，描述不会开始。

## 排查

| 现象 | 处理 |
| --- | --- |
| --list 没有 iPhone | 检查账户、连续互通开关、Wi-Fi/蓝牙、USB 信任和锁屏状态；先用 FaceTime 的“视频”菜单确认系统能选择 iPhone |
| 多台 iPhone | 用 --camera 指定本次要用的名称或 unique_id |
| 摄像头访问被拒绝 | 在 macOS 摄像头权限页允许启动程序的应用，再重启程序；无需重置所有应用权限 |
| 已列出 iPhone 但无法打开或没有画面 | 退出其他相机应用，重新锁屏并插拔 USB，再启动 --preview |
| 解锁、来电或断线后停止 | 恢复手机锁屏及连接，然后重新运行；程序不会悄悄切换到内置相机 |
| 有画面但没有检测框 | 检查画面中是否有模型支持的常见目标；必要时调整 --confidence，例如 0.25 |
| 无法连接本机 Ollama | 打开 Ollama 应用，再运行 --check-scene；如仅需 YOLO，可使用 --no-scene |
| 模型不支持图像 / 尚未安装 | 选择已安装的本地视觉模型，或按报错中的 ollama pull 命令下载 |
| 一直等待稳定 | 固定镜头、减少大幅运动；需要容忍更多运动时可尝试 --motion-threshold 0.04 |
| 已稳定但文字尚未出现 | 本地模型正在加载或分析，等待首批文字；2 秒表示开始分析，不表示完成 |
| 描述显示错误 | 解决 Ollama 问题后按 R 重试，或调整镜头让目标更清晰 |

## 验证范围

测试命令：

```bash
../.venv/bin/python -m unittest discover -s . -v
```

设备枚举可以用实际 Mac 验证。自动选择、多设备歧义、权限拒绝、断流清理和有限帧模式用测试验证。
V1.0 原有 11 项测试保留；新版增加稳定计时、慢速漂移、旧描述取消、流式本地协议、中文侧栏和 CLI 链路测试。
真实 YOLO 模型在公交车样图及用户参考照片上运行，本地视觉模型也已在参考照片上实际生成中文空间描述。
硬件模拟测试不能代替实际 iPhone 取流、系统授权弹窗及长时间运行验证。
OpenCV 设备索引通过 AVFoundation 的 uniqueID 排序匹配，依据当前项目 OpenCV 5.0 的实现；更新 OpenCV 后需重新核实。

实现依据：[OpenCV AVFoundation 后端](https://github.com/opencv/opencv/blob/5.x/modules/videoio/src/cap_avfoundation_mac.mm)。

本地视觉接口依据：[Ollama Vision](https://docs.ollama.com/capabilities/vision)、[Chat API](https://docs.ollama.com/api/chat)、[Qwen3.5 模型](https://ollama.com/library/qwen3.5)。

## 版本管理

Git 历史位于上一级项目根目录，版本标记位于 `VERSION`，改动见 `CHANGELOG.md`。
`V1.0` 标签保存新增描述功能前的完整 iPhone 程序，基线提交为 `8851d22`。
V1.1 开发分支为 `codex/iphone-scene-description`。模型、依赖环境和用户照片不进入 Git。

查看版本：`git -C .. log --oneline --decorate`。
需要切换原版时，先保存当前修改，再在项目根目录运行 `git switch --detach V1.0`；
返回新版运行 `git switch codex/iphone-scene-description`。原版也可从 V1.1 的 `--no-scene` 模式使用。

</details>
