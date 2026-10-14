"""Chinese side panel rendered with Pillow, beside the OpenCV camera view."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ScenePanel:
    def __init__(self, font_path=None):
        candidates = [font_path] if font_path else [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        path = next((path for path in candidates if path and Path(path).is_file()), None)
        if path is None:
            raise RuntimeError("没有找到中文字体，请用 --font 指定中文字体文件。")
        self.title_font = ImageFont.truetype(path, 26)
        self.text_font = ImageFont.truetype(path, 20)
        self.small_font = ImageFont.truetype(path, 16)

    @staticmethod
    def wrap(draw, text, font, width):
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for char in paragraph:
                if line and draw.textlength(line + char, font=font) > width:
                    lines.append(line)
                    line = char
                else:
                    line += char
            lines.append(line)
        return lines

    def render(self, frame, scene, fps=0, model=""):
        canvas = Image.new("RGB", (1280, 760), "#101827")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), "iPhone 实时识别 · V1.1", font=self.title_font, fill="#eef3fa")
        draw.text((24, 56), f"{fps:.1f} FPS    Q / Esc 退出    R 重新描述",
                  font=self.small_font, fill="#a5b3c6")
        height, width = frame.shape[:2]
        scale = min(800 / width, 648 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        video = Image.fromarray(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB))
        canvas.paste(video, (20 + (800 - size[0]) // 2, 90 + (648 - size[1]) // 2))
        draw.rounded_rectangle((840, 16, 1264, 744), radius=16, fill="#1b283c")
        draw.text((860, 36), "画面内容", font=self.title_font, fill="#eef3fa")
        required = scene["required_seconds"]
        state = scene["state"]
        if state == "waiting":
            status = f"等待稳定  {min(scene['stable_for'], required):.1f} / {required:g} 秒"
            message = "保持镜头和场景基本稳定，随后自动描述物体的位置与前后关系。"
        elif state in ("queued", "describing"):
            status = "画面已稳定 · 正在理解…"
            message = scene["text"] or "正在分析当前画面，文字将逐步显示。首次加载模型需要稍等。"
        elif state == "error":
            status = "描述暂时不可用"
            message = scene["error"] + "\n\n解决后按 R 重试。"
        else:
            status = "画面已稳定 · 描述完成"
            message = scene["text"]
        draw.text((860, 80), status, font=self.small_font,
                  fill="#f6be6b" if state == "error" else "#7ddfc4")
        lines = self.wrap(draw, message, self.text_font, 380)
        # Normal generation is limited to 260 Chinese characters; guard errors too.
        visible_lines = lines[:17]
        if len(lines) > 17:
            visible_lines[-1] = visible_lines[-1][:-1] + "…"
        for index, line in enumerate(visible_lines):
            draw.text((860, 126 + index * 29), line, font=self.text_font, fill="#eef3fa")
        draw.line((860, 648, 1240, 648), fill="#34455e", width=1)
        draw.text((860, 670), "远近为视觉推断，非实际距离测量。", font=self.small_font, fill="#a5b3c6")
        draw.text((860, 700), f"本地模型  {model}", font=self.small_font, fill="#a5b3c6")
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
