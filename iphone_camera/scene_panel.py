"""Current-frame object coordinate table beside the scaled camera view."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from frame_coordinates import COLORS, annotate_frame
from cup_target import cup_target, prioritize_cups


CLASS_NAMES = dict(zip(
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,"
    "stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,"
    "backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,"
    "baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,"
    "bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,"
    "oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(","),
    "人,自行车,汽车,摩托车,飞机,公交车,火车,卡车,船,交通灯,消防栓,停止标志,停车计时器,长椅,"
    "鸟,猫,狗,马,羊,牛,大象,熊,斑马,长颈鹿,背包,雨伞,手提包,领带,行李箱,飞盘,双板滑雪板,"
    "单板滑雪板,球,风筝,棒球棒,棒球手套,滑板,冲浪板,网球拍,瓶子,高脚杯,杯子,叉子,刀,勺子,"
    "碗,香蕉,苹果,三明治,橙子,西兰花,胡萝卜,热狗,披萨,甜甜圈,蛋糕,椅子,沙发,盆栽,床,餐桌,"
    "马桶,电视,笔记本电脑,鼠标,遥控器,键盘,手机,微波炉,烤箱,烤面包机,水槽,冰箱,书,时钟,"
    "花瓶,剪刀,玩具熊,吹风机,牙刷".split(","),
    strict=True,
))


class ScenePanel:
    PAGE_SIZE = 10

    def __init__(self, font_path=None):
        candidates = [font_path] if font_path else [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        path = next((path for path in candidates if path and Path(path).is_file()), None)
        if path is None:
            raise RuntimeError("没有找到中文字体，请用 --font 指定中文字体文件。")
        self.title_font = ImageFont.truetype(path, 26)
        self.text_font = ImageFont.truetype(path, 20)
        self.small_font = ImageFont.truetype(path, 16)

    def render(self, frame, detections, fps=0, page=0):
        canvas = Image.new("RGB", (1280, 760), "#101827")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), "iPhone 杯子对准 · 中央目标框", font=self.title_font, fill="#eef3fa")
        draw.text((24, 56), f"{fps:.1f} FPS    Q / Esc 退出    N / P 坐标表翻页",
                  font=self.small_font, fill="#a5b3c6")
        height, width = frame.shape[:2]
        scale = min(800 / width, 648 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        items = prioritize_cups(detections or [])
        state = cup_target(detections, width, height)
        scaled = annotate_frame(cv2.resize(frame, size), items, (width, height))
        x1, y1, x2, y2 = state["roi_xyxy"]
        roi_color = (100, 240, 100) if state["inside"] else (0, 210, 255)
        cv2.rectangle(scaled, (round(x1 * size[0] / width), round(y1 * size[1] / height)),
                      (round(x2 * size[0] / width), round(y2 * size[1] / height)), roi_color, 3)
        cv2.putText(scaled, "CUP TARGET", (round(x1 * size[0] / width),
                    max(18, round(y2 * size[1] / height) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
        if state["target"] is not None:
            box = state["target"]["xyxy"]
            cv2.rectangle(scaled, (round(box[0] * size[0] / width), round(box[1] * size[1] / height)),
                          (round(box[2] * size[0] / width), round(box[3] * size[1] / height)),
                          (255, 255, 255), 3)
        video = Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
        canvas.paste(video, (20 + (800 - size[0]) // 2, 90 + (648 - size[1]) // 2))
        draw.rounded_rectangle((840, 16, 1264, 744), radius=16, fill="#1b283c")
        draw.text((860, 36), "杯子对准状态", font=self.title_font, fill="#eef3fa")
        draw.text((860, 80), f"原始画面 {width} × {height} px · 逐帧更新",
                  font=self.small_font, fill="#7ddfc4")
        message = {"preview": "纯预览，未运行检测", "no_cup": "未检测到杯子，等待杯子",
                   "inside": "杯子框已完整进入有效区", "outside": "杯子框尚未完整进入有效区",
                   "too_large": "杯子框过大，有效区装不下"}[state["status"]]
        draw.text((860, 118), message, font=self.text_font,
                  fill="#7ddfc4" if state["inside"] else "#ffd166")
        target_text = (f"目标：杯子 #{state['target']['id']} · 共 {state['cup_count']} 个杯子"
                       if state["target"] else "其他物体不会替代杯子目标")
        draw.text((860, 152), target_text, font=self.small_font, fill="#eef3fa")
        draw.text((860, 181), "目标框：X ±250 / Y ±375 px", font=self.small_font, fill="#a5b3c6")
        draw.text((860, 211), state["instruction"], font=self.small_font, fill="#ffd166")
        draw.text((860, 250), "编号 / 物体", font=self.small_font, fill="#a5b3c6")
        draw.text((1110, 250), "X", font=self.small_font, fill="#a5b3c6")
        draw.text((1210, 250), "Y", font=self.small_font, fill="#a5b3c6")
        pages = max(1, (len(items) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        page = min(max(page, 0), pages - 1)
        start = page * self.PAGE_SIZE
        if detections is None:
            draw.text((860, 285), "纯预览，未运行物体检测。", font=self.text_font, fill="#eef3fa")
        elif not items:
            draw.text((860, 285), "当前画面未检测到物体。", font=self.text_font, fill="#eef3fa")
        for index, item in enumerate(items[start:start + self.PAGE_SIZE], start):
            y = 282 + (index - start) * 29
            color = tuple(reversed(COLORS[index % len(COLORS)]))
            label = f"#{item['id']} {CLASS_NAMES.get(item['class'], item['class'])}"
            while len(label) > 1 and draw.textlength(label, font=self.text_font) > 190:
                label = label[:-2] + "…"
            draw.text((860, y), label, font=self.text_font, fill=color)
            draw.text((1145, y), f"{item['position']['x']:+.1f}", anchor="ra",
                      font=self.text_font, fill="#eef3fa")
            draw.text((1242, y), f"{item['position']['y']:+.1f}", anchor="ra",
                      font=self.text_font, fill="#eef3fa")
        draw.text((860, 610), f"共 {len(items)} 个物体 · 第 {page + 1}/{pages} 页",
                  font=self.small_font, fill="#a5b3c6")
        draw.line((860, 648, 1240, 648), fill="#34455e", width=1)
        draw.text((860, 665), "原点：画面中心 (0, 0)", font=self.small_font, fill="#a5b3c6")
        draw.text((860, 690), "X 向右为正；Y 向上为正；单位 px", font=self.small_font, fill="#a5b3c6")
        draw.text((860, 715), "入框判断：整个杯子检测框均在有效区。", font=self.small_font, fill="#a5b3c6")
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
