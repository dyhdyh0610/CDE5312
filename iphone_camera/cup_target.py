"""Cup-only target selection and native-frame central ROI geometry."""

ROI_HALF_WIDTH = 250
ROI_HALF_HEIGHT = 375


def prioritize_cups(detections):
    # Preserve current-frame IDs and the relative order within each group.
    return sorted(detections, key=lambda item: item["class"] != "cup")


def cup_target(detections, width, height):
    bounds = [max(0, width / 2 - ROI_HALF_WIDTH),
              max(0, height / 2 - ROI_HALF_HEIGHT),
              min(width, width / 2 + ROI_HALF_WIDTH),
              min(height, height / 2 + ROI_HALF_HEIGHT)]
    state = {"status": "preview" if detections is None else "no_cup",
             "roi_xyxy": bounds, "criterion": "entire_cup_box_inside_roi",
             "target": None, "inside": False, "cup_count": 0,
             "directions": {name: False for name in ("up", "right", "down", "left")},
             "instruction": "纯预览，未运行检测" if detections is None else "未检测到杯子，停止方向提示"}
    cups = [item for item in detections or [] if item["class"] == "cup"]
    state["cup_count"] = len(cups)
    if not cups:
        return state

    def center(item):
        x1, y1, x2, y2 = item["xyxy"]
        return (x1 + x2) / 2, (y1 + y2) / 2

    target = min(cups, key=lambda item: (
        ((center(item)[0] - width / 2) / width) ** 2
        + ((center(item)[1] - height / 2) / height) ** 2,
        -item.get("confidence", 0), item["id"]))
    x1, y1, x2, y2 = target["xyxy"]
    inside = bounds[0] <= x1 and bounds[1] <= y1 and x2 <= bounds[2] and y2 <= bounds[3]
    state.update(status="inside" if inside else "outside", target=target, inside=inside)
    if inside:
        state["instruction"] = "保持相机稳定，停止方向提示"
    elif x2 - x1 > bounds[2] - bounds[0] or y2 - y1 > bounds[3] - bounds[1]:
        # Panning alone cannot fit a box larger than the ROI. Do not give opposing cues.
        state.update(status="too_large", instruction="杯子框过大，请后退或缩小画面")
    else:
        # These directions tell the user where to move the CAMERA, not the cup.
        state["directions"] = {"up": y1 < bounds[1], "right": x2 > bounds[2],
                               "down": y2 > bounds[3], "left": x1 < bounds[0]}
        names = {"up": "上", "right": "右", "down": "下", "left": "左"}
        state["instruction"] = "相机向" + "、".join(
            names[name] for name, active in state["directions"].items() if active) + "移动"
    return state
