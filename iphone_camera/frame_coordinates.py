"""Image-plane coordinates in native pixels; independent of display scaling."""

import math

import cv2


COLORS = ((100, 240, 100), (80, 190, 255), (255, 180, 100), (210, 120, 255))


def coordinate_system(width, height):
    return {"origin": [width / 2, height / 2], "unit": "px", "x_positive": "right",
            "y_positive": "up", "position_reference": "bounding_box_center"}


def frame_position(xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    return {"x": round((x1 + x2) / 2 - width / 2, 1) or 0.0,
            "y": round(height / 2 - (y1 + y2) / 2, 1) or 0.0}


def detection_items(result, width, height):
    items = []
    for number, box in enumerate(result.boxes, 1):
        bounds = box.xyxy[0].tolist()
        items.append({"id": number, "class": result.names[int(box.cls.item())],
                      "confidence": round(float(box.conf.item()), 3), "xyxy": bounds,
                      "position": frame_position(bounds, width, height)})
    return items


def draw_label(frame, text, x, y, color, scale=0.42, occupied=None):
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    label_width, label_height = width + 8, height + baseline + 8
    x = max(0, min(round(x), frame.shape[1] - label_width))
    y = max(0, min(round(y), frame.shape[0] - label_height))
    if occupied is not None:
        candidates = [y, *range(y - label_height, -1, -label_height),
                      *range(y + label_height, frame.shape[0] - label_height + 1, label_height)]
        for candidate in candidates:
            if all(x + label_width <= left or x >= right or
                   candidate + label_height <= top or candidate >= bottom
                   for left, top, right, bottom in occupied):
                y = candidate
                break
        occupied.append((x, y, x + label_width, y + label_height))
    cv2.rectangle(frame, (x, y), (x + label_width, y + label_height), (15, 20, 28), -1)
    cv2.putText(frame, text, (x + 4, y + height + 3), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def annotate_frame(frame, detections, native_size=None):
    """Draw on native or resized video while labels/grid remain in native pixels."""
    height, width = frame.shape[:2]
    native_width, native_height = native_size or (width, height)
    scale_x, scale_y = width / native_width, height / native_height
    center_x, center_y = round(width / 2), round(height / 2)
    grid = frame.copy()
    # At most ~10 divisions on the long dimension, so a 4K source stays readable.
    step = max(100, math.ceil(max(native_width, native_height) / 1000) * 100)
    ticks = []
    for offset in range(step, max(native_width, native_height), step):
        for sign in (-1, 1):
            x = round(width / 2 + sign * offset * scale_x)
            y = round(height / 2 - sign * offset * scale_y)
            if 0 < x < width - 1:
                cv2.line(grid, (x, 0), (x, height - 1), (160, 160, 160), 1)
                ticks.append((f"{sign * offset:+d}", x - 15, center_y + 6))
            if 0 < y < height - 1:
                cv2.line(grid, (0, y), (width - 1, y), (160, 160, 160), 1)
                ticks.append((f"{sign * offset:+d}", center_x + 6, y - 10))
    output = cv2.addWeighted(frame, 0.8, grid, 0.2, 0)
    axis_color = (255, 210, 60)
    cv2.arrowedLine(output, (0, center_y), (width - 1, center_y), axis_color, 1, tipLength=0.025)
    cv2.arrowedLine(output, (center_x, height - 1), (center_x, 0), axis_color, 1, tipLength=0.025)
    for text, x, y in ticks:
        draw_label(output, text, x, y, axis_color, scale=0.35)
    cv2.circle(output, (center_x, center_y), 3, axis_color, -1)
    draw_label(output, "(0,0)", center_x + 6, center_y + 6, axis_color)
    draw_label(output, "+X", width - 36, center_y - 26, axis_color)
    draw_label(output, "+Y", center_x + 6, 3, axis_color)

    labels = []
    for index, item in enumerate(detections):
        x1, y1, x2, y2 = item["xyxy"]
        center = (round((x1 + x2) / 2 * scale_x), round((y1 + y2) / 2 * scale_y))
        color = COLORS[index % len(COLORS)]
        cv2.rectangle(output, (round(x1 * scale_x), round(y1 * scale_y)),
                      (round(x2 * scale_x), round(y2 * scale_y)), color, 2)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 12, 2)
        position = item["position"]
        text = (f"#{item['id']} {item['class']} "
                f"({position['x']:+.1f}, {position['y']:+.1f})")
        labels.append((text, x1 * scale_x, y1 * scale_y - 24, color))
    occupied = []
    for text, x, y, color in labels:
        draw_label(output, text, x, y, color, occupied=occupied)
    return output
