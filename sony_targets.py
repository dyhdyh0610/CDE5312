"""Local person/cup detection plus YOLOE trees/cup variants and persistent tracking."""

from pathlib import Path

import numpy as np


TREE_LABELS = {
    "tree", "apple tree", "autumn tree", "banana tree", "banyan tree", "beech tree",
    "birch tree", "cherry tree", "coconut tree", "cypress tree", "eucalyptus tree",
    "fir tree", "fruit tree", "ginkgo tree", "oak tree", "olive tree", "orange tree",
    "palm tree", "pine tree", "maple tree", "willow tree",
}

CUP_LABELS = {"cup", "coffee cup", "paper cup", "insulated cup", "mug", "teacup"}


def target_class(label):
    if label in TREE_LABELS:
        return "tree"
    if label in CUP_LABELS:
        return "cup"
    return "person" if label == "person" else None


class TargetDetector:
    def __init__(self, model_path, tree_model_path, device="cpu", confidence=0.35,
                 cup_confidence=0.20, tree_confidence=0.20):
        for path in (model_path, tree_model_path):
            if not Path(path).is_file():
                raise RuntimeError(f"本地模型不存在：{path}。请先按 INSTALLATION.md 联网准备模型。")
        try:
            import lap  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as error:
            raise RuntimeError("缺少跟踪依赖；请先在项目虚拟环境安装 scipy 和 lap。") from error
        from ultralytics import YOLO, YOLOE
        self.base = YOLO(str(model_path))
        self.trees = YOLOE(str(tree_model_path))
        self.device = device
        self.thresholds = {"person": confidence, "cup": cup_confidence, "tree": tree_confidence}
        self.base_classes = {name: i for i, name in self.base.names.items() if name in ("person", "cup")}
        self.tree_classes = [i for i, name in self.trees.names.items() if name in TREE_LABELS]
        self.cup_classes = [i for i, name in self.trees.names.items() if name in CUP_LABELS]
        if set(self.base_classes) != {"person", "cup"} or not self.tree_classes or not self.cup_classes:
            raise RuntimeError("模型类别不完整：必须支持人、水杯和真实树木。")
        # Initialize BOTH tracker instances before assigning any visible target IDs.
        # Their underlying ID counter is shared, so initializing the second one late
        # would reset the counter while the first model already had active tracks.
        blank = np.zeros((96, 96, 3), dtype=np.uint8)
        self.track(blank)
        # BoT-SORT otherwise rejects new detections below its own 0.25 cutoff,
        # even when the detector's confidence threshold has been lowered.
        for model, threshold in ((self.base, min(confidence, cup_confidence)),
                                 (self.trees, min(tree_confidence, cup_confidence))):
            for tracker in model.predictor.trackers:
                tracker.args.track_high_thresh = min(tracker.args.track_high_thresh, threshold)
                tracker.args.new_track_thresh = min(tracker.args.new_track_thresh, threshold)
                tracker.args.track_low_thresh = min(tracker.args.track_low_thresh, threshold)
        self.reset()

    def track(self, frame, selected_class=None):
        batches = []
        if selected_class in (None, "person", "cup"):
            names = ("person", "cup") if selected_class is None else (selected_class,)
            classes = [self.base_classes[name] for name in names]
            result = self.base.track(frame, persist=True, tracker="botsort.yaml", classes=classes,
                                     device=self.device, conf=min(self.thresholds[name] for name in names),
                                     imgsz=640, verbose=False)[0]
            batches.append(("base", result))
        if selected_class in (None, "tree", "cup"):
            names = ("tree", "cup") if selected_class is None else (selected_class,)
            category_ids = {"tree": self.tree_classes, "cup": self.cup_classes}
            classes = [index for name in names for index in category_ids[name]]
            result = self.trees.track(frame, persist=True, tracker="botsort.yaml", classes=classes,
                                      device=self.device, conf=min(self.thresholds[name] for name in names),
                                      imgsz=640, verbose=False)[0]
            batches.append(("scene", result))
        filtered = []
        for source, result in batches:
            keep = [i for i, box in enumerate(result.boxes)
                    if float(box.conf) >= self.thresholds[target_class(result.names[int(box.cls)])]]
            filtered.append((source, result[keep]))
        return filtered

    def reset(self):
        for model in (self.base, self.trees):
            for tracker in getattr(model.predictor, "trackers", ()):
                tracker.reset()
