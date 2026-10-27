"""Local person/cup detection plus a tree-capable YOLOE model and persistent tracking."""

from pathlib import Path

import numpy as np


TREE_LABELS = {
    "tree", "apple tree", "autumn tree", "banana tree", "banyan tree", "beech tree",
    "birch tree", "cherry tree", "coconut tree", "cypress tree", "eucalyptus tree",
    "fir tree", "fruit tree", "ginkgo tree", "oak tree", "olive tree", "orange tree",
    "palm tree", "pine tree", "maple tree", "willow tree",
}


class TargetDetector:
    def __init__(self, model_path, tree_model_path, device="cpu", confidence=0.35):
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
        self.confidence = confidence
        self.base_classes = {name: i for i, name in self.base.names.items() if name in ("person", "cup")}
        self.tree_classes = [i for i, name in self.trees.names.items() if name in TREE_LABELS]
        if set(self.base_classes) != {"person", "cup"} or not self.tree_classes:
            raise RuntimeError("模型类别不完整：必须支持人、水杯和真实树木。")
        # Initialize BOTH tracker instances before assigning any visible target IDs.
        # Their underlying ID counter is shared, so initializing the second one late
        # would reset the counter while the first model already had active tracks.
        blank = np.zeros((96, 96, 3), dtype=np.uint8)
        self.track(blank)
        self.reset()

    def track(self, frame, selected_class=None):
        batches = []
        if selected_class in (None, "person", "cup"):
            classes = list(self.base_classes.values()) if selected_class is None else [self.base_classes[selected_class]]
            result = self.base.track(frame, persist=True, tracker="botsort.yaml", classes=classes,
                                     device=self.device, conf=self.confidence, imgsz=640, verbose=False)[0]
            batches.append(("base", result))
        if selected_class in (None, "tree"):
            result = self.trees.track(frame, persist=True, tracker="botsort.yaml", classes=self.tree_classes,
                                      device=self.device, conf=self.confidence, imgsz=640, verbose=False)[0]
            batches.append(("tree", result))
        return batches

    def reset(self):
        for model in (self.base, self.trees):
            for tracker in getattr(model.predictor, "trackers", ()):
                tracker.reset()
