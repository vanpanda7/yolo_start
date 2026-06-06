from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    dataset_dirs,
    ensure_dataset_dirs,
    get_class_colors,
    get_classes,
    iter_images,
    label_path_for,
    load_config,
    read_yolo_label,
    write_yolo_label,
)


class ImageLabeler:
    def __init__(self, config: dict[str, Any], prelabel: bool = False) -> None:
        self.config = config
        self.dirs = ensure_dataset_dirs(config)
        self.classes = get_classes(config)
        self.colors = get_class_colors(config) or {cid: [0, 180, 255] for cid in self.classes}
        self.images = iter_images(self.dirs["images"], config)
        self.index = 0
        self.current_class = min(self.classes) if self.classes else 0
        self.boxes: list[dict[str, float | int]] = []
        self.selected = -1
        self.frame = None
        self.drawing = False
        self.start_px = (0, 0)
        self.preview_px = (0, 0)
        self.window = "YOLO Labeler"
        self.prelabel = prelabel
        self.model = None

    def _init_model(self) -> None:
        if self.model is not None:
            return
        from ultralytics import YOLO
        model_name = self.config.get("prelabel", {}).get("model", "yolov8n.pt")
        self.model = YOLO(str(model_name))

    def _prelabel(self) -> None:
        if not self.prelabel or self.frame is None:
            return
        self._init_model()
        cfg = self.config.get("prelabel", {})
        results = self.model(
            self.frame,
            conf=float(cfg.get("conf_threshold", 0.25)),
            classes=cfg.get("coco_classes", [0]),
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return
        h, w = self.frame.shape[:2]
        boxes: list[dict[str, float | int]] = []
        for raw in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [float(v) for v in raw[:4]]
            boxes.append({
                "class_id": self.current_class,
                "xc": ((x1 + x2) / 2) / w,
                "yc": ((y1 + y2) / 2) / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
            })
        if boxes:
            self.boxes = boxes
            self.selected = 0

    def load_current(self) -> bool:
        if not self.images:
            print(f"No images found in {self.dirs['images']}")
            return False
        self.index = max(0, min(self.index, len(self.images) - 1))
        image = self.images[self.index]
        self.frame = cv2.imread(str(image))
        if self.frame is None:
            print(f"Cannot read image: {image}")
            return False
        self.boxes = read_yolo_label(label_path_for(image, self.dirs["labels"]))
        self.selected = 0 if self.boxes else -1
        if not self.boxes:
            self._prelabel()
        return True

    def save_current(self) -> None:
        image = self.images[self.index]
        saved = write_yolo_label(label_path_for(image, self.dirs["labels"]), self.boxes)
        print(f"Saved {saved} boxes: {image.name}")

    def next_image(self, step: int = 1) -> None:
        if not self.images:
            return
        self.index = (self.index + step) % len(self.images)
        self.load_current()

    def delete_selected(self) -> None:
        if 0 <= self.selected < len(self.boxes):
            self.boxes.pop(self.selected)
            self.selected = min(self.selected, len(self.boxes) - 1)

    def set_selected_class(self, class_id: int) -> None:
        if class_id not in self.classes:
            return
        self.current_class = class_id
        if 0 <= self.selected < len(self.boxes):
            self.boxes[self.selected]["class_id"] = class_id

    def _box_at(self, x: int, y: int) -> int:
        h, w = self.frame.shape[:2]
        for idx in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[idx]
            x1 = int((float(box["xc"]) - float(box["w"]) / 2) * w)
            y1 = int((float(box["yc"]) - float(box["h"]) / 2) * h)
            x2 = int((float(box["xc"]) + float(box["w"]) / 2) * w)
            y2 = int((float(box["yc"]) + float(box["h"]) / 2) * h)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return -1

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if self.frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._box_at(x, y)
            if hit >= 0:
                self.selected = hit
                return
            self.drawing = True
            self.start_px = (x, y)
            self.preview_px = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.preview_px = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start_px
            x2, y2 = x, y
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return
            h, w = self.frame.shape[:2]
            self.boxes.append({
                "class_id": self.current_class,
                "xc": ((x1 + x2) / 2) / w,
                "yc": ((y1 + y2) / 2) / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
            })
            self.selected = len(self.boxes) - 1

    def draw(self):
        display = self.frame.copy()
        h, w = display.shape[:2]
        image = self.images[self.index]
        for idx, box in enumerate(self.boxes):
            class_id = int(box["class_id"])
            color = self.colors.get(class_id, [0, 180, 255])
            x1 = int((float(box["xc"]) - float(box["w"]) / 2) * w)
            y1 = int((float(box["yc"]) - float(box["h"]) / 2) * h)
            x2 = int((float(box["xc"]) + float(box["w"]) / 2) * w)
            y2 = int((float(box["yc"]) + float(box["h"]) / 2) * h)
            thickness = 3 if idx == self.selected else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            label = self.classes.get(class_id, str(class_id))
            cv2.putText(display, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.drawing:
            cv2.rectangle(display, self.start_px, self.preview_px, (0, 255, 0), 1)

        help_text = (
            f"{self.index + 1}/{len(self.images)} {image.name} | "
            "drag=box s=save n/p=next/prev del=delete q=quit"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(display, help_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        return display

    def run(self) -> None:
        if not self.load_current():
            return
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        print("Controls: drag box, 1-9 set class, s save, n next, p previous, del delete, q quit")
        while True:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save_current()
            elif key == ord("n"):
                self.next_image(1)
            elif key == ord("p"):
                self.next_image(-1)
            elif key in (8, 127):
                self.delete_selected()
            elif ord("1") <= key <= ord("9"):
                self.set_selected_class(key - ord("1"))
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple offline image labeler for YOLO datasets.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to ./config.yaml")
    parser.add_argument("--prelabel", action="store_true", help="Run COCO person prelabeling when an image has no labels")
    args = parser.parse_args()

    config = load_config(args.config)
    ImageLabeler(config, prelabel=args.prelabel).run()


if __name__ == "__main__":
    main()
