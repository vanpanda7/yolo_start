from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    configured_path,
    ensure_dataset_dirs,
    load_config,
    load_sources,
    project_path,
    save_sources,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("._") or "video"


def phash(img: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size))
    dct = cv2.dct(np.float32(resized))
    avg = dct.mean()
    bits = (dct > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class Deduplicator:
    def __init__(self, threshold: int = 8, max_history: int = 200) -> None:
        self.threshold = threshold
        self.max_history = max_history
        self.hashes: list[int] = []

    def is_duplicate(self, frame: np.ndarray) -> bool:
        value = phash(frame)
        for existing in self.hashes[-self.max_history:]:
            if hamming(value, existing) < self.threshold:
                return True
        self.hashes.append(value)
        return False


class VideoExtractor:
    def __init__(
        self,
        config: dict[str, Any],
        input_dir: str | None = None,
        interval: float | None = None,
        conf: float | None = None,
        prelabel: bool | None = None,
        dedup: bool | None = None,
        dedup_threshold: int | None = None,
        save_empty_every: int | None = None,
    ) -> None:
        self.config = config
        video_cfg = config.get("video", {})
        prelabel_cfg = config.get("prelabel", {})
        raw_input = input_dir or video_cfg.get("input_dir")
        self.input_dir = project_path(raw_input) if raw_input else configured_path(config, "videos", "data/videos")
        self.interval = float(interval if interval is not None else video_cfg.get("interval", 2.0))
        self.conf = float(conf if conf is not None else prelabel_cfg.get("conf_threshold", 0.25))
        self.prelabel_enabled = bool(prelabel_cfg.get("enabled", True) if prelabel is None else prelabel)
        self.dedup_enabled = bool(video_cfg.get("dedup", True) if dedup is None else dedup)
        self.dedup = Deduplicator(int(dedup_threshold or video_cfg.get("dedup_threshold", 8)))
        self.save_empty_every = int(save_empty_every if save_empty_every is not None else video_cfg.get("save_empty_every", 8))

        self.dirs = ensure_dataset_dirs(config)
        self.sources = load_sources(config)
        self.model = None

    def _video_files(self) -> list[Path]:
        if self.input_dir.is_file() and self.input_dir.suffix.lower() in VIDEO_EXTENSIONS:
            return [self.input_dir]
        if not self.input_dir.exists():
            return []
        return sorted(p for p in self.input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)

    def _init_model(self) -> None:
        if self.model is not None or not self.prelabel_enabled:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for prelabeling. Use --no-prelabel to skip it.") from exc
        model_name = self.config.get("prelabel", {}).get("model", "yolov8n.pt")
        self.model = YOLO(str(model_name))

    def _detect(self, frame: np.ndarray) -> list[dict[str, int]]:
        if not self.prelabel_enabled:
            return []
        self._init_model()
        coco_classes = self.config.get("prelabel", {}).get("coco_classes", [0])
        results = self.model(frame, conf=self.conf, classes=coco_classes, verbose=False)
        boxes: list[dict[str, int]] = []
        if not results:
            return boxes
        result = results[0]
        if result.boxes is None:
            return boxes
        for raw in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in raw[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return boxes

    def _save_frame(self, frame: np.ndarray, boxes: list[dict[str, int]], video: Path, frame_idx: int) -> None:
        """Save one sampled frame plus its YOLO label file.

        For a beginner, this is the key handoff: raw video becomes a normal
        image in datasets/.../images and a same-stem txt file in labels.
        sources.json records the original video so train.py can split by video
        source instead of mixing near-identical frames across train and val.
        """
        base = safe_stem(video.name)
        name = f"{base}_f{frame_idx:08d}.jpg"
        image_path = self.dirs["images"] / name
        label_path = self.dirs["labels"] / f"{Path(name).stem}.txt"
        cv2.imwrite(str(image_path), frame)

        h, w = frame.shape[:2]
        lines = []
        for box in boxes:
            xc = ((box["x1"] + box["x2"]) / 2) / w
            yc = ((box["y1"] + box["y2"]) / 2) / h
            bw = (box["x2"] - box["x1"]) / w
            bh = (box["y2"] - box["y1"]) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        self.sources[name] = video.name

    def process_video(self, video: Path) -> dict[str, int | str]:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            return {"video": video.name, "saved": 0, "skipped_duplicate": 0, "skipped_empty": 0, "error": "cannot open"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip = max(1, int(fps * self.interval))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        saved = 0
        skipped_duplicate = 0
        skipped_empty = 0
        empty_seen = 0
        frame_idx = 0

        print(f"Processing {video.name}: {total_frames} frames, {fps:.1f} fps, every {frame_skip} frames")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            if self.dedup_enabled and self.dedup.is_duplicate(frame):
                skipped_duplicate += 1
                frame_idx += 1
                continue

            boxes = self._detect(frame)
            should_save_empty = self.save_empty_every > 0 and empty_seen % self.save_empty_every == 0
            if boxes or should_save_empty:
                self._save_frame(frame, boxes, video, frame_idx)
                saved += 1
            else:
                skipped_empty += 1
            if not boxes:
                empty_seen += 1
            frame_idx += 1

        cap.release()
        save_sources(self.config, self.sources)
        return {
            "video": video.name,
            "saved": saved,
            "skipped_duplicate": skipped_duplicate,
            "skipped_empty": skipped_empty,
            "error": "",
        }

    def run(self) -> int:
        videos = self._video_files()
        if not videos:
            print(f"No video files found in {self.input_dir}")
            return 0
        total = 0
        for video in videos:
            stats = self.process_video(video)
            total += int(stats["saved"])
            print(
                f"  saved={stats['saved']} duplicate={stats['skipped_duplicate']} "
                f"empty_skipped={stats['skipped_empty']}"
            )
        print(f"Done. Saved {total} frames to {self.dirs['images']}")
        return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from local videos into a YOLO dataset.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to ./config.yaml")
    parser.add_argument("--input", help="Video file or directory. Defaults to config video.input_dir")
    parser.add_argument("--interval", type=float, help="Seconds between sampled frames")
    parser.add_argument("--conf", type=float, help="Prelabel confidence threshold")
    parser.add_argument("--no-prelabel", action="store_true", help="Save sampled frames with empty labels only")
    parser.add_argument("--no-dedup", action="store_true", help="Disable perceptual-hash deduplication")
    parser.add_argument("--dedup-threshold", type=int, help="pHash hamming threshold")
    parser.add_argument("--save-empty-every", type=int, help="Save one empty frame every N empty samples; 0 skips all")
    args = parser.parse_args()

    config = load_config(args.config)
    extractor = VideoExtractor(
        config,
        input_dir=args.input,
        interval=args.interval,
        conf=args.conf,
        prelabel=not args.no_prelabel,
        dedup=not args.no_dedup,
        dedup_threshold=args.dedup_threshold,
        save_empty_every=args.save_empty_every,
    )
    extractor.run()


if __name__ == "__main__":
    main()
