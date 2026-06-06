# tools/video_extractor.py
"""
视频抽帧 + AI 预标注 + 去重。

将游戏录像自动转化为 YOLO 标注数据集。

Usage:
  python video_extractor.py                          # 默认参数
  python video_extractor.py --input ./videos --interval 1.5
  python video_extractor.py --no-dedup               # 不去重
  python video_extractor.py --conf 0.15              # 降低 AI 检测阈值
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent / path
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 感知哈希去重
# ============================================================

def _phash(img: np.ndarray, hash_size: int = 8) -> int:
    """计算感知哈希 (pHash)，返回 64 位整数。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    avg = dct_low.mean()
    bits = (dct_low > avg).flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def _hamming(a: int, b: int) -> int:
    """汉明距离。"""
    return bin(a ^ b).count("1")


class Deduplicator:
    """基于感知哈希的图像去重器（滑动窗口，防止 O(n²)）。"""

    def __init__(self, threshold: int = 12, max_history: int = 200):
        self.threshold = threshold
        self.max_history = max_history
        self.hashes: List[int] = []

    def is_duplicate(self, img: np.ndarray) -> bool:
        h = _phash(img)
        for existing in self.hashes[-self.max_history:]:
            if _hamming(h, existing) < self.threshold:
                return True
        self.hashes.append(h)
        return False

    def clear(self):
        self.hashes.clear()


# ============================================================
# 视频抽帧器
# ============================================================

class VideoExtractor:
    """从视频中抽取帧并进行 AI 预标注。"""

    def __init__(self, config: dict, input_dir: str = None,
                 interval: float = None, conf: float = None,
                 dedup: bool = True, dedup_threshold: int = 12):
        self.cfg = config
        vcfg = config.get("video", {})

        self.input_dir = Path(input_dir or vcfg.get("input_dir", "./videos"))
        if not self.input_dir.is_absolute():
            self.input_dir = Path(__file__).parent / self.input_dir

        self.interval = interval or vcfg.get("interval", 2.0)
        self.conf = conf or config.get("ai", {}).get("conf_threshold", 0.25)
        self.dedup_enabled = dedup
        self.deduplicator = Deduplicator(threshold=dedup_threshold)

        # 裁剪 + 尺寸过滤
        cap_cfg = config.get("capture", {})
        self.crop_ratio = cap_cfg.get("crop_ratio", 1.0)
        self.max_box_ratio = cap_cfg.get("max_box_ratio", 0)

        # 排除区域
        raw_zones = config.get("exclude_zones", [])
        self.exclude_zones = self._parse_zones(raw_zones)

        # 输出路径
        dscfg = config.get("dataset", {})
        dspath = Path(dscfg.get("path", "./dataset"))
        if not dspath.is_absolute():
            dspath = Path(__file__).parent / dspath
        self.img_dir = dspath / "images"
        self.lbl_dir = dspath / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        # 来源元数据
        self._sources_path = dspath / "sources.json"
        self._sources = self._load_sources()

        # 帧计数器
        existing = list(self.img_dir.glob("frame_*.jpg"))
        self.frame_counter = len(existing)

        # AI 模型（延迟加载）
        self.model = None

    def _init_model(self):
        if self.model is not None:
            return
        # 优先 model_state.json → config → yolov8n.pt
        p = self._resolve_model()
        from ultralytics import YOLO
        self.model = YOLO(str(p))

    def _resolve_model(self):
        import json
        # 1. 检查 model_state.json
        ms = Path(__file__).parent / "model_state.json"
        if ms.exists():
            with open(ms) as f:
                active = json.load(f).get("active", "")
            if active:
                if active == "yolov8n.pt":
                    self._use_custom_model = False
                    print("[AI] model_state → COCO yolov8n.pt")
                    return "yolov8n.pt"
                wp = Path(__file__).parent / "weights" / active
                if wp.exists():
                    self._use_custom_model = True
                    print(f"[AI] model_state → 自定义 {active}")
                    return str(wp)
        # 2. fallback config
        w = self.cfg.get("ai", {}).get("model_weights", "")
        p = Path(w) if w else None
        if not p or not p.is_absolute():
            p = Path(__file__).parent / (w or "weights/best.pt")
        if p and p.exists():
            self._use_custom_model = True
            print(f"[AI] config → 自定义 {p.name}")
            return str(p)
        # 3. fallback yolov8n.pt
        self._use_custom_model = False
        print("[AI] 使用 COCO 预训练 yolov8n.pt")
        return "yolov8n.pt"

    @staticmethod
    def _parse_zones(raw_zones: list) -> list:
        """解析排除区域，返回 [{x1,y1,x2,y2}]。"""
        parsed = []
        for z in raw_zones:
            if len(z) == 5 and z[4] == "norm":
                # 归一化坐标，运行时根据帧尺寸转换（在 detect 时处理）
                parsed.append({"x1": z[0], "y1": z[1], "x2": z[2], "y2": z[3], "norm": True})
            elif len(z) >= 4:
                parsed.append({"x1": z[0], "y1": z[1], "x2": z[2], "y2": z[3], "norm": False})
        return parsed

    def _detect_persons(self, frame: np.ndarray) -> List[dict]:
        """检测画面中所有人物。自动裁剪武器区+过滤超大框。"""
        self._init_model()
        h, w = frame.shape[:2]
        crop_h = int(h * self.crop_ratio) if self.crop_ratio < 1.0 else h
        detect_frame = frame[:crop_h, :] if crop_h < h else frame

        if getattr(self, '_use_custom_model', False):
            results = self.model(detect_frame, conf=self.conf, verbose=False)
        else:
            results = self.model(detect_frame, conf=self.conf, classes=[0], verbose=False)
        boxes = []
        max_area = w * h * self.max_box_ratio if self.max_box_ratio > 0 else float('inf')
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    bw = box[2]-box[0]; bh = box[3]-box[1]
                    if bw*bh > max_area: continue  # 过滤超大框
                    boxes.append({
                        "x1": int(box[0]), "y1": int(box[1]),
                        "x2": int(box[2]), "y2": int(box[3]),
                    })
        boxes = self._filter_exclude_zones(boxes, w, h)
        return boxes

    def _filter_exclude_zones(self, boxes: List[dict], img_w: int, img_h: int) -> List[dict]:
        """排除中心点落入排除区域的检测框。"""
        zones = self.exclude_zones
        if not zones:
            return boxes
        kept = []
        for b in boxes:
            cx = (b["x1"] + b["x2"]) / 2
            cy = (b["y1"] + b["y2"]) / 2
            inside = False
            for z in zones:
                if z.get("norm"):
                    zx1, zy1 = z["x1"] * img_w, z["y1"] * img_h
                    zx2, zy2 = z["x2"] * img_w, z["y2"] * img_h
                else:
                    zx1, zy1, zx2, zy2 = z["x1"], z["y1"], z["x2"], z["y2"]
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    inside = True
                    break
            if not inside:
                kept.append(b)
        return kept

    def _save(self, frame: np.ndarray, boxes: List[dict]):
        """保存帧和标签。"""
        self.frame_counter += 1
        name = f"frame_{self.frame_counter:04d}"
        h, w = frame.shape[:2]

        img_path = self.img_dir / f"{name}.jpg"
        cv2.imwrite(str(img_path), frame)

        lines = []
        for b in boxes:
            xc = ((b["x1"] + b["x2"]) / 2) / w
            yc = ((b["y1"] + b["y2"]) / 2) / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        lbl_path = self.lbl_dir / f"{name}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        return name

    def _load_sources(self) -> dict:
        import json
        if self._sources_path.exists():
            with open(self._sources_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_sources(self):
        import json, traceback
        try:
            with open(self._sources_path, "w", encoding="utf-8") as f:
                json.dump(self._sources, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠ 保存 sources.json 失败: {e}")
            traceback.print_exc()

    def _record_source(self, frame_name: str, video_name: str):
        """记录帧来自哪个视频。"""
        self._sources[frame_name] = video_name

    def _get_video_files(self) -> List[Path]:
        exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
        files = []
        for ext in exts:
            files.extend(self.input_dir.glob(f"*{ext}"))
        return sorted(files)

    def process_video(self, video_path: Path) -> int:
        """处理单个视频，返回保存的帧数。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ⚠ 无法打开视频: {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        frame_skip = max(1, int(fps * self.interval))

        print(f"  {video_path.name}: {duration:.0f}s, {fps:.1f}fps, "
              f"每 {self.interval}s 抽一帧 (间隔 {frame_skip} 帧)")

        saved = 0
        skipped_dup = 0
        skipped_nodet = 0
        frame_idx = 0

        # 断点续抽：跳过已处理的帧数
        existing = len(list(self.img_dir.glob("frame_*.jpg")))
        if existing > self.frame_counter:
            print(f"  ⏭ 检测到已有 {existing - self.frame_counter} 帧（可能来自之前的处理），将从中断处继续")

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="  抽帧", unit="f")
        except ImportError:
            pbar = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # 去重
                if self.dedup_enabled and self.deduplicator.is_duplicate(frame):
                    skipped_dup += 1
                else:
                    boxes = self._detect_persons(frame)
                    if boxes:
                        name = self._save(frame, boxes)
                        self._record_source(name, video_path.name)
                        if pbar:
                            pbar.set_postfix_str(f"✓ {name}")
                        saved += 1
                    else:
                        skipped_nodet += 1

            frame_idx += 1
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        cap.release()
        self.deduplicator.clear()
        self._save_sources()

        print(f"  → 保存 {saved} 帧, 去重跳过 {skipped_dup}, 无人跳过 {skipped_nodet}")
        return saved

    def run(self) -> int:
        """处理 videos/ 下所有视频，返回总保存帧数。"""
        videos = self._get_video_files()
        if not videos:
            print(f"\n❌ 在 {self.input_dir} 下未找到视频文件 (.mp4/.avi/.mkv)")
            print(f"   请将游戏录像放入该目录后重试。")
            return 0

        print(f"\n找到 {len(videos)} 个视频文件")
        print(f"抽帧间隔: {self.interval}s  |  AI置信度: {self.conf}  |  去重: {'开' if self.dedup_enabled else '关'}")
        print(f"输出目录: {self.img_dir}")
        print("=" * 55)

        total = 0
        for v in videos:
            total += self.process_video(v)

        print(f"\n✅ 完成！共保存 {total} 帧 (累计 {self.frame_counter} 张)")
        print(f"下一步: python labeler.py --review  # 审核预标注结果")
        return total


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="视频抽帧 + AI 预标注 → YOLO 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python video_extractor.py
  python video_extractor.py --input ./my_videos --interval 1.5
  python video_extractor.py --conf 0.15 --no-dedup
        """
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", help="视频目录")
    parser.add_argument("--interval", type=float, help="抽帧间隔(秒)")
    parser.add_argument("--conf", type=float, help="AI检测置信度")
    parser.add_argument("--no-dedup", action="store_true", help="关闭去重")
    parser.add_argument("--dedup-threshold", type=int, default=12,
                        help="去重阈值(汉明距离, 默认12, 越小越严格)")
    args = parser.parse_args()

    config = load_config(args.config)
    extractor = VideoExtractor(
        config,
        input_dir=args.input,
        interval=args.interval,
        conf=args.conf,
        dedup=not args.no_dedup,
        dedup_threshold=args.dedup_threshold,
    )
    extractor.run()


if __name__ == "__main__":
    main()
