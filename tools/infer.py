# tools/infer.py
"""
实时推理脚本 —— 屏幕捕获 + YOLOv8 推理 + 实时叠加显示。

Usage:
    python infer.py                    # 使用默认配置
    python infer.py --weights ./weights/best.pt --conf 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from screen_capture import ScreenCapture, list_monitors


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 目标跟踪（匈牙利算法 + Kalman 平滑 + 时序持久化）
# ============================================================

class KalmanTracker:
    """匈牙利算法最优匹配 + 指数平滑 + 丢失目标持久化。"""

    def __init__(self, smooth_factor: float = 0.3, persist_frames: int = 5):
        self.alpha = smooth_factor
        self.persist_frames = persist_frames
        self.tracks: dict = {}  # {id: {bbox, conf, cls, name, lost}}

    def update(self, detections: list, iou_thres: float = 0.3) -> list:
        if not detections:
            # 所有 track 丢失计数+1
            for t in self.tracks.values():
                t["lost"] = t.get("lost", 0) + 1
            # 保留未过期的 track
            return [t for t in self.tracks.values() if t.get("lost", 0) < self.persist_frames]

        # 匈牙利算法：构建 cost matrix (1 - IoU)
        n_det = len(detections)
        n_trk = len(self.tracks)
        trk_ids = list(self.tracks.keys())

        if n_trk == 0:
            # 没有历史 track，全部新建
            for i, det in enumerate(detections):
                tid = f"t{i}"
                self.tracks[tid] = {**det, "lost": 0}
            return detections

        cost = np.ones((n_det, n_trk))
        for i, det in enumerate(detections):
            for j, tid in enumerate(trk_ids):
                cost[i, j] = 1.0 - self._iou(
                    det["bbox"], self.tracks[tid]["bbox"])

        # 贪心匹配（替代 scipy.optimize.linear_sum_assignment）
        matched_det = set()
        matched_trk = set()
        pairs = []

        # 按 cost 排序，贪心配对
        candidates = []
        for i in range(n_det):
            for j in range(n_trk):
                if cost[i, j] < (1.0 - iou_thres):  # IoU > threshold
                    candidates.append((cost[i, j], i, j))
        candidates.sort()

        for c, i, j in candidates:
            if i not in matched_det and j not in matched_trk:
                pairs.append((i, trk_ids[j]))
                matched_det.add(i)
                matched_trk.add(j)

        # 更新匹配的 track
        new_tracks = {}
        for i, tid in pairs:
            det = detections[i]
            old = self.tracks[tid]
            a = self.alpha
            ox = old["bbox"]
            sx = [int(a*ox[k] + (1-a)*det["bbox"][k]) for k in range(4)]
            new_tracks[tid] = {
                "bbox": sx, "confidence": det["confidence"],
                "class_id": det["class_id"], "name": det["name"], "lost": 0}

        # 未匹配的检测 → 新建 track
        for i in range(n_det):
            if i not in matched_det:
                tid = f"t{len(new_tracks)}_{i}"
                new_tracks[tid] = {**detections[i], "lost": 0}

        # 未匹配的历史 track → 丢失计数
        for j, tid in enumerate(trk_ids):
            if j not in matched_trk:
                t = self.tracks[tid]
                t["lost"] = t.get("lost", 0) + 1
                if t["lost"] < self.persist_frames:
                    new_tracks[tid] = t

        self.tracks = new_tracks
        return list(self.tracks.values())

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
        return inter / (area_a + area_b - inter + 1e-6)


# ============================================================
# 推理器
# ============================================================

class Inferencer:
    """实时目标检测推理器。"""

    def __init__(self, config: dict, weights_path: str = None, collect: bool = False, model_idx: int = 0):
        self.cfg = config
        self.collect = collect

        # 多模型支持
        inf_cfg = config.get("inference", {})
        wlist = inf_cfg.get("weights_list", [])
        if wlist and not weights_path:
            if model_idx < len(wlist):
                weights_path = wlist[model_idx]
            print(f"[模型] 使用第 {model_idx+1}/{len(wlist)} 个模型: {Path(weights_path).name}")

        # 类别信息
        classes = config.get("classes", {0: "enemy_body", 1: "enemy_head"})
        self.class_names = [classes[k] for k in sorted(classes.keys())]
        default_colors = {0: [255, 0, 0], 1: [0, 0, 255]}
        self.colors = config.get("class_colors", default_colors)

        # 推理参数
        self.conf_thres = inf_cfg.get("conf_threshold", 0.5)
        self.iou_thres = inf_cfg.get("iou_threshold", 0.45)
        self.show_fps = inf_cfg.get("show_fps", True)
        self.show_labels = inf_cfg.get("show_labels", True)
        self.smooth = inf_cfg.get("smooth", True)

        # 屏幕捕获
        cap_cfg = config.get("capture", {})
        self.center_size = cap_cfg.get("center_size", 0)     # 中心区域边长，0=全屏
        self.max_box_ratio = cap_cfg.get("max_box_ratio", 0)
        self.capture = ScreenCapture(
            monitor=cap_cfg.get("monitor", 1),
            fps=cap_cfg.get("fps", 30),
            region=cap_cfg.get("region"),
            downsample=cap_cfg.get("downsample", 1.0),
        )

        # 加载模型
        if weights_path is None:
            weights_path = inf_cfg.get("weights", "./weights/best.pt")

        weights_path = Path(weights_path)
        if not weights_path.exists():
            # 尝试相对路径
            weights_path = Path(__file__).parent / weights_path

        if not weights_path.exists():
            raise FileNotFoundError(
                f"模型权重文件未找到: {weights_path}\n"
                f"请先运行 train.py 训练模型，或指定正确的权重路径:\n"
                f"  python infer.py --weights <path/to/weights.pt>\n\n"
                f"你也可以直接下载 YOLOv8 预训练权重做测试:\n"
                f"  python infer.py --weights yolov8n.pt"
            )

        print(f"正在加载模型: {weights_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(weights_path))
        except ImportError:
            raise ImportError("ultralytics 未安装。请运行: pip install ultralytics")

        # 性能统计
        self._fps_history = []
        self._start_time = time.time()
        self._frame_count = 0

        # Kalman 平滑跟踪
        self._kalman = KalmanTracker(smooth_factor=0.35) if self.smooth else None

        # 自动收集
        if self.collect:
            ds_cfg = config.get("dataset", {})
            ds_path = Path(ds_cfg.get("path", "./dataset"))
            if not ds_path.is_absolute():
                ds_path = Path(__file__).parent / ds_path
            self._coll_img = ds_path / "images"
            self._coll_lbl = ds_path / "labels"
            self._coll_img.mkdir(parents=True, exist_ok=True)
            self._coll_lbl.mkdir(parents=True, exist_ok=True)
            existing = list(self._coll_img.glob("frame_*.jpg"))
            self._coll_count = len(existing)
            self._coll_interval = 10  # 每N帧保存一次
            print(f"[收集模式] 检测帧将自动保存到 {self._coll_img}")

    @property
    def avg_fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    def predict(self, frame: np.ndarray) -> list:
        """对单帧推理。中心区域捕获+过滤超大框+匈牙利跟踪。"""
        h, w = frame.shape[:2]
        if self.center_size > 0:
            half = self.center_size // 2
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half - 40)  # 上方偏移（FPS视角）
            x2 = min(w, cx + half)
            y2 = min(h, cy + half - 40)
            detect_frame = frame[y1:y2, x1:x2]
        else:
            detect_frame = frame
            x1, y1 = 0, 0

        t0 = time.time()
        results = self.model(detect_frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        t1 = time.time()

        fps = 1.0 / max(t1 - t0, 0.001)
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)

        detections = []
        max_area = w * h * self.max_box_ratio if self.max_box_ratio > 0 else float('inf')
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else []
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else []

                for box, conf, cls_id in zip(boxes, confs, clss):
                    bx = box.astype(int).tolist()
                    # 中心捕获偏移回全屏坐标
                    if self.center_size > 0:
                        bx[0] += x1; bx[2] += x1
                        bx[1] += y1; bx[3] += y1
                    area = (bx[2]-bx[0]) * (bx[3]-bx[1])
                    if area > max_area: continue
                    if int(cls_id) == 2: continue  # 跳过 menu_player
                    detections.append({
                        "bbox": bx,
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "name": self.class_names[cls_id] if cls_id < len(self.class_names) else "?",
                    })

        self._frame_count += 1
        if self._kalman:
            detections = self._kalman.update(detections)
        return detections

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """ESP 风格绘制。"""
        h, w = frame.shape[:2]
        fov_r = min(w, h) // 4  # FOV 圈半径

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            conf = det["confidence"]
            name = det["name"]
            color = self.colors.get(cls_id, [0, 255, 0])

            # 框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 角标（四条短线）
            cx, cy = (x1+x2)//2, (y1+y2)//2
            for (sx, sy) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                dx = 15 if sx == x1 else -15
                dy = 15 if sy == y1 else -15
                cv2.line(frame, (sx, sy), (sx+dx, sy), color, 2)
                cv2.line(frame, (sx, sy), (sx, sy+dy), color, 2)

            # 标签
            if self.show_labels:
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
                cv2.putText(frame, label, (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # FPS
        if self.show_fps:
            fps_t = f"FPS:{self.avg_fps:.0f} | det:{len(detections)}"
            cv2.putText(frame, fps_t, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 准星
        cv2.drawMarker(frame, (w//2, h//2), (0,255,0), cv2.MARKER_CROSS, 12, 1)
        cv2.circle(frame, (w//2, h//2), fov_r, (0,255,0), 1)

        return frame

    def run(self, show_window: bool = True):
        window_name = "Inference - ESP"
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("=" * 60)
        print("  实时推理已启动")
        print(f"  类别: {self.class_names}  |  conf: {self.conf_thres}")
        if self.collect:
            print(f"  📸 自动收集模式: 每 {self._coll_interval} 帧保存检测结果")
        print(f"  按 'q' 退出")
        print("=" * 60)

        try:
            for frame in self.capture:
                detections = self.predict(frame)
                display = self.draw(frame, detections)

                # 自动收集
                if self.collect and detections and self._frame_count % self._coll_interval == 0:
                    self._coll_count += 1
                    name = f"frame_{self._coll_count:04d}"
                    cv2.imwrite(str(self._coll_img / f"{name}.jpg"), frame)
                    h, w = frame.shape[:2]
                    lines = []
                    for d in detections:
                        x1, y1, x2, y2 = d["bbox"]
                        xc = ((x1+x2)/2)/w; yc = ((y1+y2)/2)/h
                        bw = (x2-x1)/w; bh = (y2-y1)/h
                        lines.append(f"{d['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    with open(self._coll_lbl / f"{name}.txt", "w") as f:
                        f.write("\n".join(lines))

                if show_window:
                    cv2.imshow(window_name, display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\n[推理] 用户中断")
        finally:
            self.capture.close()
            cv2.destroyAllWindows()
            elapsed = time.time() - self._start_time
            print(f"\n[推理] 已退出。运行 {elapsed:.1f}s，共处理 {self._frame_count} 帧，平均 FPS: {self.avg_fps:.1f}")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 实时目标检测推理")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--weights", type=str, help="模型权重路径")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--no-window", action="store_true", help="不显示窗口")
    parser.add_argument("--collect", action="store_true", help="自动收集模式：检测帧自动保存到数据集")
    parser.add_argument("--model", type=int, default=0, help="多模型索引（对应 config 中 weights_list）")
    args = parser.parse_args()

    config = load_config(args.config)

    # 命令行覆盖配置
    if args.conf:
        config.setdefault("inference", {})["conf_threshold"] = args.conf

    # 列出显示器
    print("可用显示器：")
    list_monitors()
    print()

    inferencer = Inferencer(config, weights_path=args.weights, collect=args.collect, model_idx=args.model)
    inferencer.run(show_window=not args.no_window)


if __name__ == "__main__":
    main()
