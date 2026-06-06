# tools/labeler.py
"""
交互式标注工具 v2 —— AI 预标注 + 批量审核模式。

运行方式：
  python labeler.py              # 实时屏幕捕获模式
  python labeler.py --review     # 批量审核模式（审核已有截图）

操作说明：
  SPACE     冻结/解冻画面（live 模式）
  A         切换 AI 预标注 开/关
  R         重新执行 AI 检测
  鼠标拖拽   手动补画框
  1 / 2     给当前框分配类别（1=enemy_body, 2=enemy_head）
  左/右箭头  切换选中框
  D         删除当前框
  C         清除所有未标注框
  S         保存当前帧 + 标签
  N         跳过当前帧（不保存）
  Q / ESC   退出
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from screen_capture import ScreenCapture, list_monitors


# ============================================================
# 配置
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 标注器
# ============================================================

class Labeler:
    """AI 辅助标注工具。"""

    # 框来源常量
    SRC_AI = -2      # AI 检测的框（青色）
    SRC_MANUAL = -1  # 手动画的框（黄色）

    def __init__(self, config: dict, mode: str = "live", import_dir: str = None):
        self.cfg = config
        self.mode = mode  # "live" | "review"

        # ---- 类别 ----
        self.classes: dict = config.get("classes", {0: "enemy_body", 1: "enemy_head"})
        self.class_names = [self.classes.get(i, f"class_{i}") for i in sorted(self.classes.keys())]

        # 颜色：{class_id: [B,G,R]}
        self.colors = {
            -2: [255, 255, 0],   # AI 检测 = 青色
            -1: [0, 255, 255],   # 手动未分配 = 黄色
            0:  [255, 0, 0],     # enemy_body = 蓝色
            1:  [0, 0, 255],     # enemy_head = 红色
        }

        # ---- AI 模型 ----
        self.ai_enabled = True
        self.ai_model = None
        ai_cfg = config.get("ai", {})
        self.ai_conf = ai_cfg.get("conf_threshold", 0.25)

        # 排除区域
        self.exclude_zones = self._parse_zones(config.get("exclude_zones", []))

        # 裁剪 + 尺寸过滤
        cap_cfg = config.get("capture", {})
        self.crop_ratio = cap_cfg.get("crop_ratio", 1.0)
        self.max_box_ratio = cap_cfg.get("max_box_ratio", 0)

        # ---- 数据集路径 ----
        ds_cfg = config.get("dataset", {})
        ds_path = Path(ds_cfg.get("path", "./dataset"))
        if not ds_path.is_absolute():
            ds_path = Path(__file__).parent / ds_path
        self.img_dir = ds_path / "images"
        self.lbl_dir = ds_path / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        # 帧计数器
        existing = list(self.img_dir.glob("frame_*.jpg"))
        self.frame_counter = len(existing)

        # ---- 屏幕捕获 (live 模式) ----
        self.capture = None
        if mode == "live":
            cap_cfg = config.get("capture", {})
            self.capture = ScreenCapture(
                monitor=cap_cfg.get("monitor", 1),
                fps=cap_cfg.get("fps", 30),
                region=cap_cfg.get("region"),
                downsample=cap_cfg.get("downsample", 1.0),
            )

        # ---- 审核模式 ----
        self.review_images: List[Path] = []
        self.review_idx = 0
        if mode == "review":
            scan_dir = Path(import_dir) if import_dir else self.img_dir
            # 找所有图片（排除已有标签的，如果用户想重新标可以删掉旧标签）
            all_imgs = sorted(list(scan_dir.glob("*.jpg")) + list(scan_dir.glob("*.png")))
            # 只取还没有对应标签的
            self.review_images = [
                p for p in all_imgs
                if not (self.lbl_dir / f"{p.stem}.txt").exists()
            ]
            if not self.review_images:
                # 如果都标过了，就全部加载，用户可能想修正
                self.review_images = all_imgs
            print(f"[审核模式] 找到 {len(self.review_images)} 张待审核图片")

        # ---- 标注状态 ----
        self.frame = None           # 当前帧 (BGR numpy)
        self.frozen = (mode == "review")  # 审核模式默认冻结
        self.boxes: List[dict] = []       # [{x1, y1, x2, y2, class_id, conf?}]
        self.active_idx = -1

        # ---- 鼠标 ----
        self.drawing = False
        self.start_point = (-1, -1)
        self.current_rect = (-1, -1, -1, -1)

        # ---- 窗口 ----
        self.window_name = "Labeler v2 - AI辅助标注"

    # ================================================================
    # AI 模型
    # ================================================================

    def _init_ai(self):
        if self.ai_model is not None:
            return
        p = self._resolve_model()
        from ultralytics import YOLO
        self.ai_model = YOLO(str(p))

    def _resolve_model(self):
        import json
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
        w = self.cfg.get("ai", {}).get("model_weights", "")
        p = Path(w) if w else None
        if not p or not p.is_absolute():
            p = Path(__file__).parent / (w or "weights/best.pt")
        if p and p.exists():
            self._use_custom_model = True
            print(f"[AI] config → 自定义 {p.name}")
            return str(p)
        self._use_custom_model = False
        print("[AI] 使用 COCO 预训练 yolov8n.pt")
        return "yolov8n.pt"
        print("[AI] 模型加载完成，COCO 80类预训练")

    @staticmethod
    def _parse_zones(raw_zones: list) -> list:
        parsed = []
        for z in raw_zones:
            if len(z) == 5 and z[4] == "norm":
                parsed.append({"x1": z[0], "y1": z[1], "x2": z[2], "y2": z[3], "norm": True})
            elif len(z) >= 4:
                parsed.append({"x1": z[0], "y1": z[1], "x2": z[2], "y2": z[3], "norm": False})
        return parsed

    def _filter_exclude_zones(self, boxes: List[dict], img_w: int, img_h: int) -> List[dict]:
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

    def _ai_detect(self, frame: np.ndarray) -> List[dict]:
        """AI 检测。自动裁剪武器区+过滤超大框。"""
        if not self.ai_enabled:
            return []
        self._init_ai()
        h, w = frame.shape[:2]
        crop_h = int(h * self.crop_ratio) if self.crop_ratio < 1.0 else h
        detect_frame = frame[:crop_h, :] if crop_h < h else frame

        try:
            if getattr(self, '_use_custom_model', False):
                results = self.ai_model(detect_frame, conf=self.ai_conf, verbose=False)
            else:
                results = self.ai_model(detect_frame, conf=self.ai_conf, classes=[0], verbose=False)
        except Exception as e:
            print(f"[AI] 检测失败: {e}")
            return []

        boxes = []
        max_area = w * h * self.max_box_ratio if self.max_box_ratio > 0 else float('inf')
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                for i, box in enumerate(xyxy):
                    bw = box[2]-box[0]; bh = box[3]-box[1]
                    if bw*bh > max_area: continue
                    boxes.append({
                        "x1": int(box[0]), "y1": int(box[1]),
                        "x2": int(box[2]), "y2": int(box[3]),
                        "class_id": self.SRC_AI,
                        "conf": float(confs[i]) if i < len(confs) else 0.0,
                    })
        return self._filter_exclude_zones(boxes, w, h)

    # ================================================================
    # 帧加载
    # ================================================================

    def _load_review_frame(self) -> bool:
        """加载下一张审核图片，返回 False 表示没有更多。"""
        if self.review_idx >= len(self.review_images):
            return False
        path = self.review_images[self.review_idx]
        self.frame = cv2.imread(str(path))
        if self.frame is None:
            print(f"  ⚠ 无法读取: {path}")
            self.review_idx += 1
            return self._load_review_frame()
        print(f"\n[审核 {self.review_idx + 1}/{len(self.review_images)}] {path.name}")
        return True

    def _freeze_and_detect(self):
        """冻结当前帧并执行 AI 检测。"""
        self.frozen = True
        self.boxes = self._ai_detect(self.frame)
        self.active_idx = -1
        ai_count = sum(1 for b in self.boxes if b["class_id"] == self.SRC_AI)
        if ai_count > 0:
            print(f"  🤖 AI 检测到 {ai_count} 个目标，请确认/删除/补画")
        else:
            print(f"  🤖 AI 未检测到目标，请手动拖拽画框")

    # ================================================================
    # 鼠标
    # ================================================================

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.frozen:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点中了已有框（选中它）
            hit = -1
            for i, b in reversed(list(enumerate(self.boxes))):
                if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                    hit = i
                    break
            if hit >= 0:
                self.active_idx = hit
                self.drawing = False
                return
            # 否则开始画新框
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x1, y1 = self.start_point
            self.current_rect = (x1, y1, x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = x, y
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                self.current_rect = (-1, -1, -1, -1)
                return
            box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": self.SRC_MANUAL}
            self.boxes.append(box)
            self.active_idx = len(self.boxes) - 1
            self.current_rect = (-1, -1, -1, -1)

    # ================================================================
    # 键盘
    # ================================================================

    def _handle_key(self, key: int) -> bool:
        """返回 False = 退出。"""
        # ---- 全局快捷键 ----
        if key in (ord("q"), 27):  # Q / ESC
            return False

        elif key == ord("a"):
            self.ai_enabled = not self.ai_enabled
            print(f"[AI] 预标注已{'开启' if self.ai_enabled else '关闭'}")

        elif key == ord("r"):
            if self.frozen and self.frame is not None:
                self.boxes = self._ai_detect(self.frame)
                self.active_idx = -1
                ai_count = sum(1 for b in self.boxes if b["class_id"] == self.SRC_AI)
                print(f"  🔄 AI 重新检测: {ai_count} 个目标")

        # ---- 冻结状态快捷键 ----
        elif self.frozen:

            if key == ord(" "):  # SPACE
                if self.mode == "live":
                    self.frozen = False
                    print("[标注] 画面已解冻")
                    self.boxes.clear()
                    self.active_idx = -1

            elif key in (ord("1"), ord("2")):
                cid = int(chr(key)) - 1
                if cid in self.classes:
                    self._assign_class(cid)

            elif key in (ord("d"), 8):  # D / Backspace
                self._delete_active()

            elif key == ord("c"):
                self._clear_unassigned()

            elif key == ord("s"):
                self._save_frame()
                if self.mode == "review":
                    self._advance_review()
                else:
                    self.frozen = False
                    self.boxes.clear()
                    self.active_idx = -1

            elif key == ord("n"):
                if self.mode == "review":
                    self._advance_review()
                else:
                    print("[标注] 跳过当前帧")
                    self.frozen = False
                    self.boxes.clear()
                    self.active_idx = -1

            # 方向键切换选中框
            elif key == 81:  # 左箭头 (OpenCV: 65361 → & 0xFF → 81)
                self._cycle_active(-1)
            elif key == 83:  # 右箭头
                self._cycle_active(1)

            elif key == 13:  # Enter
                self._save_frame()
                if self.mode == "review":
                    self._advance_review()

        # ---- 直播模式：非冻结时 SPACE 冻结 ----
        elif not self.frozen and key == ord(" ") and self.mode == "live":
            self._freeze_and_detect()

        return True

    def _assign_class(self, class_id: int):
        if self.active_idx < 0 or self.active_idx >= len(self.boxes):
            print("  ⚠ 没有选中的框（点击框选中，或画一个新框）")
            return
        self.boxes[self.active_idx]["class_id"] = class_id
        name = self.classes.get(class_id, f"class_{class_id}")
        src = "AI" if self.boxes[self.active_idx].get("conf") else "手动"
        print(f"  ✓ [{src}] → {name}")

    def _delete_active(self):
        if 0 <= self.active_idx < len(self.boxes):
            self.boxes.pop(self.active_idx)
            self.active_idx = max(0, len(self.boxes) - 1) if self.boxes else -1

    def _clear_unassigned(self):
        before = len(self.boxes)
        self.boxes = [b for b in self.boxes if b["class_id"] >= 0]
        print(f"  ✗ 清除 {before - len(self.boxes)} 个未标注框")
        self.active_idx = len(self.boxes) - 1 if self.boxes else -1

    def _cycle_active(self, direction: int):
        if not self.boxes:
            return
        self.active_idx = (self.active_idx + direction) % len(self.boxes)

    def _save_frame(self):
        assigned = [b for b in self.boxes if b["class_id"] >= 0]
        if not assigned:
            print("  ⚠ 没有已标注的框")
            return

        self.frame_counter += 1
        name = f"frame_{self.frame_counter:04d}"
        img_path = self.img_dir / f"{name}.jpg"
        cv2.imwrite(str(img_path), self.frame)
        h, w = self.frame.shape[:2]
        lines = []
        for b in assigned:
            xc = ((b["x1"] + b["x2"]) / 2) / w
            yc = ((b["y1"] + b["y2"]) / 2) / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            lines.append(f"{b['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        lbl_path = self.lbl_dir / f"{name}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  💾 {name}  |  {len(lines)} 个框  |  累计 {self.frame_counter} 张")
        # 记录来源
        self._record_label_source(name)

    def _record_label_source(self, frame_name: str):
        """在 sources.json 中记录此帧为手动标注。"""
        import json
        sp = self.lbl_dir.parent / "sources.json"
        sources = {}
        if sp.exists():
            with open(sp) as f:
                sources = json.load(f)
        sources[frame_name] = "manual_label"
        with open(sp, "w") as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)

    def _advance_review(self):
        """审核模式：进入下一张图。"""
        self.review_idx += 1
        if self._load_review_frame():
            self.boxes = self._ai_detect(self.frame)
            self.active_idx = -1
        else:
            print("\n✅ 审核完成！所有图片已处理。")
            self.frozen = False

    # ================================================================
    # 绘制
    # ================================================================

    def _draw_boxes(self, display: np.ndarray):
        for i, b in enumerate(self.boxes):
            cid = b["class_id"]
            color = self.colors.get(cid, [128, 128, 128])
            is_active = (i == self.active_idx)
            thickness = 3 if is_active else 2
            style = cv2.LINE_AA if is_active else cv2.LINE_4

            cv2.rectangle(display, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thickness, style)

            # 标签
            if cid >= 0:
                label = self.classes.get(cid, "?")
            elif cid == self.SRC_AI:
                conf = b.get("conf", 0)
                label = f"AI {conf:.0%}"
            else:
                label = "?"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (b["x1"], b["y1"] - th - 6), (b["x1"] + tw + 6, b["y1"]), color, -1)
            cv2.putText(display, label, (b["x1"] + 3, b["y1"] - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 拖拽中的临时框
        if self.drawing and self.current_rect[0] >= 0:
            cv2.rectangle(display,
                         (self.current_rect[0], self.current_rect[1]),
                         (self.current_rect[2], self.current_rect[3]),
                         (0, 255, 0), 1)

    def _draw_hud(self, display: np.ndarray):
        assigned = sum(1 for b in self.boxes if b["class_id"] >= 0)
        ai_count = sum(1 for b in self.boxes if b["class_id"] == self.SRC_AI)

        # 状态行
        if self.mode == "review":
            status = f"📋 审核 {self.review_idx + 1}/{len(self.review_images)}"
        else:
            status = "■ FROZEN" if self.frozen else "□ LIVE (按SPACE冻结)"

        color = (0, 0, 255) if self.frozen else (0, 255, 0)
        ai_status = "AI:ON" if self.ai_enabled else "AI:OFF"

        lines = [
            f"{status}  |  {ai_status}  |  AI框:{ai_count}  已标注:{assigned}/{len(self.boxes)}  |  总计:{self.frame_counter}",
            "[SPACE]冻结 [A]切换AI [R]重检测 [←→]切框 [1/2]类别 [D]删 [C]清 [S]存 [N]跳 [Q]退",
            "类别: " + "  ".join(f"[{i}]={n}" for i, n in self.classes.items()),
        ]
        for i, line in enumerate(lines):
            y = 28 + i * 26
            cv2.putText(display, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ================================================================
    # 主循环
    # ================================================================

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        self._print_help()

        # 审核模式：加载第一张图
        if self.mode == "review":
            if not self._load_review_frame():
                print("没有可审核的图片。")
                return
            self.boxes = self._ai_detect(self.frame)

        first_frame = True
        try:
            while True:
                # ---- 获取帧 ----
                if self.mode == "live":
                    if self.frozen and self.frame is not None:
                        display = self.frame.copy()
                    else:
                        try:
                            self.frame = next(self.capture)
                        except StopIteration:
                            break
                        display = self.frame.copy()
                else:  # review
                    if self.frame is None:
                        break
                    display = self.frame.copy()

                # ---- 引导遮罩 ----
                if first_frame and self.mode == "live":
                    first_frame = self._draw_intro_overlay(display)
                else:
                    first_frame = False

                # ---- 绘制 ----
                self._draw_boxes(display)
                self._draw_hud(display)
                cv2.imshow(self.window_name, display)

                # ---- 键盘 ----
                key = cv2.waitKey(1) & 0xFF
                if key > 0 and first_frame:
                    first_frame = False
                if not self._handle_key(key):
                    break

                # 审核模式没有更多帧时退出
                if self.mode == "review" and self.frame is None:
                    break

        except KeyboardInterrupt:
            print("\n[标注] 用户中断")
        finally:
            if self.capture:
                self.capture.close()
            cv2.destroyAllWindows()
            print(f"[标注] 已退出。共标注 {self.frame_counter} 张图片。")

    def _draw_intro_overlay(self, display) -> bool:
        h, w = display.shape[:2]
        if hasattr(self, '_intro_shown'):
            return False
        self._intro_shown = True
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        display[:] = cv2.addWeighted(display, 0.4, overlay, 0.6, 0)
        lines = [
            ">>> 请点击此窗口激活 <<<",
            "SPACE = 冻结画面 (AI自动检测)",
            "1/2 = 确认类别  D = 删除  S = 保存",
        ]
        for i, line in enumerate(lines):
            ts = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            tx = (w - ts[0]) // 2
            ty = h // 2 - 30 + i * 35
            cv2.putText(display, line, (tx, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return True

    def _print_help(self):
        mode_label = "实时捕获" if self.mode == "live" else "批量审核"
        print("=" * 55)
        print(f"  标注工具 v2 — {mode_label}模式")
        print(f"  类别: {self.classes}")
        print(f"  AI预标注: {'开启' if self.ai_enabled else '关闭'} (置信度≥{self.ai_conf})")
        print(f"  保存到: {self.img_dir}")
        print("=" * 55)
        print("  快捷键:")
        print("    SPACE     冻结/解冻")
        print("    A         切换AI预标注 开/关")
        print("    R         重新AI检测")
        print("    鼠标拖拽    手动补画框")
        print("    点击框      选中框")
        print("    ← →       切换选中框")
        print("    1 / 2      分配类别")
        print("    D          删除当前框")
        print("    C          清除未标注框")
        print("    S / Enter  保存并下一帧")
        print("    N          跳过不保存")
        print("    Q / ESC    退出")
        print("=" * 55)


# ============================================================
# 入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI 辅助标注工具")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--review", action="store_true", help="批量审核模式")
    parser.add_argument("--import-dir", type=str, help="审核模式：导入图片的目录")
    parser.add_argument("--no-ai", action="store_true", help="关闭 AI 预标注")
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.review:
        print("可用显示器：")
        list_monitors()
        monitor = config.get("capture", {}).get("monitor", 1)
        print(f"当前选择: 显示器 {monitor}（修改 config.yaml → capture.monitor）\n")

    labeler = Labeler(
        config,
        mode="review" if args.review else "live",
        import_dir=args.import_dir,
    )
    if args.no_ai:
        labeler.ai_enabled = False

    labeler.run()


if __name__ == "__main__":
    main()
