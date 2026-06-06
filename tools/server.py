# tools/server.py
"""
Flask 数据集浏览器 —— 查看标注数据集统计和图片。

Usage:
  python server.py                    # 默认 http://0.0.0.0:5000
  python server.py --port 8080
  python server.py --host 127.0.0.1   # 仅本机访问
"""

import argparse
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)

# ============================================================
# 配置
# ============================================================

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent / path
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
classes = config.get("classes", {0: "enemy_body", 1: "enemy_head"})
class_names = [classes[k] for k in sorted(classes.keys())]
class_colors = config.get("class_colors", {0: [255, 0, 0], 1: [0, 0, 255]})

dscfg = config.get("dataset", {})
dspath = Path(dscfg.get("path", "./dataset"))
if not dspath.is_absolute():
    dspath = Path(__file__).parent / dspath
IMG_DIR = dspath / "images"
LBL_DIR = dspath / "labels"

# 来源元数据
import json as _json
SOURCES_PATH = dspath / "sources.json"
MODEL_STATE_PATH = Path(__file__).parent / "model_state.json"
def _load_sources() -> dict:
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    return {}
def _save_sources(data: dict):
    with open(SOURCES_PATH, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2, ensure_ascii=False)

# 视频目录
VIDEO_DIR = Path(__file__).parent / "videos"
VIDEO_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "avi", "mkv", "mov", "webm", "flv"}

# 简单 TTL 缓存
_cache: dict = {}
def _cached(key: str, ttl: float = 2.0):
    """如果缓存未过期返回缓存值，否则返回 None。"""
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < ttl:
        return entry["val"]
    return None
def _cache_set(key: str, val):
    _cache[key] = {"val": val, "ts": time.time()}

# 后台抽帧任务状态
_jobs: dict = {}       # {video_name: {status, progress, total, saved, error}}
_jobs_lock = threading.Lock()


# ============================================================
# 数据读取
# ============================================================

def list_labeled_images():
    """返回所有有标签的图片列表，含来源信息（2秒缓存）。"""
    cached = _cached("labeled_images")
    if cached is not None:
        return cached
    imgs = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    sources = _load_sources()
    result = []
    for img in imgs:
        lbl = LBL_DIR / f"{img.stem}.txt"
        result.append({
            "name": img.name,
            "path": str(img),
            "labeled": lbl.exists(),
            "label_path": str(lbl) if lbl.exists() else None,
            "source": sources.get(img.name, ""),
        })
    _cache_set("labeled_images", result)
    return result


def read_labels(name: str) -> list:
    """读取 YOLO 标签，返回 [{class_id, xc, yc, w, h}]。"""
    lbl_path = LBL_DIR / f"{Path(name).stem}.txt"
    if not lbl_path.exists():
        return []
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append({
                    "class_id": int(parts[0]),
                    "xc": float(parts[1]),
                    "yc": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
    return boxes


def get_stats():
    """数据集统计。"""
    images = list_labeled_images()
    labeled = [i for i in images if i["labeled"]]
    total_boxes = 0
    class_counts = {k: 0 for k in classes.keys()}
    for img in labeled:
        boxes = read_labels(img["name"])
        total_boxes += len(boxes)
        for b in boxes:
            cid = b["class_id"]
            if cid in class_counts:
                class_counts[cid] += 1
    return {
        "total_images": len(images),
        "labeled_images": len(labeled),
        "unlabeled_images": len(images) - len(labeled),
        "total_boxes": total_boxes,
        "class_counts": class_counts,
    }


def draw_boxes_on_image(img_path: str, boxes: list, max_size: int = None) -> np.ndarray:
    """在图片上绘制边界框，返回 BGR numpy 数组。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    for b in boxes:
        cid = b["class_id"]
        color = class_colors.get(cid, [0, 255, 0])
        x1 = int((b["xc"] - b["w"] / 2) * w)
        y1 = int((b["yc"] - b["h"] / 2) * h)
        x2 = int((b["xc"] + b["w"] / 2) * w)
        y2 = int((b["yc"] + b["h"] / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = classes.get(cid, f"class_{cid}")
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if max_size:
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
    return img


def img_to_png_bytes(img: np.ndarray) -> bytes:
    """numpy BGR 数组 → PNG 字节流。"""
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ============================================================
# 路由
# ============================================================

@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats, classes=classes)


@app.route("/gallery")
def gallery():
    page = request.args.get("page", 1, type=int)
    per_page = config.get("server", {}).get("per_page", 20)
    images = list_labeled_images()
    total = len(images)
    start = (page - 1) * per_page
    end = start + per_page
    page_images = images[start:end]
    total_pages = max(1, (total + per_page - 1) // per_page)

    # 只读当前页标签（不读全部图片的标签）
    for img in page_images:
        img["boxes"] = read_labels(img["name"])
        img["box_count"] = len(img["boxes"])

    return render_template(
        "gallery.html",
        images=page_images,
        page=page,
        total_pages=total_pages,
        total=total,
        classes=classes,
        class_colors=class_colors,
    )


@app.route("/image/<name>")
def image_detail(name):
    img_path = IMG_DIR / name
    if not img_path.exists():
        return "Image not found", 404

    boxes = read_labels(name)
    img = draw_boxes_on_image(str(img_path), boxes, max_size=1200)
    if img is None:
        return "Cannot read image", 500

    png_bytes = img_to_png_bytes(img)
    png_b64 = __import__("base64").b64encode(png_bytes).decode()

    return render_template(
        "image.html",
        name=name,
        boxes=boxes,
        classes=classes,
        class_colors=class_colors,
        png_b64=png_b64,
    )


@app.route("/image/<name>/thumb")
def image_thumb(name):
    """返回缩略图（带框）。"""
    img_path = IMG_DIR / name
    if not img_path.exists():
        return "Not found", 404
    boxes = read_labels(name)
    img = draw_boxes_on_image(str(img_path), boxes, max_size=400)
    if img is None:
        return "Cannot read", 500
    return send_file(BytesIO(img_to_png_bytes(img)), mimetype="image/png")


@app.route("/image/<name>/full")
def image_full(name):
    """返回原图（不带框）。"""
    img_path = IMG_DIR / name
    if not img_path.exists():
        return "Not found", 404
    return send_file(str(img_path))


# ============================================================
# API
# ============================================================

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/images")
def api_images():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    images = list_labeled_images()
    total = len(images)
    start = (page - 1) * per_page
    end = start + per_page
    result = images[start:end]
    for img in result:
        img["boxes"] = read_labels(img["name"])
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "images": result,
    })


@app.route("/api/image/<name>")
def api_image(name):
    img_path = IMG_DIR / name
    if not img_path.exists():
        return jsonify({"error": "not found"}), 404
    boxes = read_labels(name)
    return jsonify({
        "name": name,
        "boxes": boxes,
        "classes": {str(k): v for k, v in classes.items()},
    })


# ============================================================
# 视频管理
# ============================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def list_videos():
    """列出 videos/ 下所有视频文件及其状态。"""
    files = []
    for ext in ALLOWED_EXTENSIONS:
        files.extend(VIDEO_DIR.glob(f"*.{ext}"))
    result = []
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        job = _jobs.get(f.name, {})
        result.append({
            "name": f.name,
            "path": str(f),
            "size_mb": round(size_mb, 1),
            "status": job.get("status", "idle"),
            "progress": job.get("progress", 0),
            "total": job.get("total", 0),
            "saved": job.get("saved", 0),
            "error": job.get("error", ""),
        })
    return result

def _run_extraction(video_name: str):
    """后台线程：执行视频抽帧。"""
    from video_extractor import VideoExtractor
    try:
        with _jobs_lock:
            _jobs[video_name] = {"status": "running", "progress": 0, "total": 0, "saved": 0, "error": ""}

        video_path = VIDEO_DIR / video_name
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        extractor = VideoExtractor(config)
        saved = extractor.process_video(video_path)

        with _jobs_lock:
            _jobs[video_name] = {"status": "done", "progress": 100, "total": total_frames,
                                "saved": saved, "error": ""}
    except Exception as e:
        with _jobs_lock:
            _jobs[video_name] = {"status": "error", "progress": 0, "total": 0,
                                "saved": 0, "error": str(e)}


# ============================================================
# 视频路由
# ============================================================

@app.route("/videos")
def videos_page():
    return render_template("videos.html", videos=list_videos())


@app.route("/api/videos/list")
def api_videos_list():
    return jsonify(list_videos())


@app.route("/api/videos/upload", methods=["POST"])
def api_videos_upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件类型，允许: {ALLOWED_EXTENSIONS}"}), 400
    filename = secure_filename(file.filename)
    save_path = VIDEO_DIR / filename
    file.save(str(save_path))
    return jsonify({"ok": True, "name": filename, "size_mb": round(save_path.stat().st_size / (1024*1024), 1)})


@app.route("/api/videos/process/<name>", methods=["POST"])
def api_videos_process(name):
    video_path = VIDEO_DIR / name
    if not video_path.exists():
        return jsonify({"error": "video not found"}), 404

    with _jobs_lock:
        if name in _jobs and _jobs[name].get("status") == "running":
            return jsonify({"error": "already processing"}), 409

    t = threading.Thread(target=_run_extraction, args=(name,), daemon=True)
    t.start()
    return jsonify({"ok": True, "name": name})


@app.route("/api/videos/status")
def api_videos_status():
    return jsonify(dict(_jobs))


@app.route("/api/videos/delete/<name>", methods=["POST"])
def api_videos_delete(name):
    video_path = VIDEO_DIR / name
    if video_path.exists():
        video_path.unlink()
    with _jobs_lock:
        _jobs.pop(name, None)
    return jsonify({"ok": True})


# ============================================================
# 标注路由
# ============================================================

def _label_stats():
    """标注进度统计。"""
    imgs = list_labeled_images()
    total = len(imgs)
    labeled = sum(1 for i in imgs if i["labeled"])
    return {"total": total, "labeled": labeled, "unlabeled": total - labeled}

@app.route("/label")
def label_page():
    stats = _label_stats()
    return render_template("label.html", classes=classes, stats=stats)

@app.route("/api/label/list")
def api_label_list():
    """返回所有图片及其标注状态，按来源分组。"""
    imgs = list_labeled_images()
    sources = _load_sources()
    # 按来源分组
    groups = {}
    for img in imgs:
        img["boxes"] = read_labels(img["name"])
        img["box_count"] = len(img["boxes"])
        src = img["source"] or "未归类"
        groups.setdefault(src, []).append(img)
    return jsonify({
        "images": imgs,
        "groups": {k: len(v) for k, v in groups.items()},
        "stats": _label_stats(),
        "classes": {str(k): v for k, v in classes.items()},
    })

@app.route("/api/label/<name>")
def api_label_get(name):
    """获取单张图片的标注数据（base64缓存60秒）。"""
    img_path = IMG_DIR / name
    if not img_path.exists():
        return jsonify({"error": "not found"}), 404
    import base64
    ck = f"img_b64_{name}"
    b64 = _cached(ck, ttl=60)
    img = None
    if b64 is None:
        img = cv2.imread(str(img_path))
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode()
        _cache_set(ck, b64)
    if img is None:
        img = cv2.imread(str(img_path))
    boxes = read_labels(name)
    h, w = img.shape[:2]
    return jsonify({
        "name": name,
        "width": w, "height": h,
        "image_b64": b64,
        "boxes": boxes,
    })

@app.route("/api/label/<name>", methods=["POST"])
def api_label_save(name):
    """保存标注框（覆盖写入 YOLO 格式标签文件）。"""
    data = request.get_json(force=True)
    boxes = data.get("boxes", [])
    lbl_path = LBL_DIR / f"{Path(name).stem}.txt"
    lines = []
    for b in boxes:
        cid = int(b["class_id"])
        xc = float(b["xc"])
        yc = float(b["yc"])
        w = float(b["w"])
        h = float(b["h"])
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))
    return jsonify({"ok": True, "saved": len(lines)})


@app.route("/api/label/batch-ignore", methods=["POST"])
def api_label_batch_ignore():
    """批量标记：对同视频来源的所有帧，删除与指定区域重叠的框。"""
    data = request.get_json(force=True)
    source = data.get("source", "")
    zone = data.get("zone", {})  # {x1, y1, x2, y2} 归一化坐标或像素

    if not source:
        return jsonify({"error": "需要指定 source"}), 400

    sources = _load_sources()
    affected = 0
    removed = 0

    for name, src in sources.items():
        if src != source:
            continue
        lbl_path = LBL_DIR / f"{Path(name).stem}.txt"
        if not lbl_path.exists():
            continue

        boxes = read_labels(name)
        if not boxes:
            continue

        # 快速取尺寸（用第一张图缓存）
        ck = f"img_size_{name}"
        wh = _cached(ck, ttl=300)
        if wh is None:
            img = cv2.imread(str(IMG_DIR / name))
            if img is None:
                continue
            h, w = img.shape[:2]
            _cache_set(ck, (w, h))
        else:
            w, h = wh

        # 转换 zone 到像素坐标
        if zone:
            zx1 = zone.get("x1", 0) * w if zone.get("norm") else zone.get("x1", 0)
            zy1 = zone.get("y1", 0) * h if zone.get("norm") else zone.get("y1", 0)
            zx2 = zone.get("x2", w) * w if zone.get("norm") else zone.get("x2", w)
            zy2 = zone.get("y2", h) * h if zone.get("norm") else zone.get("y2", h)
        else:
            zx1 = zy1 = 0
            zx2, zy2 = w, h

        kept = []
        for b in boxes:
            bx = b["xc"] * w
            by = b["yc"] * h
            if zx1 <= bx <= zx2 and zy1 <= by <= zy2:
                removed += 1
                continue
            kept.append(b)

        if len(kept) != len(boxes):
            # 重写标签文件
            lbl_path = LBL_DIR / f"{Path(name).stem}.txt"
            lines = []
            for b in kept:
                lines.append(f"{b['class_id']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}")
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))
            affected += 1

    return jsonify({
        "ok": True,
        "source": source,
        "affected_frames": affected,
        "removed_boxes": removed,
    })


@app.route("/api/label/source-images/<source>")
def api_label_source_images(source):
    """获取指定视频来源的所有图片。"""
    sources = _load_sources()
    imgs = []
    for name, src in sources.items():
        if src == source:
            img_path = IMG_DIR / name
            if img_path.exists():
                boxes = read_labels(name)
                imgs.append({
                    "name": name,
                    "box_count": len(boxes),
                    "labeled": len(boxes) > 0,
                })
    return jsonify({"source": source, "count": len(imgs), "images": sorted(imgs, key=lambda x: x["name"])})


@app.route("/api/label/reset/<name>", methods=["POST"])
def api_label_reset(name):
    """重置单张图片的标签（删除标签文件）。"""
    lbl_path = LBL_DIR / f"{Path(name).stem}.txt"
    if lbl_path.exists():
        lbl_path.unlink()
    return jsonify({"ok": True})


# ============================================================
# 模型管理
# ============================================================

def _load_model_state() -> dict:
    if MODEL_STATE_PATH.exists():
        with open(MODEL_STATE_PATH) as f:
            return _json.load(f)
    return {"active": "yolov8n.pt", "available": ["yolov8n.pt"]}

def _save_model_state(state: dict):
    with open(MODEL_STATE_PATH, "w") as f:
        _json.dump(state, f, indent=2)

@app.route("/api/models")
def api_models():
    """列出可用模型 + 当前激活的模型。"""
    wdir = Path(__file__).parent / "weights"
    available = ["yolov8n.pt"]  # COCO 预训练始终可用
    for p in sorted(wdir.glob("*.pt")):
        available.append(p.name)
    state = _load_model_state()
    state["available"] = sorted(set(available))
    _save_model_state(state)
    return jsonify(state)

@app.route("/api/models/active", methods=["POST"])
def api_models_set_active():
    """设置当前使用的模型。"""
    data = request.get_json(force=True)
    model = data.get("model", "yolov8n.pt")
    state = _load_model_state()
    state["active"] = model
    _save_model_state(state)
    return jsonify({"ok": True, "active": model})

def get_active_model() -> str:
    """获取当前激活的模型路径（供视频抽帧和标注使用）。"""
    state = _load_model_state()
    return state.get("active", "yolov8n.pt")


# ============================================================
# 数据集去重
# ============================================================

@app.route("/api/dedup", methods=["POST"])
def api_dedup():
    """对已有数据集进行感知哈希去重，删除高度相似的图片。"""
    data = request.get_json(silent=True) or {}
    threshold = data.get("threshold", 8)
    dry_run = data.get("dry_run", True)

    imgs = sorted(IMG_DIR.glob("frame_*.jpg"))
    if len(imgs) < 2:
        return jsonify({"ok": True, "removed": 0, "message": "图片不足2张"})

    hashes = {}
    to_remove = []
    for img in imgs:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        # 感知哈希
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8))
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        avg = dct_low.mean()
        bits = (dct_low > avg).flatten()
        h = 0
        for bit in bits:
            h = (h << 1) | int(bit)
        # 检查是否与已有哈希相近
        is_dup = False
        for existing_h, existing_name in hashes.items():
            if bin(h ^ existing_h).count("1") < threshold:
                is_dup = True
                break
        if is_dup:
            to_remove.append(img)
        else:
            hashes[h] = img.name

    if not dry_run:
        sources = _load_sources()
        for img in to_remove:
            # 删除图片
            img.unlink(missing_ok=True)
            # 删除标签
            lbl = LBL_DIR / f"{img.stem}.txt"
            lbl.unlink(missing_ok=True)
            # 清理来源
            sources.pop(img.name, None)
        _save_sources(sources)
        # 清除缓存
        _cache.clear()

    return jsonify({
        "ok": True,
        "dry_run": dry_run,
        "total": len(imgs),
        "kept": len(imgs) - len(to_remove),
        "removed": len(to_remove),
        "threshold": threshold,
    })


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="数据集 Web 浏览器")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", type=str, help="监听地址")
    parser.add_argument("--port", type=int, help="端口")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # 重新加载配置以适配命令行参数
    global config, IMG_DIR, LBL_DIR, dspath
    config = load_config(args.config)
    dspath = Path(config.get("dataset", {}).get("path", "./dataset"))
    if not dspath.is_absolute():
        dspath = Path(__file__).parent / dspath
    IMG_DIR = dspath / "images"
    LBL_DIR = dspath / "labels"

    host = args.host or config.get("server", {}).get("host", "0.0.0.0")
    port = args.port or config.get("server", {}).get("port", 5000)

    print(f"\n  [Web] 数据集浏览器")
    print(f"  地址: http://{host}:{port}")
    print(f"  图片目录: {IMG_DIR}")
    print(f"  图片数量: {len(list_labeled_images())}")
    print(f"  按 Ctrl+C 退出\n")

    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
