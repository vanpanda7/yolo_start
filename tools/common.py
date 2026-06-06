from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(path)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_path(config: dict[str, Any], key: str, default: str) -> Path:
    return project_path(config.get("paths", {}).get(key, default))


def get_classes(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("classes", {0: "player"})
    return {int(k): str(v) for k, v in raw.items()}


def get_class_names(config: dict[str, Any]) -> list[str]:
    classes = get_classes(config)
    return [classes[k] for k in sorted(classes)]


def get_class_colors(config: dict[str, Any]) -> dict[int, list[int]]:
    raw = config.get("class_colors", {})
    return {int(k): [int(c) for c in v] for k, v in raw.items()}


def dataset_root(config: dict[str, Any]) -> Path:
    return configured_path(config, "dataset", "datasets/game_targets")


def dataset_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = dataset_root(config)
    sources_file = config.get("dataset", {}).get("sources_file", "sources.json")
    prepared_subdir = config.get("dataset", {}).get("prepared_subdir", "prepared")
    return {
        "root": root,
        "images": root / "images",
        "labels": root / "labels",
        "sources": root / sources_file,
        "prepared": root / prepared_subdir,
    }


def ensure_dataset_dirs(config: dict[str, Any]) -> dict[str, Path]:
    dirs = dataset_dirs(config)
    dirs["images"].mkdir(parents=True, exist_ok=True)
    dirs["labels"].mkdir(parents=True, exist_ok=True)
    return dirs


def load_sources(config: dict[str, Any]) -> dict[str, str]:
    path = dataset_dirs(config)["sources"]
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def save_sources(config: dict[str, Any], sources: dict[str, str]) -> None:
    path = dataset_dirs(config)["sources"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(sources.items())), f, indent=2, ensure_ascii=False)


def image_extensions(config: dict[str, Any]) -> set[str]:
    raw = config.get("dataset", {}).get("image_extensions", [".jpg", ".jpeg", ".png"])
    return {str(ext).lower() for ext in raw}


def iter_images(path: Path, config: dict[str, Any]) -> list[Path]:
    exts = image_extensions(config)
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def read_yolo_label(label_path: Path) -> list[dict[str, float | int]]:
    """Read one YOLO txt file.

    YOLO detection labels are plain text rows:
    class_id x_center y_center width height

    The four coordinates are normalized to 0..1, so the same label stays valid
    even when the image is displayed at a different size in the Web UI.
    """
    if not label_path.exists():
        return []
    boxes: list[dict[str, float | int]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append({
                "class_id": int(float(parts[0])),
                "xc": float(parts[1]),
                "yc": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return boxes


def write_yolo_label(label_path: Path, boxes: list[dict[str, Any]]) -> int:
    """Write frontend boxes back to YOLO txt format.

    The annotation UI uses class_id=-1 for temporary "ignore" boxes. Those are
    review helpers only and must not be saved, because Ultralytics expects every
    persisted row to reference a real configured class.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for box in boxes:
        class_id = int(box["class_id"])
        if class_id < 0:
            continue
        xc = float(box["xc"])
        yc = float(box["yc"])
        w = float(box["w"])
        h = float(box["h"])
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
