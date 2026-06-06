from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    configured_path,
    dataset_dirs,
    ensure_dataset_dirs,
    get_class_names,
    iter_images,
    label_path_for,
    load_config,
    load_sources,
    reset_dir,
    set_seed,
)


def _copy_pair(image: Path, labels_dir: Path, dst_images: Path, dst_labels: Path) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, dst_images / image.name)
    src_label = label_path_for(image, labels_dir)
    dst_label = dst_labels / f"{image.stem}.txt"
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def collect_images(config: dict[str, Any]) -> list[Path]:
    dirs = ensure_dataset_dirs(config)
    include_negative = bool(config.get("dataset", {}).get("include_negative", True))
    images = iter_images(dirs["images"], config)
    if include_negative:
        return images
    return [img for img in images if label_path_for(img, dirs["labels"]).exists()]


def split_images(config: dict[str, Any], images: list[Path]) -> tuple[list[Path], list[Path], str]:
    """Split images into train and validation sets.

    Source-aware splitting matters for game footage because adjacent video
    frames are highly similar. If one frame goes to train and the next goes to
    val, validation can look better than the model really is.
    """
    if not images:
        return [], [], "empty"

    ds_cfg = config.get("dataset", {})
    seed = int(config.get("project", {}).get("seed", ds_cfg.get("seed", 42)))
    train_split = float(ds_cfg.get("train_split", 0.85))
    strategy = ds_cfg.get("split_strategy", "source")
    rng = random.Random(seed)

    if strategy == "source":
        sources = load_sources(config)
        groups: dict[str, list[Path]] = {}
        for image in images:
            source = sources.get(image.name, image.name)
            groups.setdefault(source, []).append(image)

        group_items = list(groups.items())
        rng.shuffle(group_items)

        if len(group_items) >= 2:
            target = max(1, round(len(images) * train_split))
            train: list[Path] = []
            val: list[Path] = []
            for _, group_images in group_items:
                if len(train) < target:
                    train.extend(group_images)
                else:
                    val.extend(group_images)
            if not val:
                _, last_group = group_items[-1]
                for img in last_group:
                    train.remove(img)
                val.extend(last_group)
            if not train:
                _, first_group = group_items[0]
                for img in first_group:
                    val.remove(img)
                train.extend(first_group)
            return sorted(train), sorted(val), "source"

    shuffled = images[:]
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * train_split))
    if len(shuffled) > 1:
        split_idx = min(split_idx, len(shuffled) - 1)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:] or shuffled[:]
    return sorted(train), sorted(val), "image"


def prepare_dataset(config: dict[str, Any]) -> Path:
    """Build the Ultralytics-ready dataset directory.

    The raw dataset remains in images/labels. prepared/ is a disposable copy
    with train/val folders plus dataset.yaml, which is the file YOLO reads.
    """
    dirs = ensure_dataset_dirs(config)
    images = collect_images(config)
    if not images:
        raise RuntimeError(
            "Dataset is empty. Add images to "
            f"{dirs['images']} and labels to {dirs['labels']}."
        )

    train_images, val_images, strategy = split_images(config, images)
    prepared = dirs["prepared"]
    reset_dir(prepared)

    for subset, subset_images in (("train", train_images), ("val", val_images)):
        for image in subset_images:
            _copy_pair(
                image,
                dirs["labels"],
                prepared / subset / "images",
                prepared / subset / "labels",
            )

    names = get_class_names(config)
    dataset_yaml = prepared / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(prepared.resolve()),
                "train": "train/images",
                "val": "val/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "split_strategy": strategy,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "total_images": len(images),
        "train": [p.name for p in train_images],
        "val": [p.name for p in val_images],
    }
    (prepared / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Dataset prepared: {prepared}")
    print(f"  split: {strategy}")
    print(f"  train: {len(train_images)} images")
    print(f"  val:   {len(val_images)} images")
    print(f"  yaml:  {dataset_yaml}")
    return dataset_yaml


def train_model(config: dict[str, Any], dataset_yaml: Path, overrides: dict[str, Any]) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    train_cfg = config.get("train", {})
    seed = int(overrides.get("seed") or train_cfg.get("seed") or config.get("project", {}).get("seed", 42))
    set_seed(seed)

    params: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": train_cfg.get("epochs", 100),
        "batch": train_cfg.get("batch", 16),
        "imgsz": train_cfg.get("imgsz", 640),
        "device": train_cfg.get("device", "cpu"),
        "workers": train_cfg.get("workers", 4),
        "patience": train_cfg.get("patience", 20),
        "lr0": train_cfg.get("lr0", 0.01),
        "lrf": train_cfg.get("lrf", 0.01),
        "seed": seed,
        "deterministic": train_cfg.get("deterministic", True),
        "project": str(configured_path(config, "runs", "runs") / "train"),
        "name": train_cfg.get("name", "game_detector"),
        "exist_ok": True,
        "freeze": train_cfg.get("freeze"),
        "mosaic": train_cfg.get("mosaic", 1.0),
        "close_mosaic": train_cfg.get("close_mosaic", 10),
        "mixup": train_cfg.get("mixup", 0.0),
        "copy_paste": train_cfg.get("copy_paste", 0.0),
        "degrees": train_cfg.get("degrees", 0.0),
        "translate": train_cfg.get("translate", 0.1),
        "scale": train_cfg.get("scale", 0.5),
        "fliplr": train_cfg.get("fliplr", 0.5),
        "hsv_h": train_cfg.get("hsv_h", 0.015),
        "hsv_s": train_cfg.get("hsv_s", 0.7),
        "hsv_v": train_cfg.get("hsv_v", 0.4),
        "erasing": train_cfg.get("erasing", 0.2),
    }
    params.update({k: v for k, v in overrides.items() if v is not None and k not in {"seed", "model"}})
    params = {k: v for k, v in params.items() if v is not None}

    model_name = overrides.get("model") or train_cfg.get("model", "yolov8n.pt")
    print("Training YOLO model")
    print(f"  model: {model_name}")
    print(f"  data:  {dataset_yaml}")
    print(f"  runs:  {params['project']}/{params['name']}")

    model = YOLO(str(model_name))
    results = model.train(**params)

    weights_dir = configured_path(config, "weights", "weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    run_weights = Path(results.save_dir) / "weights"
    for name in ("best.pt", "last.pt"):
        src = run_weights / name
        if src.exists():
            shutil.copy2(src, weights_dir / name)
            print(f"Copied {name} -> {weights_dir / name}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset and train YOLOv8.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to ./config.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the prepared train/val dataset.")
    parser.add_argument("--model", help="Base model or checkpoint, e.g. yolov8n.pt")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", help="cpu, cuda:0, or other Ultralytics device value")
    parser.add_argument("--workers", type=int, help="Data loader workers")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_yaml = prepare_dataset(config)
    if args.prepare_only:
        return

    train_model(
        config,
        dataset_yaml,
        {
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
