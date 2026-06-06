# tools/train.py
"""
YOLOv8 训练脚本 —— 读取标注数据集，训练目标检测模型。

Usage:
    python train.py                # 使用 config.yaml 中的默认参数
    python train.py --epochs 200    # 覆盖训练轮数
    python train.py --device cuda:0 # 使用 GPU
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

import yaml

# 确保能找到本目录的模块
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict) -> Path:
    """检查数据集并生成 ultralytics 格式的 dataset.yaml。

    Returns:
        Path: 生成的 dataset.yaml 文件路径。
    """
    ds_cfg = config.get("dataset", {})
    ds_path = Path(ds_cfg.get("path", "./dataset"))
    if not ds_path.is_absolute():
        ds_path = Path(__file__).parent / ds_path

    img_dir = ds_path / "images"
    lbl_dir = ds_path / "labels"

    # 获取所有有对应标签的图片
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    valid_images = []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            valid_images.append(img)

    if not valid_images:
        raise RuntimeError(
            f"数据集为空！请先用 labeler.py 标注一些图片。\n"
            f"  图片目录: {img_dir}\n"
            f"  标签目录: {lbl_dir}"
        )

    print(f"  有效图片: {len(valid_images)} 张")

    # 划分训练集/验证集
    train_split = ds_cfg.get("train_split", 0.85)
    split_idx = int(len(valid_images) * train_split)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    # 创建 train/val 目录结构
    for subset, imgs in [("train", train_images), ("val", val_images)]:
        sub_img = ds_path / subset / "images"
        sub_lbl = ds_path / subset / "labels"
        sub_img.mkdir(parents=True, exist_ok=True)
        sub_lbl.mkdir(parents=True, exist_ok=True)

        # 用符号链接/复制避免重复占用磁盘
        for img_path in imgs:
            dst_img = sub_img / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            dst_lbl = sub_lbl / lbl_path.name
            if not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)

    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")
    if len(val_images) < 10:
        print(f"  ⚠ 验证集仅 {len(val_images)} 张，早停和mAP评估不可靠，建议标注更多图片")

    # 生成 dataset.yaml
    classes = config.get("classes", {0: "enemy_body", 1: "enemy_head"})
    class_names = [classes[k] for k in sorted(classes.keys())]

    ds_yaml_path = ds_path / "dataset.yaml"
    ds_yaml_content = {
        "path": str(ds_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(ds_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(ds_yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"  dataset.yaml 已生成: {ds_yaml_path}")
    return ds_yaml_path


def train(config: dict, dataset_yaml: Path, overrides: dict = None):
    """执行 YOLOv8 训练，小数据集自动启用防过拟合策略。"""
    train_cfg = config.get("train", {})

    # 检测数据集大小，自动调整策略
    img_count = len(list((dataset_yaml.parent / "train" / "images").glob("*.jpg")))
    is_small = img_count < 100

    params = {
        "model": train_cfg.get("model", "yolov8n.pt"),
        "data": str(dataset_yaml),
        "epochs": train_cfg.get("epochs", 100),
        "batch": train_cfg.get("batch", 16),
        "imgsz": train_cfg.get("imgsz", 640),
        "device": train_cfg.get("device", "cpu"),
        "workers": train_cfg.get("workers", 4),
        "lr0": train_cfg.get("lr0", 0.01),
        "lrf": train_cfg.get("lrf", 0.01),
        "patience": train_cfg.get("patience", 20),
        "save_period": train_cfg.get("save_period", 10),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3),
        "cos_lr": train_cfg.get("cos_lr", False),
        # 数据增强
        "mosaic": train_cfg.get("mosaic", 1.0),
        "close_mosaic": train_cfg.get("close_mosaic", 15),
        "mixup": train_cfg.get("mixup", 0.0),
        "copy_paste": train_cfg.get("copy_paste", 0.0),
        "degrees": train_cfg.get("degrees", 0.0),
        "translate": train_cfg.get("translate", 0.1),
        "scale": train_cfg.get("scale", 0.5),
        "fliplr": train_cfg.get("fliplr", 0.5),
        "hsv_h": train_cfg.get("hsv_h", 0.015),
        "hsv_s": train_cfg.get("hsv_s", 0.7),
        "hsv_v": train_cfg.get("hsv_v", 0.4),
        "erasing": train_cfg.get("erasing", 0.4),
    }

    # 小数据集自动激进策略（如果用户没显式设值）
    freeze = train_cfg.get("freeze")
    if freeze is None:
        params["freeze"] = 10 if is_small else 0
    else:
        params["freeze"] = freeze

    if is_small:
        if train_cfg.get("epochs") is None:
            params["epochs"] = 200
        if train_cfg.get("mixup") is None or train_cfg.get("mixup") == 0.0:
            params["mixup"] = 0.2
        if train_cfg.get("close_mosaic") is None:
            params["close_mosaic"] = 5
        print(f"\n  ⚡ 检测到小数据集 ({img_count} 张)，自动启用防过拟合策略:")
        print(f"     freeze={params['freeze']}, epochs={params['epochs']}, mixup={params['mixup']}")
        print(f"     建议标注更多图片以获得更好效果 (目标 ≥200 张)\n")

    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    # 权重输出目录
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    params["project"] = str(weights_dir)
    params["name"] = "train"

    print("\n" + "=" * 60)
    print("  开始训练")
    print(f"  模型: {params['model']}  |  数据: {img_count} 张")
    print(f"  轮数: {params['epochs']}  |  批次: {params['batch']}  |  设备: {params['device']}")
    print(f"  冻结: {params['freeze']}层  |  mosaic: {params['mosaic']}  |  mixup: {params['mixup']}")
    print(f"  输出: {weights_dir}/train/weights/")
    print("=" * 60 + "\n")

    try:
        from ultralytics import YOLO

        model = YOLO(params["model"])
        results = model.train(**{k: v for k, v in params.items() if k != "model"})

        # 训练完成后，复制最佳权重到 tools/weights/
        best_src = Path(results.save_dir) / "weights" / "best.pt"
        if best_src.exists():
            best_dst = weights_dir / "best.pt"
            shutil.copy2(best_src, best_dst)
            print(f"\n✅ 最佳权重已保存: {best_dst}")

        last_src = Path(results.save_dir) / "weights" / "last.pt"
        if last_src.exists():
            last_dst = weights_dir / "last.pt"
            shutil.copy2(last_src, last_dst)

        return results

    except ImportError:
        raise ImportError(
            "ultralytics 未安装。请运行:\n"
            "  pip install ultralytics"
        )


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 目标检测训练")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch", type=int, help="批次大小")
    parser.add_argument("--device", type=str, help="设备 (cpu / cuda:0)")
    parser.add_argument("--lr0", type=float, help="初始学习率")
    parser.add_argument("--model", type=str, help="预训练模型路径")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 准备数据集
    print("检查数据集...")
    dataset_yaml = prepare_dataset(config)

    # 训练
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "lr0": args.lr0,
        "model": args.model,
    }
    train(config, dataset_yaml, overrides)


if __name__ == "__main__":
    main()
