from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ensure_dataset_dirs, image_extensions, load_config, load_sources, project_path, save_sources


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "image"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 100000):
        candidate = path.with_name(f"{path.stem}_{idx:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a free filename for {path}")


def import_images(config: dict, source: str, source_name: str | None = None) -> int:
    """Copy screenshots into the YOLO dataset without training side effects.

    Imported images receive empty label files when negative samples are enabled.
    That makes "no target in this frame" an explicit training example instead
    of an unlabeled file that may be skipped later.
    """
    src = project_path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    dirs = ensure_dataset_dirs(config)
    exts = image_extensions(config)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    images = [p for p in files if p.suffix.lower() in exts]
    sources = load_sources(config)
    label_empty = bool(config.get("dataset", {}).get("include_negative", True))
    imported = 0

    for image in images:
        prefix = safe_name(source_name or (src.stem if src.is_file() else src.name))
        target_name = safe_name(f"{prefix}_{image.stem}") + image.suffix.lower()
        dst = unique_path(dirs["images"] / target_name)
        shutil.copy2(image, dst)
        if label_empty:
            (dirs["labels"] / f"{dst.stem}.txt").write_text("", encoding="utf-8")
        sources[dst.name] = source_name or str(src.name)
        imported += 1

    save_sources(config, sources)
    print(f"Imported {imported} images into {dirs['images']}")
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy image files into the YOLO dataset.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to ./config.yaml")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--source-name", help="Source group name stored in sources.json")
    args = parser.parse_args()

    config = load_config(args.config)
    import_images(config, args.source, args.source_name)


if __name__ == "__main__":
    main()
