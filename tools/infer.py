from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from common import configured_path, load_config, project_path


def resolve_weights(config: dict[str, Any], value: str | None) -> str:
    raw = value or config.get("inference", {}).get("weights", "weights/best.pt")
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = project_path(path)
    if candidate.exists():
        return str(candidate)
    if raw in {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}:
        return raw
    raise FileNotFoundError(
        f"Weights not found: {raw}. Train first or pass --weights yolov8n.pt for a smoke test."
    )


def run_inference(
    config: dict[str, Any],
    source: str,
    weights: str | None,
    output: str | None,
    name: str,
    conf: float | None,
    iou: float | None,
    imgsz: int | None,
    save_txt: bool | None,
    save_conf: bool | None,
) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    inf_cfg = config.get("inference", {})
    source_path = project_path(source)
    if not source_path.exists() and not source.startswith(("http://", "https://")):
        raise FileNotFoundError(f"Source not found: {source}")

    out_dir = project_path(output) if output else configured_path(config, "exports", "exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(resolve_weights(config, weights))
    return model.predict(
        source=str(source_path if source_path.exists() else source),
        conf=float(conf if conf is not None else inf_cfg.get("conf_threshold", 0.35)),
        iou=float(iou if iou is not None else inf_cfg.get("iou_threshold", 0.5)),
        imgsz=int(imgsz if imgsz is not None else inf_cfg.get("imgsz", 640)),
        project=str(out_dir),
        name=name,
        exist_ok=True,
        save=True,
        save_txt=bool(inf_cfg.get("save_txt", False) if save_txt is None else save_txt),
        save_conf=bool(inf_cfg.get("save_conf", False) if save_conf is None else save_conf),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline YOLO inference on an image, folder, or video.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to ./config.yaml")
    parser.add_argument("--source", required=True, help="Image, directory, or video file")
    parser.add_argument("--weights", help="Model weights. Defaults to config inference.weights")
    parser.add_argument("--output", help="Output directory. Defaults to config paths.exports")
    parser.add_argument("--name", default="predict", help="Run name under the output directory")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    parser.add_argument("--save-txt", action="store_true", help="Write YOLO-format prediction txt files")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in txt predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_inference(
        config=config,
        source=args.source,
        weights=args.weights,
        output=args.output,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_txt=True if args.save_txt else None,
        save_conf=True if args.save_conf else None,
    )
    print(f"Inference complete. Processed {len(results)} item(s).")


if __name__ == "__main__":
    main()
