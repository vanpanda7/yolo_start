from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from common import configured_path, dataset_dirs, load_config, project_path
from infer import resolve_weights
from train import prepare_dataset


def _serializable_metrics(metrics: Any) -> dict[str, float]:
    raw = getattr(metrics, "results_dict", {}) or {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def evaluate(config: dict[str, Any], weights: str | None = None, rebuild: bool = False) -> dict[str, float]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    dataset_yaml = dataset_dirs(config)["prepared"] / "dataset.yaml"
    if rebuild or not dataset_yaml.exists():
        dataset_yaml = prepare_dataset(config)

    val_project = configured_path(config, "runs", "runs") / "val"
    val_project.mkdir(parents=True, exist_ok=True)
    model = YOLO(resolve_weights(config, weights))
    metrics = model.val(
        data=str(dataset_yaml),
        project=str(val_project),
        name="evaluation",
        exist_ok=True,
        plots=True,
    )

    result = _serializable_metrics(metrics)
    report_dir = configured_path(config, "reports", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation report: {report_path}")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on the prepared validation split.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to ./config.yaml")
    parser.add_argument("--weights", help="Weights path. Defaults to config inference.weights")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the prepared dataset before evaluating")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, weights=args.weights, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
