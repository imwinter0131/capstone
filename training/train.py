from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Type


if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image classification training entrypoint.")
    parser.add_argument(
        "--family",
        required=True,
        choices=("convnext", "yolo"),
        help="Model family to train. Available values: convnext, yolo.",
    )
    parser.add_argument(
        "--model_key",
        required=True,
        help="Model key defined in the selected family's setting file.",
    )
    return parser.parse_args()


def load_setting_module(family: str) -> ModuleType:
    if family == "convnext":
        from training.settings import convnext_setting

        return convnext_setting
    if family == "yolo":
        from training.settings import yolo_setting

        return yolo_setting

    raise ValueError("Unsupported family. Available families: convnext, yolo.")


def load_trainer_class(family: str) -> Type[object]:
    if family == "convnext":
        from training.trainers.convnext_trainer import ConvNeXtTrainer

        return ConvNeXtTrainer
    if family == "yolo":
        from training.trainers.yolo_trainer import YOLOTrainer

        return YOLOTrainer

    raise ValueError("Unsupported family. Available families: convnext, yolo.")


def main() -> int:
    cli_args = parse_args()

    try:
        setting_module = load_setting_module(cli_args.family)
        args = setting_module.get_args(cli_args.model_key)
        trainer_class = load_trainer_class(cli_args.family)
        trainer = trainer_class(args)
        trainer.train()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"[ERROR] Missing dependency or module import failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
