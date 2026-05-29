from __future__ import annotations

import argparse
from typing import Any

COMMON_ARGS: dict[str, Any] = {
    "data_dir": "/root/project/dataset/yolo_dataset",
    "save_dir": "/root/project/outputs/yolo",
    "epochs": 250,
    "device": "cuda:0",
    "workers": 16,
    "seed": 0,
    "patience": 60,
    "optimizer": "AdamW",
    "lr0": 3e-4,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "dropout": 0.05,
    "label_smoothing": 0.01,
    "auto_augment": "none",
    "erasing": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "cache": "disk",
    "save_period": 10,
}


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "yolo11n_cls": {
        "model_name": "yolo11n-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo11s_cls": {
        "model_name": "yolo11s-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo11m_cls": {
        "model_name": "yolo11m-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo11l_cls": {
        "model_name": "yolo11l-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo11x_cls": {
        "model_name": "yolo11x-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo12n_cls": {
        "model_name": "yolo12n-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo12s_cls": {
        "model_name": "yolo12s-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo12m_cls": {
        "model_name": "yolo12m-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo12l_cls": {
        "model_name": "yolo12l-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo12x_cls": {
        "model_name": "yolo12x-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo26n_cls": {
        "model_name": "yolo26n-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo26s_cls": {
        "model_name": "yolo26s-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo26m_cls": {
        "model_name": "yolo26m-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo26l_cls": {
        "model_name": "yolo26l-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolo26x_cls": {
        "model_name": "yolo26x-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolov8n_cls": {
        "model_name": "yolov8n-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolov8s_cls": {
        "model_name": "yolov8s-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolov8m_cls": {
        "model_name": "yolov8m-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolov8l_cls": {
        "model_name": "yolov8l-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
    "yolov8x_cls": {
        "model_name": "yolov8x-cls.pt",
        "batch_size": -1,
        "img_size": 224,
    },
}


def get_args(model_key: str) -> argparse.Namespace:
    if model_key not in MODEL_CONFIGS:
        available_keys = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"Unknown YOLO model_key '{model_key}'. "
            f"Available YOLO model_key values: {available_keys}."
        )

    args = COMMON_ARGS.copy()
    args.update(MODEL_CONFIGS[model_key])
    return argparse.Namespace(**args)
