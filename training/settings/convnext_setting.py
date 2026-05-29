from __future__ import annotations

import argparse
from typing import Any


COMMON_ARGS: dict[str, Any] = {
    "data_dir": "/root/project/dataset/dataset",
    "save_dir": "/root/project/outputs/convnext",
    "num_classes": 7,
    "img_size": 224,
    "epochs": 150,
    "num_workers": 16,
    "device": "cuda:0",
    "seed": 0,
    "loss": "ce",
    "weight_decay": 0.05,
    "patience": 30,
    "amp": True,
    "save_period": 10,
    "interpolation": "bicubic",
    "train_random_resized_crop_scale": (0.8, 1.0),
    "train_hflip_prob": 0.5,
    "color_jitter": 0.0,
    "auto_augment": "none",
    "random_erasing": 0.25,
    "normalize_mean": (0.485, 0.456, 0.406),
    "normalize_std": (0.229, 0.224, 0.225),
}


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "convnext_zepto_rms": {
        "model_name": "convnext_zepto_rms.ra4_e3600_r224_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_zepto_rms_ols": {
        "model_name": "convnext_zepto_rms_ols.ra4_e3600_r224_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_atto": {
        "model_name": "convnext_atto.d2_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_atto_ols": {
        "model_name": "convnext_atto_ols.a2_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_atto_rms": {
        "model_name": "convnext_atto_rms",
        "pretrained": False,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_femto": {
        "model_name": "convnext_femto.d1_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_femto_ols": {
        "model_name": "convnext_femto_ols.d1_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_pico": {
        "model_name": "convnext_pico.d1_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_pico_ols": {
        "model_name": "convnext_pico_ols.d1_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnext_nano": {
        "model_name": "convnext_nano.in12k_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 384,
        "lr": 1e-4,
    },
    "convnext_nano_ols": {
        "model_name": "convnext_nano_ols.d1h_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 384,
        "lr": 1e-4,
    },
    "convnext_tiny": {
        "model_name": "convnext_tiny.fb_in22k_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.1,
        "batch_size": 256,
        "lr": 1e-4,
    },
    "convnext_tiny_hnf": {
        "model_name": "convnext_tiny_hnf.a2h_in1k",
        "pretrained": True,
        "drop_path_rate": 0.1,
        "batch_size": 256,
        "lr": 1e-4,
    },
    "convnext_small": {
        "model_name": "convnext_small.fb_in22k_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.1,
        "batch_size": 128,
        "lr": 1e-4,
    },
    "convnext_base": {
        "model_name": "convnext_base.fb_in22k_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.2,
        "batch_size": 64,
        "lr": 5e-5,
    },
    "convnext_large": {
        "model_name": "convnext_large.fb_in22k_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.3,
        "batch_size": 32,
        "lr": 5e-5,
    },
    "convnext_large_mlp": {
        "model_name": "convnext_large_mlp.clip_laion2b_augreg_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.3,
        "batch_size": 32,
        "lr": 5e-5,
    },
    "convnext_xlarge": {
        "model_name": "convnext_xlarge.fb_in22k_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.4,
        "batch_size": 16,
        "lr": 3e-5,
    },
    "convnext_xxlarge": {
        "model_name": "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.4,
        "batch_size": 8,
        "lr": 3e-5,
    },
    "convnextv2_atto": {
        "model_name": "convnextv2_atto.fcmae_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnextv2_femto": {
        "model_name": "convnextv2_femto.fcmae_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnextv2_pico": {
        "model_name": "convnextv2_pico.fcmae_ft_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 512,
        "lr": 1e-4,
    },
    "convnextv2_nano": {
        "model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
        "pretrained": True,
        "drop_path_rate": 0.05,
        "batch_size": 384,
        "lr": 1e-4,
    },
    "convnextv2_tiny": {
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "pretrained": True,
        "drop_path_rate": 0.1,
        "batch_size": 256,
        "lr": 1e-4,
    },
    "convnextv2_small": {
        "model_name": "convnextv2_small.fcmae",
        "pretrained": True,
        "drop_path_rate": 0.1,
        "batch_size": 128,
        "lr": 1e-4,
    },
    "convnextv2_base": {
        "model_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "pretrained": True,
        "drop_path_rate": 0.2,
        "batch_size": 64,
        "lr": 5e-5,
    },
    "convnextv2_large": {
        "model_name": "convnextv2_large.fcmae_ft_in22k_in1k",
        "pretrained": True,
        "drop_path_rate": 0.3,
        "batch_size": 32,
        "lr": 5e-5,
    },
    "convnextv2_huge": {
        "model_name": "convnextv2_huge.fcmae_ft_in22k_in1k_384",
        "pretrained": True,
        "drop_path_rate": 0.4,
        "batch_size": 8,
        "lr": 3e-5,
    },
}


def get_args(model_key: str) -> argparse.Namespace:
    if model_key not in MODEL_CONFIGS:
        available_keys = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"Unknown ConvNeXt model_key '{model_key}'. "
            f"Available ConvNeXt model_key values: {available_keys}."
        )

    args = COMMON_ARGS.copy()
    args.update(MODEL_CONFIGS[model_key])
    return argparse.Namespace(**args)
