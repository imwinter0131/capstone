from __future__ import annotations

import argparse


class YOLOTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def train(self) -> None:
        from ultralytics import YOLO

        model = YOLO(self.args.model_name)
        model.train(
            data=self.args.data_dir,
            epochs=self.args.epochs,
            imgsz=self.args.img_size,
            batch=self.args.batch_size,
            device=self.args.device,
            workers=self.args.workers,
            patience=self.args.patience,
            optimizer=self.args.optimizer,
            lr0=self.args.lr0,
            lrf=self.args.lrf,
            weight_decay=self.args.weight_decay,
            cos_lr=self.args.cos_lr,
            warmup_epochs=self.args.warmup_epochs,
            dropout=self.args.dropout,
            label_smoothing=self.args.label_smoothing,
            auto_augment=self.args.auto_augment,
            erasing=self.args.erasing,
            mosaic=self.args.mosaic,
            mixup=self.args.mixup,
            cutmix=self.args.cutmix,
            seed=self.args.seed,
            save_period=self.args.save_period,
            cache=self.args.cache,
            project=self.args.save_dir,
            name="train",
        )
