from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from training.models.convnext_model import build_convnext_model
from training.utils.seed import set_seed


class ConvNeXtTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        set_seed(int(args.seed))

        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.save_dir / "train_log.csv"

        self.device = torch.device(args.device)
        self.scaler = GradScaler(enabled=bool(args.amp))
        self.best_val_acc = 0.0
        self.best_epoch = 0

        self.train_loader: DataLoader[tuple[Tensor, Tensor]] | None = None
        self.val_loader: DataLoader[tuple[Tensor, Tensor]] | None = None
        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: Optimizer | None = None

    def build_dataloader(self) -> None:
        train_transform, val_transform = self._build_transforms()

        train_dir = Path(self.args.data_dir) / "train"
        val_dir = Path(self.args.data_dir) / "val"
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=train_transform,
        )
        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=val_transform,
        )

        if len(train_dataset) == 0:
            raise ValueError(
                f"No training samples found under '{train_dir}'. "
                "Expected ImageFolder layout: data_dir/train/class_name/image files."
            )
        if len(val_dataset) == 0:
            raise ValueError(
                f"No validation samples found under '{val_dir}'. "
                "Expected ImageFolder layout: data_dir/val/class_name/image files."
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.args.batch_size),
            shuffle=True,
            num_workers=int(self.args.num_workers),
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.args.batch_size),
            shuffle=False,
            num_workers=int(self.args.num_workers),
            pin_memory=True,
        )

    def _build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        interpolation = self._interpolation_mode(str(self.args.interpolation))
        normalize = transforms.Normalize(
            mean=tuple(float(value) for value in self.args.normalize_mean),
            std=tuple(float(value) for value in self.args.normalize_std),
        )

        train_transforms: list[Any] = [
            transforms.RandomResizedCrop(
                size=int(self.args.img_size),
                scale=tuple(float(value) for value in self.args.train_random_resized_crop_scale),
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(p=float(self.args.train_hflip_prob)),
        ]

        if float(self.args.color_jitter) > 0:
            jitter = float(self.args.color_jitter)
            train_transforms.append(
                transforms.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                )
            )

        if str(self.args.auto_augment).lower() == "randaugment":
            train_transforms.append(transforms.RandAugment(interpolation=interpolation))

        train_transforms.extend([transforms.ToTensor(), normalize])

        if float(self.args.random_erasing) > 0:
            train_transforms.append(transforms.RandomErasing(p=float(self.args.random_erasing)))

        val_transforms = [
            transforms.Resize(
                (int(self.args.img_size), int(self.args.img_size)),
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            normalize,
        ]

        return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

    @staticmethod
    def _interpolation_mode(name: str) -> InterpolationMode:
        modes = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "lanczos": InterpolationMode.LANCZOS,
        }
        normalized_name = name.lower()
        if normalized_name not in modes:
            available = ", ".join(sorted(modes))
            raise ValueError(f"Unknown interpolation '{name}'. Available values: {available}.")
        return modes[normalized_name]

    def build_model(self) -> None:
        self.model = build_convnext_model(
            model_name=self.args.model_name,
            num_classes=int(self.args.num_classes),
            pretrained=bool(self.args.pretrained),
            drop_path_rate=float(self.args.drop_path_rate),
        ).to(self.device)

    def build_loss(self) -> None:
        if self.args.loss != "ce":
            raise ValueError("ConvNeXtTrainer currently supports only loss='ce'.")
        self.criterion = nn.CrossEntropyLoss()

    def build_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be built before optimizer.")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )

    def train_one_epoch(self) -> tuple[float, float]:
        if self.model is None or self.criterion is None or self.optimizer is None:
            raise RuntimeError("Trainer components are not fully initialized.")
        if self.train_loader is None:
            raise RuntimeError("Train dataloader is not initialized.")

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(self.args.amp)):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.model is None or self.criterion is None:
            raise RuntimeError("Trainer components are not fully initialized.")
        if self.val_loader is None:
            raise RuntimeError("Validation dataloader is not initialized.")

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        confusion = torch.zeros(
            (int(self.args.num_classes), int(self.args.num_classes)),
            dtype=torch.long,
        )

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

            for label, pred in zip(labels.cpu(), preds.cpu(), strict=True):
                confusion[int(label), int(pred)] += 1

        macro_precision, macro_recall, macro_f1 = self._macro_metrics(confusion)
        return {
            "val_loss": total_loss / total_samples,
            "val_acc": total_correct / total_samples,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def save_checkpoint(self, epoch: int, path: Path, metrics: dict[str, float]) -> None:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer components are not fully initialized.")

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": vars(self.args),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)

    def train(self) -> None:
        self.build_dataloader()
        self.build_model()
        self.build_loss()
        self.build_optimizer()
        self._init_log()

        no_improve_epochs = 0
        for epoch in range(1, int(self.args.epochs) + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_metrics = self.validate()
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            }

            self._append_log(epoch, metrics)
            self.save_checkpoint(epoch, self.save_dir / "last.pth", metrics)

            if int(self.args.save_period) > 0 and epoch % int(self.args.save_period) == 0:
                self.save_checkpoint(epoch, self.save_dir / f"epoch_{epoch}.pth", metrics)

            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.best_epoch = epoch
                no_improve_epochs = 0
                self.save_checkpoint(epoch, self.save_dir / "best.pth", metrics)
            else:
                no_improve_epochs += 1

            print(
                "epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
                "val_loss={val_loss:.6f} val_acc={val_acc:.6f} "
                "macro_precision={macro_precision:.6f} macro_recall={macro_recall:.6f} "
                "macro_f1={macro_f1:.6f}".format(epoch=epoch, **metrics)
            )

            if no_improve_epochs >= int(self.args.patience):
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch}, best val_acc: {self.best_val_acc:.6f}."
                )
                break

    def _init_log(self) -> None:
        with self.log_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._log_fieldnames())
            writer.writeheader()

    def _append_log(self, epoch: int, metrics: dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics}
        with self.log_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._log_fieldnames())
            writer.writerow(row)

    @staticmethod
    def _log_fieldnames() -> list[str]:
        return [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]

    @staticmethod
    def _macro_metrics(confusion: Tensor) -> tuple[float, float, float]:
        confusion = confusion.float()
        tp = confusion.diag()
        precision = tp / confusion.sum(dim=0).clamp_min(1.0)
        recall = tp / confusion.sum(dim=1).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return (
            float(precision.mean().item()),
            float(recall.mean().item()),
            float(f1.mean().item()),
        )
