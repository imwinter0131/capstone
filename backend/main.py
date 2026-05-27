from datetime import datetime
import csv
import importlib.util
import json
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
import uuid
import zipfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, inspect, text
from sqlalchemy.orm import Session

import database
import models


app = FastAPI(title="DLOps Backend")


# React 개발 서버에서 오는 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 서버 시작 시 DB 테이블 생성
models.Base.metadata.create_all(bind=database.engine)


# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

BUILTIN_TRAINING_MODELS = [
    {"id": "yolov8n.pt", "name": "YOLOv8 Nano", "task": "detect", "description": "가볍고 빠른 YOLO 탐지 모델", "source": "builtin", "family": "YOLO"},
    {"id": "yolov8s.pt", "name": "YOLOv8 Small", "task": "detect", "description": "속도와 정확도 균형형 YOLO 탐지 모델", "source": "builtin", "family": "YOLO"},
    {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "task": "detect", "description": "정확도 우선 YOLO 탐지 모델", "source": "builtin", "family": "YOLO"},
    {"id": "yolov8l.pt", "name": "YOLOv8 Large", "task": "detect", "description": "대형 객체 탐지 실험용 YOLO 모델", "source": "builtin", "family": "YOLO"},
    {"id": "yolov8x.pt", "name": "YOLOv8 XLarge", "task": "detect", "description": "가장 큰 YOLOv8 탐지 모델", "source": "builtin", "family": "YOLO"},
    {"id": "convnext_tiny", "name": "ConvNeXt Tiny", "task": "classify", "description": "가볍고 빠른 ConvNeXt 분류 모델", "source": "builtin", "family": "ConvNeXt"},
    {"id": "convnext_small", "name": "ConvNeXt Small", "task": "classify", "description": "속도와 정확도 균형형 ConvNeXt 분류 모델", "source": "builtin", "family": "ConvNeXt"},
    {"id": "convnext_base", "name": "ConvNeXt Base", "task": "classify", "description": "정확도 우선 ConvNeXt 분류 모델", "source": "builtin", "family": "ConvNeXt"},
]

ALLOWED_OPTIMIZERS = {"SGD", "Adam", "AdamW"}
ALLOWED_TASKS = {"detect", "classify"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt"}
ALLOWED_NORMALIZATIONS = {"none", "zero_one", "imagenet"}
DEFAULT_AUGMENTATIONS = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_degrees": 0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "mosaic": True,
    "mixup": 0.0,
}

TRAINING_THREADS: Dict[int, threading.Thread] = {}
TRAINING_STOP_EVENTS: Dict[int, threading.Event] = {}
TRAINING_PROCESSES: Dict[int, subprocess.Popen] = {}
STOPPING_STALE_SECONDS = 8.0


class UserAuth(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class ProjectCreate(BaseModel):
    user_id: int
    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = None
    folder_path: Optional[str] = None


class ProjectUpdate(BaseModel):
    user_id: int
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = None
    folder_path: Optional[str] = None


class PreprocessingPipelinePayload(BaseModel):
    user_id: int
    dataset_id: Optional[int] = None
    name: str = Field(min_length=1, max_length=160)
    description: Optional[str] = ""
    task_type: str = "detect"
    image_size: int = 224
    keep_aspect_ratio: bool = True
    normalize: str = "zero_one"
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0
    augmentations: Dict[str, Any] = Field(default_factory=dict)


class TrainingJobCreate(BaseModel):
    user_id: int
    name: str = Field(min_length=1, max_length=160)
    description: Optional[str] = ""
    task_type: str = "detect"
    dataset_id: int
    preprocessing_pipeline_id: Optional[int] = None
    yolo_model: str = "yolov8n.pt"
    optimizer: str = "AdamW"
    image_size: int = 640
    epochs: int = 10
    batch_min: int = 16
    batch_max: int = 64
    lr_initial_min: float = 0.0001
    lr_initial_max: float = 0.01
    momentum_min: float = 0.8
    momentum_max: float = 0.99


def ensure_database_schema():
    inspector = inspect(database.engine)

    def add_column_if_missing(table_name: str, column_name: str, ddl: str):
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        if column_name not in existing_columns:
            with database.engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))

    add_column_if_missing("datasets", "description", "description TEXT")
    add_column_if_missing("datasets", "task_type", "task_type VARCHAR(30) NOT NULL DEFAULT 'detect'")
    add_column_if_missing("datasets", "status", "status VARCHAR(40) NOT NULL DEFAULT 'READY'")
    add_column_if_missing("datasets", "updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL")
    add_column_if_missing("training_jobs", "preprocessing_pipeline_id", "preprocessing_pipeline_id INTEGER")
    add_column_if_missing("training_jobs", "epochs", "epochs INTEGER NOT NULL DEFAULT 10")
    add_column_if_missing("training_jobs", "progress", "progress DOUBLE PRECISION NOT NULL DEFAULT 0")
    add_column_if_missing("training_jobs", "current_epoch", "current_epoch INTEGER NOT NULL DEFAULT 0")
    add_column_if_missing("training_jobs", "logs_json", "logs_json TEXT NOT NULL DEFAULT '[]'")
    add_column_if_missing("training_jobs", "result_json", "result_json TEXT NOT NULL DEFAULT '{}'")
    add_column_if_missing("training_jobs", "run_dir", "run_dir VARCHAR(700)")
    add_column_if_missing("training_jobs", "stop_requested", "stop_requested BOOLEAN NOT NULL DEFAULT false")
    add_column_if_missing("training_jobs", "started_at", "started_at TIMESTAMP")
    add_column_if_missing("training_jobs", "completed_at", "completed_at TIMESTAMP")


ensure_database_schema()


def serialize_project(project: models.Project):
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description or "",
        "folder_path": project.folder_path or "",
        "user_id": project.user_id,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }


def serialize_dataset(dataset: models.Dataset):
    try:
        report = json.loads(dataset.report_json or "{}")
    except json.JSONDecodeError:
        report = {}

    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "user_id": dataset.user_id,
        "name": dataset.name,
        "description": dataset.description or "",
        "task_type": dataset.task_type,
        "original_filename": dataset.original_filename,
        "zip_path": dataset.zip_path,
        "file_size": dataset.file_size,
        "report": report,
        "status": dataset.status,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
    }


def clamp_epoch_value(value: Optional[int], total_epochs: Optional[int]):
    total = max(1, int(total_epochs or 1))
    if value is None:
        return 0
    return max(0, min(total, int(value)))


def clamp_progress_value(value: Optional[float], status: Optional[str] = None):
    if value is None:
        return 0
    progress = max(0, min(100, float(value)))
    if (status or "").upper() in {"QUEUED", "RUNNING", "STOPPING"} and progress >= 100:
        return 99
    return progress


def serialize_training_job(
    job: models.TrainingJob,
    dataset: Optional[models.Dataset] = None,
    pipeline: Optional[models.PreprocessingPipeline] = None,
):
    try:
        logs = json.loads(job.logs_json or "[]")
    except json.JSONDecodeError:
        logs = []

    try:
        result = json.loads(job.result_json or "{}")
    except json.JSONDecodeError:
        result = {}

    if "ensure_result_metrics" in globals():
        metrics, epoch_metrics = ensure_result_metrics(job, result)
        result = {
            **result,
            "metrics": metrics,
            "epoch_metrics": epoch_metrics,
        }
    if "build_artifacts" in globals():
        result["artifacts"] = build_artifacts(job, result)
    if "get_primary_score" in globals():
        result["primary_score"] = get_primary_score(job, result.get("metrics") or {})
    if "duration_seconds" in globals():
        result["duration_seconds"] = duration_seconds(job)

    display_epoch = clamp_epoch_value(job.current_epoch, job.epochs)
    display_progress = clamp_progress_value(job.progress, job.status)

    data = {
        "id": job.id,
        "project_id": job.project_id,
        "dataset_id": job.dataset_id,
        "preprocessing_pipeline_id": job.preprocessing_pipeline_id,
        "user_id": job.user_id,
        "name": job.name,
        "description": job.description or "",
        "task_type": job.task_type,
        "yolo_model": job.yolo_model,
        "optimizer": job.optimizer,
        "image_size": job.image_size,
        "epochs": job.epochs,
        "batch_min": job.batch_min,
        "batch_max": job.batch_max,
        "lr_initial_min": job.lr_initial_min,
        "lr_initial_max": job.lr_initial_max,
        "momentum_min": job.momentum_min,
        "momentum_max": job.momentum_max,
        "status": job.status,
        "progress": display_progress,
        "current_epoch": display_epoch,
        "logs": logs,
        "result": result,
        "run_dir": job.run_dir or "",
        "stop_requested": job.stop_requested,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }
    if dataset:
        data["dataset"] = serialize_dataset(dataset)
    if pipeline:
        data["preprocessing_pipeline"] = serialize_preprocessing_pipeline(pipeline)
    return data


def serialize_training_model(model: models.TrainingModel):
    return {
        "id": model.model_key,
        "database_id": model.id,
        "name": model.name,
        "task": model.task_type,
        "description": model.description or "",
        "source": model.source,
        "original_filename": model.original_filename,
        "model_path": model.model_path,
        "file_size": model.file_size,
        "status": model.status,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "updated_at": model.updated_at.isoformat() if model.updated_at else None,
    }


def serialize_preprocessing_pipeline(pipeline: models.PreprocessingPipeline):
    try:
        config = json.loads(pipeline.config_json or "{}")
    except json.JSONDecodeError:
        config = {}

    return {
        "id": pipeline.id,
        "project_id": pipeline.project_id,
        "user_id": pipeline.user_id,
        "name": pipeline.name,
        "description": pipeline.description or "",
        "task_type": pipeline.task_type,
        "image_size": pipeline.image_size,
        "train_split": pipeline.train_split,
        "val_split": pipeline.val_split,
        "test_split": pipeline.test_split,
        "config": config,
        "source": config.get("source", "manual"),
        "dataset_id": config.get("dataset_id"),
        "status": pipeline.status,
        "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
        "updated_at": pipeline.updated_at.isoformat() if pipeline.updated_at else None,
    }


def make_safe_folder_name(name: str):
    safe = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", name.strip())
    safe = safe.strip("_")
    return safe or "project"


def make_default_folder_path(user_id: int, project_name: str):
    folder_name = make_safe_folder_name(project_name)
    return f"C:/DLOps/projects/user_{user_id}/{folder_name}"


def list_available_models(db: Session, user_id: Optional[int] = None, project_id: Optional[int] = None):
    models_list = [dict(item) for item in BUILTIN_TRAINING_MODELS]
    if user_id is None:
        return models_list

    query = db.query(models.TrainingModel).filter(models.TrainingModel.user_id == user_id)
    if project_id is not None:
        query = query.filter(models.TrainingModel.project_id == project_id)

    custom_models = query.order_by(models.TrainingModel.updated_at.desc()).all()
    return models_list + [serialize_training_model(item) for item in custom_models]


def get_training_model_info(
    db: Session,
    model_id: str,
    task_type: str,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
):
    for item in BUILTIN_TRAINING_MODELS:
        if item["id"] == model_id and item["task"] == task_type:
            return {**item, "model_path": item["id"]}

    if user_id is None or project_id is None:
        return None

    custom_model = (
        db.query(models.TrainingModel)
        .filter(
            models.TrainingModel.model_key == model_id,
            models.TrainingModel.user_id == user_id,
            models.TrainingModel.project_id == project_id,
            models.TrainingModel.task_type == task_type,
        )
        .first()
    )
    if custom_model:
        return serialize_training_model(custom_model)
    return None


def model_supports_task(
    db: Session,
    model_id: str,
    task_type: str,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
):
    return get_training_model_info(db, model_id, task_type, user_id, project_id) is not None


def normalize_bool(value: Any):
    return bool(value)


def normalize_float(value: Any, default: float):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_preprocessing_config(payload: PreprocessingPipelinePayload):
    task_type = payload.task_type.strip().lower()
    if task_type not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="task_type은 detect 또는 classify만 가능합니다.")

    normalize = payload.normalize.strip().lower()
    if normalize not in ALLOWED_NORMALIZATIONS:
        raise HTTPException(status_code=400, detail="normalize 값이 올바르지 않습니다.")

    if payload.image_size < 64 or payload.image_size > 2048:
        raise HTTPException(status_code=400, detail="Image Size는 64 이상 2048 이하로 입력하세요.")

    splits = [payload.train_split, payload.val_split, payload.test_split]
    if any(value < 0 or value > 1 for value in splits):
        raise HTTPException(status_code=400, detail="데이터 분할 비율은 0 이상 1 이하로 입력하세요.")
    if abs(sum(splits) - 1.0) > 0.001:
        raise HTTPException(status_code=400, detail="Train/Val/Test 비율의 합은 1.0이어야 합니다.")

    raw_aug = payload.augmentations or {}
    augmentations = {
        "horizontal_flip": normalize_bool(raw_aug.get("horizontal_flip", DEFAULT_AUGMENTATIONS["horizontal_flip"])),
        "vertical_flip": normalize_bool(raw_aug.get("vertical_flip", DEFAULT_AUGMENTATIONS["vertical_flip"])),
        "rotation_degrees": normalize_float(
            raw_aug.get("rotation_degrees", DEFAULT_AUGMENTATIONS["rotation_degrees"]),
            DEFAULT_AUGMENTATIONS["rotation_degrees"],
        ),
        "hsv_h": normalize_float(raw_aug.get("hsv_h", DEFAULT_AUGMENTATIONS["hsv_h"]), DEFAULT_AUGMENTATIONS["hsv_h"]),
        "hsv_s": normalize_float(raw_aug.get("hsv_s", DEFAULT_AUGMENTATIONS["hsv_s"]), DEFAULT_AUGMENTATIONS["hsv_s"]),
        "hsv_v": normalize_float(raw_aug.get("hsv_v", DEFAULT_AUGMENTATIONS["hsv_v"]), DEFAULT_AUGMENTATIONS["hsv_v"]),
        "mosaic": normalize_bool(raw_aug.get("mosaic", DEFAULT_AUGMENTATIONS["mosaic"])),
        "mixup": normalize_float(raw_aug.get("mixup", DEFAULT_AUGMENTATIONS["mixup"]), DEFAULT_AUGMENTATIONS["mixup"]),
    }

    if augmentations["rotation_degrees"] < 0 or augmentations["rotation_degrees"] > 180:
        raise HTTPException(status_code=400, detail="Rotation은 0 이상 180 이하로 입력하세요.")
    for key in ["hsv_h", "hsv_s", "hsv_v", "mixup"]:
        if augmentations[key] < 0 or augmentations[key] > 1:
            raise HTTPException(status_code=400, detail=f"{key} 값은 0 이상 1 이하로 입력하세요.")

    return {
        "keep_aspect_ratio": payload.keep_aspect_ratio,
        "normalize": normalize,
        "augmentations": augmentations,
    }


def attach_dataset_reference_to_pipeline_config(
    db: Session,
    *,
    project_id: int,
    payload: PreprocessingPipelinePayload,
    config: Dict[str, Any],
):
    if not payload.dataset_id:
        return config

    task_type = payload.task_type.strip().lower()
    dataset = (
        db.query(models.Dataset)
        .filter(
            models.Dataset.id == payload.dataset_id,
            models.Dataset.project_id == project_id,
            models.Dataset.user_id == payload.user_id,
        )
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="연결할 데이터셋을 찾을 수 없습니다.")
    if dataset.task_type != task_type:
        raise HTTPException(status_code=400, detail="데이터셋과 전처리 작업 유형이 일치해야 합니다.")

    config.update(
        {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
        }
    )
    return config


def normalize_split_values(train_split: float, val_split: float, test_split: float):
    values = [max(0.0, float(train_split)), max(0.0, float(val_split)), max(0.0, float(test_split))]
    total = sum(values)
    if total <= 0:
        return 0.8, 0.2, 0.0

    train = round(values[0] / total, 3)
    val = round(values[1] / total, 3)
    test = round(max(0.0, 1.0 - train - val), 3)
    return train, val, test


def recommend_preprocessing_pipeline(report: Dict[str, Any], task_type: str, dataset_name: str = "dataset"):
    task_type = (task_type or "detect").strip().lower()
    if task_type not in ALLOWED_TASKS:
        task_type = "detect"

    image_count = int(report.get("image_count") or 0)
    class_count = int(report.get("class_count") or 0)
    train_images = int(report.get("train_images") or 0)
    val_images = int(report.get("val_images") or 0)
    test_images = int(report.get("test_images") or 0)
    split_total = train_images + val_images + test_images
    notes = []

    if split_total > 0:
        train_split, val_split, test_split = normalize_split_values(train_images, val_images, test_images)
        notes.append("Existing train/val/test folders were used for split ratios.")
    elif image_count >= 300:
        train_split, val_split, test_split = 0.8, 0.1, 0.1
        notes.append("A validation and test split were reserved because the dataset is large enough.")
    else:
        train_split, val_split, test_split = 0.8, 0.2, 0.0
        notes.append("A simple train/validation split was selected for a smaller dataset.")

    if task_type == "classify":
        image_size = 224
        normalize = "imagenet"
        augmentations = {
            **DEFAULT_AUGMENTATIONS,
            "rotation_degrees": 10 if image_count < 300 else 5,
            "mosaic": False,
            "mixup": 0.05 if image_count >= 500 and class_count >= 2 else 0.0,
        }
        notes.append("Classification uses ImageNet normalization and lightweight color/geometric augmentation.")
    else:
        image_size = 224
        normalize = "zero_one"
        augmentations = {
            **DEFAULT_AUGMENTATIONS,
            "rotation_degrees": 5 if image_count < 300 else 0,
            "mosaic": image_count >= 50,
            "mixup": 0.05 if image_count >= 500 and class_count >= 2 else 0.0,
        }
        notes.append("Detection keeps YOLO-friendly resize, HSV, and mosaic augmentation.")

    safe_name = make_safe_folder_name(dataset_name).replace("_", "-")
    return {
        "name": f"auto-{safe_name}-pipeline",
        "description": "Auto-generated from dataset analysis.",
        "task_type": task_type,
        "image_size": image_size,
        "keep_aspect_ratio": True,
        "normalize": normalize,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "augmentations": augmentations,
        "notes": notes,
    }


def make_unique_pipeline_name(db: Session, project_id: int, user_id: int, base_name: str):
    name = (base_name or "auto-pipeline").strip()
    if not name:
        name = "auto-pipeline"

    existing = {
        item[0].lower()
        for item in db.query(models.PreprocessingPipeline.name)
        .filter(
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == user_id,
        )
        .all()
    }

    if name.lower() not in existing:
        return name

    for index in range(2, 1000):
        candidate = f"{name}-{index}"
        if candidate.lower() not in existing:
            return candidate
    return f"{name}-{uuid.uuid4().hex[:6]}"


def create_recommended_preprocessing_pipeline(
    db: Session,
    *,
    project_id: int,
    user_id: int,
    dataset: models.Dataset,
    report: Dict[str, Any],
):
    recommendation = recommend_preprocessing_pipeline(report, dataset.task_type, dataset.name)
    payload = PreprocessingPipelinePayload(
        user_id=user_id,
        name=make_unique_pipeline_name(db, project_id, user_id, recommendation["name"]),
        description=f"{recommendation['description']} Source dataset: {dataset.name}",
        task_type=recommendation["task_type"],
        image_size=recommendation["image_size"],
        keep_aspect_ratio=recommendation["keep_aspect_ratio"],
        normalize=recommendation["normalize"],
        train_split=recommendation["train_split"],
        val_split=recommendation["val_split"],
        test_split=recommendation["test_split"],
        augmentations=recommendation["augmentations"],
    )
    config = attach_dataset_reference_to_pipeline_config(
        db,
        project_id=project_id,
        payload=payload,
        config=build_preprocessing_config(payload),
    )
    config.update(
        {
            "source": "auto",
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "recommendation_notes": recommendation["notes"],
            "report_summary": {
                "image_count": report.get("image_count", 0),
                "label_count": report.get("label_count", 0),
                "class_count": report.get("class_count", 0),
                "train_images": report.get("train_images", 0),
                "val_images": report.get("val_images", 0),
                "test_images": report.get("test_images", 0),
            },
        }
    )

    pipeline = models.PreprocessingPipeline(
        project_id=project_id,
        user_id=user_id,
        name=payload.name,
        description=payload.description,
        task_type=payload.task_type,
        image_size=payload.image_size,
        train_split=payload.train_split,
        val_split=payload.val_split,
        test_split=payload.test_split,
        config_json=json.dumps(config, ensure_ascii=False),
        status="READY",
    )
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    return pipeline


def make_dataset_report(zip_path: Path):
    image_count = 0
    label_count = 0
    train_images = 0
    val_images = 0
    test_images = 0
    yaml_files = []
    class_names = []
    warnings = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [item for item in zf.namelist() if not item.endswith("/")]
            for name in names:
                path = Path(name)
                lower = name.lower()
                suffix = path.suffix.lower()

                if suffix in IMAGE_EXTENSIONS:
                    image_count += 1
                    if "/train/" in lower or "\\train\\" in lower:
                        train_images += 1
                    elif "/val/" in lower or "/valid/" in lower or "\\val\\" in lower or "\\valid\\" in lower:
                        val_images += 1
                    elif "/test/" in lower or "\\test\\" in lower:
                        test_images += 1

                if suffix in LABEL_EXTENSIONS and ("label" in lower or "/labels/" in lower or "\\labels\\" in lower):
                    label_count += 1

                if path.name.lower() in {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}:
                    yaml_files.append(name)
                    try:
                        text = zf.read(name).decode("utf-8", errors="ignore")
                        names_match = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
                        if names_match:
                            class_names = [
                                item.strip().strip("'\"")
                                for item in names_match.group(1).split(",")
                                if item.strip()
                            ]
                    except Exception:
                        warnings.append("data.yaml 파일을 읽는 중 오류가 발생했습니다.")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="올바른 zip 파일이 아닙니다.")

    if image_count == 0:
        warnings.append("이미지 파일을 찾지 못했습니다.")
    if not yaml_files:
        warnings.append("YOLO data.yaml 파일을 찾지 못했습니다. 학습 전 데이터셋 설정 확인이 필요합니다.")
    if label_count == 0:
        warnings.append("라벨 txt 파일을 찾지 못했습니다. 분류 데이터셋이라면 정상일 수 있습니다.")

    return {
        "total_files": len(names),
        "image_count": image_count,
        "label_count": label_count,
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "yaml_files": yaml_files,
        "class_names": class_names,
        "class_count": len(class_names),
        "warnings": warnings,
    }


def save_upload_file(upload: UploadFile, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)


def read_json_list(raw: str):
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def read_json_dict(raw: str):
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def to_float_or_none(value: Any):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_metric_value(data: Dict[str, Any], *keys: str):
    for key in keys:
        if key in data:
            value = to_float_or_none(data.get(key))
            if value is not None:
                return value
    return None


def normalize_metric_key(key: str):
    normalized = key.strip().lower()
    normalized = normalized.replace("metrics/", "")
    normalized = normalized.replace("(", "_").replace(")", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    aliases = {
        "map50_b": "map50",
        "map50_95_b": "map50_95",
        "precision_b": "precision",
        "recall_b": "recall",
        "train_acc": "train_acc",
        "val_acc": "val_acc",
        "train_accuracy": "train_acc",
        "val_accuracy": "val_acc",
        "accuracy_top1": "accuracy_top1",
        "accuracy_top5": "accuracy_top5",
        "train_box_loss": "train_loss",
        "train_cls_loss": "train_cls_loss",
        "train_dfl_loss": "train_dfl_loss",
        "val_box_loss": "val_loss",
        "val_cls_loss": "val_cls_loss",
        "val_dfl_loss": "val_dfl_loss",
    }
    return aliases.get(normalized, normalized)


def load_epoch_metrics_from_results_csv(run_dir: Optional[str]):
    if not run_dir:
        return []

    candidates = [
        Path(run_dir) / "results.csv",
        Path(run_dir) / "ultralytics" / "results.csv",
    ]
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            rows = []
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader, start=1):
                    item = {"epoch": int(to_float_or_none(row.get("epoch")) or index)}
                    for raw_key, raw_value in row.items():
                        if raw_key is None:
                            continue
                        metric_key = normalize_metric_key(raw_key)
                        metric_value = to_float_or_none(raw_value)
                        if metric_value is not None:
                            item[metric_key] = round(metric_value, 6)
                    rows.append(item)
            return rows
        except Exception:
            return []
    return []


def summarize_epoch_metrics(epoch_metrics):
    if not epoch_metrics:
        return {}
    last = epoch_metrics[-1]
    metrics = {}
    for key in [
        "precision",
        "recall",
        "map50",
        "map50_95",
        "train_acc",
        "val_acc",
        "accuracy_top1",
        "accuracy_top5",
        "train_loss",
        "val_loss",
        "val_cls_loss",
    ]:
        value = first_metric_value(last, key)
        if value is not None:
            metrics[key] = value
    return metrics


def build_simulated_result_metrics(job: models.TrainingJob):
    epochs = max(1, int(job.epochs or 1))
    task_bonus = 0.02 if job.task_type == "classify" else 0
    seed = (job.id % 7) * 0.006
    final_precision = min(0.97, 0.78 + task_bonus + seed)
    final_recall = min(0.96, 0.74 + task_bonus + seed)
    final_map50 = min(0.98, 0.82 + task_bonus + seed)
    final_map95 = min(0.9, 0.58 + task_bonus + seed)
    final_acc = min(0.98, 0.84 + task_bonus + seed)

    epoch_metrics = []
    for epoch in range(1, epochs + 1):
        ratio = epoch / epochs
        train_loss = round(1.3 - ratio * 0.78 + seed, 4)
        val_loss = round(1.45 - ratio * 0.72 + seed, 4)
        item = {
            "epoch": epoch,
            "train_loss": max(0.18, train_loss),
            "val_loss": max(0.2, val_loss),
        }
        if job.task_type == "classify":
            val_acc = round(0.48 + ratio * (final_acc - 0.48), 4)
            item.update(
                {
                    "train_acc": round(min(0.995, val_acc + 0.035), 4),
                    "val_acc": val_acc,
                    "accuracy_top1": val_acc,
                    "accuracy_top5": round(min(0.995, 0.72 + ratio * 0.24), 4),
                    "f1": round(0.46 + ratio * (final_precision - 0.46), 4),
                }
            )
        else:
            item.update(
                {
                    "precision": round(0.42 + ratio * (final_precision - 0.42), 4),
                    "recall": round(0.4 + ratio * (final_recall - 0.4), 4),
                    "map50": round(0.5 + ratio * (final_map50 - 0.5), 4),
                    "map50_95": round(0.28 + ratio * (final_map95 - 0.28), 4),
                }
            )
        epoch_metrics.append(item)

    metrics = summarize_epoch_metrics(epoch_metrics)
    if job.task_type == "detect":
        metrics["f1"] = round(
            2 * metrics["precision"] * metrics["recall"] / max(metrics["precision"] + metrics["recall"], 0.0001),
            4,
        )
    return metrics, epoch_metrics


def ensure_result_metrics(job: models.TrainingJob, result: Dict[str, Any]):
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    epoch_metrics = result.get("epoch_metrics") if isinstance(result.get("epoch_metrics"), list) else []

    if not epoch_metrics:
        epoch_metrics = load_epoch_metrics_from_results_csv(result.get("run_dir") or job.run_dir)
    if not metrics:
        metrics = summarize_epoch_metrics(epoch_metrics)
    if not metrics and result.get("mode") == "simulated":
        metrics, epoch_metrics = build_simulated_result_metrics(job)
    if job.task_type == "classify":
        if "val_acc" not in metrics and metrics.get("accuracy_top1") is not None:
            metrics["val_acc"] = metrics["accuracy_top1"]
        for row in epoch_metrics:
            if isinstance(row, dict) and "val_acc" not in row and row.get("accuracy_top1") is not None:
                row["val_acc"] = row["accuracy_top1"]

    return metrics, epoch_metrics


def get_primary_score(job: models.TrainingJob, metrics: Dict[str, Any]):
    if job.status != "COMPLETED":
        return None
    if job.task_type == "classify":
        return first_metric_value(metrics, "val_acc", "accuracy_top1", "f1", "accuracy_top5")
    return first_metric_value(metrics, "map50_95", "map50", "f1", "precision")


def build_artifacts(job: models.TrainingJob, result: Dict[str, Any]):
    run_dir = result.get("run_dir") or job.run_dir or ""
    artifacts = {
        "run_dir": run_dir,
        "best_model_path": result.get("best_model_path", ""),
        "last_model_path": result.get("last_model_path", ""),
        "results_csv": "",
        "confusion_matrix": "",
        "results_plot": "",
        "labels": "",
        "train_batch": "",
        "val_batch": "",
    }

    def pick_first(run_path: Path, names: List[str]):
        for name in names:
            for candidate in [run_path / name, run_path / "ultralytics" / name]:
                if candidate.exists():
                    return str(candidate)
        return ""

    def pick_glob(run_path: Path, patterns: List[str]):
        for pattern in patterns:
            for root in [run_path, run_path / "ultralytics"]:
                matches = sorted(root.glob(pattern)) if root.exists() else []
                if matches:
                    return str(matches[0])
        return ""

    if run_dir:
        run_path = Path(run_dir)
        artifacts["results_csv"] = pick_first(run_path, ["results.csv"])
        artifacts["confusion_matrix"] = pick_first(run_path, ["confusion_matrix.png", "confusion_matrix_normalized.png"])
        artifacts["results_plot"] = pick_first(run_path, ["results.png"])
        artifacts["labels"] = pick_first(run_path, ["labels.jpg", "labels_correlogram.jpg"])
        artifacts["train_batch"] = pick_glob(run_path, ["train_batch*.jpg", "train_batch*.png"])
        artifacts["val_batch"] = pick_glob(run_path, ["val_batch*.jpg", "val_batch*.png"])
    artifacts["exists"] = {
        key: bool(value and Path(value).exists())
        for key, value in artifacts.items()
        if key != "exists"
    }
    return artifacts


def duration_seconds(job: models.TrainingJob):
    if not job.started_at:
        return None
    end = job.completed_at or datetime.utcnow()
    return max(0, int((end - job.started_at).total_seconds()))


def serialize_result_run(
    job: models.TrainingJob,
    dataset: Optional[models.Dataset] = None,
    pipeline: Optional[models.PreprocessingPipeline] = None,
    include_logs: bool = False,
):
    result = read_json_dict(job.result_json)
    metrics, epoch_metrics = ensure_result_metrics(job, result)
    artifacts = build_artifacts(job, result)
    primary_score = get_primary_score(job, metrics)
    payload = serialize_training_job(job, dataset, pipeline)
    payload.update(
        {
            "metrics": metrics,
            "epoch_metrics": epoch_metrics,
            "artifacts": artifacts,
            "primary_score": primary_score,
            "duration_seconds": duration_seconds(job),
            "is_finished": job.status in {"COMPLETED", "FAILED", "STOPPED"},
        }
    )
    if not include_logs:
        payload["logs"] = payload.get("logs", [])[-5:]
    return payload


def append_training_log(db: Session, job: models.TrainingJob, message: str):
    logs = read_json_list(job.logs_json)
    logs.append(
        {
            "time": datetime.utcnow().isoformat(),
            "message": message,
        }
    )
    job.logs_json = json.dumps(logs, ensure_ascii=False)
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)


def update_training_progress(
    db: Session,
    job: models.TrainingJob,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    current_epoch: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
):
    next_status = status or job.status
    if status is not None:
        job.status = status
    if current_epoch is not None:
        next_epoch = clamp_epoch_value(current_epoch, job.epochs)
        if next_status in {"QUEUED", "RUNNING", "STOPPING", "STOPPED"}:
            next_epoch = max(clamp_epoch_value(job.current_epoch, job.epochs), next_epoch)
        job.current_epoch = next_epoch
    if progress is not None:
        next_progress = clamp_progress_value(progress, next_status)
        if next_status in {"QUEUED", "RUNNING", "STOPPING", "STOPPED"}:
            next_progress = max(clamp_progress_value(job.progress, job.status), next_progress)
        if next_status == "COMPLETED":
            next_progress = 100
        job.progress = next_progress
    if result is not None:
        job.result_json = json.dumps(result, ensure_ascii=False)
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)


def is_training_process_alive(job_id: int):
    process = TRAINING_PROCESSES.get(job_id)
    return bool(process and process.poll() is None)


def is_training_thread_alive(job_id: int):
    thread = TRAINING_THREADS.get(job_id)
    return bool(thread and thread.is_alive())


def stop_training_process(job_id: int, *, wait_timeout: float = 1.0):
    process = TRAINING_PROCESSES.get(job_id)
    if not process or process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass


def mark_training_stopped(db: Session, job: models.TrainingJob, *, message: Optional[str] = None):
    result = read_json_dict(job.result_json)
    result.setdefault("message", "사용자 요청으로 학습이 중지되었습니다.")
    update_training_progress(
        db,
        job,
        status="STOPPED",
        progress=job.progress,
        current_epoch=job.current_epoch,
        result=result,
    )
    job.stop_requested = False
    job.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    if message:
        append_training_log(db, job, message)


def reconcile_stopping_training_job(db: Session, job: models.TrainingJob, *, force: bool = False):
    if job.status != "STOPPING":
        return

    stop_event = TRAINING_STOP_EVENTS.get(job.id)
    if stop_event:
        stop_event.set()

    updated_at = job.updated_at or job.started_at or datetime.utcnow()
    stopping_seconds = max(0.0, (datetime.utcnow() - updated_at).total_seconds())
    should_force = force or stopping_seconds >= STOPPING_STALE_SECONDS

    if should_force and is_training_process_alive(job.id):
        stop_training_process(job.id, wait_timeout=1.0)

    if should_force and is_training_thread_alive(job.id):
        thread = TRAINING_THREADS.get(job.id)
        if thread:
            thread.join(timeout=0.5)

    process_alive = is_training_process_alive(job.id)
    thread_alive = is_training_thread_alive(job.id)
    if not process_alive and (not thread_alive or should_force):
        mark_training_stopped(db, job, message="학습 중지 상태를 정리했습니다.")


def find_first_dataset_config(dataset_dir: Path):
    candidates = []
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        candidates.extend(dataset_dir.rglob(name))
    return candidates[0] if candidates else None


def extract_dataset_zip(zip_path: Path, target_dir: Path):
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def write_training_runner(run_dir: Path, job: models.TrainingJob, model_path: str, data_target: str):
    runner_path = run_dir / "run_ultralytics_training.py"
    task_method = "train" if job.task_type == "detect" else "train"
    runner_path.write_text(
        "\n".join(
            [
                "from ultralytics import YOLO",
                "",
                f"TOTAL_EPOCHS = {int(job.epochs)}",
                "_last_emitted_progress = -1.0",
                "",
                "def _trainer_epoch_total(trainer):",
                "    epoch = int(getattr(trainer, 'epoch', -1)) + 1",
                "    total = int(getattr(trainer, 'epochs', TOTAL_EPOCHS) or TOTAL_EPOCHS)",
                "    return max(1, epoch), max(1, total)",
                "",
                "def _emit_progress(epoch, total, batch=0, batches=0, progress=None, force=False):",
                "    global _last_emitted_progress",
                "    if progress is None:",
                "        ratio = 1.0 if not batches else max(0.0, min(1.0, batch / batches))",
                "        progress = (((epoch - 1) + ratio) / total) * 100",
                "    progress = max(0.0, min(100.0, float(progress)))",
                "    if force or progress - _last_emitted_progress >= 0.25:",
                "        print(f'DL_PROGRESS epoch={epoch} total={total} batch={batch} batches={batches} progress={progress:.2f}', flush=True)",
                "        _last_emitted_progress = progress",
                "",
                "def emit_batch_progress(trainer):",
                "    epoch, total = _trainer_epoch_total(trainer)",
                "    batch_index = int(getattr(trainer, 'batch_i', -1)) + 1",
                "    train_loader = getattr(trainer, 'train_loader', None)",
                "    try:",
                "        batches = len(train_loader) if train_loader is not None else 0",
                "    except Exception:",
                "        batches = 0",
                "    if batch_index <= 0:",
                "        batch_index = 1",
                "    _emit_progress(epoch, total, batch_index, batches)",
                "",
                "def emit_epoch_progress(trainer):",
                "    epoch, total = _trainer_epoch_total(trainer)",
                "    _emit_progress(epoch, total, progress=(epoch / total) * 100, force=True)",
                "",
                f"model = YOLO({model_path!r})",
                "for event_name, callback in (",
                "    ('on_train_batch_end', emit_batch_progress),",
                "    ('on_fit_epoch_end', emit_epoch_progress),",
                "    ('on_train_epoch_end', emit_epoch_progress),",
                "):",
                "    try:",
                "        model.add_callback(event_name, callback)",
                "    except Exception:",
                "        pass",
                "",
                "model.train(",
                f"    data={data_target!r},",
                f"    epochs={int(job.epochs)},",
                f"    imgsz={int(job.image_size)},",
                f"    optimizer={job.optimizer!r},",
                f"    project={str(run_dir)!r},",
                f"    name='ultralytics',",
                "    exist_ok=True,",
                "    verbose=True,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    return runner_path


def write_convnext_training_runner(run_dir: Path, job: models.TrainingJob, model_id: str, data_target: str):
    runner_path = run_dir / "run_convnext_training.py"
    convnext_run_dir = run_dir / "convnext"
    runner_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import copy",
                "import csv",
                "import torch",
                "from torch import nn",
                "from torch.utils.data import DataLoader",
                "from torchvision import datasets, models, transforms",
                "",
                f"DATA_ROOT = Path({str(data_target)!r})",
                f"RUN_DIR = Path({str(convnext_run_dir)!r})",
                f"MODEL_ID = {model_id!r}",
                f"TOTAL_EPOCHS = {int(job.epochs)}",
                f"IMAGE_SIZE = {int(job.image_size)}",
                f"BATCH_SIZE = {int(job.batch_min)}",
                f"OPTIMIZER_NAME = {job.optimizer!r}",
                f"LR = {float(job.lr_initial_min)}",
                f"MOMENTUM = {float(job.momentum_min)}",
                "_last_emitted_progress = -1.0",
                "",
                "RUN_DIR.mkdir(parents=True, exist_ok=True)",
                "WEIGHTS_DIR = RUN_DIR / 'weights'",
                "WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)",
                "RESULTS_CSV = RUN_DIR / 'results.csv'",
                "",
                "def find_split(root, names):",
                "    for name in names:",
                "        candidate = root / name",
                "        if candidate.exists() and candidate.is_dir():",
                "            return candidate",
                "    return None",
                "",
                "def emit_progress(epoch, batch=0, batches=0, progress=None, force=False):",
                "    global _last_emitted_progress",
                "    if progress is None:",
                "        ratio = 1.0 if not batches else max(0.0, min(1.0, batch / batches))",
                "        progress = (((epoch - 1) + ratio) / TOTAL_EPOCHS) * 100",
                "    progress = max(0.0, min(100.0, float(progress)))",
                "    if force or progress - _last_emitted_progress >= 0.25:",
                "        print(f'DL_PROGRESS epoch={epoch} total={TOTAL_EPOCHS} batch={batch} batches={batches} progress={progress:.2f}', flush=True)",
                "        _last_emitted_progress = progress",
                "",
                "def build_model(model_id, num_classes):",
                "    builders = {",
                "        'convnext_tiny': models.convnext_tiny,",
                "        'convnext_small': models.convnext_small,",
                "        'convnext_base': models.convnext_base,",
                "    }",
                "    builder = builders.get(model_id, models.convnext_tiny)",
                "    try:",
                "        model = builder(weights=None)",
                "    except TypeError:",
                "        model = builder(pretrained=False)",
                "    in_features = model.classifier[-1].in_features",
                "    model.classifier[-1] = nn.Linear(in_features, num_classes)",
                "    if model_id not in builders and Path(model_id).exists():",
                "        checkpoint = torch.load(model_id, map_location='cpu')",
                "        state = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else None",
                "        if state is None and isinstance(checkpoint, dict):",
                "            state = checkpoint.get('state_dict') or checkpoint",
                "        if isinstance(state, dict):",
                "            state = {key.replace('module.', '', 1): value for key, value in state.items()}",
                "            current = model.state_dict()",
                "            filtered = {key: value for key, value in state.items() if key in current and tuple(current[key].shape) == tuple(value.shape)}",
                "            if filtered:",
                "                model.load_state_dict(filtered, strict=False)",
                "    return model",
                "",
                "def accuracy_topk(output, target, topk=(1, 5)):",
                "    if output.numel() == 0:",
                "        return [0.0 for _ in topk]",
                "    max_k = min(max(topk), output.size(1))",
                "    _, pred = output.topk(max_k, 1, True, True)",
                "    pred = pred.t()",
                "    correct = pred.eq(target.reshape(1, -1).expand_as(pred))",
                "    scores = []",
                "    for k in topk:",
                "        actual_k = min(k, output.size(1))",
                "        correct_k = correct[:actual_k].reshape(-1).float().sum(0).item()",
                "        scores.append(correct_k / max(1, target.size(0)))",
                "    return scores",
                "",
                "train_dir = find_split(DATA_ROOT, ['train', 'training'])",
                "val_dir = find_split(DATA_ROOT, ['val', 'valid', 'validation'])",
                "if train_dir is None:",
                "    raise RuntimeError('Classification dataset must contain a train folder with class subfolders.')",
                "if val_dir is None:",
                "    val_dir = train_dir",
                "",
                "train_transform = transforms.Compose([",
                "    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),",
                "    transforms.RandomHorizontalFlip(),",
                "    transforms.ToTensor(),",
                "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
                "])",
                "val_transform = transforms.Compose([",
                "    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),",
                "    transforms.ToTensor(),",
                "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
                "])",
                "",
                "train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)",
                "val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)",
                "num_classes = max(1, len(train_dataset.classes))",
                "batch_size = max(1, min(BATCH_SIZE, len(train_dataset)))",
                "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)",
                "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)",
                "",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
                "model = build_model(MODEL_ID, num_classes).to(device)",
                "criterion = nn.CrossEntropyLoss()",
                "if OPTIMIZER_NAME == 'SGD':",
                "    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)",
                "elif OPTIMIZER_NAME == 'Adam':",
                "    optimizer = torch.optim.Adam(model.parameters(), lr=LR)",
                "else:",
                "    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)",
                "",
                "print(f'ConvNeXt training model={MODEL_ID} classes={num_classes} train={len(train_dataset)} val={len(val_dataset)} device={device}', flush=True)",
                "best_acc = -1.0",
                "best_state = None",
                "fieldnames = ['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'accuracy_top1', 'accuracy_top5']",
                "rows = []",
                "",
                "for epoch in range(1, TOTAL_EPOCHS + 1):",
                "    model.train()",
                "    train_loss_sum = 0.0",
                "    train_items = 0",
                "    train_acc_sum = 0.0",
                "    batches = max(1, len(train_loader))",
                "    for batch_index, (images, targets) in enumerate(train_loader, start=1):",
                "        images = images.to(device)",
                "        targets = targets.to(device)",
                "        optimizer.zero_grad(set_to_none=True)",
                "        outputs = model(images)",
                "        loss = criterion(outputs, targets)",
                "        loss.backward()",
                "        optimizer.step()",
                "        train_top1, _ = accuracy_topk(outputs.detach(), targets)",
                "        train_loss_sum += loss.item() * images.size(0)",
                "        train_items += images.size(0)",
                "        train_acc_sum += train_top1 * images.size(0)",
                "        emit_progress(epoch, batch_index, batches)",
                "",
                "    model.eval()",
                "    val_loss_sum = 0.0",
                "    val_items = 0",
                "    top1_sum = 0.0",
                "    top5_sum = 0.0",
                "    with torch.no_grad():",
                "        for images, targets in val_loader:",
                "            images = images.to(device)",
                "            targets = targets.to(device)",
                "            outputs = model(images)",
                "            loss = criterion(outputs, targets)",
                "            top1, top5 = accuracy_topk(outputs, targets)",
                "            val_loss_sum += loss.item() * images.size(0)",
                "            val_items += images.size(0)",
                "            top1_sum += top1 * images.size(0)",
                "            top5_sum += top5 * images.size(0)",
                "",
                "    train_loss = train_loss_sum / max(1, train_items)",
                "    train_acc = train_acc_sum / max(1, train_items)",
                "    val_loss = val_loss_sum / max(1, val_items)",
                "    top1 = top1_sum / max(1, val_items)",
                "    top5 = top5_sum / max(1, val_items)",
                "    row = {'epoch': epoch, 'train_acc': round(train_acc, 6), 'val_acc': round(top1, 6), 'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6), 'accuracy_top1': round(top1, 6), 'accuracy_top5': round(top5, 6)}",
                "    rows.append(row)",
                "    with RESULTS_CSV.open('w', encoding='utf-8', newline='') as handle:",
                "        writer = csv.DictWriter(handle, fieldnames=fieldnames)",
                "        writer.writeheader()",
                "        writer.writerows(rows)",
                "",
                "    checkpoint = {'model_id': MODEL_ID, 'classes': train_dataset.classes, 'epoch': epoch, 'model_state_dict': model.state_dict()}",
                "    torch.save(checkpoint, WEIGHTS_DIR / 'last.pt')",
                "    if top1 >= best_acc:",
                "        best_acc = top1",
                "        best_state = copy.deepcopy(checkpoint)",
                "        torch.save(best_state, WEIGHTS_DIR / 'best.pt')",
                "",
                "    print(f'Epoch {epoch}/{TOTAL_EPOCHS} train_acc={train_acc:.4f} val_acc={top1:.4f} train_loss={train_loss:.4f} val_loss={val_loss:.4f}', flush=True)",
                "    emit_progress(epoch, progress=(epoch / TOTAL_EPOCHS) * 100, force=True)",
                "",
                "try:",
                "    import matplotlib.pyplot as plt",
                "    epochs = [item['epoch'] for item in rows]",
                "    plt.figure(figsize=(8, 4))",
                "    plt.plot(epochs, [item['val_acc'] for item in rows], label='Val Acc')",
                "    plt.plot(epochs, [item['val_loss'] for item in rows], label='Val Loss')",
                "    plt.xlabel('Epoch')",
                "    plt.legend()",
                "    plt.tight_layout()",
                "    plt.savefig(RUN_DIR / 'results.png')",
                "    plt.close()",
                "except Exception:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    return runner_path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_training_output(raw: str):
    text = ANSI_ESCAPE_RE.sub("", raw or "")
    return text.replace("\x00", "").strip()


def enqueue_training_output(pipe, output_queue: Queue):
    buffer = []
    try:
        while True:
            char = pipe.read(1)
            if not char:
                break
            if char in {"\n", "\r"}:
                text = clean_training_output("".join(buffer))
                if text:
                    output_queue.put(text)
                buffer = []
            else:
                buffer.append(char)
        text = clean_training_output("".join(buffer))
        if text:
            output_queue.put(text)
    finally:
        output_queue.put(None)


def parse_tqdm_training_progress(line: str, total_epochs: int, fallback_epoch: int = 0):
    total = max(1, int(total_epochs or 1))

    marker = re.search(
        r"DL_PROGRESS\s+epoch=(\d+)\s+total=(\d+)"
        r"(?:\s+batch=(\d+)\s+batches=(\d+))?"
        r"(?:\s+progress=(\d+(?:\.\d+)?))?",
        line,
    )
    if marker:
        marker_total = max(1, int(marker.group(2)))
        if marker_total == total:
            epoch = clamp_epoch_value(int(marker.group(1)), total)
            explicit_progress = to_float_or_none(marker.group(5))
            if explicit_progress is not None:
                progress = max(0, min(99, explicit_progress))
            else:
                batches = max(0, int(marker.group(4) or 0))
                batch = max(0, int(marker.group(3) or 0))
                batch_ratio = max(0.0, min(1.0, batch / batches)) if batches else 1.0
                progress = ((max(0, epoch - 1) + batch_ratio) / total) * 100
            return {
                "epoch": epoch,
                "progress": round(min(99, progress), 2),
                "log": f"Epoch {epoch}/{total} 진행률 {round(min(100, progress), 1)}%",
            }

    current_epoch = None
    epoch_match = re.search(r"^\s*(\d+)\s*/\s*(\d+)\b", line)
    if epoch_match and max(1, int(epoch_match.group(2))) == total:
        current_epoch = clamp_epoch_value(int(epoch_match.group(1)), total)
    elif "|" in line and "%" in line and fallback_epoch:
        current_epoch = clamp_epoch_value(fallback_epoch, total)

    if current_epoch is None:
        return None

    batch_ratio = 0.0
    percent_match = re.search(r"(\d+(?:\.\d+)?)%\|", line)
    if percent_match:
        batch_ratio = max(0.0, min(1.0, float(percent_match.group(1)) / 100))
    else:
        fraction_match = re.search(r"\|\s*(\d+)\s*/\s*(\d+)\s*(?:\[|$)", line)
        if fraction_match:
            current_step = max(0, int(fraction_match.group(1)))
            total_steps = max(1, int(fraction_match.group(2)))
            batch_ratio = max(0.0, min(1.0, current_step / total_steps))
        elif current_epoch > fallback_epoch:
            batch_ratio = 0.02

    epoch_base = max(0, current_epoch - 1)
    progress = ((epoch_base + batch_ratio) / total) * 100
    if batch_ratio >= 0.999:
        progress = (current_epoch / total) * 100

    return {
        "epoch": current_epoch,
        "progress": min(99, round(progress, 2)),
        "log": f"Epoch {current_epoch}/{total} 진행률 {round(min(100, progress), 1)}%",
    }


def run_simulated_training(db: Session, job: models.TrainingJob, stop_event: threading.Event):
    append_training_log(db, job, "개발 환경 학습 시뮬레이션을 시작합니다.")
    epochs = max(1, int(job.epochs or 1))
    for epoch in range(1, epochs + 1):
        steps_per_epoch = 6
        for step in range(1, steps_per_epoch + 1):
            db.refresh(job)
            if stop_event.is_set() or job.stop_requested:
                mark_training_stopped(db, job, message="사용자 요청으로 학습을 중지했습니다.")
                return

            time.sleep(0.08)
            progress = round((((epoch - 1) + (step / steps_per_epoch)) / epochs) * 100, 2)
            update_training_progress(db, job, status="RUNNING", progress=progress, current_epoch=epoch)
        append_training_log(db, job, f"Epoch {epoch}/{epochs} 완료")

    result = {
        "mode": "simulated",
        "run_dir": job.run_dir,
        "best_model_path": str(Path(job.run_dir or "") / "weights" / "best.pt"),
        "last_model_path": str(Path(job.run_dir or "") / "weights" / "last.pt"),
        "message": "실제 학습 실행 환경이 없거나 시뮬레이션으로 실행되어 결과 파일은 생성되지 않았습니다.",
    }
    metrics, epoch_metrics = build_simulated_result_metrics(job)
    result["metrics"] = metrics
    result["epoch_metrics"] = epoch_metrics
    update_training_progress(db, job, status="COMPLETED", progress=100, current_epoch=epochs, result=result)
    job.completed_at = datetime.utcnow()
    db.commit()
    append_training_log(db, job, "학습이 완료되었습니다.")


def run_ultralytics_training(
    db: Session,
    job: models.TrainingJob,
    model_path: str,
    data_target: str,
    stop_event: threading.Event,
):
    run_dir = Path(job.run_dir)
    runner_path = write_training_runner(run_dir, job, model_path, data_target)
    append_training_log(db, job, "ultralytics 기반 YOLO 학습 프로세스를 시작합니다.")

    process = subprocess.Popen(
        [sys.executable, "-u", str(runner_path)],
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    TRAINING_PROCESSES[job.id] = process
    output_queue = Queue()
    reader_thread = threading.Thread(
        target=enqueue_training_output,
        args=(process.stdout, output_queue),
        daemon=True,
    )
    reader_thread.start()

    total_epochs = max(1, int(job.epochs or 1))
    last_progress = max(1, float(job.progress or 0))
    last_epoch = clamp_epoch_value(job.current_epoch, total_epochs)
    last_progress_log = -1.0
    last_progress_log_epoch = -1
    last_progress_update_at = 0.0
    reader_done = False
    terminate_requested = False
    update_training_progress(db, job, status="RUNNING", progress=last_progress)

    try:
        while True:
            if stop_event.is_set() and not terminate_requested:
                stop_training_process(job.id, wait_timeout=0.5)
                terminate_requested = True
                append_training_log(db, job, "학습 프로세스 종료 요청을 보냈습니다.")

            line = ""
            try:
                item = output_queue.get(timeout=0.2)
                if item is None:
                    reader_done = True
                else:
                    line = item
            except Empty:
                pass

            if line:
                progress_info = parse_tqdm_training_progress(line, total_epochs, last_epoch)
                if progress_info:
                    current_epoch = max(last_epoch, progress_info["epoch"])
                    next_progress = max(last_progress, progress_info["progress"])
                    now = time.monotonic()

                    should_update = (
                        current_epoch > last_epoch
                        or next_progress - last_progress >= 0.35
                        or now - last_progress_update_at >= 1.0
                    )
                    if should_update:
                        last_epoch = current_epoch
                        last_progress = next_progress
                        last_progress_update_at = now
                        update_training_progress(
                            db,
                            job,
                            status="RUNNING",
                            progress=last_progress,
                            current_epoch=last_epoch,
                        )

                    should_log_progress = (
                        current_epoch > last_progress_log_epoch
                        or last_progress - last_progress_log >= 5
                        or last_progress >= 99
                    )
                    if should_log_progress:
                        append_training_log(db, job, progress_info["log"])
                        last_progress_log = last_progress
                        last_progress_log_epoch = current_epoch
                else:
                    append_training_log(db, job, line)

            if process.poll() is not None:
                if reader_done or output_queue.empty():
                    break

            if not line:
                db.refresh(job)
                if job.stop_requested:
                    stop_event.set()
                    stop_training_process(job.id, wait_timeout=0.5)

        return_code = process.wait()
        if stop_event.is_set():
            mark_training_stopped(db, job, message="학습이 중지되었습니다.")
            return

        weights_dir = run_dir / "ultralytics" / "weights"
        result = {
            "mode": "ultralytics",
            "run_dir": str(run_dir / "ultralytics"),
            "best_model_path": str(weights_dir / "best.pt"),
            "last_model_path": str(weights_dir / "last.pt"),
            "return_code": return_code,
        }
        if return_code == 0:
            epoch_metrics = load_epoch_metrics_from_results_csv(result["run_dir"])
            result["epoch_metrics"] = epoch_metrics
            result["metrics"] = summarize_epoch_metrics(epoch_metrics)
            update_training_progress(db, job, status="COMPLETED", progress=100, current_epoch=job.epochs, result=result)
            append_training_log(db, job, "YOLO 학습이 완료되었습니다.")
        else:
            result["message"] = "ultralytics 학습 프로세스가 실패했습니다."
            update_training_progress(db, job, status="FAILED", result=result)
            append_training_log(db, job, "YOLO 학습이 실패했습니다.")
    finally:
        TRAINING_PROCESSES.pop(job.id, None)
        if job.status == "STOPPING":
            mark_training_stopped(db, job, message="학습 중지 상태를 정리했습니다.")
        elif job.status in {"COMPLETED", "FAILED", "STOPPED"}:
            job.completed_at = job.completed_at or datetime.utcnow()
            db.commit()


def run_convnext_training(
    db: Session,
    job: models.TrainingJob,
    model_id: str,
    data_target: str,
    stop_event: threading.Event,
):
    run_dir = Path(job.run_dir)
    runner_path = write_convnext_training_runner(run_dir, job, model_id, data_target)
    append_training_log(db, job, "torchvision 기반 ConvNeXt 분류 학습 프로세스를 시작합니다.")

    process = subprocess.Popen(
        [sys.executable, "-u", str(runner_path)],
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    TRAINING_PROCESSES[job.id] = process
    output_queue = Queue()
    reader_thread = threading.Thread(
        target=enqueue_training_output,
        args=(process.stdout, output_queue),
        daemon=True,
    )
    reader_thread.start()

    total_epochs = max(1, int(job.epochs or 1))
    last_progress = max(1, float(job.progress or 0))
    last_epoch = clamp_epoch_value(job.current_epoch, total_epochs)
    last_progress_log = -1.0
    last_progress_log_epoch = -1
    last_progress_update_at = 0.0
    reader_done = False
    terminate_requested = False
    update_training_progress(db, job, status="RUNNING", progress=last_progress)

    try:
        while True:
            if stop_event.is_set() and not terminate_requested:
                stop_training_process(job.id, wait_timeout=0.5)
                terminate_requested = True
                append_training_log(db, job, "ConvNeXt 학습 프로세스 종료 요청을 보냈습니다.")

            line = ""
            try:
                item = output_queue.get(timeout=0.2)
                if item is None:
                    reader_done = True
                else:
                    line = item
            except Empty:
                pass

            if line:
                progress_info = parse_tqdm_training_progress(line, total_epochs, last_epoch)
                if progress_info:
                    current_epoch = max(last_epoch, progress_info["epoch"])
                    next_progress = max(last_progress, progress_info["progress"])
                    now = time.monotonic()

                    should_update = (
                        current_epoch > last_epoch
                        or next_progress - last_progress >= 0.35
                        or now - last_progress_update_at >= 1.0
                    )
                    if should_update:
                        last_epoch = current_epoch
                        last_progress = next_progress
                        last_progress_update_at = now
                        update_training_progress(
                            db,
                            job,
                            status="RUNNING",
                            progress=last_progress,
                            current_epoch=last_epoch,
                        )

                    should_log_progress = (
                        current_epoch > last_progress_log_epoch
                        or last_progress - last_progress_log >= 5
                        or last_progress >= 99
                    )
                    if should_log_progress:
                        append_training_log(db, job, progress_info["log"])
                        last_progress_log = last_progress
                        last_progress_log_epoch = current_epoch
                else:
                    append_training_log(db, job, line)

            if process.poll() is not None:
                if reader_done or output_queue.empty():
                    break

            if not line:
                db.refresh(job)
                if job.stop_requested:
                    stop_event.set()
                    stop_training_process(job.id, wait_timeout=0.5)

        return_code = process.wait()
        if stop_event.is_set():
            mark_training_stopped(db, job, message="ConvNeXt 학습이 중지되었습니다.")
            return

        weights_dir = run_dir / "convnext" / "weights"
        result = {
            "mode": "convnext",
            "run_dir": str(run_dir / "convnext"),
            "best_model_path": str(weights_dir / "best.pt"),
            "last_model_path": str(weights_dir / "last.pt"),
            "return_code": return_code,
        }
        if return_code == 0:
            epoch_metrics = load_epoch_metrics_from_results_csv(result["run_dir"])
            result["epoch_metrics"] = epoch_metrics
            result["metrics"] = summarize_epoch_metrics(epoch_metrics)
            update_training_progress(db, job, status="COMPLETED", progress=100, current_epoch=job.epochs, result=result)
            append_training_log(db, job, "ConvNeXt 분류 학습이 완료되었습니다.")
        else:
            result["message"] = "ConvNeXt 분류 학습 프로세스가 실패했습니다."
            update_training_progress(db, job, status="FAILED", result=result)
            append_training_log(db, job, "ConvNeXt 분류 학습이 실패했습니다.")
    finally:
        TRAINING_PROCESSES.pop(job.id, None)
        if job.status == "STOPPING":
            mark_training_stopped(db, job, message="학습 중지 상태를 정리했습니다.")
        elif job.status in {"COMPLETED", "FAILED", "STOPPED"}:
            job.completed_at = job.completed_at or datetime.utcnow()
            db.commit()


def training_worker(job_id: int, simulate: bool = False):
    db = database.SessionLocal()
    stop_event = TRAINING_STOP_EVENTS.setdefault(job_id, threading.Event())
    try:
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            return

        project = get_owned_project(db, job.project_id, job.user_id)
        dataset = db.query(models.Dataset).filter(models.Dataset.id == job.dataset_id).first()
        pipeline = None
        if job.preprocessing_pipeline_id:
            pipeline = db.query(models.PreprocessingPipeline).filter(models.PreprocessingPipeline.id == job.preprocessing_pipeline_id).first()

        run_dir = Path(project.folder_path) / "runs" / f"{make_safe_folder_name(job.name)}_{job.id}"
        dataset_dir = run_dir / "dataset"
        run_dir.mkdir(parents=True, exist_ok=True)

        job.run_dir = str(run_dir)
        job.status = "RUNNING"
        job.progress = 1
        job.current_epoch = 0
        job.stop_requested = False
        job.logs_json = "[]"
        job.result_json = "{}"
        job.started_at = datetime.utcnow()
        job.completed_at = None
        db.commit()
        db.refresh(job)

        append_training_log(db, job, f"학습 작업을 시작합니다: {job.name}")
        if pipeline:
            append_training_log(db, job, f"전처리 preset 적용 예정: {pipeline.name}")

        if not dataset:
            raise RuntimeError("연결된 데이터셋을 찾을 수 없습니다.")

        extract_dataset_zip(Path(dataset.zip_path), dataset_dir)
        append_training_log(db, job, f"데이터셋 압축 해제 완료: {dataset_dir}")

        data_config = find_first_dataset_config(dataset_dir)
        data_target = str(dataset_dir if job.task_type == "classify" else data_config or dataset_dir)
        if job.task_type == "detect" and not data_config:
            append_training_log(db, job, "data.yaml을 찾지 못해 실제 탐지 학습 대신 시뮬레이션으로 전환합니다.")
            simulate = True

        model_info = get_training_model_info(db, job.yolo_model, job.task_type, job.user_id, job.project_id)
        if not model_info:
            raise RuntimeError("학습 모델 정보를 찾을 수 없습니다.")
        model_path = model_info.get("model_path") or model_info["id"]

        if job.task_type == "classify":
            has_convnext_runtime = (
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("torchvision") is not None
            )
            if simulate or not has_convnext_runtime:
                if not simulate:
                    append_training_log(db, job, "torch/torchvision 패키지를 찾지 못해 시뮬레이션으로 전환합니다.")
                run_simulated_training(db, job, stop_event)
            else:
                run_convnext_training(db, job, model_path, data_target, stop_event)
        else:
            if simulate or importlib.util.find_spec("ultralytics") is None:
                if not simulate:
                    append_training_log(db, job, "ultralytics 패키지를 찾지 못해 시뮬레이션으로 전환합니다.")
                run_simulated_training(db, job, stop_event)
            else:
                run_ultralytics_training(db, job, model_path, data_target, stop_event)
    except Exception as exc:
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if job:
            result = read_json_dict(job.result_json)
            result["error"] = str(exc)
            update_training_progress(db, job, status="FAILED", result=result)
            job.completed_at = datetime.utcnow()
            db.commit()
            append_training_log(db, job, f"학습 실패: {exc}")
    finally:
        TRAINING_THREADS.pop(job_id, None)
        TRAINING_STOP_EVENTS.pop(job_id, None)
        db.close()


def get_owned_project(db: Session, project_id: int, user_id: int):
    project = (
        db.query(models.Project)
        .filter(
            models.Project.id == project_id,
            models.Project.user_id == user_id,
        )
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")

    return project


def check_project_name_duplicate(
    db: Session,
    user_id: int,
    name: str,
    exclude_project_id: Optional[int] = None,
):
    normalized_name = name.strip().lower()

    query = db.query(models.Project).filter(
        models.Project.user_id == user_id,
        func.lower(models.Project.name) == normalized_name,
    )

    if exclude_project_id is not None:
        query = query.filter(models.Project.id != exclude_project_id)

    if query.first():
        raise HTTPException(status_code=400, detail="이미 같은 이름의 프로젝트가 있습니다.")


@app.get("/")
def home():
    return {"status": "DLOps Backend is running!"}


@app.post("/signup")
def signup(user_data: UserAuth, db: Session = Depends(database.get_db)):
    email = user_data.email.lower().strip()

    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 이메일입니다.")

    hashed_pw = pwd_context.hash(user_data.password)
    new_user = models.User(email=email, hashed_password=hashed_pw)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "회원가입이 성공적으로 완료되었습니다.",
        "user_id": new_user.id,
        "email": new_user.email,
    }


@app.post("/login")
def login(user_data: UserAuth, db: Session = Depends(database.get_db)):
    email = user_data.email.lower().strip()

    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="등록되지 않은 이메일입니다.")

    if not pwd_context.verify(user_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")

    return {
        "message": "로그인 성공",
        "user_id": user.id,
        "email": user.email,
    }


@app.get("/projects")
def list_projects(
    user_id: int = Query(...),
    keyword: str = Query(default=""),
    db: Session = Depends(database.get_db),
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    query = db.query(models.Project).filter(models.Project.user_id == user_id)

    keyword = keyword.strip()
    if keyword:
        like_keyword = f"%{keyword}%"
        query = query.filter(
            models.Project.name.ilike(like_keyword)
            | models.Project.description.ilike(like_keyword)
            | models.Project.folder_path.ilike(like_keyword)
        )

    projects = query.order_by(models.Project.updated_at.desc()).all()

    return {
        "projects": [serialize_project(project) for project in projects],
        "count": len(projects),
    }


@app.get("/projects/{project_id}")
def get_project(
    project_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    return serialize_project(project)


@app.post("/projects")
def create_project(project_data: ProjectCreate, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.id == project_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    name = project_data.name.strip()
    check_project_name_duplicate(db, project_data.user_id, name)
    folder_path = (project_data.folder_path or "").strip()
    if not folder_path:
        folder_path = make_default_folder_path(project_data.user_id, name)

    new_project = models.Project(
        name=name,
        description=(project_data.description or "").strip(),
        folder_path=folder_path,
        user_id=project_data.user_id,
    )

    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return {
        "message": "프로젝트가 생성되었습니다.",
        "project": serialize_project(new_project),
    }


@app.patch("/projects/{project_id}")
def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, project_data.user_id)

    if project_data.name is not None:
        name = project_data.name.strip()
        check_project_name_duplicate(db, project_data.user_id, name, exclude_project_id=project_id)
        project.name = name

    if project_data.description is not None:
        project.description = project_data.description.strip()
    if project_data.folder_path is not None:
        folder_path = project_data.folder_path.strip()
        project.folder_path = folder_path or make_default_folder_path(project_data.user_id, project.name)

    project.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(project)

    return {
        "message": "프로젝트가 수정되었습니다.",
        "project": serialize_project(project),
    }


@app.delete("/projects/{project_id}")
def delete_project(
    project_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    datasets = (
        db.query(models.Dataset)
        .filter(models.Dataset.project_id == project_id, models.Dataset.user_id == user_id)
        .all()
    )
    training_models = (
        db.query(models.TrainingModel)
        .filter(models.TrainingModel.project_id == project_id, models.TrainingModel.user_id == user_id)
        .all()
    )
    dataset_dirs = []
    for dataset in datasets:
        try:
            dataset_dirs.append(Path(dataset.zip_path).parent)
        except TypeError:
            pass
    model_dirs = []
    for training_model in training_models:
        try:
            model_dirs.append(Path(training_model.model_path).parent)
        except TypeError:
            pass

    db.query(models.TrainingJob).filter(
        models.TrainingJob.project_id == project_id,
        models.TrainingJob.user_id == user_id,
    ).delete(synchronize_session=False)
    db.query(models.PreprocessingPipeline).filter(
        models.PreprocessingPipeline.project_id == project_id,
        models.PreprocessingPipeline.user_id == user_id,
    ).delete(synchronize_session=False)
    db.query(models.TrainingModel).filter(
        models.TrainingModel.project_id == project_id,
        models.TrainingModel.user_id == user_id,
    ).delete(synchronize_session=False)
    db.query(models.Dataset).filter(
        models.Dataset.project_id == project_id,
        models.Dataset.user_id == user_id,
    ).delete(synchronize_session=False)

    db.delete(project)
    db.commit()

    for dataset_dir in dataset_dirs:
        try:
            shutil.rmtree(dataset_dir, ignore_errors=True)
        except Exception:
            pass
    for model_dir in model_dirs:
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass

    return {
        "message": "프로젝트가 삭제되었습니다.",
        "project_id": project_id,
    }


@app.get("/training-models")
def list_training_models(
    user_id: Optional[int] = Query(default=None),
    project_id: Optional[int] = Query(default=None),
    db: Session = Depends(database.get_db),
):
    return {"models": list_available_models(db, user_id, project_id)}


@app.get("/yolo-models")
def list_yolo_models(
    user_id: Optional[int] = Query(default=None),
    project_id: Optional[int] = Query(default=None),
    db: Session = Depends(database.get_db),
):
    return list_training_models(user_id, project_id, db)


@app.post("/projects/{project_id}/training-models")
def create_training_model(
    project_id: int,
    user_id: int = Form(...),
    name: str = Form(...),
    task_type: str = Form(default="detect"),
    description: str = Form(default=""),
    model_file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    model_name = name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="모델 이름을 입력하세요.")

    task_type = task_type.strip().lower()
    if task_type not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="task_type은 detect 또는 classify만 가능합니다.")

    original_filename = Path(model_file.filename or "").name
    if not original_filename.lower().endswith((".pt", ".pth", ".onnx")):
        raise HTTPException(status_code=400, detail="학습 모델은 .pt, .pth, .onnx 파일만 추가할 수 있습니다.")

    duplicate = (
        db.query(models.TrainingModel)
        .filter(
            models.TrainingModel.project_id == project_id,
            models.TrainingModel.user_id == user_id,
            func.lower(models.TrainingModel.name) == model_name.lower(),
        )
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="이미 같은 이름의 학습 모델이 있습니다.")

    model_uid = uuid.uuid4().hex[:10]
    safe_model_name = make_safe_folder_name(model_name)
    safe_file_name = make_safe_folder_name(Path(original_filename).stem) + Path(original_filename).suffix.lower()
    model_dir = Path(project.folder_path) / "training_models" / f"{safe_model_name}_{model_uid}"
    model_path = model_dir / safe_file_name
    save_upload_file(model_file, model_path)

    model = models.TrainingModel(
        project_id=project_id,
        user_id=user_id,
        model_key=f"custom:{model_uid}",
        name=model_name,
        description=description.strip(),
        task_type=task_type,
        original_filename=original_filename,
        model_path=str(model_path),
        file_size=model_path.stat().st_size,
        source="custom",
        status="READY",
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    return {
        "message": "학습 모델이 추가되었습니다.",
        "model": serialize_training_model(model),
    }


@app.delete("/projects/{project_id}/training-models/{model_id}")
def delete_training_model(
    project_id: int,
    model_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    model = (
        db.query(models.TrainingModel)
        .filter(
            models.TrainingModel.id == model_id,
            models.TrainingModel.project_id == project_id,
            models.TrainingModel.user_id == user_id,
        )
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="학습 모델을 찾을 수 없습니다.")

    used_count = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
            models.TrainingJob.yolo_model == model.model_key,
        )
        .count()
    )
    if used_count:
        raise HTTPException(status_code=400, detail="학습 작업에서 사용 중인 모델은 삭제할 수 없습니다.")

    model_dir = Path(model.model_path).parent
    db.delete(model)
    db.commit()
    shutil.rmtree(model_dir, ignore_errors=True)
    return {"message": "학습 모델이 삭제되었습니다.", "model_id": model_id}


@app.get("/preprocessing-options")
def list_preprocessing_options():
    return {
        "task_types": sorted(ALLOWED_TASKS),
        "normalizations": sorted(ALLOWED_NORMALIZATIONS),
        "default_augmentations": DEFAULT_AUGMENTATIONS,
        "image_size": {"min": 64, "max": 2048, "default": 224},
        "split": {"default_train": 0.8, "default_val": 0.2, "default_test": 0.0},
    }


@app.get("/projects/{project_id}/preprocessing-pipelines")
def list_preprocessing_pipelines(
    project_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    pipelines = (
        db.query(models.PreprocessingPipeline)
        .filter(
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == user_id,
        )
        .order_by(models.PreprocessingPipeline.updated_at.desc())
        .all()
    )
    return {
        "project": serialize_project(project),
        "pipelines": [serialize_preprocessing_pipeline(item) for item in pipelines],
        "count": len(pipelines),
    }


@app.post("/projects/{project_id}/preprocessing-pipelines")
def create_preprocessing_pipeline(
    project_id: int,
    payload: PreprocessingPipelinePayload,
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, payload.user_id)
    name = payload.name.strip()
    duplicate = (
        db.query(models.PreprocessingPipeline)
        .filter(
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == payload.user_id,
            func.lower(models.PreprocessingPipeline.name) == name.lower(),
        )
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="이미 같은 이름의 전처리 파이프라인이 있습니다.")

    config = attach_dataset_reference_to_pipeline_config(
        db,
        project_id=project_id,
        payload=payload,
        config=build_preprocessing_config(payload),
    )
    pipeline = models.PreprocessingPipeline(
        project_id=project_id,
        user_id=payload.user_id,
        name=name,
        description=(payload.description or "").strip(),
        task_type=payload.task_type.strip().lower(),
        image_size=payload.image_size,
        train_split=payload.train_split,
        val_split=payload.val_split,
        test_split=payload.test_split,
        config_json=json.dumps(config, ensure_ascii=False),
        status="READY",
    )
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    return {
        "message": "전처리 파이프라인이 생성되었습니다.",
        "pipeline": serialize_preprocessing_pipeline(pipeline),
    }


@app.patch("/projects/{project_id}/preprocessing-pipelines/{pipeline_id}")
def update_preprocessing_pipeline(
    project_id: int,
    pipeline_id: int,
    payload: PreprocessingPipelinePayload,
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, payload.user_id)
    pipeline = (
        db.query(models.PreprocessingPipeline)
        .filter(
            models.PreprocessingPipeline.id == pipeline_id,
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == payload.user_id,
        )
        .first()
    )
    if not pipeline:
        raise HTTPException(status_code=404, detail="전처리 파이프라인을 찾을 수 없습니다.")

    name = payload.name.strip()
    duplicate = (
        db.query(models.PreprocessingPipeline)
        .filter(
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == payload.user_id,
            func.lower(models.PreprocessingPipeline.name) == name.lower(),
            models.PreprocessingPipeline.id != pipeline_id,
        )
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="이미 같은 이름의 전처리 파이프라인이 있습니다.")

    config = build_preprocessing_config(payload)
    pipeline.name = name
    pipeline.description = (payload.description or "").strip()
    pipeline.task_type = payload.task_type.strip().lower()
    pipeline.image_size = payload.image_size
    pipeline.train_split = payload.train_split
    pipeline.val_split = payload.val_split
    pipeline.test_split = payload.test_split
    pipeline.config_json = json.dumps(config, ensure_ascii=False)
    pipeline.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(pipeline)
    return {
        "message": "전처리 파이프라인이 수정되었습니다.",
        "pipeline": serialize_preprocessing_pipeline(pipeline),
    }


@app.delete("/projects/{project_id}/preprocessing-pipelines/{pipeline_id}")
def delete_preprocessing_pipeline(
    project_id: int,
    pipeline_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    pipeline = (
        db.query(models.PreprocessingPipeline)
        .filter(
            models.PreprocessingPipeline.id == pipeline_id,
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == user_id,
        )
        .first()
    )
    if not pipeline:
        raise HTTPException(status_code=404, detail="전처리 파이프라인을 찾을 수 없습니다.")

    db.delete(pipeline)
    db.commit()
    return {"message": "전처리 파이프라인이 삭제되었습니다.", "pipeline_id": pipeline_id}


@app.get("/projects/{project_id}/datasets")
def list_project_datasets(
    project_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    datasets = (
        db.query(models.Dataset)
        .filter(models.Dataset.project_id == project_id, models.Dataset.user_id == user_id)
        .order_by(models.Dataset.created_at.desc())
        .all()
    )
    return {
        "project": serialize_project(project),
        "datasets": [serialize_dataset(item) for item in datasets],
        "count": len(datasets),
    }


@app.post("/datasets/analyze-zip")
def analyze_dataset_zip(
    task_type: str = Form(default="detect"),
    dataset_zip: UploadFile = File(...),
):
    original_filename = Path(dataset_zip.filename or "").name
    if not original_filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="데이터셋은 zip 파일만 분석할 수 있습니다.")

    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / original_filename
        save_upload_file(dataset_zip, target)
        file_size = target.stat().st_size
        report = make_dataset_report(target)
        report["zip_size"] = file_size
        report["zip_filename"] = original_filename
        report["recommended_pipeline"] = recommend_preprocessing_pipeline(
            report,
            task_type,
            Path(original_filename).stem,
        )

    return {
        "message": "데이터셋 분석이 완료되었습니다.",
        "report": report,
    }


@app.post("/projects/{project_id}/datasets")
def create_dataset(
    project_id: int,
    user_id: int = Form(...),
    dataset_name: str = Form(...),
    description: str = Form(default=""),
    task_type: str = Form(default="detect"),
    create_auto_pipeline: bool = Form(default=False),
    dataset_zip: UploadFile = File(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    name = dataset_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="데이터셋 이름을 입력하세요.")

    duplicate = (
        db.query(models.Dataset)
        .filter(
            models.Dataset.project_id == project_id,
            models.Dataset.user_id == user_id,
            func.lower(models.Dataset.name) == name.lower(),
        )
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="이미 같은 이름의 데이터셋이 있습니다.")

    task_type = task_type.strip().lower()
    if task_type not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="task_type은 detect 또는 classify만 가능합니다.")

    original_filename = Path(dataset_zip.filename or "").name
    if not original_filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="데이터셋은 zip 파일만 업로드할 수 있습니다.")

    safe_dataset_name = make_safe_folder_name(name)
    safe_file_name = make_safe_folder_name(Path(original_filename).stem) + ".zip"
    dataset_uid = uuid.uuid4().hex[:10]
    project_root = Path(project.folder_path)
    dataset_dir = project_root / "datasets" / f"{safe_dataset_name}_{dataset_uid}"
    zip_path = dataset_dir / safe_file_name

    save_upload_file(dataset_zip, zip_path)
    file_size = zip_path.stat().st_size
    report = make_dataset_report(zip_path)
    report["zip_size"] = file_size
    report["zip_filename"] = original_filename
    report["recommended_pipeline"] = recommend_preprocessing_pipeline(report, task_type, name)

    dataset = models.Dataset(
        project_id=project_id,
        user_id=user_id,
        name=name,
        description=description.strip(),
        task_type=task_type,
        original_filename=original_filename,
        zip_path=str(zip_path),
        file_size=file_size,
        report_json=json.dumps(report, ensure_ascii=False),
        status="READY",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    pipeline = None
    if create_auto_pipeline:
        pipeline = create_recommended_preprocessing_pipeline(
            db,
            project_id=project_id,
            user_id=user_id,
            dataset=dataset,
            report=report,
        )

    return {
        "message": "데이터셋이 업로드되었습니다.",
        "dataset": serialize_dataset(dataset),
        "auto_pipeline": serialize_preprocessing_pipeline(pipeline) if pipeline else None,
    }


@app.post("/projects/{project_id}/datasets/{dataset_id}/auto-preprocessing-pipeline")
def create_dataset_auto_preprocessing_pipeline(
    project_id: int,
    dataset_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    dataset = (
        db.query(models.Dataset)
        .filter(
            models.Dataset.id == dataset_id,
            models.Dataset.project_id == project_id,
            models.Dataset.user_id == user_id,
        )
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    report = read_json_dict(dataset.report_json)
    report["recommended_pipeline"] = recommend_preprocessing_pipeline(report, dataset.task_type, dataset.name)
    dataset.report_json = json.dumps(report, ensure_ascii=False)
    dataset.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(dataset)

    pipeline = create_recommended_preprocessing_pipeline(
        db,
        project_id=project_id,
        user_id=user_id,
        dataset=dataset,
        report=report,
    )
    return {
        "message": "추천 전처리 파이프라인이 생성되었습니다.",
        "pipeline": serialize_preprocessing_pipeline(pipeline),
        "dataset": serialize_dataset(dataset),
    }


@app.delete("/projects/{project_id}/datasets/{dataset_id}")
def delete_dataset(
    project_id: int,
    dataset_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    dataset = (
        db.query(models.Dataset)
        .filter(
            models.Dataset.id == dataset_id,
            models.Dataset.project_id == project_id,
            models.Dataset.user_id == user_id,
        )
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    used_count = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
            models.TrainingJob.dataset_id == dataset_id,
        )
        .count()
    )
    if used_count:
        raise HTTPException(status_code=400, detail="학습에서 사용 중인 데이터셋은 삭제할 수 없습니다.")

    dataset_dir = Path(dataset.zip_path).parent
    linked_pipelines = (
        db.query(models.PreprocessingPipeline)
        .filter(
            models.PreprocessingPipeline.project_id == project_id,
            models.PreprocessingPipeline.user_id == user_id,
        )
        .all()
    )
    for pipeline in linked_pipelines:
        config = read_json_dict(pipeline.config_json)
        if int(config.get("dataset_id") or 0) == dataset_id:
            db.delete(pipeline)

    db.delete(dataset)
    db.commit()
    try:
        shutil.rmtree(dataset_dir, ignore_errors=True)
    except Exception:
        pass

    return {"message": "데이터셋이 삭제되었습니다.", "dataset_id": dataset_id}


@app.get("/projects/{project_id}/training-jobs")
def list_training_jobs(
    project_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    rows = (
        db.query(models.TrainingJob, models.Dataset, models.PreprocessingPipeline)
        .join(models.Dataset, models.TrainingJob.dataset_id == models.Dataset.id)
        .outerjoin(
            models.PreprocessingPipeline,
            models.TrainingJob.preprocessing_pipeline_id == models.PreprocessingPipeline.id,
        )
        .filter(models.TrainingJob.project_id == project_id, models.TrainingJob.user_id == user_id)
        .order_by(models.TrainingJob.created_at.desc())
        .all()
    )
    for job, _, _ in rows:
        reconcile_stopping_training_job(db, job)

    return {
        "project": serialize_project(project),
        "training_jobs": [serialize_training_job(job, dataset, pipeline) for job, dataset, pipeline in rows],
        "count": len(rows),
    }


@app.get("/projects/{project_id}/artifact-file")
def get_project_artifact_file(
    project_id: int,
    user_id: int = Query(...),
    path: str = Query(...),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    requested = Path(path).resolve()
    project_root = Path(project.folder_path).resolve()
    allowed_suffixes = {".png", ".jpg", ".jpeg"}

    if project_root not in [requested, *requested.parents]:
        raise HTTPException(status_code=403, detail="프로젝트 결과 폴더 밖의 파일은 열 수 없습니다.")
    if requested.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="이미지 결과 파일만 미리보기할 수 있습니다.")
    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="결과 이미지 파일을 찾을 수 없습니다.")

    return FileResponse(str(requested))


@app.post("/projects/{project_id}/training-jobs")
def create_training_job(
    project_id: int,
    payload: TrainingJobCreate,
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, payload.user_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="학습 이름을 입력하세요.")

    duplicate = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == payload.user_id,
            func.lower(models.TrainingJob.name) == name.lower(),
        )
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="이미 같은 이름의 학습이 있습니다.")

    task_type = payload.task_type.strip().lower()
    if task_type not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="task_type은 detect 또는 classify만 가능합니다.")

    dataset = (
        db.query(models.Dataset)
        .filter(
            models.Dataset.id == payload.dataset_id,
            models.Dataset.project_id == project_id,
            models.Dataset.user_id == payload.user_id,
        )
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")
    if dataset.task_type != task_type:
        raise HTTPException(status_code=400, detail="선택한 데이터셋의 작업 유형과 학습 작업 유형이 다릅니다.")

    pipeline = None
    if payload.preprocessing_pipeline_id:
        pipeline = (
            db.query(models.PreprocessingPipeline)
            .filter(
                models.PreprocessingPipeline.id == payload.preprocessing_pipeline_id,
                models.PreprocessingPipeline.project_id == project_id,
                models.PreprocessingPipeline.user_id == payload.user_id,
            )
            .first()
        )
        if not pipeline:
            raise HTTPException(status_code=404, detail="전처리 파이프라인을 찾을 수 없습니다.")
        if pipeline.task_type != task_type:
            raise HTTPException(status_code=400, detail="선택한 전처리 파이프라인의 작업 유형과 학습 작업 유형이 다릅니다.")

    if not model_supports_task(db, payload.yolo_model, task_type, payload.user_id, project_id):
        raise HTTPException(status_code=400, detail="선택한 학습 모델이 작업 유형과 맞지 않습니다.")
    if payload.optimizer not in ALLOWED_OPTIMIZERS:
        raise HTTPException(status_code=400, detail="지원하지 않는 optimizer입니다.")
    if payload.image_size < 64 or payload.image_size > 2048:
        raise HTTPException(status_code=400, detail="Image Size는 64 이상 2048 이하로 입력하세요.")
    if payload.epochs < 1 or payload.epochs > 1000:
        raise HTTPException(status_code=400, detail="Epochs는 1 이상 1000 이하로 입력하세요.")
    if payload.batch_min < 1 or payload.batch_max < payload.batch_min:
        raise HTTPException(status_code=400, detail="Batch 범위를 확인하세요.")
    if payload.lr_initial_min <= 0 or payload.lr_initial_max < payload.lr_initial_min:
        raise HTTPException(status_code=400, detail="LR Initial 범위를 확인하세요.")
    if payload.momentum_min < 0 or payload.momentum_max > 1 or payload.momentum_max < payload.momentum_min:
        raise HTTPException(status_code=400, detail="Momentum 범위는 0 이상 1 이하로 입력하세요.")

    job = models.TrainingJob(
        project_id=project_id,
        dataset_id=dataset.id,
        preprocessing_pipeline_id=pipeline.id if pipeline else None,
        user_id=payload.user_id,
        name=name,
        description=(payload.description or "").strip(),
        task_type=task_type,
        yolo_model=payload.yolo_model,
        optimizer=payload.optimizer,
        image_size=payload.image_size,
        epochs=payload.epochs,
        batch_min=payload.batch_min,
        batch_max=payload.batch_max,
        lr_initial_min=payload.lr_initial_min,
        lr_initial_max=payload.lr_initial_max,
        momentum_min=payload.momentum_min,
        momentum_max=payload.momentum_max,
        status="READY",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "message": "새 학습이 생성되었습니다.",
        "training_job": serialize_training_job(job, dataset, pipeline),
    }


@app.post("/projects/{project_id}/training-jobs/{job_id}/start")
def start_training_job(
    project_id: int,
    job_id: int,
    user_id: int = Query(...),
    simulate: bool = Query(default=False),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    job = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.id == job_id,
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="학습을 찾을 수 없습니다.")
    if job.status in {"RUNNING", "QUEUED"}:
        raise HTTPException(status_code=400, detail="이미 실행 중인 학습입니다.")

    job.status = "QUEUED"
    job.progress = 0
    job.current_epoch = 0
    job.stop_requested = False
    job.logs_json = json.dumps(
        [{"time": datetime.utcnow().isoformat(), "message": "학습 실행 대기열에 등록되었습니다."}],
        ensure_ascii=False,
    )
    job.result_json = "{}"
    job.started_at = None
    job.completed_at = None
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)

    stop_event = threading.Event()
    TRAINING_STOP_EVENTS[job.id] = stop_event
    thread = threading.Thread(target=training_worker, args=(job.id, simulate), daemon=True)
    TRAINING_THREADS[job.id] = thread
    thread.start()

    return {
        "message": "학습을 시작했습니다.",
        "training_job": serialize_training_job(job),
    }


@app.post("/projects/{project_id}/training-jobs/{job_id}/stop")
def stop_training_job(
    project_id: int,
    job_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    job = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.id == job_id,
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="학습을 찾을 수 없습니다.")
    if job.status not in {"QUEUED", "RUNNING", "STOPPING"}:
        raise HTTPException(status_code=400, detail="실행 중인 학습만 중지할 수 있습니다.")

    already_stopping = job.status == "STOPPING"
    job.stop_requested = True
    job.status = "STOPPING"
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)

    stop_event = TRAINING_STOP_EVENTS.get(job.id)
    if stop_event:
        stop_event.set()

    stop_training_process(job.id, wait_timeout=1.0 if already_stopping else 0.2)

    if already_stopping:
        append_training_log(db, job, "중지 중인 학습을 다시 정리합니다.")
    else:
        append_training_log(db, job, "사용자가 학습 중지를 요청했습니다.")

    reconcile_stopping_training_job(db, job, force=already_stopping)
    db.refresh(job)
    return {
        "message": "학습을 중지했습니다." if job.status == "STOPPED" else "학습 중지를 요청했습니다.",
        "training_job": serialize_training_job(job),
    }


@app.get("/projects/{project_id}/training-jobs/{job_id}/logs")
def get_training_job_logs(
    project_id: int,
    job_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    job = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.id == job_id,
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="학습을 찾을 수 없습니다.")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": clamp_progress_value(job.progress, job.status),
        "current_epoch": clamp_epoch_value(job.current_epoch, job.epochs),
        "logs": read_json_list(job.logs_json),
        "result": read_json_dict(job.result_json),
    }


@app.get("/projects/{project_id}/results")
def list_project_results(
    project_id: int,
    user_id: int = Query(...),
    status: str = Query(default="finished"),
    task_type: str = Query(default="all"),
    db: Session = Depends(database.get_db),
):
    project = get_owned_project(db, project_id, user_id)
    query = (
        db.query(models.TrainingJob, models.Dataset, models.PreprocessingPipeline)
        .join(models.Dataset, models.TrainingJob.dataset_id == models.Dataset.id)
        .outerjoin(
            models.PreprocessingPipeline,
            models.TrainingJob.preprocessing_pipeline_id == models.PreprocessingPipeline.id,
        )
        .filter(models.TrainingJob.project_id == project_id, models.TrainingJob.user_id == user_id)
    )

    normalized_status = status.strip().upper()
    if normalized_status in {"FINISHED", ""}:
        query = query.filter(models.TrainingJob.status.in_(["COMPLETED", "FAILED", "STOPPED"]))
    elif normalized_status != "ALL":
        query = query.filter(models.TrainingJob.status == normalized_status)

    normalized_task = task_type.strip().lower()
    if normalized_task in ALLOWED_TASKS:
        query = query.filter(models.TrainingJob.task_type == normalized_task)

    rows = query.order_by(models.TrainingJob.updated_at.desc()).all()
    results = [serialize_result_run(job, dataset, pipeline) for job, dataset, pipeline in rows]
    scored_results = [item for item in results if item.get("primary_score") is not None]
    best_result = max(scored_results, key=lambda item: item["primary_score"], default=None)

    return {
        "project": serialize_project(project),
        "results": results,
        "count": len(results),
        "summary": {
            "completed": sum(1 for item in results if item["status"] == "COMPLETED"),
            "failed": sum(1 for item in results if item["status"] == "FAILED"),
            "stopped": sum(1 for item in results if item["status"] == "STOPPED"),
            "best_score": best_result["primary_score"] if best_result else None,
            "best_job_id": best_result["id"] if best_result else None,
            "best_job_name": best_result["name"] if best_result else "",
        },
    }


@app.get("/projects/{project_id}/results/compare")
def compare_project_results(
    project_id: int,
    user_id: int = Query(...),
    job_ids: str = Query(default=""),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    selected_ids = []
    for raw_id in job_ids.split(","):
        raw_id = raw_id.strip()
        if raw_id.isdigit():
            selected_ids.append(int(raw_id))

    if not selected_ids:
        return {"runs": [], "metric_keys": []}

    rows = (
        db.query(models.TrainingJob, models.Dataset, models.PreprocessingPipeline)
        .join(models.Dataset, models.TrainingJob.dataset_id == models.Dataset.id)
        .outerjoin(
            models.PreprocessingPipeline,
            models.TrainingJob.preprocessing_pipeline_id == models.PreprocessingPipeline.id,
        )
        .filter(
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
            models.TrainingJob.id.in_(selected_ids),
        )
        .all()
    )
    by_id = {
        job.id: serialize_result_run(job, dataset, pipeline)
        for job, dataset, pipeline in rows
        if job.status in {"COMPLETED", "FAILED", "STOPPED"}
    }
    runs = [by_id[item] for item in selected_ids if item in by_id]
    metric_keys = sorted({key for item in runs for key in (item.get("metrics") or {}).keys()})
    return {"runs": runs, "metric_keys": metric_keys}


@app.get("/projects/{project_id}/results/{job_id}")
def get_project_result_detail(
    project_id: int,
    job_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    row = (
        db.query(models.TrainingJob, models.Dataset, models.PreprocessingPipeline)
        .join(models.Dataset, models.TrainingJob.dataset_id == models.Dataset.id)
        .outerjoin(
            models.PreprocessingPipeline,
            models.TrainingJob.preprocessing_pipeline_id == models.PreprocessingPipeline.id,
        )
        .filter(
            models.TrainingJob.id == job_id,
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")

    job, dataset, pipeline = row
    return {"result": serialize_result_run(job, dataset, pipeline, include_logs=True)}


@app.delete("/projects/{project_id}/training-jobs/{job_id}")
def delete_training_job(
    project_id: int,
    job_id: int,
    user_id: int = Query(...),
    db: Session = Depends(database.get_db),
):
    get_owned_project(db, project_id, user_id)
    job = (
        db.query(models.TrainingJob)
        .filter(
            models.TrainingJob.id == job_id,
            models.TrainingJob.project_id == project_id,
            models.TrainingJob.user_id == user_id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="학습을 찾을 수 없습니다.")
    if job.status in {"QUEUED", "RUNNING", "STOPPING"}:
        raise HTTPException(status_code=400, detail="실행 중인 학습은 삭제할 수 없습니다. 먼저 중지하세요.")

    db.delete(job)
    db.commit()
    return {"message": "학습이 삭제되었습니다.", "job_id": job_id}
