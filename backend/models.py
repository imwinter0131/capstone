from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False, index=True)
    description = Column(Text, nullable=True)
    folder_path = Column(String(500), nullable=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    owner = relationship("User", back_populates="projects")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(160), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(30), nullable=False, default="detect")
    original_filename = Column(String(255), nullable=False)
    zip_path = Column(String(700), nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    report_json = Column(Text, nullable=False, default="{}")
    status = Column(String(40), nullable=False, default="READY")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    preprocessing_pipeline_id = Column(Integer, ForeignKey("preprocessing_pipelines.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    name = Column(String(160), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(30), nullable=False, default="detect")
    yolo_model = Column(String(80), nullable=False, default="yolov8n.pt")
    optimizer = Column(String(40), nullable=False, default="AdamW")
    image_size = Column(Integer, nullable=False, default=640)
    epochs = Column(Integer, nullable=False, default=10)

    batch_min = Column(Integer, nullable=False, default=16)
    batch_max = Column(Integer, nullable=False, default=64)
    lr_initial_min = Column(Float, nullable=False, default=0.0001)
    lr_initial_max = Column(Float, nullable=False, default=0.01)
    momentum_min = Column(Float, nullable=False, default=0.8)
    momentum_max = Column(Float, nullable=False, default=0.99)

    status = Column(String(40), nullable=False, default="READY")
    progress = Column(Float, nullable=False, default=0)
    current_epoch = Column(Integer, nullable=False, default=0)
    logs_json = Column(Text, nullable=False, default="[]")
    result_json = Column(Text, nullable=False, default="{}")
    run_dir = Column(String(700), nullable=True)
    stop_requested = Column(Boolean, nullable=False, default=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class PreprocessingPipeline(Base):
    __tablename__ = "preprocessing_pipelines"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    name = Column(String(160), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(30), nullable=False, default="detect")
    image_size = Column(Integer, nullable=False, default=640)
    train_split = Column(Float, nullable=False, default=0.8)
    val_split = Column(Float, nullable=False, default=0.2)
    test_split = Column(Float, nullable=False, default=0.0)
    config_json = Column(Text, nullable=False, default="{}")
    status = Column(String(40), nullable=False, default="READY")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TrainingModel(Base):
    __tablename__ = "training_models"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    model_key = Column(String(120), unique=True, nullable=False, index=True)
    name = Column(String(160), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(30), nullable=False, default="detect")
    original_filename = Column(String(255), nullable=False)
    model_path = Column(String(700), nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    source = Column(String(40), nullable=False, default="custom")
    status = Column(String(40), nullable=False, default="READY")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
