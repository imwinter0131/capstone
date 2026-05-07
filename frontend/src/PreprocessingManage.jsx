import React, { useEffect, useMemo, useState } from "react";
import "./PreprocessingManage.css";

const API_BASE = "http://localhost:8000";

const emptyForm = {
  name: "",
  description: "",
  task_type: "detect",
  image_size: 640,
  keep_aspect_ratio: true,
  normalize: "zero_one",
  train_split: 0.8,
  val_split: 0.2,
  test_split: 0,
  augmentations: {
    horizontal_flip: true,
    vertical_flip: false,
    rotation_degrees: 0,
    hsv_h: 0.015,
    hsv_s: 0.7,
    hsv_v: 0.4,
    mosaic: true,
    mixup: 0,
    random_crop: false,
    blur: false,
    noise: false,
  },
};

function cloneForm(source = emptyForm) {
  return {
    ...source,
    augmentations: {
      ...emptyForm.augmentations,
      ...(source.augmentations || {}),
    },
  };
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function asPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function PipelineSummary({ pipeline }) {
  const config = pipeline.config || {};
  const augmentations = config.augmentations || {};
  const activeAugmentations = [
    augmentations.horizontal_flip && "H-Flip",
    augmentations.vertical_flip && "V-Flip",
    Number(augmentations.rotation_degrees || 0) > 0 && `Rotate ${augmentations.rotation_degrees}°`,
    Number(augmentations.hsv_h || 0) > 0 && "HSV",
    augmentations.mosaic && "Mosaic",
    Number(augmentations.mixup || 0) > 0 && `MixUp ${augmentations.mixup}`,
    augmentations.random_crop && "Crop",
    augmentations.blur && "Blur",
    augmentations.noise && "Noise",
  ].filter(Boolean);

  return (
    <>
      <div className="preprocess-metrics">
        <div>
          <span>Task</span>
          <strong>{pipeline.task_type === "detect" ? "Detect" : "Classify"}</strong>
        </div>
        <div>
          <span>Image</span>
          <strong>{pipeline.image_size}px</strong>
        </div>
        <div>
          <span>Normalize</span>
          <strong>{config.normalize || "none"}</strong>
        </div>
        <div>
          <span>Split</span>
          <strong>
            {asPercent(pipeline.train_split)} / {asPercent(pipeline.val_split)} / {asPercent(pipeline.test_split)}
          </strong>
        </div>
      </div>

      <div className="augmentation-tags">
        {activeAugmentations.length === 0 ? (
          <span className="muted-tag">No augmentation</span>
        ) : (
          activeAugmentations.map((item) => <span key={item}>{item}</span>)
        )}
      </div>
    </>
  );
}

function PreprocessingManage({ user, projectId }) {
  const [project, setProject] = useState(null);
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editingPipeline, setEditingPipeline] = useState(null);
  const [form, setForm] = useState(cloneForm());
  const [saving, setSaving] = useState(false);

  const userId = user?.user_id;
  const userEmail = user?.email || "user";

  const splitTotal = useMemo(() => {
    return Number(form.train_split || 0) + Number(form.val_split || 0) + Number(form.test_split || 0);
  }, [form.train_split, form.val_split, form.test_split]);

  async function loadPipelines() {
    if (!projectId || !userId) return;
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/preprocessing-pipelines?user_id=${userId}`);
      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "전처리 파이프라인을 불러오지 못했습니다.");
        setProject(null);
        setPipelines([]);
        return;
      }

      setProject(data.project);
      setPipelines(data.pipelines || []);
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
      setProject(null);
      setPipelines([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPipelines();
  }, [projectId, userId]);

  function openCreateModal() {
    setEditingPipeline(null);
    setForm(cloneForm());
    setModalOpen(true);
    setError("");
  }

  function openEditModal(pipeline) {
    const config = pipeline.config || {};
    setEditingPipeline(pipeline);
    setForm(cloneForm({
      name: pipeline.name || "",
      description: pipeline.description || "",
      task_type: pipeline.task_type || "detect",
      image_size: pipeline.image_size || 640,
      keep_aspect_ratio: config.keep_aspect_ratio ?? true,
      normalize: config.normalize || "zero_one",
      train_split: pipeline.train_split ?? 0.8,
      val_split: pipeline.val_split ?? 0.2,
      test_split: pipeline.test_split ?? 0,
      augmentations: config.augmentations || {},
    }));
    setModalOpen(true);
    setError("");
  }

  function closeModal() {
    if (saving) return;
    setModalOpen(false);
    setEditingPipeline(null);
    setForm(cloneForm());
  }

  function handleChange(event) {
    const { name, value, type, checked } = event.target;
    const numericFields = new Set(["image_size", "train_split", "val_split", "test_split"]);

    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : numericFields.has(name) ? Number(value) : value,
    }));
  }

  function handleAugmentationChange(event) {
    const { name, value, type, checked } = event.target;
    const numericFields = new Set(["rotation_degrees", "hsv_h", "hsv_s", "hsv_v", "mixup"]);

    setForm((prev) => ({
      ...prev,
      augmentations: {
        ...prev.augmentations,
        [name]: type === "checkbox" ? checked : numericFields.has(name) ? Number(value) : value,
      },
    }));
  }

  function validateForm() {
    if (!form.name.trim()) return "전처리 이름을 입력하세요.";
    if (form.image_size < 64 || form.image_size > 2048) return "Image Size는 64 이상 2048 이하로 입력하세요.";
    if (Math.abs(splitTotal - 1) > 0.001) return "Train/Val/Test 비율의 합은 1.0이어야 합니다.";
    if (form.train_split < 0 || form.val_split < 0 || form.test_split < 0) return "분할 비율은 0 이상이어야 합니다.";
    if (form.augmentations.rotation_degrees < 0 || form.augmentations.rotation_degrees > 180) {
      return "Rotation은 0 이상 180 이하로 입력하세요.";
    }
    for (const key of ["hsv_h", "hsv_s", "hsv_v", "mixup"]) {
      if (form.augmentations[key] < 0 || form.augmentations[key] > 1) {
        return `${key} 값은 0 이상 1 이하로 입력하세요.`;
      }
    }
    return "";
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const validation = validateForm();
    if (validation) {
      setError(validation);
      return;
    }

    const payload = {
      user_id: userId,
      name: form.name.trim(),
      description: form.description.trim(),
      task_type: form.task_type,
      image_size: form.image_size,
      keep_aspect_ratio: form.keep_aspect_ratio,
      normalize: form.normalize,
      train_split: form.train_split,
      val_split: form.val_split,
      test_split: form.test_split,
      augmentations: form.augmentations,
    };

    const endpoint = editingPipeline
      ? `${API_BASE}/projects/${projectId}/preprocessing-pipelines/${editingPipeline.id}`
      : `${API_BASE}/projects/${projectId}/preprocessing-pipelines`;

    setSaving(true);
    setError("");

    try {
      const response = await fetch(endpoint, {
        method: editingPipeline ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "전처리 파이프라인 저장 중 오류가 발생했습니다.");
        return;
      }

      await loadPipelines();
      closeModal();
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
    } finally {
      setSaving(false);
    }
  }

  async function deletePipeline(pipeline) {
    const ok = window.confirm(`"${pipeline.name}" 전처리 파이프라인을 삭제하시겠습니까?`);
    if (!ok) return;

    setError("");
    try {
      const response = await fetch(
        `${API_BASE}/projects/${projectId}/preprocessing-pipelines/${pipeline.id}?user_id=${userId}`,
        { method: "DELETE" },
      );
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "전처리 파이프라인 삭제 중 오류가 발생했습니다.");
        return;
      }
      setPipelines((prev) => prev.filter((item) => item.id !== pipeline.id));
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
    }
  }

  function goDatasets() {
    window.location.href = `/projects/${projectId}`;
  }

  function goPreprocessing() {
    window.location.href = `/projects/${projectId}/preprocessing`;
  }

  function goTraining() {
    window.location.href = `/projects/${projectId}/training`;
  }

  function goProjects() {
    window.location.href = "/projects";
  }

  function logout() {
    localStorage.removeItem("dlops_user");
    window.location.href = "/";
  }

  return (
    <main className="preprocess-page">
      <aside className="preprocess-sidebar">
        <button className="nav-button" type="button" onClick={goDatasets}>
          Dataset Management
        </button>
        <button className="nav-button active" type="button" onClick={goPreprocessing}>
          Preprocessing
        </button>
        <button className="nav-button" type="button" onClick={goTraining}>
          Training
        </button>
        <button className="nav-button" type="button" onClick={goProjects}>
          Back to Projects
        </button>

        <div className="preprocess-brand">
          <span className="brand-mark" />
          <div>
            <strong>{project?.name || "Project"}</strong>
            <p>{project?.folder_path || "Loading project path..."}</p>
          </div>
        </div>

        <div className="side-panel">
          <span>Signed in</span>
          <strong>{userEmail}</strong>
          <button type="button" onClick={logout}>
            Logout
          </button>
        </div>

        <div className="side-panel">
          <span>Pipelines</span>
          <strong>{pipelines.length}</strong>
          <p>학습 전에 재사용할 전처리 preset을 프로젝트 단위로 관리합니다.</p>
        </div>
      </aside>

      <section className="preprocess-workspace">
        <header className="preprocess-header">
          <div>
            <p className="eyebrow">Preprocessing Management</p>
            <h1>전처리 파이프라인</h1>
            <span>Resize, Normalize, Split, Augmentation 설정을 학습 전에 저장합니다.</span>
          </div>
          <button className="primary-action" type="button" onClick={openCreateModal}>
            New Pipeline
          </button>
        </header>

        {error && <div className="preprocess-error">{error}</div>}

        {loading ? (
          <div className="preprocess-state">전처리 파이프라인을 불러오는 중입니다.</div>
        ) : pipelines.length === 0 ? (
          <div className="preprocess-state">
            <strong>아직 전처리 파이프라인이 없습니다.</strong>
            <p>New Pipeline으로 학습 전처리 preset을 생성하세요.</p>
          </div>
        ) : (
          <div className="pipeline-grid">
            {pipelines.map((pipeline) => (
              <article className="pipeline-card" key={pipeline.id}>
                <div className="pipeline-head">
                  <div>
                    <p className="eyebrow">
                      {pipeline.source === "auto" ? "Auto" : "Manual"} {pipeline.task_type === "detect" ? "Detection" : "Classification"}
                    </p>
                    <h2>{pipeline.name}</h2>
                  </div>
                  <div className="pipeline-badges">
                    <span className={`source-pill source-${pipeline.source || "manual"}`}>
                      {pipeline.source === "auto" ? "AUTO" : "MANUAL"}
                    </span>
                    <span className="status-pill">{pipeline.status}</span>
                  </div>
                </div>

                <p className="pipeline-description">{pipeline.description || "전처리 설명이 없습니다."}</p>
                <PipelineSummary pipeline={pipeline} />

                <div className="pipeline-footer">
                  <span>Updated {formatDate(pipeline.updated_at)}</span>
                  <div>
                    <button type="button" onClick={() => openEditModal(pipeline)}>
                      Edit
                    </button>
                    <button className="danger" type="button" onClick={() => deletePipeline(pipeline)}>
                      Delete
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      {modalOpen && (
        <div className="preprocess-modal-backdrop">
          <form className="preprocess-modal" onSubmit={handleSubmit}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">{editingPipeline ? "Edit Pipeline" : "New Pipeline"}</p>
                <h2>{editingPipeline ? "전처리 수정" : "전처리 생성"}</h2>
              </div>
              <button type="button" onClick={closeModal} disabled={saving}>
                Close
              </button>
            </div>

            <div className="modal-body">
              <section className="modal-section">
                <h3>Basic</h3>
                <div className="form-grid">
                  <label className="field">
                    <span>전처리 이름</span>
                    <input name="name" value={form.name} onChange={handleChange} placeholder="default-yolo-augment" />
                  </label>
                  <label className="field">
                    <span>작업 유형</span>
                    <select name="task_type" value={form.task_type} onChange={handleChange}>
                      <option value="detect">탐지 Detect</option>
                      <option value="classify">분류 Classify</option>
                    </select>
                  </label>
                  <label className="field wide">
                    <span>설명</span>
                    <textarea
                      name="description"
                      value={form.description}
                      onChange={handleChange}
                      rows={3}
                      placeholder="데이터셋 특성이나 적용 목적을 적어두세요."
                    />
                  </label>
                </div>
              </section>

              <section className="modal-section">
                <h3>Image Pipeline</h3>
                <div className="form-grid">
                  <label className="field">
                    <span>Image Size</span>
                    <input
                      type="number"
                      name="image_size"
                      min="64"
                      max="2048"
                      value={form.image_size}
                      onChange={handleChange}
                    />
                  </label>
                  <label className="field">
                    <span>Normalize</span>
                    <select name="normalize" value={form.normalize} onChange={handleChange}>
                      <option value="zero_one">0-1 Scale</option>
                      <option value="imagenet">ImageNet Mean/Std</option>
                      <option value="none">None</option>
                    </select>
                  </label>
                  <label className="toggle-field">
                    <input
                      type="checkbox"
                      name="keep_aspect_ratio"
                      checked={form.keep_aspect_ratio}
                      onChange={handleChange}
                    />
                    <span>Keep Aspect Ratio</span>
                  </label>
                </div>
              </section>

              <section className="modal-section">
                <h3>Dataset Split</h3>
                <div className="form-grid split-grid">
                  <label className="field">
                    <span>Train</span>
                    <input type="number" name="train_split" min="0" max="1" step="0.05" value={form.train_split} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>Val</span>
                    <input type="number" name="val_split" min="0" max="1" step="0.05" value={form.val_split} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>Test</span>
                    <input type="number" name="test_split" min="0" max="1" step="0.05" value={form.test_split} onChange={handleChange} />
                  </label>
                  <div className={`split-total ${Math.abs(splitTotal - 1) > 0.001 ? "invalid" : ""}`}>
                    <span>Total</span>
                    <strong>{splitTotal.toFixed(2)}</strong>
                  </div>
                </div>
              </section>

              <section className="modal-section">
                <h3>Augmentation</h3>
                <div className="toggle-grid">
                  {[
                    ["horizontal_flip", "Horizontal Flip"],
                    ["vertical_flip", "Vertical Flip"],
                    ["mosaic", "Mosaic"],
                    ["random_crop", "Random Crop"],
                    ["blur", "Blur"],
                    ["noise", "Noise"],
                  ].map(([name, label]) => (
                    <label className="toggle-field" key={name}>
                      <input
                        type="checkbox"
                        name={name}
                        checked={Boolean(form.augmentations[name])}
                        onChange={handleAugmentationChange}
                      />
                      <span>{label}</span>
                    </label>
                  ))}
                </div>

                <div className="form-grid augmentation-numbers">
                  <label className="field">
                    <span>Rotation Degrees</span>
                    <input type="number" name="rotation_degrees" min="0" max="180" value={form.augmentations.rotation_degrees} onChange={handleAugmentationChange} />
                  </label>
                  <label className="field">
                    <span>HSV H</span>
                    <input type="number" name="hsv_h" min="0" max="1" step="0.001" value={form.augmentations.hsv_h} onChange={handleAugmentationChange} />
                  </label>
                  <label className="field">
                    <span>HSV S</span>
                    <input type="number" name="hsv_s" min="0" max="1" step="0.01" value={form.augmentations.hsv_s} onChange={handleAugmentationChange} />
                  </label>
                  <label className="field">
                    <span>HSV V</span>
                    <input type="number" name="hsv_v" min="0" max="1" step="0.01" value={form.augmentations.hsv_v} onChange={handleAugmentationChange} />
                  </label>
                  <label className="field">
                    <span>MixUp</span>
                    <input type="number" name="mixup" min="0" max="1" step="0.05" value={form.augmentations.mixup} onChange={handleAugmentationChange} />
                  </label>
                </div>
              </section>
            </div>

            <div className="modal-footer">
              <button type="button" className="secondary" onClick={closeModal} disabled={saving}>
                취소
              </button>
              <button type="submit" disabled={saving}>
                {saving ? "저장 중..." : editingPipeline ? "수정" : "생성"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default PreprocessingManage;
