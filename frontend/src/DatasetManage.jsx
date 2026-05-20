import React, { useEffect, useMemo, useState } from "react";
import "./DatasetManage.css";

const API_BASE = "http://localhost:8000";

const emptyForm = {
  dataset_name: "",
  description: "",
  task_type: "detect",
  pipeline_mode: "auto",
  dataset_zip: null,
};

const emptyPipelineForm = {
  name: "",
  description: "",
  task_type: "detect",
  image_size: 224,
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
  },
};

const supportedAugmentationKeys = [
  "horizontal_flip",
  "vertical_flip",
  "rotation_degrees",
  "hsv_h",
  "hsv_s",
  "hsv_v",
  "mosaic",
  "mixup",
];

function sanitizeAugmentations(source = {}) {
  return supportedAugmentationKeys.reduce((next, key) => {
    next[key] = source[key] ?? emptyPipelineForm.augmentations[key];
    return next;
  }, {});
}

function clonePipelineForm(source = emptyPipelineForm) {
  return {
    ...emptyPipelineForm,
    ...source,
    augmentations: sanitizeAugmentations(source.augmentations),
  };
}

function prettyBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, index);
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
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

function splitTotal(form) {
  return Number(form.train_split || 0) + Number(form.val_split || 0) + Number(form.test_split || 0);
}

function pipelineFormFromRecommendation(recommended, fallbackName, taskType) {
  if (!recommended) {
    return clonePipelineForm({
      name: fallbackName ? `${fallbackName}-pipeline` : "",
      task_type: taskType || "detect",
    });
  }

  return clonePipelineForm({
    name: recommended.name || (fallbackName ? `${fallbackName}-pipeline` : ""),
    description: recommended.description || "",
    task_type: recommended.task_type || taskType || "detect",
    image_size: recommended.image_size || 224,
    keep_aspect_ratio: recommended.keep_aspect_ratio ?? true,
    normalize: recommended.normalize || "zero_one",
    train_split: recommended.train_split ?? 0.8,
    val_split: recommended.val_split ?? 0.2,
    test_split: recommended.test_split ?? 0,
    augmentations: recommended.augmentations || {},
  });
}

function DatasetReport({ report }) {
  if (!report) {
    return (
      <div className="dataset-report muted">
        zip 데이터셋을 선택하고 분석하면 파일 수, 이미지 수, 라벨 수, 추천 전처리 값이 표시됩니다.
      </div>
    );
  }

  const warnings = report.warnings || [];
  const classes = report.class_names || [];
  const recommended = report.recommended_pipeline;

  return (
    <div className="dataset-report">
      <div className="report-grid">
        <div>
          <span>Total Files</span>
          <strong>{report.total_files ?? "-"}</strong>
        </div>
        <div>
          <span>Images</span>
          <strong>{report.image_count ?? "-"}</strong>
        </div>
        <div>
          <span>Labels</span>
          <strong>{report.label_count ?? "-"}</strong>
        </div>
        <div>
          <span>Classes</span>
          <strong>{report.class_count ?? classes.length ?? 0}</strong>
        </div>
      </div>

      <div className="split-report">
        <span>Train {report.train_images ?? 0}</span>
        <span>Val {report.val_images ?? 0}</span>
        <span>Test {report.test_images ?? 0}</span>
      </div>

      {classes.length > 0 && (
        <div className="class-tags">
          {classes.slice(0, 12).map((item) => (
            <span key={item}>{item}</span>
          ))}
          {classes.length > 12 && <span>+{classes.length - 12}</span>}
        </div>
      )}

      {recommended && (
        <div className="auto-pipeline-panel">
          <div className="auto-pipeline-head">
            <span>Recommended Preprocessing</span>
            <strong>{recommended.name}</strong>
          </div>
          <div className="auto-pipeline-grid">
            <div>
              <span>Image</span>
              <strong>{recommended.image_size}px</strong>
            </div>
            <div>
              <span>Normalize</span>
              <strong>{recommended.normalize}</strong>
            </div>
            <div>
              <span>Split</span>
              <strong>
                {Math.round(Number(recommended.train_split || 0) * 100)} /
                {Math.round(Number(recommended.val_split || 0) * 100)} /
                {Math.round(Number(recommended.test_split || 0) * 100)}
              </strong>
            </div>
          </div>
          {Array.isArray(recommended.notes) && recommended.notes.length > 0 && (
            <ul className="auto-pipeline-notes">
              {recommended.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <ul className="report-warnings">
          {warnings.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PipelineSummary({ pipeline }) {
  const config = pipeline.config || {};
  const augmentations = config.augmentations || {};
  const activeAugmentations = [
    augmentations.horizontal_flip && "H-Flip",
    augmentations.vertical_flip && "V-Flip",
    Number(augmentations.rotation_degrees || 0) > 0 && `Rotate ${augmentations.rotation_degrees}`,
    Number(augmentations.hsv_h || 0) > 0 && "HSV",
    augmentations.mosaic && "Mosaic",
    Number(augmentations.mixup || 0) > 0 && `MixUp ${augmentations.mixup}`,
  ].filter(Boolean);

  return (
    <div className="dataset-pipeline-summary">
      <div className="preprocess-metrics">
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
    </div>
  );
}

function DatasetManage({ user, projectId }) {
  const [project, setProject] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [modalError, setModalError] = useState("");
  const [pipelineModalError, setPipelineModalError] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [pipelineForm, setPipelineForm] = useState(clonePipelineForm());
  const [editingPipeline, setEditingPipeline] = useState(null);
  const [pipelineModalOpen, setPipelineModalOpen] = useState(false);
  const [pipelineSaving, setPipelineSaving] = useState(false);
  const [saving, setSaving] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedFileReport, setSelectedFileReport] = useState(null);
  const [creatingPipelineFor, setCreatingPipelineFor] = useState(null);

  const userId = user?.user_id;
  const userEmail = user?.email || "user";
  const pipelineSplitTotal = useMemo(() => splitTotal(pipelineForm), [pipelineForm]);
  const pipelinesByDataset = useMemo(() => {
    const groups = new Map();
    pipelines.forEach((pipeline) => {
      if (!pipeline.dataset_id) return;
      const key = String(pipeline.dataset_id);
      groups.set(key, [...(groups.get(key) || []), pipeline]);
    });
    return groups;
  }, [pipelines]);

  async function loadPage() {
    if (!projectId || !userId) return;
    setLoading(true);
    setError("");

    try {
      const [datasetsResponse, pipelinesResponse] = await Promise.all([
        fetch(`${API_BASE}/projects/${projectId}/datasets?user_id=${userId}`),
        fetch(`${API_BASE}/projects/${projectId}/preprocessing-pipelines?user_id=${userId}`),
      ]);
      const datasetsData = await datasetsResponse.json();
      const pipelinesData = await pipelinesResponse.json();

      if (!datasetsResponse.ok || !pipelinesResponse.ok) {
        setError(datasetsData.detail || pipelinesData.detail || "데이터셋 정보를 불러오지 못했습니다.");
        setProject(null);
        setDatasets([]);
        setPipelines([]);
        return;
      }

      setProject(datasetsData.project || pipelinesData.project);
      setDatasets(datasetsData.datasets || []);
      setPipelines(pipelinesData.pipelines || []);
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
      setProject(null);
      setDatasets([]);
      setPipelines([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPage();
  }, [projectId, userId]);

  function resetDatasetModal() {
    setModalOpen(false);
    setForm(emptyForm);
    setPipelineForm(clonePipelineForm());
    setSelectedFileReport(null);
    setModalError("");
  }

  function openCreateModal() {
    setForm(emptyForm);
    setPipelineForm(clonePipelineForm());
    setSelectedFileReport(null);
    setModalOpen(true);
    setError("");
    setNotice("");
    setModalError("");
  }

  function closeModal() {
    if (saving || analyzing) return;
    resetDatasetModal();
  }

  function handleChange(event) {
    const { name, value, type, checked } = event.target;
    const nextValue = type === "checkbox" ? checked : value;

    setForm((prev) => ({ ...prev, [name]: nextValue }));

    if (name === "task_type") {
      setSelectedFileReport(null);
      setPipelineForm((prev) => ({ ...prev, task_type: value }));
    }

    if (name === "pipeline_mode" && value === "manual" && !pipelineForm.name.trim()) {
      setPipelineForm((prev) =>
        clonePipelineForm({
          ...prev,
          name: form.dataset_name.trim() ? `${form.dataset_name.trim()}-pipeline` : "",
          task_type: form.task_type,
        }),
      );
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    setForm((prev) => ({ ...prev, dataset_zip: file }));

    if (!file) {
      setSelectedFileReport(null);
      return;
    }

    setSelectedFileReport({
      total_files: "-",
      image_count: "-",
      label_count: "-",
      class_count: "-",
      train_images: 0,
      val_images: 0,
      test_images: 0,
      warnings: file.name.toLowerCase().endsWith(".zip")
        ? [`선택됨: ${file.name} (${prettyBytes(file.size)})`]
        : ["zip 파일만 업로드할 수 있습니다."],
    });
  }

  async function analyzeDatasetZip() {
    if (!form.dataset_zip) {
      setModalError("분석할 zip 데이터셋 파일을 선택하세요.");
      return;
    }
    if (!form.dataset_zip.name.toLowerCase().endsWith(".zip")) {
      setModalError("데이터셋은 zip 파일만 분석할 수 있습니다.");
      return;
    }

    const formData = new FormData();
    formData.append("task_type", form.task_type);
    formData.append("dataset_zip", form.dataset_zip);

    setAnalyzing(true);
    setModalError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE}/datasets/analyze-zip`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setModalError(data.detail || "데이터셋 분석 중 오류가 발생했습니다.");
        return;
      }

      setSelectedFileReport(data.report);
      if (data.report?.recommended_pipeline) {
        setPipelineForm(
          pipelineFormFromRecommendation(data.report.recommended_pipeline, form.dataset_name.trim(), form.task_type),
        );
      }
    } catch {
      setModalError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    } finally {
      setAnalyzing(false);
    }
  }

  function handlePipelineFormChange(event) {
    const { name, value, type, checked } = event.target;
    const numericFields = new Set(["image_size", "train_split", "val_split", "test_split"]);

    setPipelineForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : numericFields.has(name) ? Number(value) : value,
    }));
  }

  function handlePipelineAugmentationChange(event) {
    const { name, value, type, checked } = event.target;
    const numericFields = new Set(["rotation_degrees", "hsv_h", "hsv_s", "hsv_v", "mixup"]);

    setPipelineForm((prev) => ({
      ...prev,
      augmentations: {
        ...prev.augmentations,
        [name]: type === "checkbox" ? checked : numericFields.has(name) ? Number(value) : value,
      },
    }));
  }

  function validatePipelineForm(requireName = true) {
    if (requireName && !pipelineForm.name.trim()) return "전처리 이름을 입력하세요.";
    if (pipelineForm.image_size < 64 || pipelineForm.image_size > 2048) {
      return "Image Size는 64 이상 2048 이하로 입력하세요.";
    }
    if (Math.abs(pipelineSplitTotal - 1) > 0.001) return "Train/Val/Test 비율의 합은 1.0이어야 합니다.";
    if (pipelineForm.train_split < 0 || pipelineForm.val_split < 0 || pipelineForm.test_split < 0) {
      return "분할 비율은 0 이상이어야 합니다.";
    }
    if (pipelineForm.augmentations.rotation_degrees < 0 || pipelineForm.augmentations.rotation_degrees > 180) {
      return "Rotation은 0 이상 180 이하로 입력하세요.";
    }
    for (const key of ["hsv_h", "hsv_s", "hsv_v", "mixup"]) {
      if (pipelineForm.augmentations[key] < 0 || pipelineForm.augmentations[key] > 1) {
        return `${key} 값은 0 이상 1 이하로 입력하세요.`;
      }
    }
    return "";
  }

  function validateForm() {
    if (!form.dataset_name.trim()) return "데이터셋 이름을 입력하세요.";
    if (!form.dataset_zip) return "zip 데이터셋 파일을 선택하세요.";
    if (!form.dataset_zip.name.toLowerCase().endsWith(".zip")) return "데이터셋은 zip 파일만 가능합니다.";
    if (form.pipeline_mode === "manual") return validatePipelineForm(true);
    return "";
  }

  function buildPipelinePayload(datasetId = editingPipeline?.dataset_id || null) {
    return {
      user_id: userId,
      dataset_id: datasetId ? Number(datasetId) : null,
      name: pipelineForm.name.trim(),
      description: pipelineForm.description.trim(),
      task_type: pipelineForm.task_type,
      image_size: pipelineForm.image_size,
      keep_aspect_ratio: pipelineForm.keep_aspect_ratio,
      normalize: pipelineForm.normalize,
      train_split: pipelineForm.train_split,
      val_split: pipelineForm.val_split,
      test_split: pipelineForm.test_split,
      augmentations: pipelineForm.augmentations,
    };
  }

  async function createManualPipeline(datasetId) {
    const response = await fetch(`${API_BASE}/projects/${projectId}/preprocessing-pipelines`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPipelinePayload(datasetId)),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "수동 전처리 생성 중 오류가 발생했습니다.");
    }
    return data.pipeline;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const validation = validateForm();
    if (validation) {
      setModalError(validation);
      return;
    }

    const formData = new FormData();
    formData.append("user_id", String(userId));
    formData.append("dataset_name", form.dataset_name.trim());
    formData.append("description", form.description.trim());
    formData.append("task_type", form.task_type);
    formData.append("create_auto_pipeline", String(form.pipeline_mode === "auto"));
    formData.append("dataset_zip", form.dataset_zip);

    setSaving(true);
    setModalError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/datasets`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setModalError(data.detail || "데이터셋 업로드 중 오류가 발생했습니다.");
        return;
      }

      let manualPipeline = null;
      let manualPipelineError = "";
      if (form.pipeline_mode === "manual" && data.dataset?.id) {
        try {
          manualPipeline = await createManualPipeline(data.dataset.id);
        } catch (pipelineError) {
          manualPipelineError = pipelineError.message;
        }
      }

      await loadPage();

      if (manualPipelineError) {
        setModalError(`데이터셋은 업로드되었지만 전처리 생성은 실패했습니다. ${manualPipelineError}`);
        return;
      } else if (manualPipeline) {
        setNotice(`데이터셋 "${data.dataset.name}"에 전처리 "${manualPipeline.name}"이 연결되었습니다.`);
      } else if (data.auto_pipeline) {
        setNotice(`데이터셋 "${data.dataset.name}"에 자동 전처리 "${data.auto_pipeline.name}"이 연결되었습니다.`);
      } else {
        setNotice("데이터셋이 업로드되었습니다.");
      }
      resetDatasetModal();
    } catch {
      setModalError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    } finally {
      setSaving(false);
    }
  }

  async function deleteDataset(dataset) {
    const ok = window.confirm(`"${dataset.name}" 데이터셋을 삭제하시겠습니까?`);
    if (!ok) return;

    setError("");
    setNotice("");
    try {
      const response = await fetch(
        `${API_BASE}/projects/${projectId}/datasets/${dataset.id}?user_id=${userId}`,
        { method: "DELETE" },
      );
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "데이터셋 삭제 중 오류가 발생했습니다.");
        return;
      }
      await loadPage();
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    }
  }

  async function createAutoPipeline(dataset) {
    setError("");
    setNotice("");
    setCreatingPipelineFor(dataset.id);

    try {
      const response = await fetch(
        `${API_BASE}/projects/${projectId}/datasets/${dataset.id}/auto-preprocessing-pipeline?user_id=${userId}`,
        { method: "POST" },
      );
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "자동 전처리 생성 중 오류가 발생했습니다.");
        return;
      }

      await loadPage();
      setNotice(`데이터셋 "${dataset.name}"에 자동 전처리 "${data.pipeline.name}"이 연결되었습니다.`);
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
    } finally {
      setCreatingPipelineFor(null);
    }
  }

  function openPipelineEditModal(pipeline) {
    const config = pipeline.config || {};
    setEditingPipeline(pipeline);
    setPipelineForm(
      clonePipelineForm({
        name: pipeline.name || "",
        description: pipeline.description || "",
        task_type: pipeline.task_type || "detect",
        image_size: pipeline.image_size || 224,
        keep_aspect_ratio: config.keep_aspect_ratio ?? true,
        normalize: config.normalize || "zero_one",
        train_split: pipeline.train_split ?? 0.8,
        val_split: pipeline.val_split ?? 0.2,
        test_split: pipeline.test_split ?? 0,
        augmentations: config.augmentations || {},
      }),
    );
    setPipelineModalOpen(true);
    setError("");
    setNotice("");
    setPipelineModalError("");
  }

  function closePipelineModal() {
    if (pipelineSaving) return;
    setPipelineModalOpen(false);
    setEditingPipeline(null);
    setPipelineForm(clonePipelineForm());
    setPipelineModalError("");
  }

  async function savePipeline(event) {
    event.preventDefault();
    const validation = validatePipelineForm(true);
    if (validation) {
      setPipelineModalError(validation);
      return;
    }

    setPipelineSaving(true);
    setPipelineModalError("");
    setNotice("");

    try {
      const response = await fetch(
        `${API_BASE}/projects/${projectId}/preprocessing-pipelines/${editingPipeline.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildPipelinePayload(editingPipeline.dataset_id)),
        },
      );
      const data = await response.json();

      if (!response.ok) {
        setPipelineModalError(data.detail || "전처리 저장 중 오류가 발생했습니다.");
        return;
      }

      await loadPage();
      setNotice(`전처리 "${data.pipeline.name}"이 저장되었습니다.`);
      closePipelineModal();
    } catch {
      setPipelineModalError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
    } finally {
      setPipelineSaving(false);
    }
  }

  async function deletePipeline(pipeline) {
    const ok = window.confirm(`"${pipeline.name}" 전처리를 삭제하시겠습니까?`);
    if (!ok) return;

    setError("");
    setNotice("");
    try {
      const response = await fetch(
        `${API_BASE}/projects/${projectId}/preprocessing-pipelines/${pipeline.id}?user_id=${userId}`,
        { method: "DELETE" },
      );
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "전처리 삭제 중 오류가 발생했습니다.");
        return;
      }
      await loadPage();
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
    }
  }

  function goBack() {
    window.location.href = "/projects";
  }

  function goTraining() {
    window.location.href = `/projects/${projectId}/training`;
  }

  function logout() {
    localStorage.removeItem("dlops_user");
    window.location.href = "/";
  }

  function renderPipelineFields() {
    return (
      <>
        <section className="modal-section">
          <h3>Preprocessing Basic</h3>
          <div className="form-grid">
            <label className="field">
              <span>전처리 이름</span>
              <input name="name" value={pipelineForm.name} onChange={handlePipelineFormChange} placeholder="default-yolo-augment" />
            </label>
            <label className="field">
              <span>작업 유형</span>
              <select name="task_type" value={pipelineForm.task_type} onChange={handlePipelineFormChange}>
                <option value="detect">탐지 Detect</option>
                <option value="classify">분류 Classify</option>
              </select>
            </label>
            <label className="field wide">
              <span>설명</span>
              <textarea
                name="description"
                value={pipelineForm.description}
                onChange={handlePipelineFormChange}
                rows={3}
                placeholder="데이터셋 특성이나 적용 목적을 적어주세요."
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
                value={pipelineForm.image_size}
                onChange={handlePipelineFormChange}
              />
            </label>
            <label className="field">
              <span>Normalize</span>
              <select name="normalize" value={pipelineForm.normalize} onChange={handlePipelineFormChange}>
                <option value="zero_one">0-1 Scale</option>
                <option value="imagenet">ImageNet Mean/Std</option>
                <option value="none">None</option>
              </select>
            </label>
            <label className="toggle-field">
              <input
                type="checkbox"
                name="keep_aspect_ratio"
                checked={pipelineForm.keep_aspect_ratio}
                onChange={handlePipelineFormChange}
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
              <input type="number" name="train_split" min="0" max="1" step="0.05" value={pipelineForm.train_split} onChange={handlePipelineFormChange} />
            </label>
            <label className="field">
              <span>Val</span>
              <input type="number" name="val_split" min="0" max="1" step="0.05" value={pipelineForm.val_split} onChange={handlePipelineFormChange} />
            </label>
            <label className="field">
              <span>Test</span>
              <input type="number" name="test_split" min="0" max="1" step="0.05" value={pipelineForm.test_split} onChange={handlePipelineFormChange} />
            </label>
            <div className={`split-total ${Math.abs(pipelineSplitTotal - 1) > 0.001 ? "invalid" : ""}`}>
              <span>Total</span>
              <strong>{pipelineSplitTotal.toFixed(2)}</strong>
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
            ].map(([name, label]) => (
              <label className="toggle-field" key={name}>
                <input
                  type="checkbox"
                  name={name}
                  checked={Boolean(pipelineForm.augmentations[name])}
                  onChange={handlePipelineAugmentationChange}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>

          <div className="form-grid augmentation-numbers">
            <label className="field">
              <span>Rotation Degrees</span>
              <input type="number" name="rotation_degrees" min="0" max="180" value={pipelineForm.augmentations.rotation_degrees} onChange={handlePipelineAugmentationChange} />
            </label>
            <label className="field">
              <span>HSV H</span>
              <input type="number" name="hsv_h" min="0" max="1" step="0.001" value={pipelineForm.augmentations.hsv_h} onChange={handlePipelineAugmentationChange} />
            </label>
            <label className="field">
              <span>HSV S</span>
              <input type="number" name="hsv_s" min="0" max="1" step="0.01" value={pipelineForm.augmentations.hsv_s} onChange={handlePipelineAugmentationChange} />
            </label>
            <label className="field">
              <span>HSV V</span>
              <input type="number" name="hsv_v" min="0" max="1" step="0.01" value={pipelineForm.augmentations.hsv_v} onChange={handlePipelineAugmentationChange} />
            </label>
            <label className="field">
              <span>MixUp</span>
              <input type="number" name="mixup" min="0" max="1" step="0.05" value={pipelineForm.augmentations.mixup} onChange={handlePipelineAugmentationChange} />
            </label>
          </div>
        </section>
      </>
    );
  }

  function renderAutoRecommendationPreview() {
    const recommendation = selectedFileReport?.recommended_pipeline;
    if (!recommendation) {
      return (
        <div className="readonly-preprocess muted">
          분석 버튼을 누르면 이 데이터셋에 맞춘 자동 전처리 추천값이 여기에 표시됩니다.
        </div>
      );
    }

    const augmentations = recommendation.augmentations || {};
    const items = [
      ["Task", recommendation.task_type || form.task_type],
      ["Image Size", `${recommendation.image_size || 224}px`],
      ["Keep Ratio", recommendation.keep_aspect_ratio === false ? "Off" : "On"],
      ["Normalize", recommendation.normalize || "zero_one"],
      ["Train", asPercent(recommendation.train_split)],
      ["Val", asPercent(recommendation.val_split)],
      ["Test", asPercent(recommendation.test_split)],
      ["H Flip", augmentations.horizontal_flip ? "On" : "Off"],
      ["V Flip", augmentations.vertical_flip ? "On" : "Off"],
      ["Rotation", `${augmentations.rotation_degrees ?? 0}`],
      ["HSV H", `${augmentations.hsv_h ?? 0}`],
      ["HSV S", `${augmentations.hsv_s ?? 0}`],
      ["HSV V", `${augmentations.hsv_v ?? 0}`],
      ["Mosaic", augmentations.mosaic ? "On" : "Off"],
      ["MixUp", `${augmentations.mixup ?? 0}`],
    ];

    return (
      <div className="readonly-preprocess">
        <div className="readonly-preprocess-head">
          <span>Auto Recommendation</span>
          <strong>{recommendation.name}</strong>
        </div>
        <div className="readonly-preprocess-grid">
          {items.map(([label, value]) => (
            <div key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
        {Array.isArray(recommendation.notes) && recommendation.notes.length > 0 && (
          <ul className="auto-pipeline-notes">
            {recommendation.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  return (
    <main className="dataset-page">
      <aside className="dataset-sidebar">
        <button className="back-button active" type="button">
          Dataset Management
        </button>
        <button className="back-button" type="button" onClick={goTraining}>
          Training
        </button>
        <button className="back-button" type="button" onClick={goBack}>
          Back to Projects
        </button>

        <div className="dataset-brand">
          <span className="brand-dot" />
          <div>
            <strong>{project?.name || "Project"}</strong>
            <p>{project?.folder_path || "Loading project path..."}</p>
          </div>
        </div>

        <div className="sidebar-panel">
          <span>Signed in</span>
          <strong>{userEmail}</strong>
          <button type="button" onClick={logout}>
            Logout
          </button>
        </div>

        <div className="sidebar-panel">
          <span>Datasets</span>
          <strong>{datasets.length}</strong>
          <p>YOLO 학습에 사용할 zip 데이터셋과 해당 데이터셋의 전처리 설정을 함께 관리합니다.</p>
        </div>
      </aside>

      <section className="dataset-workspace">
        <header className="dataset-header">
          <div>
            <p className="eyebrow">Dataset Management</p>
            <h1>데이터셋 관리</h1>
            <span>데이터셋 업로드 창에서 구조 분석과 전처리 생성을 함께 처리합니다.</span>
          </div>
          <button className="primary-action" type="button" onClick={openCreateModal}>
            New Dataset
          </button>
        </header>

        {error && <div className="dataset-error">{error}</div>}
        {notice && <div className="dataset-notice">{notice}</div>}

        {loading ? (
          <div className="dataset-state">프로젝트 데이터셋 정보를 불러오는 중입니다.</div>
        ) : datasets.length === 0 ? (
          <div className="dataset-state">
            <strong>아직 업로드된 데이터셋이 없습니다.</strong>
            <p>New Dataset을 눌러 YOLO zip 데이터셋을 추가하세요.</p>
          </div>
        ) : (
          <div className="training-grid">
            {datasets.map((dataset) => {
              const attachedPipelines = pipelinesByDataset.get(String(dataset.id)) || [];
              return (
                <article className="training-card" key={dataset.id}>
                  <div className="training-card-head">
                    <div>
                      <p className="eyebrow">{dataset.task_type === "detect" ? "Detection Dataset" : "Classification Dataset"}</p>
                      <h2>{dataset.name}</h2>
                    </div>
                    <span className="status-pill">{dataset.status}</span>
                  </div>

                  <p className="training-description">{dataset.description || "데이터셋 설명이 없습니다."}</p>

                  <div className="dataset-file-line">
                    <span>{dataset.original_filename}</span>
                    <strong>{prettyBytes(dataset.file_size || 0)}</strong>
                  </div>

                  <DatasetReport report={dataset.report} />

                  <div className="dataset-attached-preprocessing">
                    <div className="attached-head">
                      <span>전처리 설정</span>
                      <button
                        type="button"
                        onClick={() => createAutoPipeline(dataset)}
                        disabled={creatingPipelineFor === dataset.id}
                      >
                        {creatingPipelineFor === dataset.id ? "Creating..." : "Auto Add"}
                      </button>
                    </div>
                    {attachedPipelines.length === 0 ? (
                      <p>연결된 전처리가 없습니다. 데이터셋 업로드 창에서 함께 만들거나 Auto Add로 추천값을 생성하세요.</p>
                    ) : (
                      attachedPipelines.map((pipeline) => (
                        <div className="attached-pipeline-card" key={pipeline.id}>
                          <div className="attached-pipeline-head">
                            <div>
                              <strong>{pipeline.name}</strong>
                              <span>{pipeline.source === "auto" ? "Auto" : "Manual"}</span>
                            </div>
                            <div>
                              <button type="button" onClick={() => openPipelineEditModal(pipeline)}>
                                Edit
                              </button>
                              <button className="danger-action" type="button" onClick={() => deletePipeline(pipeline)}>
                                Delete
                              </button>
                            </div>
                          </div>
                          <PipelineSummary pipeline={pipeline} />
                        </div>
                      ))
                    )}
                  </div>

                  <div className="training-card-footer">
                    <span>Uploaded {formatDate(dataset.created_at)}</span>
                    <div className="dataset-card-actions">
                      <button className="danger-action" type="button" onClick={() => deleteDataset(dataset)}>
                        Delete Dataset
                      </button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>

      {modalOpen && (
        <div className="dataset-modal-backdrop">
          <form className="dataset-modal" onSubmit={handleSubmit}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">New Dataset</p>
                <h2>데이터셋 업로드</h2>
              </div>
              <button type="button" onClick={closeModal} disabled={saving}>
                Close
              </button>
            </div>

            {modalError && <div className="dataset-error modal-message">{modalError}</div>}

            <div className="modal-sections">
              <section className="modal-section">
                <h3>Basic</h3>
                <div className="form-grid">
                  <label className="field">
                    <span>데이터셋 이름</span>
                    <input
                      name="dataset_name"
                      value={form.dataset_name}
                      onChange={handleChange}
                      placeholder="예: surface-defect-dataset"
                      required
                    />
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
                      placeholder="데이터 출처, 클래스 구성, 라벨 기준을 적어주세요."
                      rows={3}
                    />
                  </label>
                </div>
              </section>

              <section className="modal-section">
                <h3>Dataset Upload</h3>
                <label className="upload-zone">
                  <input type="file" accept=".zip" onChange={handleFileChange} />
                  <strong>{form.dataset_zip ? form.dataset_zip.name : "zip 데이터셋 선택"}</strong>
                  <span>YOLO data.yaml, images, labels 구조를 포함한 zip 파일을 권장합니다.</span>
                </label>
                <div className="upload-actions">
                  <button type="button" onClick={analyzeDatasetZip} disabled={!form.dataset_zip || analyzing || saving}>
                    {analyzing ? "분석 중..." : "분석"}
                  </button>
                  <span>업로드 전에 zip 내부 구조를 확인하고 전처리 추천값을 자동으로 계산합니다.</span>
                </div>
                <DatasetReport report={selectedFileReport} />
              </section>

              <section className="modal-section">
                <h3>Dataset Preprocessing</h3>
                <div className="pipeline-mode-grid">
                  {[
                    ["auto", "Auto", "분석 추천값으로 이 데이터셋의 전처리 자동 생성"],
                    ["manual", "Manual", "추천값을 기반으로 직접 수정해서 생성"],
                    ["none", "Dataset Only", "이번에는 데이터셋만 업로드"],
                  ].map(([value, title, description]) => (
                    <label className={`mode-card ${form.pipeline_mode === value ? "active" : ""}`} key={value}>
                      <input type="radio" name="pipeline_mode" value={value} checked={form.pipeline_mode === value} onChange={handleChange} />
                      <span>
                        <strong>{title}</strong>
                        <small>{description}</small>
                      </span>
                    </label>
                  ))}
                </div>

                {form.pipeline_mode === "auto" && (
                  <>
                    <div className="auto-pipeline-toggle">
                      <span>
                        이 데이터셋에 자동 전처리 연결
                        <small>분석 버튼을 누르면 이미지 수, 라벨 수, 클래스 수에 맞춘 추천값을 확인할 수 있습니다.</small>
                      </span>
                    </div>
                    {renderAutoRecommendationPreview()}
                  </>
                )}
              </section>

              {form.pipeline_mode === "manual" && renderPipelineFields()}
            </div>

            <div className="modal-footer">
              <button type="button" className="secondary" onClick={closeModal} disabled={saving}>
                취소
              </button>
              <button type="submit" disabled={saving}>
                {saving ? "업로드 중..." : "데이터셋 업로드"}
              </button>
            </div>
          </form>
        </div>
      )}

      {pipelineModalOpen && (
        <div className="dataset-modal-backdrop">
          <form className="dataset-modal" onSubmit={savePipeline}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">Dataset Preprocessing</p>
                <h2>전처리 수정</h2>
              </div>
              <button type="button" onClick={closePipelineModal} disabled={pipelineSaving}>
                Close
              </button>
            </div>

            {pipelineModalError && <div className="dataset-error modal-message">{pipelineModalError}</div>}

            <div className="modal-sections">{renderPipelineFields()}</div>

            <div className="modal-footer">
              <button type="button" className="secondary" onClick={closePipelineModal} disabled={pipelineSaving}>
                취소
              </button>
              <button type="submit" disabled={pipelineSaving}>
                {pipelineSaving ? "저장 중..." : "저장"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default DatasetManage;
