import React, { useEffect, useMemo, useRef, useState } from "react";
import "./TrainingManage.css";

const API_BASE = "http://localhost:8000";

const fallbackModels = [
  { id: "yolov8n.pt", name: "YOLOv8 Nano", task: "detect", source: "builtin" },
  { id: "yolov8s.pt", name: "YOLOv8 Small", task: "detect", source: "builtin" },
  { id: "yolov8m.pt", name: "YOLOv8 Medium", task: "detect", source: "builtin" },
  { id: "yolov8l.pt", name: "YOLOv8 Large", task: "detect", source: "builtin" },
  { id: "yolov8x.pt", name: "YOLOv8 XLarge", task: "detect", source: "builtin" },
  { id: "convnext_tiny", name: "ConvNeXt Tiny", task: "classify", source: "builtin" },
  { id: "convnext_small", name: "ConvNeXt Small", task: "classify", source: "builtin" },
  { id: "convnext_base", name: "ConvNeXt Base", task: "classify", source: "builtin" },
];

const defaultImageSizeByTask = {
  detect: 640,
  classify: 224,
};

const emptyForm = {
  name: "",
  description: "",
  task_type: "detect",
  dataset_id: "",
  preprocessing_pipeline_id: "",
  yolo_model: "yolov8n.pt",
  optimizer: "AdamW",
  image_size: 640,
  epochs: 10,
  batch_min: 16,
  batch_max: 64,
  lr_initial_min: 0.0001,
  lr_initial_max: 0.01,
  momentum_min: 0.8,
  momentum_max: 0.99,
};

const emptyModelForm = {
  name: "",
  description: "",
  task_type: "detect",
  model_file: null,
};

const metricLabels = {
  map50_95: "mAP50-95",
  map50: "mAP50",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  accuracy_top1: "Top-1 Acc",
  accuracy_top5: "Top-5 Acc",
  train_loss: "Train Loss",
  val_loss: "Val Loss",
};

const artifactItems = [
  { key: "run_dir", label: "Run Dir" },
  { key: "best_model_path", label: "Best Model" },
  { key: "last_model_path", label: "Last Model" },
  { key: "results_csv", label: "Results CSV" },
  { key: "results_plot", label: "Results Plot" },
  { key: "confusion_matrix", label: "Confusion Matrix" },
  { key: "labels", label: "Labels" },
  { key: "train_batch", label: "Train Batch" },
  { key: "val_batch", label: "Val Batch" },
];

const previewArtifactKeys = ["results_plot", "confusion_matrix", "labels", "train_batch", "val_batch"];

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

function formatMetric(value, isLoss = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  if (isLoss) return number.toFixed(4);
  return `${(number * 100).toFixed(1)}%`;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "-";
  const total = Number(seconds);
  const minutes = Math.floor(total / 60);
  const remain = Math.floor(total % 60);
  if (minutes <= 0) return `${remain}s`;
  return `${minutes}m ${remain}s`;
}

function getDisplayProgress(job) {
  const rawProgress = Number(job?.progress || 0);
  const progress = Math.max(0, Math.min(100, rawProgress));
  if (["QUEUED", "RUNNING", "STOPPING"].includes(job?.status) && progress >= 100) return 99;
  return progress;
}

function getDisplayEpoch(job) {
  const total = Math.max(1, Number(job?.epochs || 1));
  return Math.max(0, Math.min(total, Number(job?.current_epoch || 0)));
}

function ProgressMeter({ job }) {
  const targetProgress = getDisplayProgress(job);
  const [displayProgress, setDisplayProgress] = useState(targetProgress);

  useEffect(() => {
    setDisplayProgress(getDisplayProgress(job));
  }, [job?.id]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setDisplayProgress((prev) => {
        const current = Number.isFinite(Number(prev)) ? Number(prev) : 0;
        const target = Number.isFinite(Number(targetProgress)) ? Number(targetProgress) : 0;
        if (Math.abs(target - current) < 0.12) return target;
        if (target < current) return target;

        const step = Math.max((target - current) * 0.18, 0.35);
        return Math.min(target, current + step);
      });
    }, 80);

    return () => window.clearInterval(timer);
  }, [targetProgress]);

  return (
    <>
      <div className="progress-head">
        <span>Progress</span>
        <strong>{Math.round(displayProgress)}%</strong>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: `${displayProgress}%` }} />
      </div>
      <small>Epoch {getDisplayEpoch(job)} / {job?.epochs || 0}</small>
    </>
  );
}

function getPrimaryMetricKey(job) {
  const metrics = job?.result?.metrics || {};
  if (job?.task_type === "classify") {
    return metrics.accuracy_top1 !== undefined ? "accuracy_top1" : "f1";
  }
  return metrics.map50_95 !== undefined ? "map50_95" : metrics.map50 !== undefined ? "map50" : "f1";
}

function getPrimaryScore(job) {
  const metrics = job?.result?.metrics || {};
  const key = getPrimaryMetricKey(job);
  const value = metrics[key] ?? job?.result?.primary_score;
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function formatDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const sign = Number(value) >= 0 ? "+" : "";
  return `${sign}${(Number(value) * 100).toFixed(1)}%p`;
}

function artifactImageUrl(projectId, userId, path) {
  if (!projectId || !userId || !path) return "";
  return `${API_BASE}/projects/${projectId}/artifact-file?user_id=${userId}&path=${encodeURIComponent(path)}`;
}

function TrainingMetricCards({ job }) {
  const metrics = job?.result?.metrics || {};
  const keys = ["map50_95", "map50", "precision", "recall", "f1", "accuracy_top1", "accuracy_top5", "train_loss", "val_loss"].filter(
    (key) => metrics[key] !== undefined,
  );

  if (!keys.length) {
    return <div className="live-empty-box">아직 표시할 성능 지표가 없습니다.</div>;
  }

  return (
    <div className="live-metric-grid">
      {keys.map((key) => (
        <div key={key}>
          <span>{metricLabels[key] || key}</span>
          <strong>{formatMetric(metrics[key], key.includes("loss"))}</strong>
        </div>
      ))}
    </div>
  );
}

function TrainingMiniChart({ job }) {
  const rows = job?.result?.epoch_metrics || [];
  const primaryMetricKey = getPrimaryMetricKey(job);
  const lines = [
    { key: "train_loss", className: "line-loss" },
    { key: "val_loss", className: "line-val" },
    { key: primaryMetricKey, className: "line-score" },
  ];
  const width = 420;
  const height = 160;
  const padding = 18;
  const values = rows.flatMap((row) => lines.map((line) => Number(row[line.key])).filter((value) => Number.isFinite(value)));

  if (!rows.length || values.length === 0) {
    return <div className="live-empty-box">epoch 그래프는 학습 로그가 쌓이면 표시됩니다.</div>;
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = maxValue - minValue || 1;

  function point(row, index, key) {
    const value = Number(row[key]);
    if (!Number.isFinite(value)) return null;
    const x = padding + (rows.length === 1 ? 0 : (index / (rows.length - 1)) * (width - padding * 2));
    const y = padding + (height - padding * 2) - ((value - minValue) / range) * (height - padding * 2);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }

  return (
    <div className="live-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="training live metric chart">
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        {lines.map((line) => {
          const points = rows.map((row, index) => point(row, index, line.key)).filter(Boolean).join(" ");
          return <polyline key={line.key} points={points} className={line.className} />;
        })}
      </svg>
      <div className="live-chart-legend">
        <span><i className="line-loss" />Train Loss</span>
        <span><i className="line-val" />Val Loss</span>
        <span><i className="line-score" />{metricLabels[primaryMetricKey] || "Score"}</span>
      </div>
    </div>
  );
}

function TrainingManage({ user, projectId }) {
  const [project, setProject] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [pipelines, setPipelines] = useState([]);
  const [models, setModels] = useState(fallbackModels);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [modelModalOpen, setModelModalOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [modelForm, setModelForm] = useState(emptyModelForm);
  const [saving, setSaving] = useState(false);
  const [modelSaving, setModelSaving] = useState(false);
  const [expandedLogs, setExpandedLogs] = useState({});
  const [selectedJobId, setSelectedJobId] = useState("");
  const liveLogBoxRef = useRef(null);

  const userId = user?.user_id;
  const userEmail = user?.email || "user";

  const modelOptions = useMemo(
    () => models.filter((model) => model.task === form.task_type),
    [form.task_type, models],
  );

  const datasetOptions = useMemo(
    () => datasets.filter((dataset) => dataset.task_type === form.task_type),
    [form.task_type, datasets],
  );

  const pipelineOptions = useMemo(
    () =>
      pipelines.filter(
        (pipeline) =>
          pipeline.task_type === form.task_type &&
          form.dataset_id &&
          String(pipeline.dataset_id) === String(form.dataset_id),
      ),
    [form.task_type, form.dataset_id, pipelines],
  );

  const hasActiveJob = jobs.some((job) => ["QUEUED", "RUNNING", "STOPPING"].includes(job.status));
  const selectedJob = jobs.find((job) => String(job.id) === String(selectedJobId)) || jobs.find((job) => ["QUEUED", "RUNNING", "STOPPING"].includes(job.status)) || jobs[0];
  const selectedLogs = selectedJob?.logs || [];
  const selectedResult = selectedJob?.result || {};
  const selectedArtifacts = selectedResult.artifacts || selectedResult;
  const completedJobs = useMemo(() => {
    return jobs
      .filter((job) => job.status === "COMPLETED" && getPrimaryScore(job) !== null)
      .sort((a, b) => getPrimaryScore(b) - getPrimaryScore(a));
  }, [jobs]);
  const bestJob = completedJobs[0] || null;
  const topRuns = completedJobs.slice(0, 5);
  const selectedPrimaryKey = getPrimaryMetricKey(selectedJob);
  const selectedPrimaryScore = getPrimaryScore(selectedJob);
  const bestPrimaryScore = getPrimaryScore(bestJob);
  const selectedBestDelta =
    selectedPrimaryScore !== null && bestPrimaryScore !== null ? selectedPrimaryScore - bestPrimaryScore : null;
  const previewArtifacts = previewArtifactKeys
    .map((key) => ({
      key,
      label: artifactItems.find((item) => item.key === key)?.label || key,
      path: selectedArtifacts?.[key] || "",
      exists: selectedArtifacts?.exists?.[key],
    }))
    .filter((item) => item.path && item.exists !== false);

  async function loadPage(silent = false) {
    if (!projectId || !userId) return;
    if (!silent) setLoading(true);
    setError("");

    try {
      const [jobsResponse, datasetsResponse, pipelinesResponse, modelsResponse] = await Promise.all([
        fetch(`${API_BASE}/projects/${projectId}/training-jobs?user_id=${userId}`),
        fetch(`${API_BASE}/projects/${projectId}/datasets?user_id=${userId}`),
        fetch(`${API_BASE}/projects/${projectId}/preprocessing-pipelines?user_id=${userId}`),
        fetch(`${API_BASE}/training-models?user_id=${userId}&project_id=${projectId}`),
      ]);

      const jobsData = await jobsResponse.json();
      const datasetsData = await datasetsResponse.json();
      const pipelinesData = await pipelinesResponse.json();

      if (!jobsResponse.ok || !datasetsResponse.ok || !pipelinesResponse.ok) {
        setError(jobsData.detail || datasetsData.detail || pipelinesData.detail || "학습 정보를 불러오지 못했습니다.");
        setJobs([]);
        setProject(null);
        return;
      }

      setProject(jobsData.project || datasetsData.project || pipelinesData.project);
      const nextJobs = jobsData.training_jobs || [];
      setJobs(nextJobs);
      setSelectedJobId((prev) => {
        if (nextJobs.some((job) => String(job.id) === String(prev))) return prev;
        const activeJob = nextJobs.find((job) => ["QUEUED", "RUNNING", "STOPPING"].includes(job.status));
        return activeJob ? String(activeJob.id) : nextJobs[0] ? String(nextJobs[0].id) : "";
      });
      setDatasets(datasetsData.datasets || []);
      setPipelines(pipelinesData.pipelines || []);

      if (modelsResponse.ok) {
        const modelData = await modelsResponse.json();
        if (Array.isArray(modelData.models) && modelData.models.length > 0) {
          setModels(modelData.models);
        }
      }
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
      setProject(null);
      setJobs([]);
      setDatasets([]);
      setPipelines([]);
    } finally {
      if (!silent) setLoading(false);
    }
  }

  useEffect(() => {
    loadPage();
  }, [projectId, userId]);

  useEffect(() => {
    if (!hasActiveJob) return undefined;
    const timer = window.setInterval(() => loadPage(true), 600);
    return () => window.clearInterval(timer);
  }, [hasActiveJob, projectId, userId]);

  useEffect(() => {
    if (!selectedJob) return;
    if (!["QUEUED", "RUNNING", "STOPPING"].includes(selectedJob.status)) return;
    const node = liveLogBoxRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [selectedJob?.id, selectedJob?.status, selectedLogs.length]);

  useEffect(() => {
    const compatibleModel = modelOptions.some((model) => model.id === form.yolo_model);
    const compatibleDataset = datasetOptions.some((dataset) => String(dataset.id) === String(form.dataset_id));
    const compatiblePipeline = pipelineOptions.some(
      (pipeline) => String(pipeline.id) === String(form.preprocessing_pipeline_id),
    );

    setForm((prev) => ({
      ...prev,
      yolo_model: compatibleModel || !modelOptions[0] ? prev.yolo_model : modelOptions[0].id,
      dataset_id: compatibleDataset || !datasetOptions[0] ? prev.dataset_id : String(datasetOptions[0].id),
      preprocessing_pipeline_id:
        prev.preprocessing_pipeline_id === "" || compatiblePipeline ? prev.preprocessing_pipeline_id : "",
    }));
  }, [modelOptions, datasetOptions, pipelineOptions, form.yolo_model, form.dataset_id, form.preprocessing_pipeline_id]);

  function openCreateModal() {
    const createTask = emptyForm.task_type;
    const createModels = models.filter((model) => model.task === createTask);
    const createDatasets = datasets.filter((dataset) => dataset.task_type === createTask);
    setForm({
      ...emptyForm,
      image_size: defaultImageSizeByTask[createTask] || emptyForm.image_size,
      dataset_id: createDatasets[0] ? String(createDatasets[0].id) : "",
      yolo_model: createModels[0] ? createModels[0].id : emptyForm.yolo_model,
    });
    setModalOpen(true);
    setError("");
  }

  function openRetuneModal(job = selectedJob) {
    if (!job) return;
    const retryIndex = jobs.length + 1;
    setForm({
      ...emptyForm,
      name: `${job.name}-retune-${retryIndex}`.slice(0, 150),
      description: "",
      task_type: job.task_type || "detect",
      dataset_id: job.dataset_id ? String(job.dataset_id) : "",
      preprocessing_pipeline_id: job.preprocessing_pipeline_id ? String(job.preprocessing_pipeline_id) : "",
      yolo_model: job.yolo_model || emptyForm.yolo_model,
      optimizer: job.optimizer || emptyForm.optimizer,
      image_size: job.image_size || emptyForm.image_size,
      epochs: Math.min(1000, Math.max(1, Number(job.epochs || emptyForm.epochs))),
      batch_min: job.batch_min || emptyForm.batch_min,
      batch_max: job.batch_max || emptyForm.batch_max,
      lr_initial_min: job.lr_initial_min || emptyForm.lr_initial_min,
      lr_initial_max: job.lr_initial_max || emptyForm.lr_initial_max,
      momentum_min: job.momentum_min || emptyForm.momentum_min,
      momentum_max: job.momentum_max || emptyForm.momentum_max,
    });
    setModalOpen(true);
    setError("");
  }

  function closeModal() {
    if (saving) return;
    setModalOpen(false);
    setForm(emptyForm);
  }

  function openModelModal(taskType = form.task_type) {
    setModelForm({ ...emptyModelForm, task_type: taskType });
    setModelModalOpen(true);
    setError("");
  }

  function closeModelModal() {
    if (modelSaving) return;
    setModelModalOpen(false);
    setModelForm(emptyModelForm);
  }

  function handleChange(event) {
    const { name, value } = event.target;
    const numericFields = new Set([
      "image_size",
      "epochs",
      "batch_min",
      "batch_max",
      "lr_initial_min",
      "lr_initial_max",
      "momentum_min",
      "momentum_max",
    ]);

    setForm((prev) => {
      if (name === "task_type") {
        return {
          ...prev,
          task_type: value,
          image_size: defaultImageSizeByTask[value] || prev.image_size,
          yolo_model: "",
          dataset_id: "",
          preprocessing_pipeline_id: "",
        };
      }

      return {
        ...prev,
        [name]: numericFields.has(name) ? Number(value) : value,
      };
    });
  }

  function handleModelChange(event) {
    const { name, value, files } = event.target;
    setModelForm((prev) => ({
      ...prev,
      [name]: files ? files[0] || null : value,
    }));
  }

  function validateForm() {
    if (!form.name.trim()) return "학습 이름을 입력하세요.";
    if (!form.dataset_id) return "학습에 사용할 데이터셋을 선택하세요.";
    if (!form.yolo_model) return "학습 모델을 선택하세요.";
    if (form.image_size < 64 || form.image_size > 2048) return "Image Size는 64 이상 2048 이하로 입력하세요.";
    if (form.epochs < 1 || form.epochs > 1000) return "Epochs는 1 이상 1000 이하로 입력하세요.";
    if (form.batch_min < 1 || form.batch_max < form.batch_min) return "Batch Min/Max 범위를 확인하세요.";
    if (form.lr_initial_min <= 0 || form.lr_initial_max < form.lr_initial_min) return "LR Initial 범위를 확인하세요.";
    if (form.momentum_min < 0 || form.momentum_max > 1 || form.momentum_max < form.momentum_min) {
      return "Momentum 범위는 0 이상 1 이하로 입력하세요.";
    }
    return "";
  }

  function validateModelForm() {
    if (!modelForm.name.trim()) return "모델 이름을 입력하세요.";
    if (!modelForm.model_file) return "모델 파일을 선택하세요.";
    const lowerName = modelForm.model_file.name.toLowerCase();
    if (!lowerName.endsWith(".pt") && !lowerName.endsWith(".pth") && !lowerName.endsWith(".onnx")) {
      return "모델 파일은 .pt, .pth, .onnx만 가능합니다.";
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
      dataset_id: Number(form.dataset_id),
      preprocessing_pipeline_id: form.preprocessing_pipeline_id ? Number(form.preprocessing_pipeline_id) : null,
      yolo_model: form.yolo_model,
      optimizer: form.optimizer,
      image_size: form.image_size,
      epochs: form.epochs,
      batch_min: form.batch_min,
      batch_max: form.batch_max,
      lr_initial_min: form.lr_initial_min,
      lr_initial_max: form.lr_initial_max,
      momentum_min: form.momentum_min,
      momentum_max: form.momentum_max,
    };

    setSaving(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/training-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "새 학습 생성 중 오류가 발생했습니다.");
        return;
      }

      if (data.training_job?.id) {
        setSelectedJobId(String(data.training_job.id));
      }
      await loadPage();
      closeModal();
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    } finally {
      setSaving(false);
    }
  }

  async function handleModelSubmit(event) {
    event.preventDefault();
    const validation = validateModelForm();
    if (validation) {
      setError(validation);
      return;
    }

    const formData = new FormData();
    formData.append("user_id", String(userId));
    formData.append("name", modelForm.name.trim());
    formData.append("description", modelForm.description.trim());
    formData.append("task_type", modelForm.task_type);
    formData.append("model_file", modelForm.model_file);

    setModelSaving(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/training-models`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "학습 모델 추가 중 오류가 발생했습니다.");
        return;
      }

      await loadPage(true);
      setForm((prev) => ({
        ...prev,
        task_type: data.model.task,
        yolo_model: data.model.id,
        image_size: defaultImageSizeByTask[data.model.task] || prev.image_size,
      }));
      closeModelModal();
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    } finally {
      setModelSaving(false);
    }
  }

  async function startJob(job) {
    setError("");
    setSelectedJobId(String(job.id));
    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/training-jobs/${job.id}/start?user_id=${userId}`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "학습 시작 중 오류가 발생했습니다.");
        return;
      }
      await loadPage(true);
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    }
  }

  async function stopJob(job) {
    setError("");
    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/training-jobs/${job.id}/stop?user_id=${userId}`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "학습 중지 중 오류가 발생했습니다.");
        return;
      }
      await loadPage(true);
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    }
  }

  async function deleteJob(job) {
    const ok = window.confirm(`"${job.name}" 학습을 삭제하시겠습니까?`);
    if (!ok) return;

    setError("");
    try {
      const response = await fetch(
        `${API_BASE}/projects/${projectId}/training-jobs/${job.id}?user_id=${userId}`,
        { method: "DELETE" },
      );
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "학습 삭제 중 오류가 발생했습니다.");
        return;
      }
      setJobs((prev) => {
        const nextJobs = prev.filter((item) => item.id !== job.id);
        if (String(selectedJobId) === String(job.id)) {
          setSelectedJobId(nextJobs[0] ? String(nextJobs[0].id) : "");
        }
        return nextJobs;
      });
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    }
  }

  function goProjects() {
    window.location.href = "/projects";
  }

  function goDatasets() {
    window.location.href = `/projects/${projectId}`;
  }

  function goTraining() {
    window.location.href = `/projects/${projectId}/training`;
  }

  function logout() {
    localStorage.removeItem("dlops_user");
    window.location.href = "/";
  }

  return (
    <main className="training-page">
      <aside className="training-sidebar">
        <button className="nav-button" type="button" onClick={goDatasets}>
          Dataset Management
        </button>
        <button className="nav-button active" type="button" onClick={goTraining}>
          Training
        </button>
        <button className="nav-button" type="button" onClick={goProjects}>
          Back to Projects
        </button>

        <div className="training-brand">
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
          <span>Training Models</span>
          <strong>{models.length}</strong>
          <p>탐지 YOLO 모델, 분류 ConvNeXt 모델, 직접 추가한 모델을 함께 사용합니다.</p>
        </div>
      </aside>

      <section className="training-workspace">
        <header className="training-header">
          <div>
            <p className="eyebrow">Training Management</p>
            <h1>학습 관리</h1>
            <span>학습 생성, 실행, 중지, 진행률, 로그, 결과 경로를 관리합니다.</span>
          </div>
          <button className="primary-action" type="button" onClick={openCreateModal}>
            New Training
          </button>
        </header>

        {error && <div className="training-error">{error}</div>}

        {loading ? (
          <div className="training-state">학습 정보를 불러오는 중입니다.</div>
        ) : jobs.length === 0 ? (
          <div className="training-state">
            <strong>아직 생성된 학습이 없습니다.</strong>
            <p>데이터셋 업로드와 전처리 preset 생성 후 New Training으로 학습 작업을 추가하세요.</p>
          </div>
        ) : (
          <div className="training-main-grid">
            <div className="job-grid">
              {jobs.map((job) => {
                const isActive = ["QUEUED", "RUNNING", "STOPPING"].includes(job.status);
                const isStopping = job.status === "STOPPING";
                const logs = job.logs || [];
                const showLogs = Boolean(expandedLogs[job.id]);
                const isSelected = String(selectedJob?.id) === String(job.id);

                return (
                <article className={`job-card ${isSelected ? "selected" : ""}`} key={job.id} onClick={() => setSelectedJobId(String(job.id))}>
                  <div className="job-card-head">
                    <div>
                      <p className="eyebrow">{job.task_type === "detect" ? "Detection" : "Classification"}</p>
                      <h2>{job.name}</h2>
                    </div>
                    <span className={`status-pill status-${job.status.toLowerCase()}`}>{job.status}</span>
                  </div>

                  <p className="job-description">{job.description || "학습 설명이 없습니다."}</p>

                  <div className="job-source">
                    <div>
                      <span>Dataset</span>
                      <strong>{job.dataset?.name || "-"}</strong>
                      <small>{job.dataset?.original_filename} · {prettyBytes(job.dataset?.file_size || 0)}</small>
                    </div>
                    <div>
                      <span>Dataset Preprocessing</span>
                      <strong>{job.preprocessing_pipeline?.name || "None"}</strong>
                      <small>
                        {job.preprocessing_pipeline
                          ? `${job.preprocessing_pipeline.source === "auto" ? "Auto" : "Manual"} · ${job.preprocessing_pipeline.image_size}px`
                          : "원본 설정 사용"}
                      </small>
                    </div>
                  </div>

                  <div className="progress-panel">
                    <ProgressMeter job={job} />
                  </div>

                  <div className="config-grid">
                    <div>
                      <span>Training Model</span>
                      <strong>{job.yolo_model}</strong>
                    </div>
                    <div>
                      <span>Optimizer</span>
                      <strong>{job.optimizer}</strong>
                    </div>
                    <div>
                      <span>Image Size</span>
                      <strong>{job.image_size}</strong>
                    </div>
                    <div>
                      <span>Batch</span>
                      <strong>{job.batch_min} - {job.batch_max}</strong>
                    </div>
                    <div>
                      <span>LR Initial</span>
                      <strong>{job.lr_initial_min} - {job.lr_initial_max}</strong>
                    </div>
                    <div>
                      <span>Momentum</span>
                      <strong>{job.momentum_min} - {job.momentum_max}</strong>
                    </div>
                  </div>

                  <div className="log-panel">
                    <button
                      type="button"
                      className="text-action"
                      onClick={() => setExpandedLogs((prev) => ({ ...prev, [job.id]: !prev[job.id] }))}
                    >
                      {showLogs ? "Hide Logs" : "Show Logs"}
                    </button>
                    {showLogs && (
                      <div className="log-box">
                        {logs.length === 0 ? (
                          <p>아직 로그가 없습니다.</p>
                        ) : (
                          logs.map((log, index) => (
                            <p key={`${log.time || "log"}-${index}`}>
                              <span>{formatDate(log.time)}</span>
                              {log.message}
                            </p>
                          ))
                        )}
                      </div>
                    )}
                  </div>

                  <div className="job-footer">
                    <span>Created {formatDate(job.created_at)}</span>
                    <div className="job-actions">
                      {isActive ? (
                        <button type="button" onClick={() => stopJob(job)}>
                          {isStopping ? "Force Stop" : "Stop"}
                        </button>
                      ) : (
                        <button type="button" onClick={() => startJob(job)}>
                          Start
                        </button>
                      )}
                      <button type="button" onClick={() => setSelectedJobId(String(job.id))}>
                        Result
                      </button>
                      <button type="button" onClick={() => openRetuneModal(job)}>
                        Retune
                      </button>
                      <button className="danger" type="button" onClick={() => deleteJob(job)} disabled={isActive}>
                        Delete
                      </button>
                    </div>
                  </div>
                </article>
                );
              })}
            </div>

            <aside className="live-result-panel">
              <div className="live-result-head">
                <div>
                  <p className="eyebrow">Live Result</p>
                  <h2>{selectedJob?.name || "학습 선택"}</h2>
                  <span>{selectedJob ? `${selectedJob.task_type === "detect" ? "Detection" : "Classification"} · ${selectedJob.yolo_model}` : "학습 작업을 선택하세요."}</span>
                </div>
                {selectedJob && <span className={`status-pill status-${selectedJob.status.toLowerCase()}`}>{selectedJob.status}</span>}
              </div>

              <div className="live-controls">
                <button type="button" onClick={() => loadPage(true)}>
                  Refresh
                </button>
                <button type="button" className="tune-action" onClick={() => openRetuneModal(selectedJob)} disabled={!selectedJob}>
                  Retune from Result
                </button>
              </div>

              {selectedJob ? (
                <>
                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Latest Logs</h3>
                      <span>{selectedLogs.length} lines</span>
                    </div>
                    <div className="live-log-box" ref={liveLogBoxRef}>
                      {selectedLogs.length === 0 ? (
                        <p>아직 로그가 없습니다.</p>
                      ) : (
                        selectedLogs.map((log, index) => (
                          <p key={`${log.time || "log"}-${index}`}>
                            <span>{formatDate(log.time)}</span>
                            {log.message}
                          </p>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="live-progress-card">
                    <ProgressMeter job={selectedJob} />
                  </div>

                  <TrainingMetricCards job={selectedJob} />

                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Result Summary</h3>
                      <span>{metricLabels[selectedPrimaryKey] || selectedPrimaryKey}</span>
                    </div>
                    <div className="result-summary-grid">
                      <div>
                        <span>Primary Score</span>
                        <strong>{formatMetric(selectedPrimaryScore)}</strong>
                        <small>{metricLabels[selectedPrimaryKey] || selectedPrimaryKey}</small>
                      </div>
                      <div>
                        <span>Dataset</span>
                        <strong>{selectedJob.dataset?.name || "-"}</strong>
                        <small>{selectedJob.dataset?.original_filename || "-"}</small>
                      </div>
                      <div>
                        <span>Dataset Preprocessing</span>
                        <strong>{selectedJob.preprocessing_pipeline?.name || "None"}</strong>
                        <small>{selectedJob.preprocessing_pipeline?.source || "original"}</small>
                      </div>
                      <div>
                        <span>Duration</span>
                        <strong>{formatDuration(selectedResult.duration_seconds)}</strong>
                        <small>{formatDate(selectedJob.completed_at || selectedJob.updated_at)}</small>
                      </div>
                    </div>
                  </div>

                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Best Run Compare</h3>
                      <span>{bestJob ? bestJob.name : "no completed run"}</span>
                    </div>
                    {bestJob ? (
                      <div className="best-compare-grid">
                        <div>
                          <span>Selected</span>
                          <strong>{formatMetric(selectedPrimaryScore)}</strong>
                          <small>{selectedJob.name}</small>
                        </div>
                        <div>
                          <span>Best</span>
                          <strong>{formatMetric(bestPrimaryScore)}</strong>
                          <small>{bestJob.name}</small>
                        </div>
                        <div className={selectedBestDelta >= 0 ? "delta-up" : "delta-down"}>
                          <span>Delta</span>
                          <strong>{formatDelta(selectedBestDelta)}</strong>
                          <small>selected - best</small>
                        </div>
                      </div>
                    ) : (
                      <div className="live-empty-box">완료된 학습이 생기면 최고 결과와 비교합니다.</div>
                    )}
                  </div>

                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Epoch Trend</h3>
                      <span>{selectedResult.epoch_metrics?.length || 0} epochs</span>
                    </div>
                    <TrainingMiniChart job={selectedJob} />
                  </div>

                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Top Runs</h3>
                      <span>{topRuns.length} completed</span>
                    </div>
                    {topRuns.length > 0 ? (
                      <div className="top-run-table">
                        {topRuns.map((job, index) => {
                          const key = getPrimaryMetricKey(job);
                          return (
                            <button type="button" key={job.id} onClick={() => setSelectedJobId(String(job.id))}>
                              <span>#{index + 1}</span>
                              <strong>{job.name}</strong>
                              <em>{formatMetric(getPrimaryScore(job))}</em>
                              <small>{metricLabels[key] || key}</small>
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="live-empty-box">완료된 학습 결과가 아직 없습니다.</div>
                    )}
                  </div>

                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Artifacts</h3>
                      <span>result paths</span>
                    </div>
                    <div className="live-artifact-list">
                      {artifactItems.map((item) => {
                        const path = selectedArtifacts?.[item.key] || "";
                        const exists = selectedArtifacts?.exists?.[item.key];
                        return (
                          <p key={item.key}>
                            <span>{item.label}</span>
                            {path ? `${path}${exists === false ? " (missing)" : ""}` : "-"}
                          </p>
                        );
                      })}
                    </div>
                  </div>

                  <div className="live-section">
                    <div className="live-section-title">
                      <h3>Result Images</h3>
                      <span>{previewArtifacts.length} previews</span>
                    </div>
                    {previewArtifacts.length > 0 ? (
                      <div className="artifact-preview-grid">
                        {previewArtifacts.map((item) => (
                          <figure key={item.key}>
                            <img src={artifactImageUrl(projectId, userId, item.path)} alt={`${item.label} preview`} />
                            <figcaption>{item.label}</figcaption>
                          </figure>
                        ))}
                      </div>
                    ) : (
                      <div className="live-empty-box">학습 결과 이미지가 생성되면 이곳에서 미리 볼 수 있습니다.</div>
                    )}
                  </div>

                  {selectedResult.error && (
                    <div className="live-error-box">
                      <span>Error</span>
                      <p>{selectedResult.error}</p>
                    </div>
                  )}

                </>
              ) : (
                <div className="live-empty-box">왼쪽에서 학습 작업을 선택하세요.</div>
              )}
            </aside>
          </div>
        )}
      </section>

      {modalOpen && (
        <div className="training-modal-backdrop">
          <form className="training-modal" onSubmit={handleSubmit}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">New Training</p>
                <h2>새 학습 생성</h2>
              </div>
              <button type="button" onClick={closeModal} disabled={saving}>
                Close
              </button>
            </div>

            {error && <div className="training-error modal-message">{error}</div>}

            <div className="modal-body">
              <section className="modal-section">
                <h3>Basic</h3>
                <div className="form-grid">
                  <label className="field">
                    <span>학습 이름</span>
                    <input name="name" value={form.name} onChange={handleChange} placeholder="예: surface-defect-v1" required />
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
                      placeholder="실험 목적, 비교 기준, 목표 지표를 적어두세요."
                    />
                  </label>
                </div>
              </section>

              <section className="modal-section">
                <h3>Source</h3>
                <div className="form-grid">
                  <label className="field">
                    <span>데이터셋</span>
                    <select name="dataset_id" value={form.dataset_id} onChange={handleChange}>
                      <option value="">데이터셋 선택</option>
                      {datasetOptions.map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.original_filename})
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>데이터셋 전처리</span>
                    <select name="preprocessing_pipeline_id" value={form.preprocessing_pipeline_id} onChange={handleChange}>
                      <option value="">원본 데이터셋 사용</option>
                      {pipelineOptions.map((pipeline) => (
                        <option key={pipeline.id} value={pipeline.id}>
                          [{pipeline.source === "auto" ? "Auto" : "Manual"}] {pipeline.name} ({pipeline.image_size}px)
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </section>

              <section className="modal-section">
                <h3>Training Config</h3>
                <div className="form-grid">
                  <label className="field">
                    <span>학습 모델</span>
                    <div className="model-select-row">
                      <select name="yolo_model" value={form.yolo_model} onChange={handleChange}>
                        {modelOptions.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name} ({model.source || "builtin"})
                          </option>
                        ))}
                      </select>
                      <button type="button" onClick={() => openModelModal(form.task_type)}>
                        Add Model
                      </button>
                    </div>
                  </label>
                  <label className="field">
                    <span>Optimizer</span>
                    <select name="optimizer" value={form.optimizer} onChange={handleChange}>
                      <option value="AdamW">AdamW</option>
                      <option value="Adam">Adam</option>
                      <option value="SGD">SGD</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>Image Size</span>
                    <input type="number" name="image_size" min="64" max="2048" value={form.image_size} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>Epochs</span>
                    <input type="number" name="epochs" min="1" max="1000" value={form.epochs} onChange={handleChange} />
                  </label>
                </div>
              </section>

              <section className="modal-section">
                <h3>Optuna Search Space</h3>
                <div className="form-grid optuna-grid">
                  <label className="field">
                    <span>Batch Min</span>
                    <input type="number" name="batch_min" min="1" value={form.batch_min} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>Batch Max</span>
                    <input type="number" name="batch_max" min="1" value={form.batch_max} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>LR Initial Min</span>
                    <input type="number" step="0.0001" name="lr_initial_min" value={form.lr_initial_min} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>LR Initial Max</span>
                    <input type="number" step="0.0001" name="lr_initial_max" value={form.lr_initial_max} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>Momentum Min</span>
                    <input type="number" step="0.01" name="momentum_min" value={form.momentum_min} onChange={handleChange} />
                  </label>
                  <label className="field">
                    <span>Momentum Max</span>
                    <input type="number" step="0.01" name="momentum_max" value={form.momentum_max} onChange={handleChange} />
                  </label>
                </div>
              </section>
            </div>

            <div className="modal-footer">
              <button type="button" className="secondary" onClick={closeModal} disabled={saving}>
                취소
              </button>
              <button type="submit" disabled={saving}>
                {saving ? "생성 중..." : "학습 생성"}
              </button>
            </div>
          </form>
        </div>
      )}

      {modelModalOpen && (
        <div className="training-modal-backdrop model-modal-backdrop">
          <form className="training-modal model-modal" onSubmit={handleModelSubmit}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">Custom Training Model</p>
                <h2>학습 모델 추가</h2>
              </div>
              <button type="button" onClick={closeModelModal} disabled={modelSaving}>
                Close
              </button>
            </div>

            {error && <div className="training-error modal-message">{error}</div>}

            <div className="modal-body">
              <section className="modal-section">
                <div className="form-grid">
                  <label className="field">
                    <span>모델 이름</span>
                    <input name="name" value={modelForm.name} onChange={handleModelChange} placeholder="예: defect-base-v1" />
                  </label>
                  <label className="field">
                    <span>작업 유형</span>
                    <select name="task_type" value={modelForm.task_type} onChange={handleModelChange}>
                      <option value="detect">탐지 Detect</option>
                      <option value="classify">분류 Classify</option>
                    </select>
                  </label>
                  <label className="field wide">
                    <span>설명</span>
                    <textarea
                      name="description"
                      value={modelForm.description}
                      onChange={handleModelChange}
                      rows={3}
                      placeholder="모델 출처, 버전, 사용 목적을 적어두세요."
                    />
                  </label>
                  <label className="upload-zone wide">
                    <input type="file" name="model_file" accept=".pt,.pth,.onnx" onChange={handleModelChange} />
                    <strong>{modelForm.model_file ? modelForm.model_file.name : "모델 파일 선택"}</strong>
                    <span>탐지는 YOLO .pt, 분류는 ConvNeXt 계열 .pt 또는 .pth 파일을 권장합니다.</span>
                  </label>
                </div>
              </section>
            </div>

            <div className="modal-footer">
              <button type="button" className="secondary" onClick={closeModelModal} disabled={modelSaving}>
                취소
              </button>
              <button type="submit" disabled={modelSaving}>
                {modelSaving ? "추가 중..." : "모델 추가"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default TrainingManage;
