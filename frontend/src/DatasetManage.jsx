import React, { useEffect, useState } from "react";
import "./DatasetManage.css";

const API_BASE = "http://localhost:8000";

const emptyForm = {
  dataset_name: "",
  description: "",
  task_type: "detect",
  create_auto_pipeline: true,
  dataset_zip: null,
};

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

function DatasetReport({ report }) {
  if (!report) {
    return (
      <div className="dataset-report muted">
        zip 파일을 선택하거나 데이터셋을 업로드하면 리포트가 표시됩니다.
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
            <span>Recommended Pipeline</span>
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

function DatasetManage({ user, projectId }) {
  const [project, setProject] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedFileReport, setSelectedFileReport] = useState(null);
  const [creatingPipelineFor, setCreatingPipelineFor] = useState(null);

  const userId = user?.user_id;
  const userEmail = user?.email || "user";

  async function loadPage() {
    if (!projectId || !userId) return;
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/datasets?user_id=${userId}`);
      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "데이터셋 정보를 불러오지 못했습니다.");
        setProject(null);
        setDatasets([]);
        return;
      }

      setProject(data.project);
      setDatasets(data.datasets || []);
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
      setProject(null);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPage();
  }, [projectId, userId]);

  function openCreateModal() {
    setForm(emptyForm);
    setSelectedFileReport(null);
    setModalOpen(true);
    setError("");
    setNotice("");
  }

  function closeModal() {
    if (saving || analyzing) return;
    setModalOpen(false);
    setForm(emptyForm);
    setSelectedFileReport(null);
  }

  function handleChange(event) {
    const { name, value, type, checked } = event.target;
    setForm((prev) => ({ ...prev, [name]: type === "checkbox" ? checked : value }));
    if (name === "task_type") {
      setSelectedFileReport(null);
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
      setError("분석할 zip 데이터셋 파일을 선택하세요.");
      return;
    }
    if (!form.dataset_zip.name.toLowerCase().endsWith(".zip")) {
      setError("데이터셋은 zip 파일만 분석할 수 있습니다.");
      return;
    }

    const formData = new FormData();
    formData.append("task_type", form.task_type);
    formData.append("dataset_zip", form.dataset_zip);

    setAnalyzing(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE}/datasets/analyze-zip`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "데이터셋 분석 중 오류가 발생했습니다.");
        return;
      }
      setSelectedFileReport(data.report);
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    } finally {
      setAnalyzing(false);
    }
  }

  function validateForm() {
    if (!form.dataset_name.trim()) return "데이터셋 이름을 입력하세요.";
    if (!form.dataset_zip) return "zip 데이터셋 파일을 선택하세요.";
    if (!form.dataset_zip.name.toLowerCase().endsWith(".zip")) return "데이터셋은 zip 파일만 가능합니다.";
    return "";
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const validation = validateForm();
    if (validation) {
      setError(validation);
      return;
    }

    const formData = new FormData();
    formData.append("user_id", String(userId));
    formData.append("dataset_name", form.dataset_name.trim());
    formData.append("description", form.description.trim());
    formData.append("task_type", form.task_type);
    formData.append("create_auto_pipeline", String(Boolean(form.create_auto_pipeline)));
    formData.append("dataset_zip", form.dataset_zip);

    setSaving(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE}/projects/${projectId}/datasets`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "데이터셋 업로드 중 오류가 발생했습니다.");
        return;
      }

      await loadPage();
      if (data.auto_pipeline) {
        setNotice(`자동 전처리 파이프라인 "${data.auto_pipeline.name}"이 생성되었습니다.`);
      }
      closeModal();
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
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
      setDatasets((prev) => prev.filter((item) => item.id !== dataset.id));
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
        setError(data.detail || "자동 전처리 파이프라인 생성 중 오류가 발생했습니다.");
        return;
      }

      await loadPage();
      setNotice(`자동 전처리 파이프라인 "${data.pipeline.name}"이 생성되었습니다.`);
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
    } finally {
      setCreatingPipelineFor(null);
    }
  }

  function goBack() {
    window.location.href = "/projects";
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

  function logout() {
    localStorage.removeItem("dlops_user");
    window.location.href = "/";
  }

  return (
    <main className="dataset-page">
      <aside className="dataset-sidebar">
        <button className="back-button active" type="button" onClick={goDatasets}>
          Dataset Management
        </button>
        <button className="back-button" type="button" onClick={goPreprocessing}>
          Preprocessing
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
          <p>YOLO 학습에 사용할 zip 데이터셋과 리포트를 관리합니다.</p>
        </div>
      </aside>

      <section className="dataset-workspace">
        <header className="dataset-header">
          <div>
            <p className="eyebrow">Dataset Management</p>
            <h1>데이터셋 관리</h1>
            <span>프로젝트 안에서 YOLO zip 데이터셋을 업로드하고 구조 리포트를 확인합니다.</span>
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
            {datasets.map((dataset) => (
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

                <div className="training-card-footer">
                  <span>Uploaded {formatDate(dataset.created_at)}</span>
                  <div className="dataset-card-actions">
                    <button
                      type="button"
                      onClick={() => createAutoPipeline(dataset)}
                      disabled={creatingPipelineFor === dataset.id}
                    >
                      {creatingPipelineFor === dataset.id ? "Creating..." : "Auto Pipeline"}
                    </button>
                    <button className="danger-action" type="button" onClick={() => deleteDataset(dataset)}>
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
                      placeholder="데이터 출처, 클래스 구성, 라벨링 기준을 적어두세요."
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
                  <span>업로드 전에 zip 내부 구조를 먼저 확인합니다.</span>
                </div>
                <DatasetReport report={selectedFileReport} />
                <label className="auto-pipeline-toggle">
                  <input
                    type="checkbox"
                    name="create_auto_pipeline"
                    checked={form.create_auto_pipeline}
                    onChange={handleChange}
                  />
                  <span>
                    분석 추천값으로 전처리 파이프라인도 함께 생성
                    <small>생성된 파이프라인은 전처리 관리와 학습 생성의 preset 선택 목록에서 사용할 수 있습니다.</small>
                  </span>
                </label>
              </section>
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
    </main>
  );
}

export default DatasetManage;
