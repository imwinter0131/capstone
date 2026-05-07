import React, { useEffect, useMemo, useState } from "react";
import "./Projects.css";

const API_BASE = "http://localhost:8000";

const emptyForm = {
  name: "",
  description: "",
  folder_path: "",
};

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

function Projects({ user }) {
  const [projects, setProjects] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editingProject, setEditingProject] = useState(null);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);

  const userId = user?.user_id;
  const userEmail = user?.email || "user";

  const filteredProjects = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    if (!keyword) return projects;

    return projects.filter((project) => {
      return [
        project.name,
        project.description,
        project.folder_path,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(keyword));
    });
  }, [projects, search]);

  const projectCountText = `${filteredProjects.length} / ${projects.length}`;

  async function loadProjects() {
    if (!userId) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/projects?user_id=${userId}`);
      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "프로젝트 목록을 불러오지 못했습니다.");
        setProjects([]);
        return;
      }

      setProjects(data.projects || []);
    } catch {
      setError("서버와 연결할 수 없습니다. FastAPI 서버를 확인하세요.");
      setProjects([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadProjects();
  }, [userId]);

  function openCreateModal() {
    setEditingProject(null);
    setForm(emptyForm);
    setModalOpen(true);
    setError("");
  }

  function openEditModal(project) {
    setEditingProject(project);
    setForm({
      name: project.name || "",
      description: project.description || "",
      folder_path: project.folder_path || "",
    });
    setModalOpen(true);
    setError("");
  }

  function closeModal() {
    if (saving) return;
    setModalOpen(false);
    setEditingProject(null);
    setForm(emptyForm);
  }

  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();

    if (!form.name.trim()) {
      setError("프로젝트명을 입력하세요.");
      return;
    }

    setSaving(true);
    setError("");

    const payload = {
      ...form,
      user_id: userId,
      name: form.name.trim(),
      folder_path: form.folder_path.trim(),
    };

    const isEdit = Boolean(editingProject);
    const endpoint = isEdit
      ? `${API_BASE}/projects/${editingProject.id}`
      : `${API_BASE}/projects`;

    try {
      const response = await fetch(endpoint, {
        method: isEdit ? "PATCH" : "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "프로젝트 저장 중 오류가 발생했습니다.");
        return;
      }

      await loadProjects();
      closeModal();
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(project) {
    const ok = window.confirm(`"${project.name}" 프로젝트를 삭제하시겠습니까?`);
    if (!ok) return;

    setError("");

    try {
      const response = await fetch(`${API_BASE}/projects/${project.id}?user_id=${userId}`, {
        method: "DELETE",
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "프로젝트 삭제 중 오류가 발생했습니다.");
        return;
      }

      setProjects((prev) => prev.filter((item) => item.id !== project.id));
    } catch {
      setError("서버와 연결할 수 없습니다. 백엔드 서버를 확인하세요.");
    }
  }

  function handleLogout() {
    localStorage.removeItem("dlops_user");
    window.location.href = "/";
  }

  function openProject(project) {
    window.location.href = `/projects/${project.id}`;
  }

  return (
    <main className="projects-page">
      <aside className="project-sidebar">
        <div className="sidebar-brand">
          <span className="brand-dot" />
          <div>
            <strong>DLOps</strong>
            <p>Project Workspace</p>
          </div>
        </div>

        <button className="new-project-button" type="button" onClick={openCreateModal}>
          New Project
        </button>

        <div className="tree-block">
          <div className="tree-title">PROJECT TREE</div>

          {projects.length === 0 ? (
            <div className="tree-empty">생성된 프로젝트가 없습니다.</div>
          ) : (
            <div className="tree-list">
              {projects.map((project) => (
                <button
                  key={project.id}
                  className="tree-item"
                  type="button"
                  onClick={() => setSearch(project.name)}
                  title={project.name}
                >
                  <span className="folder-mark" />
                  <span>{project.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>

      <section className="project-workspace">
        <header className="project-topbar">
          <div className="search-wrap">
            <span>Search</span>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="프로젝트명, 설명, 저장 경로 검색"
            />
            {search && (
              <button type="button" onClick={() => setSearch("")}>
                Clear
              </button>
            )}
          </div>

          <div className="user-box">
            <div className="avatar">{userEmail.charAt(0).toUpperCase()}</div>
            <div>
              <strong>{userEmail}</strong>
              <p>Signed in</p>
            </div>
            <button type="button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </header>

        <section className="workspace-heading">
          <div>
            <p className="eyebrow">Project Management</p>
            <h1>프로젝트 관리</h1>
            <span>프로젝트명, 설명, 저장 경로를 기준으로 프로젝트를 관리합니다.</span>
          </div>
          <div className="count-card">
            <span>Visible</span>
            <strong>{projectCountText}</strong>
          </div>
        </section>

        {error && <div className="project-error">{error}</div>}

        {loading ? (
          <div className="project-state">프로젝트 목록을 불러오는 중입니다.</div>
        ) : filteredProjects.length === 0 ? (
          <div className="project-state">
            <strong>{projects.length === 0 ? "아직 프로젝트가 없습니다." : "검색 결과가 없습니다."}</strong>
            <p>
              {projects.length === 0
                ? "New Project 버튼을 눌러 첫 프로젝트를 생성하세요."
                : "검색어를 다시 입력하거나 Clear 버튼으로 초기화하세요."}
            </p>
          </div>
        ) : (
          <div className="project-grid">
            {filteredProjects.map((project) => (
              <article className="project-card" key={project.id}>
                <div className="card-top">
                  <div>
                    <p className="project-label">PROJECT</p>
                    <h2>{project.name}</h2>
                  </div>
                  <span className="status-pill">Active</span>
                </div>

                <p className="project-description">
                  {project.description || "프로젝트 설명이 없습니다."}
                </p>

                <dl className="project-meta">
                  <div>
                    <dt>Folder</dt>
                    <dd>{project.folder_path || "-"}</dd>
                  </div>
                </dl>

                <div className="card-footer">
                  <span>Updated {formatDate(project.updated_at)}</span>
                  <div className="card-actions">
                    <button type="button" onClick={() => openProject(project)}>
                      Open
                    </button>
                    <button type="button" onClick={() => openEditModal(project)}>
                      Edit
                    </button>
                    <button className="danger" type="button" onClick={() => handleDelete(project)}>
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
        <div className="modal-backdrop" role="presentation">
          <form className="project-modal" onSubmit={handleSubmit}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">{editingProject ? "Edit Project" : "New Project"}</p>
                <h2>{editingProject ? "프로젝트 수정" : "프로젝트 생성"}</h2>
              </div>
              <button type="button" onClick={closeModal} disabled={saving}>
                Close
              </button>
            </div>

            <div className="form-grid">
              <label className="field wide">
                <span>프로젝트명</span>
                <input
                  name="name"
                  value={form.name}
                  onChange={handleChange}
                  placeholder="예: defect-detection"
                  required
                />
              </label>

              <label className="field wide">
                <span>설명</span>
                <textarea
                  name="description"
                  value={form.description}
                  onChange={handleChange}
                  placeholder="프로젝트 목적과 데이터 특징을 입력하세요."
                  rows={3}
                />
              </label>

              <label className="field wide">
                <span>저장 경로</span>
                <input
                  name="folder_path"
                  value={form.folder_path}
                  onChange={handleChange}
                  placeholder="비워두면 기본 경로가 자동으로 지정됩니다."
                />
              </label>
            </div>

            <div className="modal-footer">
              <button type="button" className="secondary" onClick={closeModal} disabled={saving}>
                취소
              </button>
              <button type="submit" disabled={saving}>
                {saving ? "저장 중..." : editingProject ? "수정 저장" : "프로젝트 생성"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default Projects;
