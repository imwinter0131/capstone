const API_BASE = "http://localhost:8000";

export async function getModelRegistry(filters = {}) {
  const params = new URLSearchParams();
  if (filters.q) params.set("q", filters.q);
  if (filters.status) params.set("status", filters.status);
  if (filters.file_ext) params.set("file_ext", filters.file_ext);

  const res = await fetch(`${API_BASE}/models/registry?${params.toString()}`);
  if (!res.ok) throw new Error("모델 목록 조회 실패");
  return await res.json();
}

export async function uploadModelRegistry(payload) {
  const formData = new FormData();
  formData.append("file", payload.file);
  formData.append("name", payload.name);
  formData.append("version", payload.version);
  formData.append("task", payload.task);
  formData.append("framework", payload.framework);
  formData.append("description", payload.description || "");

  const res = await fetch(`${API_BASE}/models/registry/upload`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "모델 등록 실패");
  return data;
}

export async function updateModelRegistry(id, payload) {
  const params = new URLSearchParams(payload);

  const res = await fetch(`${API_BASE}/models/registry/${id}?${params.toString()}`, {
    method: "PATCH",
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "모델 메타데이터 수정 실패");
  return data;
}

export async function deployModelRegistry(id, runtimeTarget) {
  const params = new URLSearchParams({ runtime_target: runtimeTarget });

  const res = await fetch(`${API_BASE}/models/registry/${id}/deploy?${params.toString()}`, {
    method: "POST",
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "모델 배포 실패");
  return data;
}

export async function rollbackModelRegistry(id) {
  const res = await fetch(`${API_BASE}/models/registry/${id}/rollback`, {
    method: "POST",
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "모델 롤백 실패");
  return data;
}

export async function deleteModelRegistry(id) {
  const res = await fetch(`${API_BASE}/models/registry/${id}`, {
    method: "DELETE",
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "모델 삭제 실패");
  return data;
}
