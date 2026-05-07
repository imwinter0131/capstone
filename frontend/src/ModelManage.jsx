import React, { useEffect, useState } from "react";
import {
  getModelRegistry,
  uploadModelRegistry,
  updateModelRegistry,
  deployModelRegistry,
  rollbackModelRegistry,
  deleteModelRegistry,
} from "./apis/modelRegistry";

const allowedExtensions = [".py", ".pt", ".pth", ".onnx", ".engine", ".trt", ".bin", ".h5", ".zip"];

function ModelManage() {
  const [models, setModels] = useState([]);
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [version, setVersion] = useState("v1");
  const [task, setTask] = useState("detect");
  const [framework, setFramework] = useState("PyTorch");
  const [description, setDescription] = useState("");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("ALL");
  const [fileExt, setFileExt] = useState("ALL");
  const [runtimeTarget, setRuntimeTarget] = useState("local-runtime");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await getModelRegistry({ q: search, status, file_ext: fileExt });
      setModels(list);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const saveModel = async () => {
    if (!file) {
      setError("모델 파일을 선택해 주세요.");
      return;
    }

    if (!name.trim()) {
      setError("모델 이름을 입력해 주세요.");
      return;
    }

    const lowerName = file.name.toLowerCase();
    const valid = allowedExtensions.some((ext) => lowerName.endsWith(ext));
    if (!valid) {
      setError("허용되지 않은 모델 파일 형식입니다.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await uploadModelRegistry({
        file,
        name,
        version,
        task,
        framework,
        description,
      });
      setFile(null);
      setName("");
      setVersion("v1");
      setDescription("");
      await loadModels();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renameModel = async (item) => {
    const nextName = window.prompt("수정할 모델 이름을 입력하세요.", item.name);
    if (!nextName) return;
    await updateModelRegistry(item.id, { name: nextName });
    loadModels();
  };

  const deployModel = async (id) => {
    await deployModelRegistry(id, runtimeTarget);
    loadModels();
  };

  const rollbackModel = async (id) => {
    await rollbackModelRegistry(id);
    loadModels();
  };

  const removeModel = async (id) => {
    if (!window.confirm("모델을 삭제할까요?")) return;
    await deleteModelRegistry(id);
    loadModels();
  };

  return (
    <div style={{ minHeight: "100vh", background: "#fff", color: "#111", padding: "24px", fontFamily: "sans-serif" }}>
      <h1>모델 관리</h1>
      <p>배포용 AI 모델을 등록하고 메타데이터, 배포 상태, 런타임 적용을 관리합니다.</p>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "16px", marginBottom: "16px" }}>
        <h2>배포용 AI 모델 등록 및 메타데이터 중앙관리</h2>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="모델 이름" />
        <input value={version} onChange={(e) => setVersion(e.target.value)} placeholder="버전" />
        <select value={task} onChange={(e) => setTask(e.target.value)}>
          <option value="detect">detect</option>
          <option value="classify">classify</option>
          <option value="segment">segment</option>
          <option value="xai">xai</option>
        </select>
        <select value={framework} onChange={(e) => setFramework(e.target.value)}>
          <option value="PyTorch">PyTorch</option>
          <option value="ONNX">ONNX</option>
          <option value="TensorFlow">TensorFlow</option>
          <option value="Custom">Custom</option>
        </select>
        <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="모델 설명" />
        <button onClick={saveModel} disabled={loading}>모델 등록</button>
      </section>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "16px" }}>
        <h2>모델 목록 통합 조회 및 조건부 탐색</h2>
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="이름, 버전, 설명 검색" />
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="ALL">전체 상태</option>
          <option value="REGISTERED">REGISTERED</option>
          <option value="DEPLOYED">DEPLOYED</option>
        </select>
        <select value={fileExt} onChange={(e) => setFileExt(e.target.value)}>
          <option value="ALL">전체 포맷</option>
          <option value="py">py</option>
          <option value="pt">pt</option>
          <option value="pth">pth</option>
          <option value="onnx">onnx</option>
          <option value="engine">engine</option>
          <option value="trt">trt</option>
          <option value="bin">bin</option>
          <option value="h5">h5</option>
          <option value="zip">zip</option>
        </select>
        <button onClick={loadModels}>검색</button>

        <div style={{ marginTop: "12px" }}>
          <span>배포 대상 </span>
          <select value={runtimeTarget} onChange={(e) => setRuntimeTarget(e.target.value)}>
            <option value="local-runtime">local-runtime</option>
            <option value="edge-device">edge-device</option>
            <option value="staging">staging</option>
            <option value="production">production</option>
          </select>
        </div>

        {error && <div style={{ color: "red" }}>{error}</div>}
        {loading && <div>처리 중...</div>}

        {models.map((item) => (
          <div key={item.id} style={{ border: "1px solid #ddd", padding: "12px", marginTop: "10px" }}>
            <h3>{item.name} {item.version}</h3>
            <div>작업 유형: {item.task}</div>
            <div>프레임워크: {item.framework}</div>
            <div>파일: {item.file_name}</div>
            <div>상태: {item.status}</div>
            <div>런타임: {item.runtime_target || "-"}</div>
            <button onClick={() => renameModel(item)} disabled={item.status === "DEPLOYED"}>수정</button>
            <button onClick={() => deployModel(item.id)} disabled={item.status === "DEPLOYED"}>배포</button>
            <button onClick={() => rollbackModel(item.id)} disabled={item.status !== "DEPLOYED"}>롤백</button>
            <button onClick={() => removeModel(item.id)} disabled={item.status === "DEPLOYED"}>삭제</button>
          </div>
        ))}
      </section>
    </div>
  );
}

export default ModelManage;
