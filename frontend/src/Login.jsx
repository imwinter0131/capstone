import React, { useMemo, useState } from "react";
import "./Login.css";
const API_BASE = "http://localhost:8000";
function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState("");
  const title = isLogin ? "DLOps 로그인" : "DLOps 회원가입";
  const subtitle = isLogin
    ? "모델 학습 프로젝트를 계속 관리하세요."
    : "새 계정을 만들고 DLOps 작업을 시작하세요.";
  const canSubmit = useMemo(() => {
    return email.trim().length > 0 && password.trim().length >= 8 && !loading;
  }, [email, password, loading]);
  const resetMessage = () => {
    setMessage("");
    setMessageType("");
  };
  const handleModeChange = () => {
    setIsLogin((prev) => !prev);
    setPassword("");
    resetMessage();
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    const endpoint = isLogin ? "/login" : "/signup";
    setLoading(true);
    resetMessage();
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim(),
          password,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setMessage(data.detail || "요청 처리 중 오류가 발생했습니다.");
        setMessageType("error");
        return;
      }
      setMessage(data.message || "요청이 완료되었습니다.");
      setMessageType("success");
      if (isLogin) {
        localStorage.setItem(
          "dlops_user",
          JSON.stringify({
            user_id: data.user_id,
            email: data.email,
          })
        );
        setTimeout(() => {
          window.location.href = "/projects";
        }, 700);
      } else {
        setTimeout(() => {
          setIsLogin(true);
          setPassword("");
          setMessage("회원가입이 완료되었습니다. 로그인해 주세요.");
          setMessageType("success");
        }, 500);
      }
    } catch (error) {
      setMessage("서버와 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요.");
      setMessageType("error");
    } finally {
      setLoading(false);
    }
  };
  return (
    <main className="login-page">
      <section className="login-shell">
        <div className="brand-panel">
          <div className="brand-badge">DLOps</div>
          <h1>AI 모델 개발 흐름을 한 곳에서 관리합니다.</h1>
          <p>
            프로젝트, 데이터셋, 학습 실행, 결과 분석까지 이어지는 작업을
            안정적으로 관리하기 위한 시작 화면입니다.
          </p>
          <div className="brand-stats">
            <div>
              <span>FastAPI</span>
              <strong>Backend</strong>
            </div>
            <div>
              <span>React</span>
              <strong>Frontend</strong>
            </div>
            <div>
              <span>PostgreSQL</span>
              <strong>Database</strong>
            </div>
          </div>
        </div>
        <div className="auth-card">
          <div className="mode-tabs" aria-label="로그인 모드 선택">
            <button
              type="button"
              className={isLogin ? "active" : ""}
              onClick={() => {
                if (!isLogin) handleModeChange();
              }}
            >
              로그인
            </button>
            <button
              type="button"
              className={!isLogin ? "active" : ""}
              onClick={() => {
                if (isLogin) handleModeChange();
              }}
            >
              회원가입
            </button>
          </div>
          <div className="auth-heading">
            <p className="eyebrow">Secure Access</p>
            <h2>{title}</h2>
            <p>{subtitle}</p>
          </div>
          <form onSubmit={handleSubmit} className="auth-form">
            <label className="field">
              <span>이메일</span>
              <input
                type="email"
                placeholder="name@example.com"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  resetMessage();
                }}
                required
              />
            </label>
            <label className="field">
              <span>비밀번호</span>
              <input
                type="password"
                placeholder="8자 이상 입력"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  resetMessage();
                }}
                required
                minLength={8}
              />
            </label>
            {message && (
              <div className={`notice ${messageType}`}>
                {message}
              </div>
            )}
            <button className="submit-button" type="submit" disabled={!canSubmit}>
              {loading ? "처리 중..." : isLogin ? "Continue" : "가입하기"}
            </button>
          </form>
          <button className="switch-button" type="button" onClick={handleModeChange}>
            {isLogin ? "계정이 없으신가요? 회원가입" : "이미 계정이 있나요? 로그인"}
          </button>
        </div>
      </section>
    </main>
  );
}
export default Login;
