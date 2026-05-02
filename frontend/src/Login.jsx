import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLogin, setIsLogin] = useState(true);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const endpoint = isLogin ? '/login' : '/signup';
    try {
      const response = await fetch("http://localhost:8000" + endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (response.ok) {
        alert(data.message);
        if (isLogin) {
          localStorage.setItem('user_id', data.user_id); // 유저 정보 저장
          navigate('/projects'); // 프로젝트 리스트로 이동
        }
      } else { alert(data.detail); }
    } catch (error) { alert("서버 연결 실패"); }
  };

  return (
    <div style={{ maxWidth: '400px', margin: '100px auto', textAlign: 'center', padding: '20px', border: '1px solid #ccc', borderRadius: '10px' }}>
      <h1>{isLogin ? 'DLops 로그인' : '회원가입'}</h1>
      <form onSubmit={handleSubmit}>
        <input type="email" placeholder="이메일" value={email} onChange={(e) => setEmail(e.target.value)} required style={{ width: '100%', marginBottom: '10px', padding: '10px' }} />
        <input type="password" placeholder="비밀번호" value={password} onChange={(e) => setPassword(e.target.value)} required style={{ width: '100%', marginBottom: '10px', padding: '10px' }} />
        <button type="submit" style={{ width: '100%', padding: '10px', backgroundColor: '#007bff', color: 'white', border: 'none' }}>{isLogin ? 'Continue' : '가입하기'}</button>
      </form>
      <p onClick={() => setIsLogin(!isLogin)} style={{ cursor: 'pointer', marginTop: '20px', color: 'blue' }}>{isLogin ? '계정이 없으신가요? 회원가입' : '이미 계정이 있나요? 로그인'}</p>
    </div>
  );
}

export default Login;
