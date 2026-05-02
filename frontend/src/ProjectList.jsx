import React, { useState, useEffect } from 'react';

function ProjectList() {
  const [projects, setProjects] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('newest');
  
  // 탐색기 내비게이션 상태
  const [currentFolderId, setCurrentFolderId] = useState(null);
  const [navHistory, setNavHistory] = useState([null]); 
  const [historyIndex, setHistoryIndex] = useState(0); 

  // 생성 모달 상태
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [isFolderType, setIsFolderType] = useState(false);
  const [selectedFolderId, setSelectedFolderId] = useState('root');

  // 드래그 앤 드롭 상태
  const [draggedItemId, setDraggedItemId] = useState(null);

  const userId = localStorage.getItem('user_id');

  useEffect(() => { fetchProjects(); }, []);

  const fetchProjects = async () => {
    const res = await fetch(`http://localhost:8000/projects/${userId}`);
    const data = await res.json();
    setProjects(data);
  };

  // 내비게이션 함수
  const navigateToFolder = (folderId) => {
    if (folderId === currentFolderId) return;
    const newHistory = navHistory.slice(0, historyIndex + 1);
    newHistory.push(folderId);
    setNavHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    setCurrentFolderId(folderId);
  };

  const goBack = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setCurrentFolderId(navHistory[newIndex]);
    }
  };

  const goForward = () => {
    if (historyIndex < navHistory.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setCurrentFolderId(navHistory[newIndex]);
    }
  };

  // --- API 호출 함수들 ---

  const handleCreate = async (e) => {
    e.preventDefault();
    await fetch(`http://localhost:8000/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        name: newName, 
        description: newDesc,
        is_folder: isFolderType, 
        parent_id: selectedFolderId === 'root' ? null : parseInt(selectedFolderId),
        user_id: userId 
      })
    });
    setNewName(''); setNewDesc(''); setShowCreateModal(false); setSelectedFolderId('root');
    fetchProjects();
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation(); // 아이콘 클릭 이벤트 전파 방지
    if (!confirm("정말 삭제하시겠습니까? 폴더 삭제 시 내부 파일도 관리 대상에서 제외될 수 있습니다.")) return;
    try {
      await fetch(`http://localhost:8000/projects/${id}`, { method: 'DELETE' });
      fetchProjects();
    } catch (error) {
      alert("삭제 중 오류가 발생했습니다.");
    }
  };

  const handleMove = async (projectId, targetFolderId) => {
    if (projectId === targetFolderId) return; // 자기 자신에게 이동 방지
    const parent_id = targetFolderId === 'root' ? null : parseInt(targetFolderId);
    
    await fetch(`http://localhost:8000/projects/${projectId}/move`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parent_id })
    });
    fetchProjects();
  };

  // --- 드래그 앤 드롭 핸들러 ---

  const onDragStart = (e, id) => {
    setDraggedItemId(id);
    e.dataTransfer.setData("projectId", id);
  };

  const onDragOver = (e) => {
    e.preventDefault(); // 드롭 허용을 위해 필수
  };

  const onDrop = (e, targetFolderId) => {
    e.preventDefault();
    const projectId = e.dataTransfer.getData("projectId");
    handleMove(projectId, targetFolderId);
    setDraggedItemId(null);
  };

  // 주소 표시줄 계산
  const getBreadcrumbs = () => {
    const path = [];
    let currId = currentFolderId;
    while (currId !== null) {
      const folder = projects.find(p => p.id === currId);
      if (folder) {
        path.unshift(folder);
        currId = folder.parent_id;
      } else break;
    }
    return path;
  };

  const folders = projects.filter(p => p.is_folder);
  const currentContent = projects.filter(p => p.parent_id === currentFolderId);
  const displayContent = currentContent
    .filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      return new Date(b.created_at) - new Date(a.created_at);
    });

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#fff', fontFamily: 'sans-serif' }}>
      {/* 1. 좌측 사이드바 (트리 뷰 + 드롭 타겟) */}
      <div style={{ width: '280px', borderRight: '1px solid #e0e0e0', backgroundColor: '#f6f8fa', padding: '15px' }}>
        <h3 style={{ fontSize: '1.2rem', color: '#0969da', marginBottom: '20px' }}>DLops Explorer</h3>
        <div>
          <div 
            onDragOver={onDragOver}
            onDrop={(e) => onDrop(e, 'root')}
            onClick={() => navigateToFolder(null)} 
            style={{ 
              cursor: 'pointer', padding: '8px', borderRadius: '4px',
              backgroundColor: currentFolderId === null ? '#e1ecf4' : 'transparent',
              fontWeight: currentFolderId === null ? 'bold' : 'normal'
            }}
          >
            🏠 최상위 (Root)
          </div>
          {folders.map(folder => (
            <div key={folder.id} style={{ marginLeft: '15px', marginTop: '5px' }}>
              <div 
                draggable="true"
                onDragStart={(e) => onDragStart(e, folder.id)}
                onDragOver={onDragOver}
                onDrop={(e) => onDrop(e, folder.id)}
                onClick={() => navigateToFolder(folder.id)} 
                style={{ 
                  cursor: 'pointer', padding: '5px', borderRadius: '4px',
                  color: currentFolderId === folder.id ? '#0969da' : '#333',
                  backgroundColor: currentFolderId === folder.id ? '#e1ecf4' : 'transparent',
                  border: '1px dashed transparent'
                }}
                onDragEnter={(e) => e.currentTarget.style.border = '1px dashed #0969da'}
                onDragLeave={(e) => e.currentTarget.style.border = '1px dashed transparent'}
              >
                📁 {folder.name}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 2. 메인 화면 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* 상단 툴바 */}
        <div style={{ padding: '15px 20px', borderBottom: '1px solid #e0e0e0', display: 'flex', alignItems: 'center', gap: '15px', backgroundColor: '#fafafa' }}>
          <div style={{ display: 'flex', gap: '5px' }}>
            <button onClick={goBack} disabled={historyIndex === 0}>←</button>
            <button onClick={goForward} disabled={historyIndex === navHistory.length - 1}>→</button>
          </div>
          <div style={{ flex: 1, backgroundColor: '#fff', padding: '8px 15px', border: '1px solid #ccc', borderRadius: '4px', display: 'flex', alignItems: 'center' }}>
            <span onClick={() => navigateToFolder(null)} style={{ cursor: 'pointer', color: '#0969da' }}>Root</span>
            {getBreadcrumbs().map(f => (
              <React.Fragment key={f.id}>
                <span style={{ margin: '0 8px', color: '#999' }}>&gt;</span>
                <span onClick={() => navigateToFolder(f.id)} style={{ cursor: 'pointer', color: '#0969da' }}>{f.name}</span>
              </React.Fragment>
            ))}
          </div>
          <input type="text" placeholder="검색..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} style={{ padding: '8px', width: '220px' }} />
        </div>

        <div style={{ padding: '15px 20px', display: 'flex', gap: '10px' }}>
          <button onClick={() => { setIsFolderType(true); setShowCreateModal(true); }}>+ 새 폴더</button>
          <button onClick={() => { setIsFolderType(false); setShowCreateModal(true); }} style={{ backgroundColor: '#2da44e', color: 'white' }}>+ 새 프로젝트</button>
        </div>

        {/* 중앙 콘텐츠 (그리드 뷰) */}
        <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '25px' }}>
            {displayContent.map(item => (
              <div 
                key={item.id} 
                draggable="true"
                onDragStart={(e) => onDragStart(e, item.id)}
                onDoubleClick={() => item.is_folder && navigateToFolder(item.id)}
                style={{ 
                  textAlign: 'center', padding: '20px 15px', borderRadius: '8px', 
                  border: '1px solid #eaeaea', position: 'relative', transition: 'all 0.2s'
                }}
              >
                {/* 삭제 버튼 추가 */}
                <button 
                  onClick={(e) => handleDelete(e, item.id)}
                  style={{ position: 'absolute', top: '5px', right: '5px', border: 'none', background: 'none', color: '#ff4d4f', cursor: 'pointer', fontWeight: 'bold' }}
                >
                  ✕
                </button>

                <div style={{ fontSize: '3.5rem', marginBottom: '15px' }}>{item.is_folder ? '📁' : '📄'}</div>
                <div style={{ fontWeight: 'bold' }}>{item.name}</div>
                {!item.is_folder && <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '5px' }}>{item.description}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 생성 모달 (동일) */}
      {showCreateModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <div style={{ backgroundColor: '#fff', padding: '30px', borderRadius: '10px', width: '400px' }}>
            <h3>{isFolderType ? '📁 새 폴더' : '📄 새 프로젝트'} 생성</h3>
            <form onSubmit={handleCreate}>
              <label>저장 위치</label>
              <select value={selectedFolderId} onChange={(e) => setSelectedFolderId(e.target.value)} style={{ width: '100%', padding: '10px', marginBottom: '15px' }}>
                <option value="root">Root (최상위)</option>
                {folders.map(f => <option key={f.id} value={f.id}>{f.name}</option>)}
              </select>
              <input type="text" placeholder="이름" value={newName} onChange={(e) => setNewName(e.target.value)} required style={{ width: '100%', padding: '10px', marginBottom: '15px' }} />
              {!isFolderType && <textarea placeholder="설명" value={newDesc} onChange={(e) => setNewDesc(e.target.value)} style={{ width: '100%', padding: '10px', height: '80px', marginBottom: '15px' }} />}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button type="button" onClick={() => setShowCreateModal(false)}>취소</button>
                <button type="submit" style={{ backgroundColor: '#0969da', color: '#fff' }}>생성</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProjectList;