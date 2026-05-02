from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
import models, database

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=database.engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==========================================
# 1. 스키마 (데이터 형식) 정의
# ==========================================
class UserAuth(BaseModel):
    email: EmailStr
    password: str

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_folder: bool = False
    parent_id: Optional[int] = None
    user_id: int

class ProjectMove(BaseModel):
    parent_id: Optional[int] = None

# ==========================================
# 2. 인증(로그인/회원가입) API (이게 빠져있었습니다!)
# ==========================================
@app.post("/signup")
def signup(user_data: UserAuth, db: Session = Depends(database.get_db)):
    if db.query(models.User).filter(models.User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다.")
    new_user = models.User(email=user_data.email, hashed_password=pwd_context.hash(user_data.password))
    db.add(new_user)
    db.commit()
    return {"message": "회원가입 성공"}

@app.post("/login")
def login(user_data: UserAuth, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == user_data.email).first()
    if not user or not pwd_context.verify(user_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="정보가 일치하지 않습니다.")
    return {"user_id": user.id, "message": "로그인 성공"}

# ==========================================
# 3. 프로젝트 관리 API
# ==========================================
@app.get("/projects/{user_id}")
def get_projects(user_id: int, db: Session = Depends(database.get_db)):
    return db.query(models.Project).filter(models.Project.user_id == user_id).all()

@app.post("/projects")
def create_project(project: ProjectCreate, db: Session = Depends(database.get_db)):
    new_p = models.Project(**project.dict())
    db.add(new_p)
    db.commit()
    db.refresh(new_p)
    return new_p

@app.patch("/projects/{project_id}/move")
def move_project(project_id: int, move: ProjectMove, db: Session = Depends(database.get_db)):
    proj = db.query(models.Project).filter(models.Project.id == project_id).first()
    proj.parent_id = move.parent_id
    db.commit()
    return {"message": "이동 성공"}

@app.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(database.get_db)):
    proj = db.query(models.Project).filter(models.Project.id == project_id).first()
    db.delete(proj)
    db.commit()
    return {"message": "삭제 성공"}