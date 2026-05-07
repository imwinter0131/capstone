from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 본인의 PostgreSQL 비밀번호와 DB 이름에 맞게 수정하세요.
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:0101@localhost/DLops"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB 세션을 가져오는 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()