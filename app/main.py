from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.v1 import data_management_router
from .api.endpoints import chat  # 채팅 라우터 import
from .services.scheduler_service import SchedulerService
from .services.vector_db_service import VectorDBService
from .services.code_service import CodeService  # CodeService import 추가

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"  # React 개발 서버 포트 추가
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(data_management_router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])  # 채팅 라우터 등록 수정

# 서비스 초기화
scheduler = SchedulerService()
vector_db = VectorDBService()
code_service = CodeService()  # CodeService 인스턴스 생성

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    # 스케줄러 시작
    scheduler.start()
    
    # DB에서 데이터 로드
    await vector_db.initialize_data()
    
    # 벡터 DB 상태 확인
    stats = await vector_db.get_stats()
    print(f"Vector DB Stats: {stats}")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    scheduler.scheduler.shutdown()

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    stats = await vector_db.get_stats()
    codes = code_service.get_cached_codes()
    return {
        "status": "healthy",
        "vector_db_stats": stats,
        "code_stats": {k: len(v) for k, v in codes.items()}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 