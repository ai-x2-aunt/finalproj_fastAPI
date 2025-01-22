from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.v1 import data_management_router
from .api.endpoints import chat  # 채팅 라우터 import
from .services.scheduler_service import SchedulerService

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 주소 명시
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(data_management_router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])  # 채팅 라우터 등록 수정

# 스케줄러 초기화 및 시작
scheduler = SchedulerService()

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    scheduler.scheduler.shutdown()

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 