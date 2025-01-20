import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import voice

app = FastAPI(debug=True)  # debug 모드 활성화

# CORS 설정 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱 주소를 명시적으로 지정
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

try:
    # 정적 파일 서빙 설정
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"Static files directory mounted: {static_dir}")
except Exception as e:
    print(f"Error mounting static files: {e}")

# 라우터 등록 - 디버깅 로그 추가
print("Registering voice router...")
app.include_router(voice.router, prefix="/api/voice")
print("Voice router registered successfully")

@app.get("/")
async def root():
    return {"message": "Server is running"} 