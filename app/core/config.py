from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 기본 설정
    PROJECT_NAME: str = "Senior Job Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS 설정
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # MongoDB 설정
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "senior_job_db"
    
    # JWT 설정
    SECRET_KEY: str = "your-secret-key"  # 실제 운영 환경에서는 환경 변수로 관리
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AWS 설정
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "ap-northeast-2"
    
    # Vector DB 설정
    VECTOR_DB_URL: Optional[str] = None
    
    # vLLM 설정
    VLLM_MODEL_PATH: str = "your-model-path"
    VLLM_MAX_LENGTH: int = 2048
    VLLM_TOP_K: int = 50
    VLLM_TOP_P: float = 0.95
    VLLM_TEMPERATURE: float = 0.7

    # 고용24 API 설정
    WORK24_API_KEY: str
    WORK24_API_BASE_URL: str

    # HRD-Net API 설정
    HRD_API_KEY: str
    HRD_API_BASE_URL: str

    # 서버 설정
    HOST: str = "localhost"
    PORT: int = 8000

    # 벡터 DB 설정
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Ollama 설정
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() 