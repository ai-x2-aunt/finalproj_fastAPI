from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    # 기본 설정
    PROJECT_NAME: str = "Senior Job Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS 설정
    BACKEND_CORS_ORIGINS: List[str]
    
    # JWT 설정
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Vector DB 설정
    VECTOR_DB_URL: str

    # LangChain 설정
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGCHAIN_PROJECT: str

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    """
    # MongoDB 설정 (현재 미사용)
    MONGODB_URL: str
    DATABASE_NAME: str
    
    # AWS 설정 (현재 미사용)
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-northeast-2"
    
    # vLLM 설정 (현재 미사용)
    VLLM_MODEL_PATH: str
    VLLM_MAX_LENGTH: int = 2048
    VLLM_TOP_K: int = 50
    VLLM_TOP_P: float = 0.95
    VLLM_TEMPERATURE: float = 0.7

    # 고용24 공통코드 API 설정 (현재 미사용)
    WORK24_COMMON_CODE_API_KEY: str
    WORK24_API_COMMON_BASE_URL: str
    WORK24_API_COMMON_TRAINING_URL: str

    # 국민내일배움카드 API 설정 (현재 미사용)
    TOMORROW_LEARNING_CARD_API_KEY: str
    TOMORROW_LEARNING_CARD_API_BASE_URL: str
    TOMORROW_LEARNING_CARD_LIST_ENDPOINT: str
    TOMORROW_LEARNING_CARD_DETAIL_ENDPOINT: str
    TOMORROW_LEARNING_CARD_SCHEDULE_ENDPOINT: str

    # API 엔드포인트 설정 (현재 미사용)
    TRAINING_COMMON_CODE_ENDPOINT: str
    REGION_CODE_ENDPOINT: str
    JOB_CODE_ENDPOINT: str
    CERTIFICATE_CODE_ENDPOINT: str
    INDUSTRIAL_COMPLEX_ENDPOINT: str
    SUBWAY_CODE_ENDPOINT: str
    MAJOR_CODE_ENDPOINT: str
    LANGUAGE_CODE_ENDPOINT: str
    DEPARTMENT_CODE_ENDPOINT: str
    SMALL_GIANT_CODE_ENDPOINT: str

    # 벡터 DB 설정 (현재 미사용)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Ollama 설정 (현재 미사용)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    """

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings() 