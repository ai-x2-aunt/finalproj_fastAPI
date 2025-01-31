from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    # 기본 설정
    PROJECT_NAME: str = "Senior Job Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS 설정
    BACKEND_CORS_ORIGINS: List[str]
    
    # MongoDB 설정
    MONGODB_URL: str
    DATABASE_NAME: str
    
    # JWT 설정
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AWS 설정
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-northeast-2"
    
    # Vector DB 설정
<<<<<<< HEAD
    VECTOR_DB_URL: str
=======
    VECTOR_DB_PATH: str = "db"  # Chroma 데이터 저장 경로
>>>>>>> origin/master
    
    # vLLM 설정
    VLLM_MODEL_PATH: str
    VLLM_MAX_LENGTH: int = 2048
    VLLM_TOP_K: int = 50
    VLLM_TOP_P: float = 0.95
    VLLM_TEMPERATURE: float = 0.7

<<<<<<< HEAD
    # 고용24 공통코드 API 설정
    WORK24_COMMON_CODE_API_KEY: str
    WORK24_API_COMMON_BASE_URL: str
    WORK24_API_COMMON_TRAINING_URL: str
=======
    # Ollama 설정
    OLLAMA_BASE_URL: str

    # 고용24 API 설정
    WORK24_API_KEY: str
    WORK24_API_BASE_URL: str
    WORK24_API_COMMON_BASE_URL: str
    WORK24_API_COMMON_TRAINING_URL: str
    WORK24_COMMON_CODE_API_KEY: str
>>>>>>> origin/master

    # 국민내일배움카드 API 설정
    TOMORROW_LEARNING_CARD_API_KEY: str
    TOMORROW_LEARNING_CARD_API_BASE_URL: str
    TOMORROW_LEARNING_CARD_LIST_ENDPOINT: str
    TOMORROW_LEARNING_CARD_DETAIL_ENDPOINT: str
    TOMORROW_LEARNING_CARD_SCHEDULE_ENDPOINT: str

<<<<<<< HEAD
    # API 엔드포인트 설정
=======
    # 사업주훈련 API 설정
    EMPLOYER_TRAINING_API_KEY: str
    EMPLOYER_TRAINING_API_BASE_URL: str
    EMPLOYER_TRAINING_LIST_ENDPOINT: str
    EMPLOYER_TRAINING_DETAIL_ENDPOINT: str
    EMPLOYER_TRAINING_SCHEDULE_ENDPOINT: str

    # 일학습병행 API 설정
    WORK_LEARNING_API_KEY: str
    WORK_LEARNING_API_BASE_URL: str
    WORK_LEARNING_LIST_ENDPOINT: str
    WORK_LEARNING_DETAIL_ENDPOINT: str
    WORK_LEARNING_SCHEDULE_ENDPOINT: str

    # 컨소시엄 API 설정
    CONSORTIUM_API_KEY: str
    CONSORTIUM_API_BASE_URL: str
    CONSORTIUM_LIST_ENDPOINT: str
    CONSORTIUM_DETAIL_ENDPOINT: str
    CONSORTIUM_SCHEDULE_ENDPOINT: str

    # 공통코드 API 엔드포인트
>>>>>>> origin/master
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

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000

<<<<<<< HEAD
    # 벡터 DB 설정
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Ollama 설정
    OLLAMA_BASE_URL: str = "http://localhost:11434"

=======
>>>>>>> origin/master
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings() 