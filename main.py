from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import logging
import sys
from pathlib import Path
import torch
from contextlib import asynccontextmanager

# 250122_b 디렉토리와 vector_db 디렉토리를 Python path에 추가
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "vector_db"))

from chatbot.job_assistant import JobAssistantBot

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수로 챗봇 인스턴스 생성
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global chatbot
    logger.info("Starting chatbot initialization...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        chatbot = JobAssistantBot(use_chroma=True)
        logger.info("Chatbot initialization completed")
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        raise
    
    yield  # FastAPI 애플리케이션 실행
    
    # 종료 시 실행
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shutting down, cleaned up resources")

app = FastAPI(lifespan=lifespan)

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    explanation: str
    jobs: List[Dict]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_input: ChatInput):
    try:
        logger.info(f"Received chat request: {chat_input.message}")
        response = chatbot.recommend_jobs(chat_input.message)
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 