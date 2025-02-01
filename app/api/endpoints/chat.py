from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.llm_service import LLMService

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 메시지를 처리하고 응답을 반환합니다."""
    service = LLMService(model_name=request.model_name)
    response = await service.get_response(request.message)
    return response

@router.post("/reset")
async def reset_chat(model_name: str = "gpt-4o-mini"):
    """채팅 대화 기록을 초기화합니다."""
    service = LLMService(model_name=model_name)
    service.reset_conversation()
    return {"message": "대화가 초기화되었습니다."} 