from fastapi import APIRouter
from .endpoints import chat, job_postings

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(job_postings.router, prefix="/job-postings", tags=["job-postings"]) 