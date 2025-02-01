from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from .job_posting import JobSearchResult

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "gpt-4o-mini"  

class ChatResponse(BaseModel):
    message: str
    keywords: Dict[str, List[str]]
    embeddings: List[float]
    similar_jobs: List[Any] 