from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from .job_posting import JobSearchResult

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "llama2"  # "llama2" 또는 "phi4"

class ChatResponse(BaseModel):
    message: str
    keywords: Dict[str, List[str]]
    embeddings: List[float]
    similar_jobs: List[Any] 