from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from ...schemas.job_posting import JobPostingCreate, JobPosting, JobSearchResult
from ...services.job_posting_service import JobPostingService

router = APIRouter()

async def get_job_posting_service() -> JobPostingService:
    return JobPostingService()

@router.post("/", response_model=JobPosting)
async def create_job_posting(
    job_data: JobPostingCreate,
    service: JobPostingService = Depends(get_job_posting_service)
):
    job_posting = await service.create_job_posting(job_data)
    if not job_posting:
        raise HTTPException(status_code=500, detail="채용 공고 생성에 실패했습니다")
    return job_posting

@router.post("/search", response_model=List[JobSearchResult])
async def search_jobs(
    query: str,
    limit: int = 5,
    service: JobPostingService = Depends(get_job_posting_service)
):
    results = await service.search_similar_jobs(query, limit)
    return results

@router.post("/sample", response_model=List[JobPosting])
async def create_sample_jobs(
    service: JobPostingService = Depends(get_job_posting_service)
):
    """샘플 채용 공고 데이터를 생성합니다"""
    return await service.create_sample_job_postings()

@router.delete("/{job_id}")
async def delete_job_posting(
    job_id: str,
    service: JobPostingService = Depends(get_job_posting_service)
):
    success = await service.delete_job_posting(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="채용 공고를 찾을 수 없습니다")
    return {"message": "채용 공고가 삭제되었습니다"} 