from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from ...schemas.job_posting import JobPosting, TrainingProgram
from ...services.vector_db_service import VectorDBService
from ...services.llm_service import LLMService
# from ...services.work24_service import Work24Service  # 현재 미사용
# from ...services.hrd_service import HRDService  # 현재 미사용
# from ...services.scheduler_service import SchedulerService  # 현재 미사용
# from ...services.code_service import CodeService  # 현재 미사용
# from ...services.data_collection_service import DataCollectionService  # 현재 미사용
from ...core.config import settings
from datetime import datetime

router = APIRouter()
vector_db = VectorDBService()
# work24_service = Work24Service()  # 현재 미사용
# hrd_service = HRDService()  # 현재 미사용
# scheduler = SchedulerService()  # 현재 미사용
# code_service = CodeService()  # 현재 미사용
# data_collection_service = DataCollectionService()  # 현재 미사용

async def get_llm_service():
    return LLMService(model_name="llama2")  # 임베딩용으로 llama2 사용

@router.post("/jobs/", response_model=bool)
async def create_job_posting(job: JobPosting, llm_service: LLMService = Depends(get_llm_service)):
    """채용 공고 등록"""
    try:
        # 임베딩 생성
        job_text = f"""
        제목: {job.title}
        회사: {job.company_name}
        위치: {job.location}
        직무 설명: {job.description}
        자격 요건: {job.requirements}
        필요 기술: {', '.join(job.required_skills)}
        """
        vector = await llm_service.embeddings.aembed_query(job_text)
        
        # 벡터 DB에 저장
        success = await vector_db.upsert_job_posting(
            job_posting=job.model_dump(),
            vector=vector
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="채용 공고 저장 실패")
        
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training-programs/", response_model=bool)
async def create_training_program(program: TrainingProgram, llm_service: LLMService = Depends(get_llm_service)):
    """훈련 프로그램 등록"""
    try:
        # 임베딩 생성
        program_text = f"""
        프로그램: {program.title}
        기관: {program.institution}
        설명: {program.description}
        자격 요건: {program.requirements}
        취득 자격증: {program.certificate if program.certificate else '없음'}
        """
        vector = await llm_service.embeddings.aembed_query(program_text)
        
        # 벡터 DB에 저장
        success = await vector_db.upsert_training_program(
            program=program.model_dump(),
            vector=vector
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="훈련 프로그램 저장 실패")
        
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}", response_model=bool)
async def delete_job_posting(job_id: str):
    """채용 공고 삭제"""
    success = await vector_db.delete_job_posting(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="채용 공고 삭제 실패")
    return True

@router.delete("/training-programs/{program_id}", response_model=bool)
async def delete_training_program(program_id: str):
    """훈련 프로그램 삭제"""
    success = await vector_db.delete_training_program(program_id)
    if not success:
        raise HTTPException(status_code=500, detail="훈련 프로그램 삭제 실패")
    return True

"""
# 현재 미사용 엔드포인트들
@router.post("/collect-jobs/", response_model=int)
async def collect_job_postings(
    start_page: int = Query(1, description="시작 페이지"),
    display: int = Query(100, description="페이지당 결과 수"),
    region: Optional[str] = Query(None, description="지역 코드"),
    occupation: Optional[str] = Query(None, description="직종 코드"),
    keyword: Optional[str] = Query(None, description="검색 키워드"),
    llm_service: LLMService = Depends(get_llm_service)
):
    # ... 생략 ...

@router.post("/collect-training-programs/", response_model=int)
async def collect_training_programs(
    start_page: int = Query(1, description="시작 페이지"),
    display: int = Query(100, description="페이지당 결과 수"),
    region: Optional[str] = Query(None, description="지역 코드"),
    keyword: Optional[str] = Query(None, description="검색 키워드"),
    training_type: Optional[str] = Query(None, description="훈련 유형"),
    training_target: Optional[str] = Query(None, description="훈련 대상"),
    llm_service: LLMService = Depends(get_llm_service)
):
    # ... 생략 ...

@router.get("/collection-status/", response_model=Dict)
async def get_collection_status():
    # ... 생략 ...

@router.post("/collect/start/")
async def start_collection():
    # ... 생략 ...

@router.get("/codes/{code_type}")
async def get_codes(code_type: str):
    # ... 생략 ...

@router.get("/codes/")
async def get_all_codes():
    # ... 생략 ...

@router.post("/collect/job-postings", response_model=Dict[str, Any])
async def collect_job_postings():
    # ... 생략 ...

@router.post("/collect/training-programs", response_model=Dict[str, Any])
async def collect_training_programs():
    # ... 생략 ...
""" 