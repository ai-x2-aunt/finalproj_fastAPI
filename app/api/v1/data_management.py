from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from ...schemas.job_posting import JobPosting, TrainingProgram
from ...services.vector_db_service import VectorDBService
from ...services.llm_service import LLMService
from ...services.work24_service import Work24Service
from ...services.hrd_service import HRDService
from ...services.scheduler_service import SchedulerService
from ...services.code_service import CodeService
from datetime import datetime

router = APIRouter()
vector_db = VectorDBService()
work24_service = Work24Service()
hrd_service = HRDService()
scheduler = SchedulerService()
code_service = CodeService()

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

@router.post("/collect-jobs/", response_model=int)
async def collect_job_postings(
    start_page: int = Query(1, description="시작 페이지"),
    display: int = Query(100, description="페이지당 결과 수"),
    region: Optional[str] = Query(None, description="지역 코드"),
    occupation: Optional[str] = Query(None, description="직종 코드"),
    keyword: Optional[str] = Query(None, description="검색 키워드"),
    llm_service: LLMService = Depends(get_llm_service)
):
    """고용24 API에서 채용 정보를 수집하여 벡터 DB에 저장"""
    try:
        # 채용 정보 수집
        job_postings = await work24_service.fetch_job_postings(
            start_page=start_page,
            display=display,
            region=region,
            occupation=occupation,
            keyword=keyword
        )

        success_count = 0
        for job in job_postings:
            try:
                # 임베딩 생성
                job_text = f"""
                제목: {job['title']}
                회사: {job['company_name']}
                위치: {job['location']}
                직무 설명: {job['description']}
                자격 요건: {job['requirements']}
                """
                vector = await llm_service.embeddings.aembed_query(job_text)
                
                # 벡터 DB에 저장
                success = await vector_db.upsert_job_posting(
                    job_posting=job,
                    vector=vector
                )
                
                if success:
                    success_count += 1
                
            except Exception as e:
                print(f"채용 공고 처리 중 오류 발생: {str(e)}")
                continue

        return success_count

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채용 정보 수집 실패: {str(e)}")

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
    """HRD-Net API에서 훈련 프로그램 정보를 수집하여 벡터 DB에 저장"""
    try:
        # 훈련 프로그램 수집
        programs = await hrd_service.fetch_training_programs(
            start_page=start_page,
            display=display,
            region=region,
            keyword=keyword,
            training_type=training_type,
            training_target=training_target
        )

        success_count = 0
        for program in programs:
            try:
                # 임베딩 생성
                program_text = f"""
                프로그램: {program['title']}
                기관: {program['institution']}
                설명: {program['description']}
                자격 요건: {program['requirements']}
                지원금: {program['support_info']}
                """
                vector = await llm_service.embeddings.aembed_query(program_text)
                
                # 벡터 DB에 저장
                success = await vector_db.upsert_training_program(
                    program=program,
                    vector=vector
                )
                
                if success:
                    success_count += 1
                
            except Exception as e:
                print(f"훈련 프로그램 처리 중 오류 발생: {str(e)}")
                continue

        return success_count

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"훈련 프로그램 수집 실패: {str(e)}")

@router.get("/collection-status/", response_model=Dict)
async def get_collection_status():
    """데이터 수집 상태 조회"""
    return scheduler.get_status()

@router.post("/collect/start/")
async def start_collection():
    """데이터 수집 수동 시작"""
    try:
        # 채용 정보 수집 시작
        await scheduler.collect_jobs()
        # 훈련 프로그램 수집 시작
        await scheduler.collect_training_programs()
        return {"message": "데이터 수집이 시작되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/codes/{code_type}")
async def get_codes(code_type: str):
    """공통코드 정보 조회"""
    try:
        codes = await code_service.fetch_code(code_type)
        return {code_type: codes}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/codes/")
async def get_all_codes():
    """모든 공통코드 정보 조회"""
    try:
        return await code_service.get_all_codes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 