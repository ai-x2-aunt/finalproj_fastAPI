from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from typing import Optional
from ..core.config import settings
from .work24_service import Work24Service
from .hrd_service import HRDService
from .vector_db_service import VectorDBService
from .llm_service import LLMService

class SchedulerService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.work24_service = Work24Service()
        self.hrd_service = HRDService()
        self.vector_db = VectorDBService()
        self.llm_service = LLMService(model_name="llama2")  # 임베딩용
        
        # 작업 상태 추적
        self.last_job_collection: Optional[datetime] = None
        self.last_training_collection: Optional[datetime] = None
        self.collection_status = {
            "jobs": {"status": "idle", "count": 0, "error": None},
            "training": {"status": "idle", "count": 0, "error": None}
        }

    async def collect_jobs(self):
        """채용 정보 수집 작업"""
        try:
            self.collection_status["jobs"]["status"] = "running"
            total_count = 0
            
            # 여러 페이지의 데이터 수집
            for page in range(1, 11):  # 최대 10페이지
                job_postings = await self.work24_service.fetch_job_postings(
                    start_page=page,
                    display=100
                )
                
                if not job_postings:
                    break
                
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
                        vector = await self.llm_service.embeddings.aembed_query(job_text)
                        
                        # 벡터 DB에 저장
                        success = await self.vector_db.upsert_job_posting(
                            job_posting=job,
                            vector=vector
                        )
                        
                        if success:
                            total_count += 1
                            
                    except Exception as e:
                        print(f"채용 공고 처리 중 오류 발생: {str(e)}")
                        continue
            
            self.collection_status["jobs"]["status"] = "completed"
            self.collection_status["jobs"]["count"] = total_count
            self.collection_status["jobs"]["error"] = None
            self.last_job_collection = datetime.now()
            
        except Exception as e:
            self.collection_status["jobs"]["status"] = "error"
            self.collection_status["jobs"]["error"] = str(e)
            print(f"채용 정보 수집 중 오류 발생: {str(e)}")

    async def collect_training_programs(self):
        """훈련 프로그램 수집 작업"""
        try:
            self.collection_status["training"]["status"] = "running"
            total_count = 0
            
            # 여러 페이지의 데이터 수집
            for page in range(1, 11):  # 최대 10페이지
                programs = await self.hrd_service.fetch_training_programs(
                    start_page=page,
                    display=100
                )
                
                if not programs:
                    break
                
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
                        vector = await self.llm_service.embeddings.aembed_query(program_text)
                        
                        # 벡터 DB에 저장
                        success = await self.vector_db.upsert_training_program(
                            program=program,
                            vector=vector
                        )
                        
                        if success:
                            total_count += 1
                            
                    except Exception as e:
                        print(f"훈련 프로그램 처리 중 오류 발생: {str(e)}")
                        continue
            
            self.collection_status["training"]["status"] = "completed"
            self.collection_status["training"]["count"] = total_count
            self.collection_status["training"]["error"] = None
            self.last_training_collection = datetime.now()
            
        except Exception as e:
            self.collection_status["training"]["status"] = "error"
            self.collection_status["training"]["error"] = str(e)
            print(f"훈련 프로그램 수집 중 오류 발생: {str(e)}")

    def start(self):
        """스케줄러 시작"""
        # 매일 새벽 3시에 채용 정보 수집
        self.scheduler.add_job(
            self.collect_jobs,
            CronTrigger(hour=3, minute=0),
            id="collect_jobs",
            replace_existing=True
        )
        
        # 매일 새벽 4시에 훈련 프로그램 수집
        self.scheduler.add_job(
            self.collect_training_programs,
            CronTrigger(hour=4, minute=0),
            id="collect_training_programs",
            replace_existing=True
        )
        
        self.scheduler.start()
    
    def get_status(self):
        """수집 상태 조회"""
        return {
            "jobs": {
                **self.collection_status["jobs"],
                "last_collection": self.last_job_collection.isoformat() if self.last_job_collection else None
            },
            "training": {
                **self.collection_status["training"],
                "last_collection": self.last_training_collection.isoformat() if self.last_training_collection else None
            }
        } 