from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from langchain_community.embeddings import OllamaEmbeddings
from .vector_db_service import VectorDBService
from ..schemas.job_posting import JobPostingCreate, JobPosting, JobSearchResult

class JobPostingService:
    def __init__(self):
        self.vector_db = VectorDBService()
        self.embeddings = OllamaEmbeddings(
            model="llama2",
            base_url="http://localhost:11434"
        )

    async def create_job_posting(self, job_data: JobPostingCreate) -> Optional[JobPosting]:
        try:
            # UUID 생성
            job_id = str(uuid.uuid4())
            
            # 채용 공고 설명을 임베딩
            description_text = f"{job_data.title} {job_data.description} {' '.join(job_data.required_skills)}"
            vector = await self.embeddings.aembed_query(description_text)
            
            # JobPosting 객체 생성
            job_posting = JobPosting(
                id=job_id,
                **job_data.dict(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                vector=vector
            )
            
            # 벡터 DB에 저장
            metadata = job_posting.dict(exclude={'vector'})
            success = await self.vector_db.upsert_job_posting(
                job_posting=metadata,
                vector=vector
            )
            
            if success:
                return job_posting
            return None
            
        except Exception as e:
            print(f"Error creating job posting: {str(e)}")
            return None

    async def search_similar_jobs(
        self,
        query_text: str,
        limit: int = 5
    ) -> List[JobSearchResult]:
        try:
            # 검색어를 임베딩
            query_vector = await self.embeddings.aembed_query(query_text)
            
            # 유사한 채용 공고 검색
            results = await self.vector_db.search_similar_jobs(
                vector=query_vector,
                limit=limit
            )
            
            # 검색 결과를 JobSearchResult 형식으로 변환
            search_results = []
            for result in results:
                job_posting = JobPosting(
                    **result['metadata'],
                    id=result['job_id'],
                    vector=None  # 벡터는 제외
                )
                search_result = JobSearchResult(
                    job=job_posting,
                    similarity_score=result['score']
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching jobs: {str(e)}")
            return []

    async def batch_create_job_postings(
        self,
        job_postings: List[JobPostingCreate]
    ) -> List[Optional[JobPosting]]:
        results = []
        for job_data in job_postings:
            result = await self.create_job_posting(job_data)
            results.append(result)
        return results

    async def delete_job_posting(self, job_id: str) -> bool:
        return await self.vector_db.delete_job_posting(job_id)

    # 샘플 데이터 생성 메서드
    async def create_sample_job_postings(self) -> List[JobPosting]:
        sample_jobs = [
            JobPostingCreate(
                title="시니어 경비원",
                company_name="안전제일 시큐리티",
                location="서울시 강남구",
                job_type="계약직",
                salary="시급 12,000원",
                required_skills=["보안경비신임교육 이수자", "신체 건강한 자"],
                description="""
                50대 이상 시니어 경비원을 모집합니다.
                - 근무시간: 4조 3교대
                - 업무내용: 출입관리, 순찰, 주차관리
                - 우대사항: 경비업무 유경험자
                """,
                working_hours="8시간/일",
                benefits=["4대보험", "중식제공", "교통비지원"]
            ),
            JobPostingCreate(
                title="아파트 관리소장",
                company_name="한마음 주택관리",
                location="경기도 분당구",
                job_type="정규직",
                salary="월 300만원",
                required_skills=["주택관리사", "전기안전관리자"],
                description="""
                경력있는 시니어 관리소장님을 모십니다.
                - 아파트 단지 관리소장
                - 입주민 민원처리 및 시설관리 총괄
                - 필수: 주택관리사 자격증
                """,
                working_hours="9:00-18:00 (주5일)",
                benefits=["4대보험", "퇴직금", "명절수당"]
            ),
            JobPostingCreate(
                title="실버택배 배송기사",
                company_name="안심물류",
                location="서울시 송파구",
                job_type="계약직",
                salary="건당 수수료",
                required_skills=["1종 운전면허", "스마트폰 활용가능자"],
                description="""
                50대 이상 시니어를 위한 실버택배 배송기사를 모집합니다.
                - 담당구역 내 택배 배송
                - 시간 자율선택제
                - 배송건당 수수료 지급
                """,
                working_hours="자율출퇴근",
                benefits=["유류비지원", "상해보험가입"]
            ),
            JobPostingCreate(
                title="식당 주방보조",
                company_name="맛있는 식당",
                location="서울시 중구",
                job_type="파트타임",
                salary="시급 11,000원",
                required_skills=["주방 경험자", "식품위생교육 이수자"],
                description="""
                오전 주방보조 구인
                - 업무: 식재료 손질, 주방 청소
                - 근무시간: 오전 9시 ~ 오후 2시
                - 우대: 단체급식 경험자
                """,
                working_hours="5시간/일",
                benefits=["중식제공", "교통비지원", "근무복지급"]
            ),
            JobPostingCreate(
                title="시니어 모델",
                company_name="실버스타 미디어",
                location="서울시 마포구",
                job_type="프리랜서",
                salary="회당 20만원",
                required_skills=["카메라 촬영 가능자"],
                description="""
                50대 이상 시니어 모델 모집
                - 온라인 쇼핑몰 의류 촬영
                - 월 2-3회 촬영
                - 경험무관
                """,
                working_hours="촬영일정 협의",
                benefits=["촬영의상 제공", "메이크업 지원"]
            )
        ]
        
        return await self.batch_create_job_postings(sample_jobs) 