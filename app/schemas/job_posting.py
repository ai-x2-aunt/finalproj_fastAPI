from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class JobPostingCreate(BaseModel):
    """채용 공고 생성 요청"""
    title: str = Field(..., description="채용 공고 제목")
    company_name: str = Field(..., description="회사명")
    location: str = Field(..., description="근무지")
    job_type: str = Field(..., description="고용 형태")
    salary: str = Field(..., description="급여")
    required_skills: List[str] = Field(..., description="필요 기술")
    description: str = Field(..., description="직무 설명")
    working_hours: str = Field(..., description="근무 시간")
    benefits: List[str] = Field(..., description="복리후생")

class JobPosting(JobPostingCreate):
    """채용 공고 정보"""
    id: Optional[str] = None
    requirements: Optional[str] = Field(None, description="자격 요건")
    age_limit: Optional[str] = Field(None, description="연령 제한")
    education: Optional[str] = Field(None, description="학력 요건")
    experience: Optional[str] = Field(None, description="경력 요건")
    deadline: Optional[datetime] = Field(None, description="마감일")
    is_active: bool = Field(default=True, description="모집 중 여부")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    vector: Optional[List[float]] = Field(None, description="임베딩 벡터")

class TrainingProgram(BaseModel):
    """훈련 프로그램 정보"""
    title: str = Field(..., description="훈련 프로그램명")
    institution: str = Field(..., description="교육 기관")
    duration: str = Field(..., description="교육 기간")
    description: str = Field(..., description="프로그램 설명")
    requirements: str = Field(..., description="지원 자격")
    cost: str = Field(..., description="교육 비용")
    support_info: str = Field(..., description="지원금 정보")
    location: str = Field(..., description="교육 장소")
    schedule: str = Field(..., description="교육 시간")
    certificate: Optional[str] = Field(None, description="취득 가능 자격증")
    start_date: datetime = Field(..., description="시작일")
    end_date: datetime = Field(..., description="종료일")
    is_active: bool = Field(default=True, description="모집 중 여부")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    vector: Optional[List[float]] = Field(None, description="임베딩 벡터")

class JobSearchResult(BaseModel):
    """검색 결과"""
    job: JobPosting
    similarity_score: float = Field(..., description="유사도 점수")

class TrainingSearchResult(BaseModel):
    """훈련 프로그램 검색 결과"""
    program: TrainingProgram
    similarity_score: float = Field(..., description="유사도 점수") 