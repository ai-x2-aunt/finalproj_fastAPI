from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
import os
from datetime import datetime
import json
import uuid
from qdrant_client.models import PointStruct

# Pinecone 클라우드 옵션 (현재는 주석 처리)
"""
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Pinecone 초기화
def init_pinecone():
    pc = Pinecone(
        api_key="your-api-key",
        environment="your-environment"
    )
    
    # 서버리스 인덱스 생성 (비용 효율적)
    if "senior-jobs" not in pc.list_indexes():
        pc.create_index(
            name="senior-jobs",
            dimension=384,  # llama2 임베딩 차원
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    
    return pc.Index("senior-jobs")
"""

class VectorDBService:
    def __init__(self):
        # 로컬 Qdrant 클라이언트 초기화
        self.client = QdrantClient(host="localhost", port=6333)
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """필요한 컬렉션이 존재하는지 확인하고 없으면 생성"""
        collections = ["job_postings", "training_programs"]
        vector_size = 4096  # llama2의 기본 임베딩 크기

        for collection_name in collections:
            try:
                # 컬렉션 존재 여부 확인
                collection_info = self.client.get_collection(collection_name)
                print(f"Collection {collection_name} exists")
                
            except Exception as e:
                print(f"Creating new collection: {collection_name}")
                # 컬렉션이 없는 경우에만 새로 생성
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                
                # 컬렉션 생성 후 인덱스 생성
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="keywords",
                    field_schema="keyword"
                )
                
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="location",
                    field_schema="keyword"
                )
                
                print(f"Created indices for {collection_name}")

    async def get_collection_info(self, collection_name: str) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(collection_name)
            points_count = self.client.count(collection_name).count
            return {
                "name": collection_name,
                "points_count": points_count,
                "config": info.config
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}

    async def upsert_job_posting(self, job_posting: Dict[str, Any], vector: List[float]):
        """채용 공고 저장 또는 업데이트"""
        try:
            # 메타데이터에서 벡터 필드 제거
            metadata = job_posting.copy()
            metadata.pop("vector", None)
            
            # datetime 객체를 문자열로 변환
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
            
            # UUID 생성
            point_id = str(uuid.uuid4())
            
            self.client.upsert(
                collection_name="job_postings",
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=metadata
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Error upserting job posting: {str(e)}")
            return False

    async def upsert_training_program(self, program: Dict[str, Any], vector: List[float]):
        """훈련 프로그램 저장 또는 업데이트"""
        try:
            metadata = program.copy()
            metadata.pop("vector", None)
            
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
            
            self.client.upsert(
                collection_name="training_programs",
                points=[
                    models.PointStruct(
                        id=str(hash(f"{program['institution']}_{program['title']}_{datetime.now().timestamp()}")),
                        vector=vector,
                        payload=metadata
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Error upserting training program: {str(e)}")
            return False

    async def search_similar_jobs(
        self, 
        vector: List[float], 
        limit: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """유사한 채용 공고 검색"""
        try:
            # 필터 조건 구성
            must_conditions = []
            
            if filter_conditions:
                # 키워드 기반 필터링 (선택적)
                if "keywords" in filter_conditions and filter_conditions["keywords"]:
                    keywords = filter_conditions["keywords"]
                    if keywords != ["일반"]:  # "일반" 키워드는 무시
                        must_conditions.append(
                            models.FieldCondition(
                                key="keywords",
                                match=models.MatchAny(any=keywords)
                            )
                        )
                
                # 지역 기반 필터링 (선택적)
                if "location" in filter_conditions and filter_conditions["location"]:
                    locations = filter_conditions["location"]
                    if locations != ["서울"]:  # "서울" 키워드는 무시
                        must_conditions.append(
                            models.FieldCondition(
                                key="location",
                                match=models.MatchAny(any=locations)
                            )
                        )

            # 검색 실행 (필터가 없어도 검색 가능)
            results = self.client.search(
                collection_name="job_postings",
                query_vector=vector,
                limit=limit,
                score_threshold=0.1,  # 임계값을 낮춰서 더 많은 결과를 얻음
                query_filter=models.Filter(
                    must=must_conditions
                ) if must_conditions else None
            )
            
            return [
                {
                    "metadata": point.payload,
                    "score": point.score
                }
                for point in results
            ]
        except Exception as e:
            print(f"Error searching jobs: {str(e)}")
            return []

    async def search_similar_programs(
        self, 
        vector: List[float], 
        limit: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """유사한 훈련 프로그램 검색"""
        try:
            # 필터 조건 구성
            must_conditions = []
            
            if filter_conditions:
                # 키워드 기반 필터링
                if "keywords" in filter_conditions and filter_conditions["keywords"]:
                    keywords = filter_conditions["keywords"]
                    must_conditions.append(
                        models.FieldCondition(
                            key="keywords",
                            match=models.MatchAny(any=keywords)
                        )
                    )

            # 검색 실행
            results = self.client.search(
                collection_name="training_programs",
                query_vector=vector,
                limit=limit,
                score_threshold=0.7,
                query_filter=models.Filter(
                    must=must_conditions
                ) if must_conditions else None
            )
            
            return [
                {
                    "metadata": point.payload,
                    "score": point.score
                }
                for point in results
            ]
        except Exception as e:
            print(f"Error searching training programs: {str(e)}")
            return []

    async def delete_job_posting(self, job_id: str) -> bool:
        """채용 공고 삭제"""
        try:
            self.client.delete(
                collection_name="job_postings",
                points_selector=models.PointIdsList(
                    points=[job_id]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting job posting: {str(e)}")
            return False

    async def delete_training_program(self, program_id: str) -> bool:
        """훈련 프로그램 삭제"""
        try:
            self.client.delete(
                collection_name="training_programs",
                points_selector=models.PointIdsList(
                    points=[program_id]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting training program: {str(e)}")
            return False

    # 클라우드 Pinecone 구현 (주석 처리)
    """
    async def upsert_job_posting_cloud(
        self,
        index: Any,  # Pinecone 인덱스
        job_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        try:
            index.upsert(
                vectors=[
                    {
                        "id": job_id,
                        "values": vector,
                        "metadata": metadata
                    }
                ]
            )
            return True
        except Exception as e:
            print(f"Error upserting job posting to cloud: {str(e)}")
            return False

    async def search_similar_jobs_cloud(
        self,
        index: Any,  # Pinecone 인덱스
        vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            results = index.query(
                vector=vector,
                top_k=limit,
                include_metadata=True
            )
            
            return [{
                "job_id": match.id,
                "score": match.score,
                "metadata": match.metadata
            } for match in results.matches]
        except Exception as e:
            print(f"Error searching jobs in cloud: {str(e)}")
            return []
    """ 

    async def add_job_posting(self, vector: List[float], payload: Dict[str, Any]) -> None:
        """채용공고 데이터를 벡터 DB에 저장"""
        try:
            point_id = uuid.uuid4().hex
            
            # 키워드 필드 추가
            if "keywords" not in payload:
                # 제목과 설명에서 키워드 추출
                keywords = []
                if "title" in payload:
                    keywords.extend(payload["title"].split())
                if "description" in payload:
                    keywords.extend(payload["description"].split())
                payload["keywords"] = list(set(keywords))  # 중복 제거
            
            self.client.upsert(
                collection_name="job_postings",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            print(f"Successfully added job posting with ID: {point_id}")
        except Exception as e:
            print(f"Error adding job posting to vector DB: {str(e)}")
            raise e

    async def add_training_program(self, vector: List[float], payload: Dict[str, Any]) -> None:
        """훈련과정 데이터를 벡터 DB에 저장"""
        try:
            point_id = uuid.uuid4().hex
            
            # 키워드 필드 추가
            if "keywords" not in payload:
                # 제목과 설명에서 키워드 추출
                keywords = []
                if "title" in payload:
                    keywords.extend(payload["title"].split())
                if "description" in payload:
                    keywords.extend(payload["description"].split())
                payload["keywords"] = list(set(keywords))  # 중복 제거
            
            self.client.upsert(
                collection_name="training_programs",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            print(f"Successfully added training program with ID: {point_id}")
        except Exception as e:
            print(f"Error adding training program to vector DB: {str(e)}")
            raise e

    async def initialize_data(self):
        """벡터 DB 초기 데이터 로드"""
        try:
            # 컬렉션 정보 확인
            job_info = await self.get_collection_info("job_postings")
            training_info = await self.get_collection_info("training_programs")
            
            # 데이터가 없는 경우에만 초기화
            if job_info.get("points_count", 0) == 0:
                print("Initializing job postings data...")
                from .job_posting_service import JobPostingService
                job_service = JobPostingService()
                await job_service.create_sample_job_postings()
                
            if training_info.get("points_count", 0) == 0:
                print("Initializing training programs data...")
                # TODO: 훈련 프로그램 샘플 데이터 추가
            
            return True
        except Exception as e:
            print(f"Error initializing data: {str(e)}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보 조회"""
        try:
            job_info = await self.get_collection_info("job_postings")
            training_info = await self.get_collection_info("training_programs")
            
            return {
                "job_postings_count": job_info.get("points_count", 0),
                "training_programs_count": training_info.get("points_count", 0)
            }
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {
                "job_postings_count": 0,
                "training_programs_count": 0
            } 