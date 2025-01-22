from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
import os
from datetime import datetime
import json

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
        self.client = QdrantClient("localhost", port=6333)
        
        # 채용 공고 컬렉션 초기화
        self._init_collection("job_postings", vector_size=384)
        # 훈련 프로그램 컬렉션 초기화
        self._init_collection("training_programs", vector_size=384)

    def _init_collection(self, collection_name: str, vector_size: int):
        """컬렉션 초기화 또는 생성"""
        collections = self.client.get_collections().collections
        exists = any(collection.name == collection_name for collection in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

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
            
            self.client.upsert(
                collection_name="job_postings",
                points=[
                    models.PointStruct(
                        id=str(hash(f"{job_posting['company_name']}_{job_posting['title']}_{datetime.now().timestamp()}")),
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

    async def search_similar_jobs(self, vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """유사한 채용 공고 검색"""
        try:
            results = self.client.search(
                collection_name="job_postings",
                query_vector=vector,
                limit=limit,
                score_threshold=0.7
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

    async def search_similar_programs(self, vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """유사한 훈련 프로그램 검색"""
        try:
            results = self.client.search(
                collection_name="training_programs",
                query_vector=vector,
                limit=limit,
                score_threshold=0.7
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