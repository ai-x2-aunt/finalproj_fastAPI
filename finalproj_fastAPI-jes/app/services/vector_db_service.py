from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import chromadb
from chromadb import Client, Settings
import json

class VectorDBService:
    def __init__(self):
        self.client = Client(Settings(
            persist_directory="db",  # 데이터를 저장할 디렉토리
            is_persistent=True      # 영구 저장 활성화
        ))
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """필요한 컬렉션이 존재하는지 확인하고 없으면 생성"""
        self.collections = {
            "job_postings": self.client.get_or_create_collection("job_postings"),
            "training_programs": self.client.get_or_create_collection("training_programs")
        }
        print("Collections initialized successfully")

    async def upsert_job_posting(self, job_posting: Dict[str, Any], vector: List[float]):
        """채용 공고 저장 또는 업데이트"""
        try:
            # 메타데이터 준비
            metadata = job_posting.copy()
            metadata.pop("vector", None)
            
            # None 값과 datetime, list 처리
            processed_metadata = {}
            for key, value in metadata.items():
                if value is None:
                    processed_metadata[key] = ""  # None을 빈 문자열로 변환
                elif isinstance(value, datetime):
                    processed_metadata[key] = value.isoformat()
                elif isinstance(value, list):
                    processed_metadata[key] = ", ".join(value)
                else:
                    processed_metadata[key] = value

            # ID 생성
            doc_id = str(uuid.uuid4())
            
            # 문서 내용 생성 (검색에 사용될 텍스트)
            document = f"{processed_metadata.get('title', '')} {processed_metadata.get('description', '')}"

            self.collections["job_postings"].add(
                documents=[document],
                metadatas=[processed_metadata],  # 처리된 메타데이터 사용
                ids=[doc_id],
                embeddings=[vector]
            )
            return True
        except Exception as e:
            print(f"Error upserting job posting: {str(e)}")
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
            where = {}
            if filter_conditions:
                if "keywords" in filter_conditions and filter_conditions["keywords"]:
                    where["keywords"] = {"$in": filter_conditions["keywords"]}
                if "location" in filter_conditions and filter_conditions["location"]:
                    where["location"] = {"$in": filter_conditions["location"]}

            results = self.collections["job_postings"].query(
                query_embeddings=[vector],
                n_results=limit,
                where=where if where else None
            )
            
            return [
                {
                    "metadata": metadata,
                    "score": 1 - distance  # Chroma returns distances, convert to similarity
                }
                for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
            ]
        except Exception as e:
            print(f"Error searching jobs: {str(e)}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보 조회"""
        try:
            return {
                "job_postings_count": self.collections["job_postings"].count(),
                "training_programs_count": self.collections["training_programs"].count()
            }
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {
                "job_postings_count": 0,
                "training_programs_count": 0
            }

    async def upsert_training_program(self, program: Dict[str, Any], vector: List[float]):
        """훈련 프로그램 저장 또는 업데이트"""
        try:
            metadata = program.copy()
            metadata.pop("vector", None)
            
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()

            doc_id = str(uuid.uuid4())
            document = f"{metadata.get('title', '')} {metadata.get('description', '')}"

            self.collections["training_programs"].add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id],
                embeddings=[vector]
            )
            return True
        except Exception as e:
            print(f"Error upserting training program: {str(e)}")
            return False

    async def search_similar_programs(
        self, 
        vector: List[float], 
        limit: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """유사한 훈련 프로그램 검색"""
        try:
            where = {}
            if filter_conditions and "keywords" in filter_conditions:
                where["keywords"] = {"$in": filter_conditions["keywords"]}

            results = self.collections["training_programs"].query(
                query_embeddings=[vector],
                n_results=limit,
                where=where if where else None
            )
            
            return [
                {
                    "metadata": metadata,
                    "score": 1 - distance
                }
                for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
            ]
        except Exception as e:
            print(f"Error searching programs: {str(e)}")
            return []

    async def initialize_data(self):
        """벡터 DB 초기 데이터 로드"""
        try:
            # 컬렉션 정보 확인
            job_count = self.collections["job_postings"].count()
            training_count = self.collections["training_programs"].count()
            
            # 데이터가 없는 경우에만 초기화
            if job_count == 0:
                print("Initializing job postings data...")
                from .job_posting_service import JobPostingService
                job_service = JobPostingService()
                await job_service.create_sample_job_postings()
                
            if training_count == 0:
                print("Initializing training programs data...")
                # TODO: 훈련 프로그램 샘플 데이터 추가
            
            return True
        except Exception as e:
            print(f"Error initializing data: {str(e)}")
            return False

    async def get_all_job_postings(self) -> List[Dict[str, Any]]:
        """모든 채용 공고 조회"""
        try:
            results = self.collections["job_postings"].get()
            jobs = []
            for i in range(len(results['ids'])):
                job = {
                    'id': results['ids'][i],
                    **results['metadatas'][i],
                    'content': results['documents'][i]
                }
                jobs.append(job)
            return jobs
        except Exception as e:
            print(f"Error getting all job postings: {str(e)}")
            return []

    async def get_all_job_postings_raw(self) -> Dict[str, Any]:
        """ChromaDB에서 모든 채용공고 원본 데이터 조회"""
        try:
            results = self.collections["job_postings"].get()
            return {
                "ids": results["ids"],
                "embeddings": results["embeddings"],
                "metadatas": results["metadatas"],
                "documents": results["documents"]
            }
        except Exception as e:
            print(f"Error getting raw job postings: {str(e)}")
            return {
                "ids": [],
                "embeddings": [],
                "metadatas": [],
                "documents": []
            }

    # 나머지 메서드들도 비슷한 방식으로 구현... 