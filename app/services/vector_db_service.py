from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
import os
from datetime import datetime
import json
import uuid
from sentence_transformers import SentenceTransformer

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
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # SentenceTransformer 모델 초기화
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """필요한 컬렉션이 존재하는지 확인하고 없으면 생성"""
        collections = ["job_postings", "training_programs"]
        
        for collection_name in collections:
            try:
                # 컬렉션 존재 여부 확인
                collection = self.client.get_collection(collection_name)
                print(f"Collection {collection_name} exists")
            except:
                print(f"Creating new collection: {collection_name}")
                # 컬렉션이 없는 경우에만 새로 생성
                self.client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Collection for {collection_name}"}
                )

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            collection = self.client.get_collection(collection_name)
            # 컬렉션의 전체 데이터 가져오기
            result = collection.get()
            
            # 샘플 데이터 출력 (처음 3개)
            print(f"\n=== {collection_name} Sample Data ===")
            for i in range(min(3, len(result['metadatas']))):
                print(f"\nDocument {i+1}:")
                print("Metadata:", result['metadatas'][i])
                print("Document:", result['documents'][i][:200] + "..." if len(result['documents'][i]) > 200 else result['documents'][i])
            
            return {
                f"{collection_name}_count": len(result['ids']),
                "sample_data": result['metadatas'][:3] if result['metadatas'] else []
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {f"{collection_name}_count": 0}

    async def upsert_job_posting(self, job_posting: Dict[str, Any], vector: List[float]):
        """채용 공고 저장 또는 업데이트"""
        try:
            collection = self.client.get_collection("job_postings")
            
            # 메타데이터 준비
            metadata = job_posting.copy()
            metadata.pop("vector", None)
            
            # datetime 객체를 문자열로 변환
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
            
            # UUID 생성
            doc_id = str(uuid.uuid4())
            
            # ChromaDB에 데이터 추가
            collection.add(
                ids=[doc_id],
                embeddings=[vector],
                metadatas=[metadata],
                documents=[json.dumps(metadata)]
            )
            return True
        except Exception as e:
            print(f"Error upserting job posting: {str(e)}")
            return False

    async def upsert_training_program(self, program: Dict[str, Any], vector: List[float]):
        """훈련 프로그램 저장 또는 업데이트"""
        try:
            collection = self.client.get_collection("training_programs")
            
            metadata = program.copy()
            metadata.pop("vector", None)
            
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
            
            doc_id = str(hash(f"{program['institution']}_{program['title']}_{datetime.now().timestamp()}"))
            
            collection.add(
                ids=[doc_id],
                embeddings=[vector],
                metadatas=[metadata],
                documents=[json.dumps(metadata)]
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
            collection = self.client.get_collection("job_postings")
            
            # 필터 조건 구성
            where = {}
            if filter_conditions:
                if "keywords" in filter_conditions and filter_conditions["keywords"]:
                    where["keywords"] = {"$in": filter_conditions["keywords"]}
                if "location" in filter_conditions and filter_conditions["location"]:
                    where["location"] = {"$in": filter_conditions["location"]}
            
            # 검색 실행
            results = collection.query(
                query_embeddings=[vector],
                n_results=limit,
                where=where if where else None
            )
            
            # 결과 포맷팅
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "metadata": json.loads(results['documents'][0][i]),
                    "score": float(results['distances'][0][i])
                })
            
            return formatted_results
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
            collection = self.client.get_collection("training_programs")
            
            where = {}
            if filter_conditions:
                if "keywords" in filter_conditions and filter_conditions["keywords"]:
                    where["keywords"] = {"$in": filter_conditions["keywords"]}
            
            results = collection.query(
                query_embeddings=[vector],
                n_results=limit,
                where=where if where else None
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "metadata": json.loads(results['documents'][0][i]),
                    "score": float(results['distances'][0][i])
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching training programs: {str(e)}")
            return []

    async def delete_job_posting(self, job_id: str) -> bool:
        """채용 공고 삭제"""
        try:
            collection = self.client.get_collection("job_postings")
            collection.delete(ids=[job_id])
            return True
        except Exception as e:
            print(f"Error deleting job posting: {str(e)}")
            return False

    async def delete_training_program(self, program_id: str) -> bool:
        """훈련 프로그램 삭제"""
        try:
            collection = self.client.get_collection("training_programs")
            collection.delete(ids=[program_id])
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

    async def add_job_posting(self, vector, job_data):
        """채용 공고 추가"""
        try:
            # 메타데이터를 단순화
            metadata = {
                "title": str(job_data.get("title", "")),
                "company_name": str(job_data.get("company_name", "")),
                "location": str(job_data.get("location", "")),
                "job_type": str(job_data.get("job_type", "")),
                "experience_level": str(job_data.get("experience_level", "")),
                "education_level": str(job_data.get("education_level", "")),
                "salary": str(job_data.get("salary", "")),
                "description": str(job_data.get("description", ""))
            }
            
            # 벡터 DB에 추가
            collection = self.client.get_collection("job_postings")
            collection.add(
                embeddings=[vector],
                metadatas=[metadata],
                documents=[str(job_data)],  # 전체 데이터는 documents에 문자열로 저장
                ids=[str(uuid.uuid4())]  # 고유 ID 생성
            )
            return True
        except Exception as e:
            print(f"Error adding job posting to vector DB: {str(e)}")
            return False

    async def add_training_program(self, vector: List[float], payload: Dict[str, Any]) -> None:
        """훈련과정 데이터를 벡터 DB에 저장"""
        try:
            collection = self.client.get_collection("training_programs")
            doc_id = uuid.uuid4().hex
            
            if "keywords" not in payload:
                keywords = []
                if "title" in payload:
                    keywords.extend(payload["title"].split())
                if "description" in payload:
                    keywords.extend(payload["description"].split())
                payload["keywords"] = list(set(keywords))
            
            collection.add(
                ids=[doc_id],
                embeddings=[vector],
                metadatas=[payload],
                documents=[json.dumps(payload)]
            )
            print(f"Successfully added training program with ID: {doc_id}")
        except Exception as e:
            print(f"Error adding training program to vector DB: {str(e)}")
            raise e

    async def initialize_data(self):
        """벡터 DB 초기 데이터 로드"""
        try:
            # 컬렉션 정보 확인
            job_info = await self.get_collection_info("job_postings")
            training_info = await self.get_collection_info("training_programs")
            
            try:
                # 기존 컬렉션 삭제
                print("Deleting existing collections...")
                self.client.delete_collection("job_postings")
                print("Creating new collections...")
                self.client.create_collection("job_postings")
            except Exception as e:
                print(f"Error resetting collections: {str(e)}")
            
            print("Initializing job postings data...")
            
            # jobs_with_details.json 파일 로드
            json_path = r"C:\Users\201\dev\250127\250127_f\client\src\assets\json\jobs_with_details.json"
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 구조 확인을 위한 출력
                print("Data keys:", list(data.keys()))
                jobs_data = data.get("채용공고목록", [])
                if jobs_data:
                    print("First job keys:", list(jobs_data[0].keys()))
                    print("First job data:", json.dumps(jobs_data[0], indent=2, ensure_ascii=False))
                
                print(f"Loaded {len(jobs_data)} jobs from JSON file")
                
                # 중복 제거 (채용제목과 회사명으로 중복 체크)
                unique_jobs = {}
                for job in jobs_data:
                    if isinstance(job, dict):
                        key = (job.get("채용제목", ""), job.get("회사명", ""))
                        if key not in unique_jobs:
                            unique_jobs[key] = job
                
                print(f"Found {len(unique_jobs)} unique jobs after deduplication")
                
                # 각 unique job을 벡터 DB에 추가
                for job in unique_jobs.values():
                    try:
                        detail_info = job.get("상세정보", {})
                        
                        # 메타데이터 형식에 맞게 변환
                        job_data = {
                            "title": job.get("채용제목", ""),
                            "company_name": job.get("회사명", ""),
                            "location": job.get("근무지역", ""),
                            "salary": job.get("급여조건", ""),
                            "description": str(detail_info) if detail_info else "",
                            "job_type": detail_info.get("모집직종", [""])[0] if isinstance(detail_info.get("모집직종"), list) else "",
                            "experience_level": detail_info.get("경력조건", [""])[0] if isinstance(detail_info.get("경력조건"), list) else "",
                            "education_level": detail_info.get("학력", [""])[0] if isinstance(detail_info.get("학력"), list) else ""
                        }

                        # 임베딩 생성을 위한 텍스트 준비
                        embedding_text = f"{job_data['title']} {job_data['description']} {job_data['job_type']} {job_data['experience_level']}"
                        # 임베딩 벡터 생성
                        vector = self.model.encode(embedding_text).tolist()
                        
                        await self.add_job_posting(vector, job_data)
                        print(f"Added job: {job_data['title']}")
                    except Exception as e:
                        print(f"Error adding individual job: {str(e)}")
                        continue
                
            except FileNotFoundError:
                print(f"JSON file not found at: {json_path}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {str(e)}")
            
            return True
        except Exception as e:
            print(f"Error initializing data: {str(e)}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보 조회"""
        try:
            job_collection = self.client.get_collection("job_postings")
            training_collection = self.client.get_collection("training_programs")
            
            return {
                "job_postings_count": job_collection.count(),
                "training_programs_count": training_collection.count()
            }
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {
                "job_postings_count": 0,
                "training_programs_count": 0
            } 