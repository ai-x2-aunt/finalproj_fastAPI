from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.v1 import data_management_router
from .api.endpoints import chat  # 채팅 라우터 import
# from .services.scheduler_service import SchedulerService  # 현재 미사용
from .services.vector_db_service import VectorDBService
# from .services.code_service import CodeService  # 현재 미사용
from pydantic import BaseModel
from vector_db.chroma_operations import ChromaOperations

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 3000 포트로 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(data_management_router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])  # 채팅 라우터 등록 수정

# 서비스 초기화
# scheduler = SchedulerService()
vector_db = VectorDBService()
# code_service = CodeService()  # 현재 미사용

class SearchQuery(BaseModel):
    query: str

@app.post("/search")
async def search_jobs(query: SearchQuery):
    try:
        # LLMService를 통해 쿼리 텍스트를 벡터로 변환
        from .services.llm_service import LLMService
        llm_service = LLMService(model_name="llama2")
        query_vector = await llm_service.embeddings.aembed_query(query.query)
        
        # 벡터 DB에서 유사한 채용 공고 검색
        results = await vector_db.search_similar_jobs(
            vector=query_vector,
            limit=5  # 상위 5개 결과 반환
        )
        
        return {"results": results}
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return {"error": str(e), "message": "Search failed"}

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    # 스케줄러 시작
    # scheduler.start()
    
    # 스케줄러를 통해 첫 데이터 수집 시작
    # await scheduler.collect_codes()
    
    # 벡터 DB 초기화
    await vector_db.initialize_data()
    
    # 벡터 DB 상태 확인
    stats = await vector_db.get_stats()
    print(f"Vector DB Stats: {stats}")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    # scheduler.scheduler.shutdown()

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    stats = await vector_db.get_stats()
    # codes = code_service.get_cached_codes()  # 현재 미사용
    return {
        "status": "healthy",
        "vector_db_stats": stats,
        # "code_stats": {k: len(v) for k, v in codes.items()}  # 현재 미사용
    }

@app.get("/api/v1/vector-db/sample")
async def get_vector_db_sample():
    collection = vector_db.client.get_collection("job_postings")
    result = collection.get(limit=5)  # 처음 5개 문서만 가져오기
    return {
        "count": len(result['ids']),
        "sample_data": [
            {
                "metadata": meta,
                "document": doc[:200] + "..." if len(doc) > 200 else doc
            }
            for meta, doc in zip(result['metadatas'], result['documents'])
        ]
    }

@app.get("/api/v1/vector-db/jobs")
async def get_jobs(skip: int = 0, limit: int = 10):
    """채용공고 목록을 페이지네이션하여 조회"""
    collection = vector_db.client.get_collection("job_postings")
    result = collection.get(limit=limit, offset=skip, include=['embeddings'])  # embeddings 포함
    return {
        "total": collection.count(),
        "jobs": [
            {
                "id": id,
                "metadata": meta,
                "document": doc[:200] + "..." if len(doc) > 200 else doc,
                "vector": emb  # 임베딩 벡터 추가
            }
            for id, meta, doc, emb in zip(
                result['ids'], 
                result['metadatas'], 
                result['documents'],
                result['embeddings']
            )
        ]
    }

@app.get("/api/v1/vector-db/jobs/{job_id}")
async def get_job_detail(job_id: str):
    """특정 채용공고의 상세 정보 조회"""
    collection = vector_db.client.get_collection("job_postings")
    result = collection.get(ids=[job_id])
    if not result['ids']:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": result['ids'][0],
        "metadata": result['metadatas'][0],
        "document": result['documents'][0]
    }

@app.get("/api/v1/vector-db/search")
async def search_jobs(query: str, limit: int = 10):
    """채용공고 검색"""
    collection = vector_db.client.get_collection("job_postings")
    results = collection.query(
        query_texts=[query],
        n_results=limit
    )
    return {
        "results": [
            {
                "id": id,
                "metadata": meta,
                "document": doc[:200] + "..." if len(doc) > 200 else doc,
                "distance": distance
            }
            for id, meta, doc, distance in zip(
                results['ids'][0], 
                results['metadatas'][0], 
                results['documents'][0],
                results['distances'][0]
            )
        ]
    }

@app.get("/api/v1/vector-db/test-similarity")
async def test_similarity(query1: str, query2: str):
    """두 쿼리의 벡터 유사도 테스트"""
    vector1 = vector_db.model.encode(query1).tolist()
    vector2 = vector_db.model.encode(query2).tolist()
    
    # 코사인 유사도 계산
    import numpy as np
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    return {
        "query1": query1,
        "query2": query2,
        "similarity": float(similarity)
    }

@app.get("/api/v1/vector-db/test-search")
async def test_search(query: str, limit: int = 5):
    """임베딩 기반 유사도 검색 테스트"""
    # 쿼리 텍스트를 벡터로 변환
    query_vector = vector_db.model.encode(query).tolist()
    
    # 벡터 DB에서 검색
    collection = vector_db.client.get_collection("job_postings")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=limit
    )
    
    return {
        "query": query,
        "results": [
            {
                "id": id,
                "metadata": meta,
                "distance": distance
            }
            for id, meta, distance in zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
    }

@app.get("/api/v1/vector-db/semantic-search")
async def semantic_search(query: str, limit: int = 5):
    """의미 기반 검색 테스트"""
    try:
        # 쿼리 텍스트를 벡터로 변환
        query_vector = vector_db.model.encode(query).tolist()
        
        # 벡터 DB에서 유사도 검색
        collection = vector_db.client.get_collection("job_postings")
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=['metadatas', 'documents', 'distances']  # 필요한 필드 모두 포함
        )
        
        # 결과가 없는 경우 처리
        if not results['ids'][0]:
            return {
                "query": query,
                "results": []
            }
        
        return {
            "query": query,
            "results": [
                {
                    "id": id,
                    "metadata": meta,
                    "similarity_score": float(1 - distance),  # 거리를 유사도로 변환
                    "document": doc[:200] + "..." if len(doc) > 200 else doc
                }
                for id, meta, doc, distance in zip(
                    results['ids'][0],
                    results['metadatas'][0],
                    results['documents'][0],
                    results['distances'][0]
                )
            ]
        }
    except Exception as e:
        print(f"Error in semantic search: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "results": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 