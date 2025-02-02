from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 벡터스토어 로드
def load_vectorstore(persist_dir: str = "./chroma_data") -> Chroma:
    return Chroma(
        embedding_function=embedding_model,
        collection_name="job_postings",
        persist_directory=persist_dir,
    )

# 검색 함수
def search_jobs(query: str, vectorstore: Chroma, top_k: int = 5) -> List[Document]:
    try:
        # 쿼리 전처리
        query_terms = query.lower().replace(',', ' ').split()
        logger.info(f"Query terms: {query_terms}")
        
        # 벡터 검색으로 후보 결과 가져오기
        results = vectorstore.similarity_search(query, k=50)
        logger.info(f"Initial results count: {len(results)}")
        
        filtered_results = []
        seen_ids = set()
        
        for doc in results:
            metadata = doc.metadata
            job_id = metadata.get('채용공고ID')
            
            # 검색 대상 필드 추출
            job_title = metadata.get('채용제목', '').lower()
            job_desc = metadata.get('상세정보', '').lower()
            job_type = metadata.get('모집직종', '').lower()
            
            # 각 필드별로 검색어 포함 여부 확인
            for term in query_terms:
                # 정확한 키워드 매칭 확인
                found = False
                
                # 제목에서 검색
                if term in job_title:
                    found = True
                    logger.info(f"Found '{term}' in title: {job_title}")
                # 모집직종에서 검색
                elif term in job_type:
                    found = True
                    logger.info(f"Found '{term}' in job type: {job_type}")
                # 상세정보에서 검색 (단어 단위로 매칭)
                elif f" {term} " in f" {job_desc} ":
                    found = True
                    logger.info(f"Found '{term}' in description")
                
                # 검색어를 찾지 못한 경우 이 문서는 제외
                if not found:
                    break
            else:  # 모든 검색어가 매칭된 경우
                if job_id not in seen_ids:
                    filtered_results.append(doc)
                    seen_ids.add(job_id)
                    logger.info(f"Added matching job: {metadata.get('채용제목')}")

        logger.info(f"Found {len(filtered_results)} exact matches")
        return filtered_results[:top_k]

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# 채용공고 포매팅 함수
def format_job_posting(doc: Document) -> str:
    metadata = doc.metadata
    return (
        f"🔹 채용공고\n"
        f"- 제목: {metadata.get('채용제목', '정보 없음')}\n"
        f"- 회사명: {metadata.get('회사명', '정보 없음')}\n"
        f"- 근무지: {metadata.get('근무지역', '정보 없음')}\n"
        f"- 급여조건: {metadata.get('급여조건', '정보 없음')}\n"
        f"- 채용공고 URL: {metadata.get('채용공고URL', '정보 없음')}\n"
        f"\n[세부요건]\n{metadata.get('세부요건', '정보 없음')}"
    )

# 요청 모델
class ChatRequest(BaseModel):
    user_message: str
    user_profile: Optional[Dict] = None
    session_id: Optional[str] = None

# 응답 모델
class ChatResponse(BaseModel):
    responses: List[str]
    user_profile: Optional[Dict] = None

# 벡터스토어 인스턴스 생성
vectorstore = load_vectorstore()

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.user_message
        user_profile = request.user_profile
        logger.info(f"Received message: {user_message}")
        logger.info(f"Received profile: {user_profile}")
        
        # 사용자가 입력한 메시지로 검색
        if user_message:
            search_query = user_message
        # 프로필 정보로 검색
        elif user_profile:
            search_terms = []
            if user_profile.get('jobType'):
                search_terms.append(user_profile['jobType'])
            if user_profile.get('location'):
                search_terms.append(user_profile['location'])
            search_query = ' '.join(search_terms)
        else:
            return ChatResponse(
                responses=["검색어를 입력해주세요."],
                user_profile=user_profile
            )
            
        logger.info(f"Searching with query: '{search_query}'")
        
        # 검색 실행
        results = search_jobs(search_query, vectorstore)
        
        if not results:
            return ChatResponse(
                responses=["검색 조건에 맞는 채용공고를 찾지 못했습니다. 다른 검색어로 시도해보시겠어요?"],
                user_profile=user_profile
            )
        
        # 검색 결과 포매팅
        formatted_results = [format_job_posting(doc) for doc in results]
        responses = [
            f"검색하신 조건에 맞는 채용공고를 {len(results)}건 찾았습니다:",
            *formatted_results
        ]
        
        return ChatResponse(responses=responses, user_profile=user_profile)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 