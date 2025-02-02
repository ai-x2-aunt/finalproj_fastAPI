import json
import re
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv
from chromadb.config import Settings

# 환경 변수 로드
load_dotenv()

# 1️⃣ 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 2️⃣ 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 3️⃣ 텍스트 전처리
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'<[^>]+>', '', str(text)).replace("\n", " ").strip()

# 4️⃣ JSON 데이터 로드
def load_data(json_file: str = "jobs.json") -> dict:
    logger.info(f"Loading data from {json_file}")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data.get('채용공고목록', []))} job postings")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return {"채용공고목록": []}

# 5️⃣ 채용 공고를 Document 객체로 변환
def prepare_documents(data: dict) -> List[Document]:
    logger.info("Preparing documents from data")
    documents = []

    for item in data.get("채용공고목록", []):
        # 기본 정보 추출
        공고번호 = item.get("공고번호", "no_id")
        채용제목 = clean_text(item.get("채용제목", ""))
        회사명 = clean_text(item.get("회사명", ""))
        근무지역 = clean_text(item.get("근무지역", ""))
        급여조건 = clean_text(item.get("급여조건", ""))
        채용공고ID = item.get("채용공고ID", "정보 없음")
        채용공고URL = item.get("채용공고URL", "정보 없음")
        
        # 상세정보 추출 및 전처리
        상세정보 = item.get("상세정보", {})
        직무내용 = ""
        세부요건_텍스트 = ""
        
        if isinstance(상세정보, dict):
            직무내용 = clean_text(상세정보.get("직무내용", ""))
            
            # 세부요건 처리
            세부요건_리스트 = 상세정보.get("세부요건", [])
            중요_필드 = {
                "모집직종": "모집직종",
                "경력조건": "경력조건",
                "학력": "학력",
                "고용형태": "고용형태",
                "임금조건": "임금조건",
                "근무예정지": "근무예정지",
                "근무시간": "근무시간",
                "근무형태": "근무형태",
                "접수마감일": "접수마감일",
                "전형방법": "전형방법"
            }
            
            for 요건 in 세부요건_리스트:
                for key, value in 요건.items():
                    if key in 중요_필드:
                        if isinstance(value, list):
                            세부요건_텍스트 += f"{중요_필드[key]}: {' '.join(value)}\n"
                        else:
                            세부요건_텍스트 += f"{중요_필드[key]}: {value}\n"
        else:
            직무내용 = clean_text(str(상세정보))

        # 데이터 검증
        if not 직무내용:
            logger.warning(f"직무내용이 비어있음: {채용제목}")
            직무내용 = "상세정보 없음"
        
        if not 채용공고URL:
            logger.warning(f"URL이 비어있음: {채용제목}")

        metadata = {
            "공고번호": 공고번호,
            "채용제목": 채용제목,
            "회사명": 회사명,
            "근무지역": 근무지역,
            "급여조건": 급여조건,
            "채용공고ID": 채용공고ID,
            "채용공고URL": 채용공고URL,
            "상세정보": 직무내용,
            "세부요건": 세부요건_텍스트
        }

        # 검색용 통합 텍스트
        combined_content = f"{채용제목} {회사명} {근무지역} {급여조건} {직무내용} {세부요건_텍스트}"
        
        doc = Document(page_content=combined_content, metadata=metadata)
        documents.append(doc)
        
        # 데이터 로깅
        logger.info(f"\n=== 문서 생성 ===")
        logger.info(f"제목: {채용제목}")
        logger.info(f"회사: {회사명}")
        logger.info(f"URL: {채용공고URL}")
        logger.info(f"상세정보 길이: {len(직무내용)}")
        logger.info(f"세부요건 길이: {len(세부요건_텍스트)}")
        logger.info("-" * 50)

    logger.info(f"총 {len(documents)}개의 문서가 생성되었습니다.")
    return documents

# 9️⃣ 벡터 DB 저장 함수
def build_vectorstore(documents: List[Document], persist_dir: str = "./chroma_data") -> Chroma:
    logger.info("Building vector store")
    
    try:
        # Chroma 설정
        client_settings = Settings(
            anonymized_telemetry=False
        )
        
        # 벡터스토어 생성
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name="job_postings",
            persist_directory=persist_dir,
            client_settings=client_settings
        )
        
        # 저장 확인
        collection = vectorstore._collection
        total_docs = collection.count()
        logger.info(f"Successfully stored {total_docs} documents")
        
        if total_docs == 0:
            logger.error("No documents were stored!")
            
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        raise

# 🔟 실행 함수
def main():
    try:
        # 데이터 로드
        data = load_data()
        if not data.get('채용공고목록'):
            logger.error("No job postings found in data!")
            return
            
        # 문서 준비
        docs = prepare_documents(data)
        if not docs:
            logger.error("No documents were prepared!")
            return
            
        # 벡터스토어 생성
        vectorstore = build_vectorstore(docs)
        
        # 저장 확인
        collection = vectorstore._collection
        total_docs = collection.count()
        logger.info(f"\n=== 최종 확인 ===")
        logger.info(f"총 저장된 문서 수: {total_docs}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
