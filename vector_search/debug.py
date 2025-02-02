import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def debug_search():
    # 임베딩 모델 설정
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 벡터스토어 로드
    vectorstore = Chroma(
        persist_directory="./chroma_data",
        embedding_function=embedding_model,
        collection_name="job_postings"
    )
    
    # 컬렉션 가져오기
    collection = vectorstore._collection
    
    # 전체 문서 수 확인
    total_docs = collection.count()
    logger.info(f"\n총 문서 수: {total_docs}")
    
    # 모든 문서의 메타데이터 확인
    all_docs = collection.get(
        include=["metadatas", "documents"],
        limit=total_docs
    )
    
    # "경비" 키워드 검색
    keyword = "경비"
    found = 0
    
    logger.info(f"\n=== '{keyword}' 키워드 포함 문서 검색 ===")
    for i, (meta, doc) in enumerate(zip(all_docs['metadatas'], all_docs['documents'])):
        title = meta.get('채용제목', '').lower()
        description = meta.get('상세정보', '').lower()
        content = doc.lower()
        
        if keyword in title or keyword in description or keyword in content:
            found += 1
            logger.info(f"\n=== 문서 {found} ===")
            logger.info(f"제목: {meta.get('채용제목')}")
            logger.info(f"회사: {meta.get('회사명')}")
            logger.info(f"상세정보: {meta.get('상세정보')[:100]}...")
            logger.info(f"제목에 포함: {keyword in title}")
            logger.info(f"상세정보에 포함: {keyword in description}")
            logger.info(f"내용에 포함: {keyword in content}")
            logger.info("-" * 80)
    
    logger.info(f"\n총 {found}개의 '{keyword}' 관련 문서를 찾았습니다.")

if __name__ == "__main__":
    debug_search() 