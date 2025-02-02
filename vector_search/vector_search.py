import logging
from typing import List, Set
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set up the embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Function to load the existing vector store
def load_vectorstore(persist_dir: str = "./chroma_data") -> Chroma:
    logger.info("Loading existing vector store")
    return Chroma(
        embedding_function=embedding_model,
        collection_name="job_postings",
        persist_directory=persist_dir,
    )

# Function to perform vector search with filtering and duplicate removal
def search_documents(vectorstore: Chroma, query: str, top_k: int = 5) -> List[Document]:
    try:
        # 쿼리 전처리
        query_terms = query.lower().replace(',', ' ').split()
        
        # 벡터 검색으로 후보 결과 가져오기
        results = vectorstore.similarity_search(query, k=50)
        
        filtered_results = []
        seen_ids = set()
        
        for doc in results:
            metadata = doc.metadata
            job_id = metadata.get('채용공고ID')
            
            # 검색 대상 텍스트 준비
            search_text = ' '.join([
                str(metadata.get('채용제목', '')),
                str(metadata.get('상세정보', '')),
                str(metadata.get('회사명', '')),
                str(metadata.get('근무지역', '')),
                str(metadata.get('세부요건', '')),
                doc.page_content
            ]).lower()
            
            # 모든 검색어 조건을 만족하는지 확인
            matches_all_terms = all(term in search_text for term in query_terms)
            
            if matches_all_terms and job_id not in seen_ids:
                filtered_results.append(doc)
                seen_ids.add(job_id)

        # 결과가 없으면 부분 매칭으로 다시 시도
        if not filtered_results:
            for doc in results:
                metadata = doc.metadata
                job_id = metadata.get('채용공고ID')
                
                search_text = ' '.join([
                    str(metadata.get('채용제목', '')),
                    str(metadata.get('상세정보', '')),
                    str(metadata.get('회사명', '')),
                    str(metadata.get('근무지역', '')),
                    str(metadata.get('세부요건', '')),
                    doc.page_content
                ]).lower()
                
                # 검색어 중 하나라도 포함되면 결과에 추가
                if any(term in search_text for term in query_terms) and job_id not in seen_ids:
                    filtered_results.append(doc)
                    seen_ids.add(job_id)

        return filtered_results[:top_k]

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# Function to display search results
def display_results(search_results: List[Document]):
    if not search_results:
        print("\n⚠ 검색된 결과가 없습니다. 다른 검색어를 시도해 주세요.")
        return

    print("\n==== 📌 검색된 채용공고 ====")
    for i, doc in enumerate(search_results):
        metadata = doc.metadata
        print(f"\n🔹 **공고 {i+1}**")
        print(f"- **제목**: {metadata.get('채용제목', '정보 없음')}")
        print(f"- **회사명**: {metadata.get('회사명', '정보 없음')}")
        print(f"- **근무지**: {metadata.get('근무지역', '정보 없음')}")
        print(f"- **급여조건**: {metadata.get('급여조건', '정보 없음')}")
        print(f"- **채용공고 URL**: {metadata.get('채용공고URL', '정보 없음')}")
        print(f"- **상세정보**:\n{metadata.get('상세정보', '정보 없음')}")
        print(f"\n- **세부요건**:\n{metadata.get('세부요건', '정보 없음')}\n")

# Main function to execute the search
def main():
    vectorstore = load_vectorstore()
    query = input("\n🔍 검색어를 입력하세요: ")
    results = search_documents(vectorstore, query)
    display_results(results)

if __name__ == "__main__":
    main()
