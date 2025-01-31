from chromadb import Client, Settings
import json

def check_chroma_db():
    # ChromaDB 클라이언트 생성
    client = Client(Settings(
        persist_directory="db",  # 현재 사용 중인 db 디렉토리
        is_persistent=True
    ))

    try:
        # 1. 모든 컬렉션 목록 확인
        collections = client.list_collections()
        print("\n=== 컬렉션 목록 ===")
        for collection in collections:
            print(f"컬렉션 이름: {collection.name}")
            print(f"컬렉션 크기: {collection.count()}")

        # 2. job_postings 컬렉션 상세 확인
        job_collection = client.get_collection("job_postings")
        results = job_collection.get()

        print("\n=== 채용공고 데이터 ===")
        for i in range(len(results['ids'])):
            print(f"\n--- 채용공고 {i+1} ---")
            print(f"ID: {results['ids'][i]}")
            print(f"메타데이터: {json.dumps(results['metadatas'][i], indent=2, ensure_ascii=False)}")
            print(f"문서내용: {results['documents'][i]}")
            # print(f"임베딩: {results['embeddings'][i][:5]}...")  # 벡터값 앞부분만 출력

    except Exception as e:
        print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    check_chroma_db() 