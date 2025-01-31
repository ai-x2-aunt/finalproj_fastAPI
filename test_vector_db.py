from vector_db.chroma_operations import ChromaOperations

def test_load_and_search():
    # ChromaDB 인스턴스 생성
    ops = ChromaOperations()
    
    # 다양한 검색 쿼리로 테스트
    test_queries = [
        "개발자 채용",
        "Python 개발자",
        "웹 개발자",
        "고령자 가능한 일자리",
        "서울 지역 사무직",
        "경비직 채용",
        "요양보호사 구인"
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*60}")
        print(f"검색어: {query}")
        print('='*60)
        
        results = ops.search_jobs(query)
        
        if not results:
            print("검색 결과가 없습니다.")
            continue
            
        for i, job in enumerate(results, 1):
            print(f"\n[검색결과 {i}]")
            print(f"채용제목: {job['title']}")
            print(f"회사명: {job['company']}")
            print(f"근무지역: {job['location']}")
            print(f"급여조건: {job['salary']}")
            if job.get('detail'):
                print(f"직무내용: {job['detail'][:150]}...")  # 직무내용 표시 길이 증가
            print('-' * 60)

if __name__ == "__main__":
    test_load_and_search()