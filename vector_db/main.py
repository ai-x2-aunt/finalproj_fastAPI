import json
from chroma_operations import ChromaOperations

def main():
    # ChromaOperations 인스턴스 생성
    ops = ChromaOperations()

    # JSON 파일에서 데이터 로드
    with open('C:/Users/201/dev/250122/250122_f/client/src/assets/json/jobs_with_details.json', 'r', encoding='utf-8') as f:
        jobs_data = json.load(f)

    # 데이터베이스에 채용공고 추가
    ops.insert_job_postings(jobs_data)

    # 미리 자주 사용될 만한 쿼리들에 대한 결과를 저장
    common_queries = [
        "파이썬 백엔드 개발자",
        "프론트엔드 개발자",
        "데이터 엔지니어",
        # ... 더 많은 쿼리 추가 가능
    ]

    # 결과를 저장할 딕셔너리
    precomputed_results = {}
    
    for query in common_queries:
        results = ops.search_similar_jobs(query)
        precomputed_results[query] = results

    # 결과를 JSON 파일로 저장
    with open('../250122_f/data/precomputed_job_results.json', 'w', encoding='utf-8') as f:
        json.dump(precomputed_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 