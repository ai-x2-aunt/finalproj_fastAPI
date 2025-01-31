import json
from pathlib import Path
import sys

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from chroma_operations import ChromaOperations

def load_jobs_to_chroma():
    # JSON 파일 경로
    json_path = Path(__file__).parent.parent.parent / "250122_f/client/src/assets/json/jobs_with_details.json"
    
    try:
        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            jobs_data = json.load(f)
        
        # ChromaDB 연결
        chroma_ops = ChromaOperations()
        
        # 채용공고 데이터 삽입
        chroma_ops.insert_job_postings(jobs_data)
        print("채용공고 데이터가 성공적으로 ChromaDB에 저장되었습니다.")
        
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {e}")

if __name__ == "__main__":
    load_jobs_to_chroma() 