import chromadb
import json
from typing import List, Dict, Any
from chromadb.utils import embedding_functions

class ChromaOperations:
    def __init__(self, json_path: str = "C:/Users/201/dev/250127/250127_f/client/src/assets/json/jobs_with_details.json"):
        # 임베딩 함수 설정
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # ChromaDB 클라이언트 생성
        self.client = chromadb.Client()
        
        # 기존 컬렉션이 있다면 삭제
        try:
            self.client.delete_collection("jobs")
        except:
            pass
            
        # 새 컬렉션 생성
        self.collection = self.client.create_collection(
            name="jobs",
            embedding_function=self.embedding_fn
        )
        
        # JSON 데이터 로드
        self.load_jobs_from_json(json_path)

    def load_jobs_from_json(self, json_path: str):
        """JSON 파일에서 채용정보를 로드하고 ChromaDB에 저장"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                jobs = data.get("채용공고목록", [])
                
            documents = []
            metadatas = []
            ids = []
            
            for job in jobs:
                job_id = str(job["공고번호"])
                
                # 상세정보에서 더 많은 정보 추출
                detail_info = job.get('상세정보', {})
                job_detail = detail_info.get('직무내용', '')
                
                # 세부요건에서 추가 정보 추출
                requirements = detail_info.get('세부요건', [])
                additional_info = []
                
                for req in requirements:
                    for key, value in req.items():
                        if isinstance(value, list):
                            additional_info.extend(value)
                
                # 검색용 텍스트 생성 - 구조화된 형태로
                content = f"""
                채용공고: {job['채용제목']}
                회사명: {job['회사명']}
                지역: {job['근무지역']}
                급여: {job['급여조건']}
                직무내용: {job_detail}
                추가정보: {' '.join(additional_info)}
                """
                
                metadata = {
                    "title": job["채용제목"],
                    "company": job["회사명"],
                    "location": job["근무지역"],
                    "salary": job["급여조건"],
                    "detail": job_detail,
                    "additional_info": ' '.join(additional_info)
                }
                
                documents.append(content)
                metadatas.append(metadata)
                ids.append(job_id)
            
            # 벌크로 데이터 추가
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return True
            
        except Exception as e:
            print(f"Error loading jobs: {e}")
            return False

    def search_jobs(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """채용정보 검색"""
        try:
            # 검색어 확장
            expanded_query = self._expand_query(query)
            
            results = self.collection.query(
                query_texts=[expanded_query],
                n_results=limit * 2,  # 더 많은 결과를 가져와서 필터링
                include=["metadatas", "distances"]
            )
            
            if results and results['metadatas']:
                filtered_results = []
                seen = set()
                
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    if distance > 1.5:  # 유사도 임계값 조정
                        continue
                        
                    key = metadata['title']
                    if key not in seen and self._is_relevant(query, metadata):
                        seen.add(key)
                        filtered_results.append(metadata)
                
                return filtered_results[:limit]
            return []
            
        except Exception as e:
            print(f"Error searching jobs: {e}")
            return []

    def _expand_query(self, query: str) -> str:
        """검색어 확장"""
        # 직무 관련 동의어 매핑
        job_synonyms = {
            "개발자": "개발 프로그래머 소프트웨어 엔지니어 IT",
            "운전": "운전기사 운전원 기사 드라이버",
            "경비": "경비원 보안 보안요원 시설관리",
            "요양": "요양보호사 간병 간병인 케어",
            "사무": "사무직 행정 총무 경리 회계",
        }
        
        expanded = query
        for key, synonyms in job_synonyms.items():
            if key in query:
                expanded = f"{expanded} {synonyms}"
        
        return expanded

    def _is_relevant(self, query: str, metadata: Dict) -> bool:
        """검색 결과의 관련성 체크"""
        # 제목, 직무내용, 추가정보에서 검색어 관련성 확인
        search_text = f"{metadata['title']} {metadata['detail']} {metadata.get('additional_info', '')}"
        search_text = search_text.lower()
        
        # 검색어의 주요 키워드 추출
        keywords = query.lower().split()
        
        # 키워드 매칭 점수 계산
        match_score = sum(1 for keyword in keywords if keyword in search_text)
        
        return match_score >= len(keywords) * 0.5  # 50% 이상 매칭되어야 함 