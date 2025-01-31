from typing import List, Dict, Any
import httpx
import asyncio
import xmltodict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from .vector_db_service import VectorDBService
import requests
from chromadb import Client, Settings

class DataCollectionService:
    def __init__(self):
        # 공통코드 API 설정
        self.work24_common_code_api_key = settings.WORK24_COMMON_CODE_API_KEY
        self.work24_api_common_base_url = settings.WORK24_API_COMMON_BASE_URL
        self.work24_api_common_training_url = settings.WORK24_API_COMMON_TRAINING_URL
        
        # Vector DB 초기화
        self.vector_db = VectorDBService()
        
        # sentence-transformers 모델 로드
        self.model = SentenceTransformer('jhgan/ko-sbert-nli')
        self.model_name = "nomic-embed-text"  # llama2에서 변경

    async def collect_job_postings(self):
        """고용24 API에서 채용공고 데이터 수집"""
        try:
            # 채용 정보 조회
            url = f"{self.work24_api_common_base_url}{settings.REGION_CODE_ENDPOINT}"
            params = {
                "authKey": self.work24_common_code_api_key,
                "returnType": "XML",
                "target": "CMCD",
                "dtlGb": "1"  # 지역코드
            }
            
            print(f"API URL: {url}")
            print(f"API Parameters: {params}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                print(f"API Response Status: {response.status_code}")
                print(f"API Response: {response.text}")
                
                if response.status_code == 200:
                    data = xmltodict.parse(response.text)
                    print(f"Parsed Data: {data}")
                    
                    # 데이터 벡터화 및 저장
                    await self._process_job_postings(data)
                    
                    return {"status": "success", "message": "채용공고 데이터 수집 완료"}
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    return {"status": "error", "message": f"채용정보 API 호출 실패: {response.status_code}"}
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def collect_training_programs(self):
        """고용24 API에서 훈련과정 데이터 수집"""
        try:
            url = f"{self.work24_api_common_training_url}{settings.TRAINING_COMMON_CODE_ENDPOINT}"
            params = {
                "authKey": self.work24_common_code_api_key,
                "returnType": "XML",
                "outType": "1",      # 출력형태 (1: 리스트)
                "srchType": "00"     # 공통코드 구분 (00: 훈련지역 대분류 코드)
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                print(f"Training API Response: {response.text}")
                
                if response.status_code == 200:
                    data = xmltodict.parse(response.text)
                    print(f"Parsed Training Data: {data}")
                    
                    # 데이터 벡터화 및 저장
                    await self._process_training_programs(data)
                    
                    return {"status": "success", "message": "훈련과정 데이터 수집 완료"}
                else:
                    print(f"Training API Error: {response.status_code} - {response.text}")
                    return {"status": "error", "message": f"훈련과정 API 호출 실패: {response.status_code}"}
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def collect_common_codes(self, code_type: str):
        """공통코드 데이터 수집"""
        try:
            # 코드 타입에 따른 엔드포인트와 파라미터 설정
            if code_type == "training":
                url = f"{self.work24_api_common_training_url}{settings.TRAINING_COMMON_CODE_ENDPOINT}"
                params = {
                    "authKey": self.work24_common_code_api_key,
                    "returnType": "XML",
                    "outType": "1",
                    "srchType": "00"  # 훈련지역 대분류 코드
                }
            else:
                # 채용 관련 공통코드
                endpoints = {
                    "region": settings.REGION_CODE_ENDPOINT,
                    "job": settings.JOB_CODE_ENDPOINT,
                    "certificate": settings.CERTIFICATE_CODE_ENDPOINT,
                    "industrial": settings.INDUSTRIAL_COMPLEX_ENDPOINT,
                    "subway": settings.SUBWAY_CODE_ENDPOINT,
                    "major": settings.MAJOR_CODE_ENDPOINT,
                    "language": settings.LANGUAGE_CODE_ENDPOINT,
                    "department": settings.DEPARTMENT_CODE_ENDPOINT,
                    "small_giant": settings.SMALL_GIANT_CODE_ENDPOINT
                }
                
                if code_type not in endpoints:
                    return {"status": "error", "message": f"잘못된 코드 타입: {code_type}"}
                
                url = f"{self.work24_api_common_base_url}{endpoints[code_type]}"
                params = {
                    "authKey": self.work24_common_code_api_key,
                    "returnType": "XML",
                    "target": "CMCD",
                    "dtlGb": self._get_detail_code(code_type)
                }

            print(f"API URL: {url}")
            print(f"Parameters: {params}")

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                print(f"Response Status: {response.status_code}")
                print(f"Response: {response.text}")

                if response.status_code == 200:
                    data = xmltodict.parse(response.text)
                    return {"status": "success", "data": data}
                else:
                    return {"status": "error", "message": f"API 호출 실패: {response.status_code}"}

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_detail_code(self, code_type: str) -> str:
        """코드 타입에 따른 상세 구분 코드 반환"""
        detail_codes = {
            "region": "1",
            "job": "2",
            "certificate": "3",
            "industrial": "4",
            "subway": "5",
            "major": "6",
            "language": "7",
            "department": "8",
            "small_giant": "9"
        }
        return detail_codes.get(code_type, "1")

    async def _process_job_postings(self, data: Dict[str, Any]):
        """채용공고 데이터 처리 및 벡터 DB 저장"""
        try:
            # XML 응답 구조에 맞게 데이터 추출
            regions = data.get("cmcdRegion", {}).get("oneDepth", [])
            if not isinstance(regions, list):
                regions = [regions]
            
            for region in regions:
                # 텍스트 데이터 결합
                text_to_vectorize = f"""
                지역코드: {region.get('regionCd', '')}
                지역명: {region.get('regionNm', '')}
                """
                
                # 텍스트 벡터화
                vector = self.model.encode(text_to_vectorize)
                
                # 벡터 DB에 저장
                await self.vector_db.add_job_posting(
                    vector=vector.tolist(),
                    payload={
                        "region_code": region.get("regionCd"),
                        "region_name": region.get("regionNm"),
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # 하위 지역 처리
                two_depth = region.get("twoDepth", [])
                if not isinstance(two_depth, list):
                    two_depth = [two_depth]
                
                for sub_region in two_depth:
                    text_to_vectorize = f"""
                    지역코드: {sub_region.get('regionCd', '')}
                    지역명: {sub_region.get('regionNm', '')}
                    상위지역코드: {sub_region.get('superCd', '')}
                    """
                    
                    vector = self.model.encode(text_to_vectorize)
                    
                    await self.vector_db.add_job_posting(
                        vector=vector.tolist(),
                        payload={
                            "region_code": sub_region.get("regionCd"),
                            "region_name": sub_region.get("regionNm"),
                            "parent_code": sub_region.get("superCd"),
                            "created_at": datetime.now().isoformat()
                        }
                    )
        except Exception as e:
            print(f"Error processing job postings: {str(e)}")
            raise e

    async def _process_training_programs(self, data: Dict[str, Any]):
        """훈련과정 데이터 처리 및 벡터 DB 저장"""
        try:
            # XML 응답 구조에 맞게 데이터 추출
            items = data.get("HRDNet", {}).get("srchList", {}).get("scn_list", [])
            if not isinstance(items, list):
                items = [items]
            
            for item in items:
                # 텍스트 데이터 결합
                text_to_vectorize = f"""
                지역코드: {item.get('rsltCode', '')}
                지역명: {item.get('rsltName', '')}
                사용여부: {item.get('useYn', '')}
                """
                
                # 텍스트 벡터화
                vector = self.model.encode(text_to_vectorize)
                
                # 벡터 DB에 저장
                await self.vector_db.add_training_program(
                    vector=vector.tolist(),
                    payload={
                        "region_code": item.get("rsltCode"),
                        "region_name": item.get("rsltName"),
                        "use_yn": item.get("useYn"),
                        "created_at": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            print(f"Error processing training programs: {str(e)}")
            raise e

    async def get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error raised by inference endpoint: {str(e)}")
            return []

client = Client(Settings(
    persist_directory="db",
    is_persistent=True
))

# 컬렉션 가져오기
collection = client.get_collection("job_postings")

# 모든 데이터 조회
results = collection.get()

# 결과 출력
for i in range(len(results['ids'])):
    print(f"\n--- 채용공고 {i+1} ---")
    print(f"ID: {results['ids'][i]}")
    print(f"메타데이터: {results['metadatas'][i]}")
    print(f"문서내용: {results['documents'][i]}")
