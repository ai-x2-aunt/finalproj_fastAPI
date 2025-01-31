from typing import Dict, List, Optional
import httpx
import xmltodict
from ..core.config import settings

class CodeService:
    def __init__(self):
        self.base_url = settings.WORK24_API_COMMON_BASE_URL
        self.auth_key = settings.WORK24_COMMON_CODE_API_KEY
        
        # API 엔드포인트 매핑
        self.endpoints = {
            "region": settings.REGION_CODE_ENDPOINT,
            "occupation": settings.JOB_CODE_ENDPOINT,
            "license": settings.CERTIFICATE_CODE_ENDPOINT,
            "industrial": settings.INDUSTRIAL_COMPLEX_ENDPOINT,
            "subway": settings.SUBWAY_CODE_ENDPOINT,
            "major": settings.MAJOR_CODE_ENDPOINT,
            "language": settings.LANGUAGE_CODE_ENDPOINT,
            "department": settings.DEPARTMENT_CODE_ENDPOINT,
            "company": settings.SMALL_GIANT_CODE_ENDPOINT
        }
        
        # dtlGb 매핑
        self.detail_codes = {
            "region": "1",
            "occupation": "2", 
            "license": "3",
            "industrial": "4",
            "subway": "5",
            "major": "6",
            "language": "7",
            "department": "8",
            "company": "9"
        }
        
        # 수집된 코드 캐시
        self.code_cache: Dict[str, List] = {}

    async def fetch_code(self, code_type: str) -> List[Dict]:
        """공통코드 정보 조회"""
        if code_type not in self.endpoints:
            raise ValueError(f"지원하지 않는 코드 타입: {code_type}")
            
        # 캐시된 데이터가 있으면 반환
        if code_type in self.code_cache:
            return self.code_cache[code_type]
            
        params = {
            "authKey": self.auth_key,
            "returnType": "XML",
            "target": "CMCD",
            "dtlGb": self.detail_codes[code_type]
        }
        
        try:
            url = f"{self.base_url}{self.endpoints[code_type]}"
            print(f"요청 URL: {url}")
            print(f"파라미터: {params}")
            
            async with httpx.AsyncClient(verify=False) as client:  # SSL 검증 비활성화
                response = await client.get(
                    url,
                    params=params,
                    headers={
                        "Accept": "application/xml",
                        "User-Agent": "Mozilla/5.0",
                        "Content-Type": "application/xml"
                    },
                    follow_redirects=True  # 리다이렉트 자동 처리
                )
                print(f"응답 상태 코드: {response.status_code}")
                print(f"응답 내용: {response.text}")
                response.raise_for_status()
                
                # 응답이 비어있거나 에러 메시지가 포함된 경우
                if not response.text or "인증키가 존재하지 않습니다" in response.text:
                    print("API 응답 오류: 인증키가 유효하지 않거나 응답이 비어있습니다.")
                    return []
                
                # XML을 딕셔너리로 변환
                data = xmltodict.parse(response.text)
                print(f"파싱된 데이터: {data}")
                codes = []
                
                # 코드 타입에 따라 다른 처리
                if code_type == "region":
                    root = data.get("cmcdRegion", {})
                    one_depth_items = root.get("oneDepth", [])
                    
                    if not isinstance(one_depth_items, list):
                        one_depth_items = [one_depth_items]
                    
                    for item in one_depth_items:
                        code_info = {
                            "code": item.get("regionCd", ""),
                            "name": item.get("regionNm", ""),
                            "description": "",
                            "sub_codes": []
                        }
                        
                        # twoDepth 항목이 있으면 처리
                        two_depth_items = item.get("twoDepth", [])
                        if two_depth_items:
                            if not isinstance(two_depth_items, list):
                                two_depth_items = [two_depth_items]
                                
                            for sub_item in two_depth_items:
                                sub_code = {
                                    "code": sub_item.get("regionCd", ""),
                                    "name": sub_item.get("regionNm", ""),
                                    "description": "",
                                    "super_code": sub_item.get("superCd", "")
                                }
                                code_info["sub_codes"].append(sub_code)
                        
                        codes.append(code_info)
                
                print(f"파싱 결과: {codes}")  # 디버그 로그 추가
                
                # 결과를 캐시에 저장
                self.code_cache[code_type] = codes
                return codes
                
        except httpx.HTTPError as e:
            print(f"HTTP 오류 발생: {str(e)}")
            return []
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            print(f"오류 상세: {e.__class__.__name__}")  # 오류 타입 출력
            import traceback
            print(f"스택 트레이스: {traceback.format_exc()}")  # 스택 트레이스 출력
            return []
            
    async def get_all_codes(self) -> Dict[str, List[Dict]]:
        """모든 공통코드 정보 조회"""
        result = {}
        for code_type in self.endpoints.keys():
            result[code_type] = await self.fetch_code(code_type)
        return result
        
    def get_cached_codes(self, code_type: Optional[str] = None) -> Dict[str, List[Dict]]:
        """캐시된 코드 정보 조회"""
        if code_type:
            return {code_type: self.code_cache.get(code_type, [])}
        return self.code_cache 