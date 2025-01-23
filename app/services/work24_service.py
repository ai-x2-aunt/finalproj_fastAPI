from typing import Dict, List, Optional
import httpx
import xmltodict
import json
from datetime import datetime
from ..core.config import settings

class Work24Service:
    def __init__(self):
        self.base_url = settings.WORK24_API_COMMON_BASE_URL
        self.api_key = settings.WORK24_COMMON_CODE_API_KEY

    async def fetch_job_postings(
        self,
        start_page: int = 1,
        display: int = 100,
        region: Optional[str] = None,
        occupation: Optional[str] = None,
        salary_type: Optional[str] = None,
        min_salary: Optional[int] = None,
        max_salary: Optional[int] = None,
        education: Optional[str] = None,
        career: Optional[str] = None,
        employment_type: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[Dict]:
        """고용24 API에서 채용 정보 조회"""
        params = {
            "authKey": self.api_key,
            "callTp": "L",
            "returnType": "xml",
            "startPage": start_page,
            "display": display
        }

        # 선택적 파라미터 추가
        if region:
            params["region"] = region
        if occupation:
            params["occupation"] = occupation
        if salary_type:
            params["salTp"] = salary_type
            if min_salary:
                params["minPay"] = min_salary
            if max_salary:
                params["maxPay"] = max_salary
        if education:
            params["education"] = education
        if career:
            params["career"] = career
        if employment_type:
            params["empTp"] = employment_type
        if keyword:
            params["keyword"] = keyword

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/callOpenApiSvcInfo210L01.do",
                    params=params
                )
                response.raise_for_status()

                # XML을 딕셔너리로 변환
                data = xmltodict.parse(response.text)
                print(f"XML 응답: {data}")  # 디버그 로그 추가
                wanted_root = data.get("wantedRoot", {})
                wanted_list = wanted_root.get("wanted", [])

                # 단일 결과인 경우 리스트로 변환
                if isinstance(wanted_list, dict):
                    wanted_list = [wanted_list]

                # 채용 정보를 JobPosting 스키마에 맞게 변환
                job_postings = []
                for wanted in wanted_list:
                    job_posting = {
                        "title": wanted.get("title", ""),
                        "company_name": wanted.get("company", ""),
                        "location": wanted.get("region", ""),
                        "job_type": self._convert_emp_type(wanted.get("empTpCd", "")),
                        "salary": f"{wanted.get('salTpNm', '')} {wanted.get('sal', '')}",
                        "required_skills": [],  # API에서 제공하지 않음
                        "description": "",  # 상세 정보 API 호출 필요
                        "requirements": f"학력: {wanted.get('minEdubg', '')}~{wanted.get('maxEdubg', '')}, 경력: {wanted.get('career', '')}",
                        "benefits": "",  # 상세 정보 API 호출 필요
                        "working_hours": "",  # 상세 정보 API 호출 필요
                        "age_limit": "",
                        "education": wanted.get("minEdubg", ""),
                        "experience": wanted.get("career", ""),
                        "deadline": wanted.get("closeDt", ""),
                        "is_active": True,
                        "created_at": datetime.strptime(wanted.get("regDt", ""), "%Y-%m-%d"),
                        "updated_at": datetime.now()
                    }
                    job_postings.append(job_posting)

                print(f"파싱 결과: {job_postings}")  # 디버그 로그 추가
                return job_postings

        except httpx.HTTPError as e:
            print(f"HTTP 오류 발생: {str(e)}")
            return []
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return []

    def _convert_emp_type(self, emp_type_code: str) -> str:
        """고용형태 코드를 문자열로 변환"""
        emp_type_map = {
            "10": "정규직",
            "11": "정규직(시간선택제)",
            "20": "계약직",
            "21": "계약직(시간선택제)",
            "4": "파견직",
            "Y": "대체인력"
        }
        return emp_type_map.get(emp_type_code, "기타")

    async def fetch_codes(self, code_type: str) -> List[Dict[str, str]]:
        try:
            params = {
                "authKey": self.api_key,
                "returnType": "XML",
                "target": "CMCD",
                "dtlGb": "1"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/callOpenApiSvcInfo21L01.do", params=params)
                response.raise_for_status()
                
                data = xmltodict.parse(response.text)
                print(f"XML 응답: {data}")  # 디버그 로그 추가
                
                codes = []
                if code_type == "region":
                    root = data.get("cmcdRegion", {})
                    one_depth_items = root.get("oneDepth", [])
                    if not isinstance(one_depth_items, list):
                        one_depth_items = [one_depth_items]
                    
                    for item in one_depth_items:
                        code = {
                            "code": item.get("regionCd", ""),
                            "name": item.get("regionNm", ""),
                            "sub_codes": []
                        }
                        
                        two_depth_items = item.get("twoDepth", [])
                        if not isinstance(two_depth_items, list):
                            two_depth_items = [two_depth_items]
                        
                        for sub_item in two_depth_items:
                            sub_code = {
                                "code": sub_item.get("regionCd", ""),
                                "name": sub_item.get("regionNm", ""),
                                "super_code": sub_item.get("superCd", "")
                            }
                            code["sub_codes"].append(sub_code)
                        
                        codes.append(code)
                
                print(f"파싱 결과: {codes}")  # 디버그 로그 추가
                return codes
                
        except Exception as e:
            import traceback
            print(f"오류 발생: {str(e)}")
            print(f"오류 상세: {e.__class__.__name__}")
            print(f"스택 트레이스: {traceback.format_exc()}")
            return [] 