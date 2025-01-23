from typing import Dict, List, Optional
import httpx
import xmltodict
import json
from datetime import datetime
from ..core.config import settings

class HRDService:
    def __init__(self):
        self.base_url = settings.WORK24_API_COMMON_TRAINING_URL
        self.auth_key = settings.TOMORROW_LEARNING_CARD_API_KEY

    async def fetch_training_programs(
        self,
        start_page: int = 1,
        display: int = 100,
        region: Optional[str] = None,
        keyword: Optional[str] = None,
        training_type: Optional[str] = None,
        training_target: Optional[str] = None
    ) -> List[Dict]:
        """HRD-Net API에서 훈련 프로그램 정보 조회"""
        params = {
            "authKey": self.auth_key,
            "returnType": "XML",
            "pageNum": start_page,
            "pageSize": display,
            "srchTraArea": region or "",  # 훈련지역
            "srchKwd": keyword or "",     # 검색 키워드
            "srchTraStDt": datetime.now().strftime("%Y%m%d"),  # 훈련시작일 (오늘 이후)
            "sort": "ASC",                # 정렬 방식
            "outType": "1"                # 출력 형식 (1: 리스트)
        }

        # 선택적 파라미터 추가
        if training_type:
            params["srchTraGbn"] = training_type  # 훈련 유형
        if training_target:
            params["srchTraTarget"] = training_target  # 훈련 대상

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/2/dataList.do",
                    params=params
                )
                response.raise_for_status()

                # XML을 딕셔너리로 변환
                data = xmltodict.parse(response.text)
                programs = data.get("HRDNet", {}).get("srchList", [])

                # 단일 결과인 경우 리스트로 변환
                if isinstance(programs, dict):
                    programs = [programs]

                # 훈련 프로그램 정보를 TrainingProgram 스키마에 맞게 변환
                training_programs = []
                for program in programs:
                    training_program = {
                        "title": program.get("subTitle", ""),
                        "institution": program.get("trainstCstmr", ""),
                        "duration": f"{program.get('traStartDate', '')} ~ {program.get('traEndDate', '')}",
                        "description": program.get("trainTarget", ""),  # 훈련 대상을 설명으로 사용
                        "requirements": program.get("trainTarget", ""),
                        "cost": program.get("courseMan", "0"),  # 수강료
                        "support_info": self._get_support_info(program),
                        "location": f"{program.get('address', '')}",
                        "schedule": f"훈련 기간: {program.get('traStartDate', '')} ~ {program.get('traEndDate', '')}",
                        "certificate": "",  # API에서 제공하지 않음
                        "start_date": datetime.strptime(program.get("traStartDate", ""), "%Y%m%d"),
                        "end_date": datetime.strptime(program.get("traEndDate", ""), "%Y%m%d"),
                        "is_active": True,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    }
                    training_programs.append(training_program)

                return training_programs

        except httpx.HTTPError as e:
            print(f"HTTP 오류 발생: {str(e)}")
            return []
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return []

    def _get_support_info(self, program: Dict) -> str:
        """지원금 정보 생성"""
        support_info = []
        
        # 국민내일배움카드 훈련비 지원
        if program.get("cardTrainSupport"):
            support_info.append(f"국민내일배움카드 지원: {program['cardTrainSupport']}%")
        
        # 실업자 훈련비 지원
        if program.get("unempSupport"):
            support_info.append(f"실업자 훈련비 지원: {program['unempSupport']}%")
        
        # 근로자 훈련비 지원
        if program.get("empSupport"):
            support_info.append(f"근로자 훈련비 지원: {program['empSupport']}%")

        return "\n".join(support_info) if support_info else "지원금 정보 없음" 