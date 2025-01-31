from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
from pathlib import Path
import soundfile as sf
import os
from dotenv import load_dotenv
from app.models.tts_model import TTSModel
from app.models.stt_model import STTModel
import librosa
import numpy as np
import subprocess  # ffmpeg 실행을 위해 추가
import mysql.connector
from typing import List
import requests  # ollama 대신 requests 사용
import json
import re  # 정규 표현식 사용

load_dotenv()

router = APIRouter()

# 모델 초기화
tts_model = TTSModel()
stt_model = STTModel()

class TTSRequest(BaseModel):
    text: str

class AddressMatchRequest(BaseModel):
    voice_text: str

@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        print("STT 요청 받음")
        
        # 임시 파일들 생성
        temp_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        try:
            # webm 파일 저장
            content = await audio.read()
            temp_webm.write(content)
            temp_webm.flush()
            
            # ffmpeg로 변환
            subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_webm.name,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                temp_wav.name
            ], check=True, capture_output=True)
            
            # wav 파일 읽기
            audio_data, sample_rate = sf.read(temp_wav.name)
            
            # 숫자 변환 함수
            def convert_korean_number(text):
                number_map = {
                    '공': '0', '영': '0',
                    '일': '1', '하나': '1', '한': '1',
                    '이': '2', '둘': '2',
                    '삼': '3', '셋': '3',
                    '사': '4', '넷': '4',
                    '오': '5', '다섯': '5',
                    '육': '6', '여섯': '6',
                    '칠': '7', '일곱': '7',
                    '팔': '8', '여덟': '8',
                    '구': '9', '아홉': '9',
                    '열': ''  # '열' 제거 (열하나 -> 11)
                }
                
                # 공백 제거 및 중복된 '하나' 처리
                text = text.replace(' ', '')
                text = text.replace('하나하나', '11')
                text = text.replace('둘하나', '21')
                
                # '열' 처리를 위한 특별 케이스
                text = text.replace('열', '1')
                for k in ['하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉']:
                    text = text.replace(f'열{k}', f'1{number_map[k]}')
                
                # 나머지 숫자 변환
                result = ''
                i = 0
                while i < len(text):
                    # 연속된 '하나' 처리
                    if i < len(text) - 1 and text[i:i+2] == '하나':
                        result += '1'
                        i += 2
                    elif text[i] in number_map:
                        result += number_map[text[i]]
                        i += 1
                    elif text[i].isdigit():
                        result += text[i]
                        i += 1
                    else:
                        i += 1
                
                print(f"숫자 변환 결과: {result}")  # 디버깅용
                return result

            # STT 모델로 텍스트 변환
            text = stt_model.transcribe(audio_data, language='ko')
            print("원본 변환 결과: ", text)
            
            # 숫자가 포함된 문자열이고, 대부분이 숫자 관련 단어인 경우에만 숫자 변환 수행
            number_keywords = ['공', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '영', '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉']
            number_word_count = sum(1 for word in text.split() if any(keyword in word for keyword in number_keywords))
            total_word_count = len(text.split())
            
            if number_word_count > 0 and number_word_count / total_word_count >= 0.5:  # 50% 이상이 숫자 관련 단어일 때
                numbers_only = convert_korean_number(''.join(
                    char for char in text.lower() 
                    if char in '공일이삼사오육칠팔구영하나둘셋넷다섯여섯일곱여덟아홉열한' 
                    or char.isdigit()
                ))
                
                print("숫자 변환 결과:", numbers_only)
                
                # 전화번호 형식으로 변환
                if len(numbers_only) >= 10:
                    formatted_number = f"{numbers_only[:3]}-{numbers_only[3:7]}-{numbers_only[7:]}"
                    print(f"전화번호 형식: {formatted_number}")
                    return JSONResponse(content={'text': formatted_number})
                else:
                    # 숫자만 있는 경우 그대로 반환
                    return JSONResponse(content={'text': numbers_only})
            
            # 일반 텍스트는 그대로 반환
            return JSONResponse(content={'text': text})
            
        finally:
            # 임시 파일들 정리
            temp_webm.close()
            temp_wav.close()
            os.unlink(temp_webm.name)
            os.unlink(temp_wav.name)
            
    except Exception as e:
        print(f"STT 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Kokoro TTS 모델로 음성 생성
        audio_data = tts_model.generate_speech(request.text)
        
        # 오디오 파일 저장
        audio_path = Path("app/static/audio")
        audio_path.mkdir(exist_ok=True, parents=True)
        
        file_name = f"{hash(request.text)}.wav"
        file_path = audio_path / file_name
        
        # 샘플링 레이트는 모델의 기본값인 24000Hz 사용
        sf.write(file_path, audio_data, 24000)
        
        print(f"TTS 오디오 파일 생성됨: {file_path}")  # 디버깅용
        
        return JSONResponse(content={
            'audioUrl': f'/static/audio/{file_name}'
        })
        
    except Exception as e:
        print(f"TTS 에러: {str(e)}")  # 디버깅을 위한 상세 에러 출력
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resume")
async def submit_resume(resume_data: dict):
    try:
        # 여기서 resume_data를 데이터베이스에 저장하거나 필요한 처리를 수행
        print("받은 이력서 데이터:", resume_data)
        return JSONResponse(content={"success": True})
    except Exception as e:
        print(f"이력서 저장 중 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_similar_addresses(city: str, district: str, town: str, limit: int = 5) -> List[dict]:
    conn = mysql.connector.connect(
        host="localhost",
        user="hmk",
        password="manager",
        database="senior_jobgo_db"
    )
    cursor = conn.cursor(dictionary=True)
    
    # road_address 테이블을 사용한 검색 쿼리
    query = """
    SELECT 
        road_code,
        road_name,
        city,
        district,
        town,
        is_basement,
        building_main_num,
        building_sub_num,
        postal_code,
        building_name
    FROM road_address
    WHERE 
        city LIKE %s AND
        district LIKE %s AND
        town LIKE %s
    LIMIT %s
    """
    
    search_city = f"%{city}%"
    search_district = f"%{district}%"
    search_town = f"%{town}%"
    
    cursor.execute(query, (search_city, search_district, search_town, limit))
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    print(f"검색어 시: {city}, 구: {district}, 동: {town}")  # 디버깅용
    print(f"검색 결과: {results}")    # 디버깅용
    
    return results

@router.post("/match_address")
async def match_address(request: AddressMatchRequest):
    try:
        print(f"Received request with voice_text: {request.voice_text}")
        
        # 행정구역 키워드
        administrative_keywords = {
            'city': ['시', '특별시', '광역시', '자치시'],
            'district': ['구', '군'],
            'town': ['동', '읍', '면']
        }
        
        # 주소 파싱
        address_parts = {'city': None, 'district': None, 'town': None}
        words = request.voice_text.split()
        
        for word in words:
            # 시/도 확인
            if any(keyword in word for keyword in administrative_keywords['city']):
                address_parts['city'] = word
            # 구/군 확인
            elif any(keyword in word for keyword in administrative_keywords['district']):
                address_parts['district'] = word
            # 동/읍/면 확인
            elif any(keyword in word for keyword in administrative_keywords['town']):
                address_parts['town'] = word
        
        print(f"파싱된 주소: {address_parts}")  # 디버깅용
        
        if not any(address_parts.values()):
            return JSONResponse(content={
                "original_text": request.voice_text,
                "similar_addresses": [],
                "llm_response": "주소를 인식할 수 없습니다. 시/군/구/동을 포함하여 말씀해주세요.",
                "matched_address": None
            })
        
        # DB 검색 - 각 부분이 있는 경우에만 검색 조건에 추가
        conn = mysql.connector.connect(
            host="localhost",
            user="hmk",
            password="manager",
            database="senior_jobgo_db"
        )
        cursor = conn.cursor(dictionary=True)
        
        # 동적 쿼리 생성
        query = """
        SELECT 
            road_code,
            road_name,
            city,
            district,
            town,
            is_basement,
            building_main_num,
            building_sub_num,
            postal_code,
            building_name
        FROM road_address
        WHERE 1=1
        """
        params = []
        
        if address_parts['city']:
            query += " AND city LIKE %s"
            params.append(f"%{address_parts['city']}%")
        if address_parts['district']:
            query += " AND district LIKE %s"
            params.append(f"%{address_parts['district']}%")
        if address_parts['town']:
            query += " AND town LIKE %s"
            params.append(f"%{address_parts['town']}%")
        
        query += " LIMIT 5"
        
        cursor.execute(query, params)
        similar_addresses = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"DB 검색 결과: {similar_addresses}")  # 디버깅용
        
        if not similar_addresses:
            return JSONResponse(content={
                "original_text": request.voice_text,
                "similar_addresses": [],
                "llm_response": "일치하는 주소를 찾을 수 없습니다.",
                "matched_address": None
            })
        
        # Phi-4에 주소 유사도 비교 요청
        prompt = f"""
당신은 주소 매칭 전문가입니다. 사용자가 말한 주소와 가장 유사한 실제 주소를 찾아주세요.

사용자 입력: "{request.voice_text}"

후보 주소 목록:
{[f"{i+1}. {addr['road_name']} {addr['city']} {addr['district']} {addr['town']} {addr['building_main_num']}{'-'+str(addr['building_sub_num']) if addr['building_sub_num'] else ''}" for i, addr in enumerate(similar_addresses)]}

작업:
1. 각 후보 주소와 사용자 입력을 비교하여 유사도를 평가하세요.
2. 가장 유사한 주소 하나를 선택하고 번호로 표시하세요.
3. 선택한 이유를 간단히 설명하세요.

답변 형식:
선택: [번호]
이유: [설명]
"""
        
        try:
            response = requests.post(
                'http://localhost:11434/api/completions',  # 엔드포인트 수정
                json={
                    "model": "phi4",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                llm_response = response.json()
                response_text = llm_response.get('response', "첫 번째 주소 선택")
                print(f"Phi4 응답: {response_text}")  # 디버깅용 로그 추가
            else:
                print(f"Phi4 모델 응답 실패: {response.status_code}")
                print(f"응답 내용: {response.text}")  # 에러 응답 내용 확인
                response_text = "첫 번째 주소 선택"
                
        except Exception as e:
            print(f"Phi4 모델 오류: {str(e)}")
            response_text = "첫 번째 주소 선택"
        
        return JSONResponse(content={
            "original_text": request.voice_text,
            "similar_addresses": similar_addresses,
            "llm_response": response_text,
            "matched_address": similar_addresses[0]
        })
        
    except Exception as e:
        print(f"주소 매칭 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 