from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vector_db.chroma_operations import ChromaOperations

class JobAssistantBot:
    def __init__(self, use_chroma: bool = False):
        # Phi-4 모델 초기화
        print("Loading Phi-4 model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-4",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu",  # 명시적으로 CPU 사용
                trust_remote_code=True
            )
            print("Phi-4 model loaded successfully")
        except Exception as e:
            print(f"Error loading Phi-4 model: {e}")
            # 모델 로드 실패시 더미 응답을 반환하는 간단한 모델로 대체
            self.model = None
            self.tokenizer = None
        
        # Chroma 연결은 선택적으로
        self.use_chroma = use_chroma
        if use_chroma:
            try:
                self.chroma_ops = ChromaOperations()
                print("ChromaDB connected successfully")
            except Exception as e:
                print(f"Chroma connection failed: {e}")
                print("Running without Chroma...")
                self.use_chroma = False
        
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Phi-4 모델을 사용하여 응답 생성"""
        if self.model is None:
            return "죄송합니다. 현재 AI 모델을 사용할 수 없습니다. 기본 응답을 제공해드립니다."
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
        
    def recommend_jobs(self, user_preferences: str) -> Dict:
        """사용자 선호도에 기반한 채용정보 추천"""
        if self.use_chroma:
            jobs = self.chroma_ops.search_similar_jobs(user_preferences)
        else:
            # Chroma 없이 테스트용 더미 데이터 반환
            jobs = [
                {
                    "job_title": "테스트 직무",
                    "company": "테스트 회사",
                    "description": "이것은 테스트용 직무 설명입니다.",
                    "requirements": "테스트 요구사항"
                }
            ]
        
        recommendations_prompt = f"""
사용자 선호: {user_preferences}

다음 채용 정보들을 분석하고 사용자에게 맞춤형 추천 설명을 제공해주세요:

{jobs[:3]}

응답 형식:
1. 추천 이유
2. 각 채용공고의 장점
3. 지원 시 주의사항
"""
        explanation = self.generate_response(recommendations_prompt)
        
        return {
            "jobs": jobs,
            "explanation": explanation
        }
    
    def create_resume(self, user_info: Dict) -> str:
        """이력서 작성 지원"""
        resume_prompt = f"""
다음 정보를 바탕으로 노인 구직자를 위한 전문적인 이력서를 작성해주세요:

이름: {user_info.get('name')}
연락처: {user_info.get('contact')}
이메일: {user_info.get('email')}
경력: {user_info.get('experience')}
학력: {user_info.get('education')}
자격증: {user_info.get('certificates')}
자기소개: {user_info.get('self_introduction')}

이력서는 다음 사항을 강조해주세요:
1. 풍부한 인생 경험과 성실성
2. 책임감과 신뢰성
3. 구체적인 업무 경험
"""
        return self.generate_response(resume_prompt)

    def get_training_recommendations(self, user_background: str) -> str:
        """교육/훈련 정보 추천"""
        training_prompt = f"""
다음 배경을 가진 노인 구직자에게 적합한 교육/훈련 프로그램을 추천해주세요:

배경 정보:
{user_background}

다음 사항을 포함해서 추천해주세요:
1. 추천하는 교육/훈련 프로그램
2. 예상 소요 기간
3. 교육 후 가능한 직종
4. 지원 가능한 정부 지원 제도
"""
        return self.generate_response(training_prompt) 