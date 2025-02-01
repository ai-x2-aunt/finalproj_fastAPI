from typing import List, Dict, Any, Optional
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain.memory import ConversationBufferMemory
from .vector_db_service import VectorDBService
import asyncio
from sentence_transformers import SentenceTransformer

# .env 파일 로드
load_dotenv()

class LLMService:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Args:
            model_name: 사용할 모델 이름 ("gpt-4o" 또는 "gpt-4o-mini")
        """
        self.model_name = model_name  # 원래대로 복구
        
        # OpenAI API 키 설정
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # 대화 기록 관리
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            return_messages=True
        )

        self.vector_db = VectorDBService()

    async def get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            model = SentenceTransformer('jhgan/ko-sbert-nli')
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error in get_embedding: {str(e)}")
            return []

    async def get_response(self, user_input: str) -> Dict[str, Any]:
        try:
            # asyncio.timeout() 대신 asyncio.wait_for() 사용
            async def process_response():
                # 대화 내용에서 키워드 추출
                chat_history = self.conversation_memory.load_memory_variables({})
                
                # 키워드 추출을 위한 프롬프트 생성
                keyword_extraction_prompt = f"""
                다음 대화 내용에서 구직 관련 정보를 추출해주세요:
                {user_input}
                
                다음 JSON 형식으로 반환해주세요:
                {{
                    "직무_키워드": [],
                    "기술_자격_키워드": [],
                    "선호도_키워드": [],
                    "제약사항_키워드": []
                }}
                """
                
                # 키워드 추출
                keywords_response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": keyword_extraction_prompt}],
                    temperature=0.7
                )
                
                try:
                    keywords = json.loads(keywords_response.choices[0].message.content)
                except json.JSONDecodeError:
                    keywords = {
                        "직무_키워드": [],
                        "기술_자격_키워드": [],
                        "선호도_키워드": [],
                        "제약사항_키워드": []
                    }

                # 임베딩 생성 및 유사 채용정보 검색
                conversation_embedding = await self.get_embedding(user_input)
                
                similar_jobs = await self.vector_db.search_similar_jobs(
                    vector=conversation_embedding,
                    limit=5,
                    filter_conditions={
                        "keywords": keywords['직무_키워드'],
                        "location": keywords['선호도_키워드']
                    }
                )
                
                # 챗봇 응답 생성
                chat_prompt = f"""당신은 시니어 구직자를 위한 AI 취업 상담사입니다.
                이전 대화: {chat_history}
                사용자 입력: {user_input}
                추천 채용정보: {similar_jobs if similar_jobs else '없음'}
                
                친절하고 공감하는 톤으로 응답해주세요."""

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": chat_prompt}],
                    temperature=0.7
                )

                bot_response = response.choices[0].message.content

                # 대화 기록 저장
                self.conversation_memory.save_context(
                    {"input": user_input},
                    {"output": bot_response}
                )

                return {
                    "message": bot_response,
                    "keywords": keywords,
                    "similar_jobs": similar_jobs,
                    "embeddings": conversation_embedding
                }

            # 60초 타임아웃 설정
            return await asyncio.wait_for(process_response(), timeout=60)

        except asyncio.TimeoutError:
            return {
                "message": "죄송합니다. 요청 처리 시간이 초과되었습니다. 다시 시도해 주시겠어요?",
                "keywords": {"직무_키워드": [], "기술_자격_키워드": [], "선호도_키워드": [], "제약사항_키워드": []},
                "similar_jobs": [],
                "embeddings": []
            }
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return {
                "message": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 한 번 말씀해 주시겠어요?",
                "keywords": {"직무_키워드": [], "기술_자격_키워드": [], "선호도_키워드": [], "제약사항_키워드": []},
                "similar_jobs": [],
                "embeddings": []
            }

    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation_memory.clear() 