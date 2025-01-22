from typing import List, Dict, Any
import os
import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from .vector_db_service import VectorDBService

class LLMService:
    def __init__(self, model_name: str = "phi4"):
        """
        Args:
            model_name: 사용할 모델 이름 ("phi4" 또는 "llama2")
        """
        self.model_name = model_name
        # 대화용 LLM은 선택한 모델 사용
        self.llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
        )
        # 임베딩은 항상 llama2를 사용 (384 차원 유지)
        self.embeddings = OllamaEmbeddings(
            model="llama2",
            base_url="http://localhost:11434"
        )
        
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            return_messages=True
        )

        self.vector_db = VectorDBService()

        # 번역용 LLM (항상 한국어로 번역)
        self.translator = Ollama(
            model="phi4",  # 번역용으로는 phi4가 더 자연스러움
            base_url="http://localhost:11434",
        )

        self.interview_prompt = PromptTemplate(
            input_variables=["chat_history", "input", "similar_jobs"],
            template="""You are an AI career counselor for senior job seekers. Have a natural conversation to understand their background and preferences.

            Key information to gather:
            - Work experience
            - Skills and certifications
            - Job preferences
            - Work environment preferences
            - Salary expectations

            Rules:
            1. Ask only ONE question at a time
            2. Keep responses under 3 sentences
            3. Be friendly and empathetic
            4. Progress naturally based on their answers
            5. Suggest jobs only when you have enough context

            Previous conversation:
            {chat_history}

            Available recommendations:
            {similar_jobs}

            User: {input}
            Assistant: """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.interview_prompt,
            memory=self.conversation_memory,
            verbose=True
        )

    async def translate_to_korean(self, text: str) -> str:
        """영어 텍스트를 한국어로 번역"""
        translation_prompt = f"""Translate to natural Korean, keeping the career counselor's friendly tone.
        Make it concise and easy to understand.
        
        Text: {text}
        
        Korean:"""
        
        translation = await self.translator.agenerate([translation_prompt])
        return translation.generations[0][0].text.strip()

    async def get_response(self, user_input: str) -> Dict[str, Any]:
        try:
            # 대화 내용 임베딩 (llama2 사용)
            conversation_embedding = await self.embeddings.aembed_query(user_input)
            
            # 유사한 채용 공고 검색
            similar_jobs = await self.vector_db.search_similar_jobs(
                vector=conversation_embedding,
                limit=3
            )
            
            # 유사한 훈련 프로그램 검색
            similar_programs = await self.vector_db.search_similar_programs(
                vector=conversation_embedding,
                limit=2
            )
            
            # 채용 공고 정보를 문자열로 변환
            similar_jobs_text = "\n".join([
                f"- {job['metadata']['title']} ({job['metadata']['company_name']})"
                f"\n  위치: {job['metadata']['location']}"
                f"\n  급여: {job['metadata']['salary']}"
                f"\n  근무형태: {job['metadata']['job_type']}"
                for job in similar_jobs
            ]) if similar_jobs else "아직 맞춤 채용 공고를 찾고 있습니다."

            # 훈련 프로그램 정보를 문자열로 변환
            similar_programs_text = "\n".join([
                f"- {program['metadata']['title']} ({program['metadata']['institution']})"
                f"\n  기간: {program['metadata']['duration']}"
                f"\n  비용: {program['metadata']['cost']}"
                f"\n  장소: {program['metadata']['location']}"
                for program in similar_programs
            ]) if similar_programs else "아직 맞춤 훈련 프로그램을 찾고 있습니다."

            # 프롬프트에 훈련 프로그램 정보 추가
            combined_recommendations = f"""
            추천 채용 공고:
            {similar_jobs_text}
            
            추천 훈련 프로그램:
            {similar_programs_text}
            """

            # LLM 응답 생성 (영어)
            response = await self.chain.ainvoke({
                "input": user_input,
                "similar_jobs": combined_recommendations
            })
            
            # 응답을 한국어로 번역
            korean_response = await self.translate_to_korean(response['text'])
            
            # 현재까지의 대화 내용에서 키워드 추출
            chat_history = self.conversation_memory.load_memory_variables({})["chat_history"]
            full_context = f"""
            지금까지의 대화:
            {chat_history}
            
            마지막 대화:
            사용자: {user_input}
            AI: {korean_response}
            """
            
            keyword_extraction_prompt = f"""
            다음 대화 내용에서 발견된 구직 관련 정보를 추출해주세요.
            발견된 정보만 추출하고, 없는 정보는 빈 리스트로 남겨주세요.
            
            {full_context}
            
            다음 JSON 형식으로 반환해주세요:
            {{
                "직무_키워드": [],
                "기술_자격_키워드": [],
                "선호도_키워드": [],
                "제약사항_키워드": []
            }}
            """
            
            keywords_response = await self.llm.agenerate([keyword_extraction_prompt])
            keywords_text = keywords_response.generations[0][0].text
            
            try:
                keywords = json.loads(keywords_text)
            except json.JSONDecodeError:
                keywords = {
                    "직무_키워드": [],
                    "기술_자격_키워드": [],
                    "선호도_키워드": [],
                    "제약사항_키워드": []
                }
            
            return {
                "message": korean_response,
                "keywords": keywords,
                "similar_jobs": similar_jobs,
                "similar_programs": similar_programs,
                "embeddings": conversation_embedding
            }
            
        except Exception as e:
            print(f"Error in LLM service: {str(e)}")
            return {
                "message": "안녕하세요! 취업 상담을 도와드리겠습니다. 어떤 도움이 필요하신가요?",
                "keywords": {
                    "직무_키워드": [],
                    "기술_자격_키워드": [],
                    "선호도_키워드": [],
                    "제약사항_키워드": []
                },
                "similar_jobs": [],
                "similar_programs": [],
                "embeddings": []
            }

    def reset_conversation(self):
        self.conversation_memory.clear() 