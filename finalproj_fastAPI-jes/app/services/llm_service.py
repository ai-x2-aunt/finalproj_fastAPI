from typing import List, Dict, Any, Optional
import os
import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from .vector_db_service import VectorDBService
import asyncio
import httpx

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
        # 임베딩은 항상 llama2를 사용
        self.embeddings = OllamaEmbeddings(
            model="llama2",
            base_url="http://localhost:11434"
        )
        
        # 대화 기록 관리 개선
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
            template="""당신은 시니어 구직자를 위한 AI 취업 상담사입니다. 

            다음 규칙을 따라 상담을 진행하세요:
            1. 채용정보가 있다면 먼저 2-3개만 간단히 소개하세요
            2. 답변은 2문장으로 제한하세요
            3. 친근하고 공감하는 톤을 유지하세요
            4. 자연스럽게 추가 정보를 요청하세요
            5. 가상의 대화를 만들지 마세요
            6. 사용자가 요청한 분야나 조건에 맞춰 상담하세요
            7. 한 번에 하나의 정보만 요청하세요
            
            수집해야 할 정보:
            - 연령대
            - 성별
            - 선호 지역
            - 희망 연봉
            - 경력 사항
            - 자격증/교육 이력
            - 선호하는 직종/업종
            - 근무 형태(정규직/계약직/파트타임)
            
            이전 대화:
            {chat_history}
            
            추천 가능한 정보:
            {similar_jobs}
            
            사용자: {input}
            Assistant: """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.interview_prompt,
            memory=self.conversation_memory,
            verbose=True,
            output_key="text"  # 출력 키를 text로 통일
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
            # 타임아웃 설정
            async with asyncio.timeout(60):  # 60초 타임아웃
                print("1. Starting get_response...")
                
                # 대화 내용에서 키워드 추출
                chat_history = self.conversation_memory.load_memory_variables({})
                print(f"2. Loaded chat history: {chat_history}")
                
                history_str = "\n".join([
                    f"사용자: {msg.content}" if msg.type == "human" else f"AI: {msg.content}"
                    for msg in chat_history.get("chat_history", [])
                ])
                print(f"3. Formatted history: {history_str}")
                
                full_context = f"""
                지금까지의 대화:
                {history_str}
                
                마지막 대화:
                사용자: {user_input}
                """
                print("4. Created full context")
                
                # 키워드 추출
                keyword_extraction_prompt = f"""
                다음 대화 내용에서 구직 관련 정보를 최대한 추출해주세요.
                빈 값이라도 관련 있을 수 있는 키워드는 모두 포함해주세요.
                
                {full_context}
                
                다음 JSON 형식으로 반환해주세요:
                {{
                    "직무_키워드": [],
                    "기술_자격_키워드": [],
                    "선호도_키워드": [],
                    "제약사항_키워드": []
                }}
                """
                
                print("5. Extracting keywords...")
                keywords_response = await self.llm.agenerate([keyword_extraction_prompt])
                keywords_text = keywords_response.generations[0][0].text
                print(f"6. Extracted keywords text: {keywords_text}")
                
                try:
                    keywords = json.loads(keywords_text)
                    print(f"7. Parsed keywords: {keywords}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing keywords JSON: {str(e)}")
                    keywords = {
                        "직무_키워드": [],
                        "기술_자격_키워드": [],
                        "선호도_키워드": [],
                        "제약사항_키워드": []
                    }
                    print(f"7. Using default keywords: {keywords}")

                # 검색 실행
                print("8. Starting vector search...")
                conversation_embedding = await self.embeddings.aembed_query(user_input)
                print("9. Created embedding")
                
                similar_jobs = await self.vector_db.search_similar_jobs(
                    vector=conversation_embedding,
                    limit=5,
                    filter_conditions={
                        "keywords": keywords['직무_키워드'],
                        "location": keywords['선호도_키워드']
                    }
                )
                print(f"10. Found similar jobs: {len(similar_jobs)}")
                
                similar_programs = await self.vector_db.search_similar_programs(
                    vector=conversation_embedding,
                    limit=3,
                    filter_conditions={
                        "keywords": keywords['직무_키워드']
                    }
                )
                print(f"11. Found similar programs: {len(similar_programs)}")
                
                # 응답 생성
                print("12. Generating response...")
                response = await self.chain.ainvoke({
                    "input": user_input,
                    "similar_jobs": self._format_recommendations(similar_jobs, similar_programs)
                })
                print(f"13. Generated response: {response}")
                
                # 대화 기록 저장
                print("14. Saving conversation...")
                self.conversation_memory.save_context(
                    {"input": user_input},
                    {"text": response['text']}
                )
                print("15. Saved conversation")
                
                return {
                    "message": response['text'],
                    "keywords": keywords,
                    "similar_jobs": similar_jobs,
                    "similar_programs": similar_programs,
                    "embeddings": conversation_embedding
                }
                
        except asyncio.TimeoutError:
            print("Request timed out after 60 seconds")
            return {
                "message": "죄송합니다. 요청 처리 시간이 초과되었습니다. 다시 시도해 주시겠어요?",
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
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "message": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 한 번 말씀해 주시겠어요?",
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

    def _format_recommendations(self, jobs: List[Dict], programs: List[Dict]) -> str:
        """추천 정보 포맷팅"""
        jobs_text = "\n".join([
            f"- {job['metadata']['title']} ({job['metadata']['company_name']})"
            f"\n  위치: {job['metadata']['location']}"
            f"\n  급여: {job['metadata']['salary']}"
            f"\n  근무형태: {job['metadata']['job_type']}"
            for job in jobs
        ]) if jobs else "현재 조건에 맞는 채용 공고를 찾고 있습니다."

        programs_text = "\n".join([
            f"- {program['metadata']['title']} ({program['metadata']['institution']})"
            f"\n  기간: {program['metadata']['duration']}"
            f"\n  비용: {program['metadata']['cost']}"
            f"\n  장소: {program['metadata']['location']}"
            for program in programs
        ]) if programs else "현재 조건에 맞는 훈련 프로그램을 찾고 있습니다."

        return f"""
        추천 채용 공고:
        {jobs_text}
        
        추천 훈련 프로그램:
        {programs_text}
        """

    def reset_conversation(self):
        self.conversation_memory.clear() 