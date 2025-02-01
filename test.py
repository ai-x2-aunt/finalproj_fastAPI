import os
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv() # .env 파일을 로드하여 환경변수에 저장
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # .env에서 'OPENAI_API_KEY' 값을 가져온다

# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader, JSONLoader  # 한 줄로 import 가능
from langchain.docstore.document import Document


import codecs
import pathlib
import json

# OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY # OpenAI API 사용을 위한 API 키 환경변수 설정

# JSON 구조에 맞게 jq_schema 정의
jq_schema = '.' # 모든 json 데이터를 그대로 사용하려면 이렇게 정의

# JSON 파일을 'utf-8'로 읽기
with open("data/jobs_with_details.json", "r", encoding="utf-8") as file:
  json_data = json.load(file) # jobs_with_details.json 파일을 읽어서 JSON 데이터를 파싱

# JSON 구조에 맞게 데이터 접근
job_listings = json_data.get("채용공고목록", [])  # "채용공고목록" 키로 채용 공고 목록을 가져옴

# JSON 데이터를 Document 객체로 변환
documents = [] # Document 객체를 담을 리스트

for item in job_listings: # 각 채용공고에 대해 반복
  if isinstance(item, dict): # 각 항목이 딕셔너리 형식인지 확인
	  # 채용공고 데이터를 문자열로 결합하여 content 변수에 저장
    # content = f"제목: {item.get('채용공고제목', '')}\n"
    # content += f"회사명: {item.get('회사명', '')}\n"
    # content += f"근무지역: {item.get('근무지역', '')}\n"
    # content += f"모집직종: {item.get('모집직종', '')}\n"
    # content += f"고용형태: {item.get('고용형태', '')}\n"
    # content += f"경력조건: {item.get('경력조건', '')}\n"
    # content += f"학력: {item.get('학력', '')}\n"
    # content += f"연령: {item.get('연령', '')}\n"
    content = ""
    for key, value in item.items():
        if key != '상세정보':
            content += f"{key}: {value}\n"

		# '상세정보' 키가 있으면 직무내용과 세부요건을 추가
    if '상세정보' in item and isinstance(item['상세정보'], dict):
        detail = item['상세정보']
    #   content += f"직무내용: {detail.get('직무내용', '')[:500]}...\n" # 직무내용의 처음 500자만 포함
        for key, value in detail.items():
            if key == '직무내용':
                content += f"{key}: {value[:100]}...\n" # 직무내용의 처음 100자만 포함
      # 세부요건 정보 추가
        # for req in detail.get('세부요건', []): # 세부요건이 리스트라면
        #     for key, value in req.items():      # 각 세부요건의 항목을 순차적으로 가져옴
        #         if isinstance(value, list) and value:
        #             content += f"{key}: {value[0]}\n" # 첫 번째 값을 내용에 추가
            elif key == '세부요건':
                for req in value:
                    for req_key, req_value in req.items():
                        # 리스트 형태의 값을 처리하는 방법 (첫 번째 값만 사용)
                        # if isinstance(req_value, list) and req_value:
                        #     content += f"{req_key}: {req_value[0]}\n"
                        # 또는 모든 값을 쉼표로 구분하여 나열하는 방법
                        if isinstance(req_value, list) and req_value:
                            content += f"{req_key}: {', '.join(req_value)}\n"
            else:
                content += f"{key}: {value}\n"
    documents.append(Document(page_content=content, metadata={"source": "jobs_with_details.json"}))

# # for문 안으로 이동 # 완성된 content를 Document 객체로 변환하여 documents 리스트에 추가
# documents.append(Document(page_content=content, metadata={"source": "jobs_with_details.json"}))

# 문서 분할 -- 문서를 작은 조각으로 분할
# 텍스트 분할기 생성 (chunk_size=250, chunk_overlap=10)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=10) 
# 문서들을 500자 크기의 작은 덩어리로 분할
docs = text_splitter.split_documents(documents) 

# 벡터화 -- OpenAI Embeddings 초기화
embedding_function = OpenAIEmbeddings() # OpenAI의 Embedding 기능을 사용하여 벡터화를 위한 함수 생성

# 벡터 스토어 생성 및 저장
persist_directory = 'db/speech_embedding_db' # 벡터 스토어를 저장할 디렉토리
# Chroma 벡터 스토어 생성
vectordb = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)

# 데이터베이스 저장
vectordb.persist() # 데이터베이스를 디스크에 저장한다

# 벡터 스토어 로드
# 저장된 벡터 스토어 로드
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
# ㄴ 컴퓨터를 재시작해도 저장된 데이터를 로드할 수 있다.

# 유사성 검색
# 유사성 검색 예시
query = "군산에 50대이상이 지원할 수 있는 일자리를 알아봐줄래? 난 여자야."
results = vectordb.similarity_search(query) # similarity_search 쿼리와 유사한 문서를 검색

# 결과 출력 -- 검색된 문서의 내용을 출력
# for idx, doc in enumerate(results):
#   print(f"결과: {idx+1}:\n{doc.page_content}\n")

# Chroma 벡터 DB 객체에서 `get()`메서드로 저장된 데이터 확인하기
# persist_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
# stored_data = persist_db.get()
# print(stored_data)

# 결과 파일 저장
with open("results.txt", "w", encoding="utf-8") as file: # 파일 열기 (쓰기 모드)
    for idx, doc in enumerate(results):
        file.write(f"결과: {idx+1}:\n{doc.page_content}\n") # 파일에 내용 쓰기

print("검색 결과가 results.txt 파일에 저장되었습니다.")