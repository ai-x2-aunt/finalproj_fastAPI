from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import os
from typing import List

class CustomEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class ChromaVectorStore:
    def __init__(self):
        self.embedding_function = CustomEmbeddingFunction('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.collection_name = "job_postings"
        self.persist_directory = "chroma_db"
        
    def create_db(self):
        """ChromaDB 인스턴스 생성"""
        try:
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            print("ChromaDB 생성 성공")
            return db
        except Exception as e:
            print(f"ChromaDB 생성 실패: {e}")
            raise
            
    def add_documents(self, documents):
        """문서 추가"""
        db = self.create_db()
        try:
            db.add_documents(documents)
            print(f"{len(documents)}개의 문서가 성공적으로 추가됨")
        except Exception as e:
            print(f"문서 추가 실패: {e}")
            raise 