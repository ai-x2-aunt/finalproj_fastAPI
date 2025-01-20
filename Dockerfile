FROM python:3.9

WORKDIR /app

# Ollama 설치
RUN curl -fsSL https://ollama.com/install.sh | sh

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Phi-4 모델 파일 복사
COPY models/phi-4 /root/.ollama/models/

# Modelfile 복사 및 커스텀 모델 생성
COPY app/models/Modelfile /app/Modelfile
RUN ollama create custom-phi -f /app/Modelfile

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 