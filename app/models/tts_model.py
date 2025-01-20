import torch
from pathlib import Path
import os
import sys
import subprocess
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# 환경변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# espeak 관련 경로 직접 지정
ESPEAK_PATH = r'C:\Program Files\eSpeak NG'
os.environ['PATH'] = ESPEAK_PATH + os.pathsep + os.environ['PATH']
os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(ESPEAK_PATH, 'espeak-ng.exe')
os.environ['PHONEMIZER_ESPEAK_LIB'] = os.path.join(ESPEAK_PATH, 'libespeak-ng.dll')

# espeak 라이브러리 경로 설정
EspeakWrapper.set_library(os.environ['PHONEMIZER_ESPEAK_LIB'])

class TTSModel:
    def __init__(self):
        # 절대 경로 설정
        current_dir = Path(__file__).parent
        kokoro_path = current_dir / "kokoro"  # Kokoro 모델 디렉토리
        
        if not kokoro_path.exists():
            kokoro_path.mkdir(exist_ok=True, parents=True)
            print(f"Kokoro 모델을 다운로드합니다. 경로: {kokoro_path}")
            
            # Git LFS 설치 및 모델 다운로드
            subprocess.run(['git', 'lfs', 'install'])
            subprocess.run(['git', 'clone', 'https://huggingface.co/hexgrad/Kokoro-82M', str(kokoro_path)])
            
            # 필요한 패키지 설치
            subprocess.run(['pip', 'install', '-q', 'phonemizer', 'munch', 'scipy'])
            
            print("Kokoro 모델 다운로드 완료")

        # Kokoro 모델 경로를 Python 경로에 추가
        sys.path.append(str(kokoro_path))
        
        # 모델 및 보이스팩 로드
        from models import build_model  # kokoro/models.py
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = build_model(str(kokoro_path / 'kokoro-v0_19.pth'), self.device)
        self.voicepack = torch.load(str(kokoro_path / 'voices/af.pt'), weights_only=True).to(self.device)
        
        from kokoro import generate  # kokoro/kokoro.py
        self.generate = generate
        
        print(f"Kokoro 모델 로드 완료 (디바이스: {self.device})")

    def generate_speech(self, text):
        # 음성 생성
        audio, _ = self.generate(self.model, text, self.voicepack, lang='a')
        return audio  # 24kHz 샘플레이트의 오디오 데이터 반환 