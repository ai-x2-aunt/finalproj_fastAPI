import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import os

# 환경변수 설정 추가
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class STTModel:
    def __init__(self):
        # 절대 경로 설정
        current_dir = Path(__file__).parent
        model_path = current_dir / "whisper_model"
        cache_dir = model_path / "cache"  # 캐시 디렉토리 추가
        
        try:
            print(f"STT 모델을 로드합니다. 경로: {model_path}")
            # 모델과 프로세서 로드
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                cache_dir=cache_dir,
                local_files_only=False  # 온라인에서 다운로드 허용
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3-turbo",
                cache_dir=cache_dir,
                local_files_only=False  # 온라인에서 다운로드 허용
            )
            
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print("GPU 사용 가능: STT 모델을 GPU로 이동했습니다.")
            else:
                print("GPU 사용 불가: CPU를 사용합니다.")
                
        except Exception as e:
            print(f"STT 모델 로드 중 에러 발생: {str(e)}")
            raise

    def transcribe(self, audio_data, language='ko'):
        try:
            print(f"오디오 데이터 형태: {audio_data.shape}, 타입: {audio_data.dtype}")
            
            # 오디오 데이터 처리
            input_features = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features

            if torch.cuda.is_available():
                input_features = input_features.to('cuda')

            # 음성을 텍스트로 변환 (한국어 설정)
            predicted_ids = self.model.generate(
                input_features,
                language="<|ko|>",  # 한국어 설정
                task="transcribe"   # 음성을 텍스트로 변환
            )
            
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]

            print(f"변환 결과: {transcription}")
            return transcription
            
        except Exception as e:
            print(f"음성 인식 중 에러 발생: {str(e)}")
            raise 