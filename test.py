import torch
print(torch.__version__)           # 파이토치 버전 확인
print(torch.cuda.is_available())   # True면 CUDA 사용 가능
print(torch.cuda.device_count())   # 사용 가능한 GPU 개수 출력
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # GPU 모델명 출력
    print(torch.cuda.current_device())    # 현재 사용중인 GPU 번호 출력

