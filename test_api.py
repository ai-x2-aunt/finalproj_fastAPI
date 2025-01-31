import requests
import json

def test_chat_api():
    url = "http://localhost:8000/chat"
    
    # 테스트용 메시지
    payload = {
        "message": "경비직 채용 정보를 찾고 있습니다."
    }
    
    # POST 요청 보내기
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chat_api() 