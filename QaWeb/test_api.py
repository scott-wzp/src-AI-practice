import requests
import json

def test_ask_question():
    url = "http://127.0.0.1:5000/api/answer"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "question": "客户经理被投诉了，投诉一次扣多少分"
    }
    response = requests.post(url, headers=headers, json=data)
    print("Request URL:", url)
    print("Request Headers:", headers)
    print("Request Data:", json.dumps(data, ensure_ascii=False))
    print("Response Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Content:", response.text)
    try:
        print("Response JSON:", response.json())
    except json.JSONDecodeError as e:
        print("Failed to decode JSON response:", str(e))

if __name__ == "__main__":
    # 测试提问
    test_ask_question() 