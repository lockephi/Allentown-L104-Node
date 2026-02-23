import urllib.request
import json
import time

def test_health():
    print("Testing health...")
    try:
        url = "http://localhost:8081/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print(f"Health Response: {data}")
    except Exception as e:
        print(f"Health Error: {e}")

def test_chat():
    print("Testing chat...")
    try:
        url = "http://localhost:8081/api/v6/chat"
        data = json.dumps({"message": "Hello", "local_only": True}).encode()
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"Chat Response: {data}")
    except Exception as e:
        print(f"Chat Error: {e}")

if __name__ == "__main__":
    test_health()
    test_chat()
