import requests
import threading
import time

def send_request(label, message, max_tokens=100):
    start = time.time()
    resp = requests.post("http://localhost:8081/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens
        })
    elapsed = time.time() - start
    print(f"[{label}] finished in {elapsed:.2f}s")
    print(f"[{label}] reply: {resp.json()['choices'][0]['message']['content'][:80]}")
    print()

# Send two requests simultaneously
t1 = threading.Thread(target=send_request, args=("Request 1", "What is the capital of France?", 50))
t2 = threading.Thread(target=send_request, args=("Request 2", "What is the capital of Japan?", 50))

print("Sending two requests simultaneously...")
t1.start()
t2.start()
t1.join()
t2.join()