import requests
import threading
import time

SERVER = "http://localhost:8081/v1/chat/completions"

def send_request(label, message, max_tokens):
    start = time.time()
    resp = requests.post(SERVER, json={
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens
    }, stream=True)
    
    # Measure time to first token
    first_token_time = None
    for chunk in resp.iter_lines():
        if chunk and first_token_time is None:
            first_token_time = time.time() - start
            break
    
    total_time = time.time() - start
    print(f"[{label}]")
    print(f"  Time to first token: {first_token_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

# Simulate: chat (short, urgent) vs background (long, not urgent)
t1 = threading.Thread(target=send_request, 
    args=("FOREGROUND chat", "Hi, how are you?", 30))
t2 = threading.Thread(target=send_request, 
    args=("BACKGROUND summarize", 
          "Write a detailed 200 word summary of the history of the Roman Empire", 
          200))

print("Sending foreground chat + background task simultaneously...")
print()
# Start background first, then foreground 0.1s later
t2.start()
time.sleep(0.1)
t1.start()

t1.join()
t2.join()