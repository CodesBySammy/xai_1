import requests
import json

# The URL where your local FastAPI server is running
LOCAL_API_URL = "http://127.0.0.1:8000/api/webhook"

# A mock payload simulating a Pull Request being opened.
# We use a real public repo so your github_client.py can fetch actual diff data!
mock_payload = {
    "action": "opened",
    "pull_request": {
        "number": 1
    },
    "repository": {
        "full_name": "CodesBySammy/xai-test-repo"
    }
}

def simulate_github_webhook():
    print(f"🚀 Simulating GitHub Webhook...")
    print(f"🎯 Target: {LOCAL_API_URL}")
    
    # GitHub always sends specific headers with their webhooks
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": "pull_request"
    }
    
    try:
        # Send the POST request to your local FastAPI server
        response = requests.post(LOCAL_API_URL, json=mock_payload, headers=headers)
        
        print(f"\n📡 Response Status Code: {response.status_code}")
        print(f"📨 Response Body: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ Success! Your API accepted the webhook.")
            print("👉 Now, look at the terminal where your FastAPI server is running to watch the AI pipeline execute in the background!")
        else:
            print("\n⚠️ API rejected the webhook. Check your FastAPI code.")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not reach the server.")
        print("Did you forget to start your API? Run this in another terminal first:")
        print("uvicorn api.index:app --reload")

if __name__ == "__main__":
    simulate_github_webhook()