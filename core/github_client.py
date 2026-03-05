import os
import requests
from dotenv import load_dotenv

# Load the variables from the .env file into Python
load_dotenv()

# Load the GitHub token from environment variables
# (You will set this in your Vercel dashboard later)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Set up the headers required by GitHub's REST API
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else ""
}

def get_pr_files(repo_full_name: str, pr_number: int) -> list:
    """
    Fetches the list of files changed in a Pull Request.
    
    Args:
        repo_full_name (str): The owner and repo name (e.g., "octocat/Hello-World").
        pr_number (int): The ID of the Pull Request.
        
    Returns:
        list: A list of file dictionaries containing additions and deletions.
    """
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
    
    if not GITHUB_TOKEN:
        print("⚠️ Warning: GITHUB_TOKEN is not set. API rate limits will be severely restricted.")

    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error fetching PR files: {response.status_code} - {response.text}")
        return []

def post_pr_comment(repo_full_name: str, pr_number: int, comment_body: str) -> bool:
    """
    Posts a formatted markdown comment back to the GitHub Pull Request.
    
    Args:
        repo_full_name (str): The owner and repo name (e.g., "octocat/Hello-World").
        pr_number (int): The ID of the Pull Request.
        comment_body (str): The markdown-formatted SHAP explanation text.
        
    Returns:
        bool: True if the comment was posted successfully, False otherwise.
    """
    if not GITHUB_TOKEN:
        print("❌ Error: Cannot post comment without a GITHUB_TOKEN.")
        return False

    # Note: GitHub PR comments are technically tied to the "issues" endpoint
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    
    payload = {
        "body": comment_body
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    
    if response.status_code == 201:
        print(f"✅ Successfully posted review to PR #{pr_number}")
        return True
    else:
        print(f"❌ Error posting comment: {response.status_code} - {response.text}")
        return False

# --- Quick Local Test ---
if __name__ == "__main__":
    # Test this by replacing these with a real, public repository and PR number
    # For example, let's look at a random public PR from FastAPI
    test_repo = "tiangolo/fastapi"
    test_pr = 1000  # Just an example PR number
    
    print(f"Fetching files for {test_repo} PR #{test_pr}...")
    files = get_pr_files(test_repo, test_pr)
    
    if files:
        print(f"✅ Found {len(files)} files modified in this PR.")
        for f in files[:2]: # Print just the first two to keep terminal clean
            print(f" - {f.get('filename')} (+{f.get('additions')} / -{f.get('deletions')})")