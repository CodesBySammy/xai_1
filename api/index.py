from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
import joblib
import os
import sys
from core import github_client, feature_extractor, model_runner, xai_explainer, logic_reviewer
# --- Path Configuration for Vercel ---
# This ensures that Python can find the 'core' folder no matter how Vercel boots it up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Now we can safely import our custom modules
from core import github_client, feature_extractor, model_runner, xai_explainer

app = FastAPI(title="XAI Commit Reviewer API")

MODEL_PATH = os.path.join(BASE_DIR, "models", "jit_defect_model.pkl")
ml_model = None

@app.on_event("startup")
def startup_event():
    """Loads the model into memory during Vercel's cold start."""
    global ml_model
    try:
        if os.path.exists(MODEL_PATH):
            ml_model = joblib.load(MODEL_PATH)
            print("✅ ML Model loaded successfully.")
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "system": "XAI Code Reviewer Engine"}

# Don't forget to import the new module at the top of api/index.py!
# from core import github_client, feature_extractor, model_runner, xai_explainer, logic_reviewer

async def process_pull_request(repo_name: str, pr_number: int):
    """
    The background task that runs the actual AI pipeline.
    """
    print(f"⚙️ Starting Hybrid XAI Pipeline for {repo_name} PR #{pr_number}")
    
    # Step 1: Fetch the files changed in this PR
    pr_files_data = github_client.get_pr_files(repo_name, pr_number)
    if not pr_files_data:
        print("No files found or error fetching files. Aborting.")
        return

    # Step 2: Convert the raw code changes into mathematical metrics
    metrics = feature_extractor.extract_features(pr_files_data)
    
    # Step 3: Run the Random Forest model to predict risk
    risk_result = model_runner.predict_risk(ml_model, metrics)
    if "error" in risk_result:
        print("Model error. Aborting.")
        return

    # Step 4: Generate the Explainable AI (SHAP) Markdown report
    explanation_markdown = xai_explainer.generate_explanation(ml_model, metrics, risk_result)

    # ---------------------------------------------------------
    # NEW STEP 5: Run the Rule-Based Logic Reviewer (AST)
    # ---------------------------------------------------------
    logic_warnings = logic_reviewer.review_code_logic(pr_files_data)
    logic_markdown = logic_reviewer.generate_logic_report(logic_warnings)
    
    # Combine the ML prediction and the Logic Review into one comment
    final_comment = f"{explanation_markdown}\n\n---\n\n{logic_markdown}"
    # ---------------------------------------------------------

    # Step 6: Post the explanation back to GitHub as a comment
    github_client.post_pr_comment(repo_name, pr_number, final_comment)
    print("🎉 Hybrid Pipeline complete!")


@app.post("/api/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives the payload from GitHub. We return a 200 OK immediately and 
    process the AI logic as a background task to prevent timeouts.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    action = payload.get("action")
    
    # We only trigger the AI when a PR is first opened, or when new code is pushed to it (synchronize)
    if "pull_request" in payload and action in ["opened", "synchronize"]:
        pr_number = payload["pull_request"]["number"]
        repo_name = payload["repository"]["full_name"]
        
        # Add the heavy lifting to a background task so GitHub gets a fast response
        background_tasks.add_task(process_pull_request, repo_name, pr_number)
        
        return {"status": "processing", "message": f"Reviewing PR #{pr_number} in the background."}

    return {"status": "ignored", "message": "Event is not PR creation or update."}