import shap
import numpy as np

# A dictionary to make our raw metric names look professional in the GitHub comment
FEATURE_NAMES = {
    'la': 'Lines Added',
    'ld': 'Lines Deleted',
    'nf': 'Files Modified',
    'entropy': 'Change Entropy (Scattered Logic)'
}

def generate_explanation(model, metrics: dict, risk_result: dict) -> str:
    """
    Uses SHAP to explain exactly why the model generated its risk score
    and formats the output as a Markdown comment for GitHub.
    """
    feature_order = ['la', 'ld', 'nf', 'entropy']
    X_instance = np.array([[metrics.get(f, 0) for f in feature_order]])
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)
    
    # --- THE FIX: Safely extract the 1D array of weights for the "Buggy" class ---
    if isinstance(shap_values, list):
        buggy_impacts = shap_values[1][0] 
    elif len(np.array(shap_values).shape) == 3:
        # Handles SHAP v0.45+ output shape: (samples, features, classes) -> (1, 4, 2)
        buggy_impacts = shap_values[0, :, 1]
    else:
        buggy_impacts = shap_values[0] 
        
    # Ensure it is a completely flat list of native floats
    buggy_impacts = np.array(buggy_impacts).flatten()
    # -----------------------------------------------------------------------------

    impact_data = []
    for i, feature_key in enumerate(feature_order):
        impact_data.append({
            'name': FEATURE_NAMES[feature_key],
            'actual_value': metrics[feature_key],
            'weight': float(buggy_impacts[i]) # Force conversion to native Python float
        })

    # This will now sort perfectly without crashing!
    impact_data.sort(key=lambda x: abs(x['weight']), reverse=True)

    is_buggy = risk_result['is_buggy']
    risk_score = risk_result['risk_score']
    
    if is_buggy:
        markdown = f"## ⚠️ XAI Code Review: High Risk Commit Detected\n"
        markdown += f"**Risk Score:** {risk_score}% probability of introducing a defect.\n\n"
    else:
        markdown = f"## ✅ XAI Code Review: Clean Commit\n"
        markdown += f"**Risk Score:** {risk_score}% (Low Risk).\n\n"

    markdown += "### 🧠 Explainable AI Breakdown (SHAP)\n"
    markdown += "Here is exactly how the AI weighted your changes:\n\n"

    for item in impact_data:
        weight = item['weight']
        impact_percent = round(weight * 100, 2)
        
        if weight > 0:
            icon = "🔴"
            action = "Increased risk by"
        else:
            icon = "🟢"
            action = "Decreased risk by"
            
        markdown += f"* {icon} **{item['name']}** ({item['actual_value']}): {action} {abs(impact_percent)}%\n"

    markdown += "\n---\n*Powered by a local Random Forest model + SHAP. No third-party APIs were used.*"
    
    return markdown

# --- Quick Local Test ---
if __name__ == "__main__":
    import joblib
    import os
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "jit_defect_model.pkl")
    
    if os.path.exists(model_path):
        test_model = joblib.load(model_path)
        
        # Test with a highly risky commit
        test_metrics = {'la': 450, 'ld': 5, 'nf': 8, 'entropy': 2.8}
        test_risk = {"is_buggy": True, "risk_score": 85.5, "clean_score": 14.5}
        
        print(generate_explanation(test_model, test_metrics, test_risk))
    else:
        print("Model not found. Cannot run test.")