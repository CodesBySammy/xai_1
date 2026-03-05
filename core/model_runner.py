import warnings

def predict_risk(model, metrics: dict) -> dict:
    """
    Feeds the extracted GitHub metrics into the trained Machine Learning model.
    
    Args:
        model: The loaded scikit-learn RandomForestClassifier.
        metrics (dict): The dictionary containing 'la', 'ld', 'nf', and 'entropy'.
        
    Returns:
        dict: A dictionary containing the binary decision and confidence scores.
    """
    if not model:
        print("❌ Error: ML Model is not loaded.")
        return {"error": "Model not loaded."}

    # 1. Format the data for scikit-learn
    # The model was trained with features in this exact order: ['la', 'ld', 'nf', 'entropy']
    feature_values = [[
        metrics.get('la', 0),
        metrics.get('ld', 0),
        metrics.get('nf', 0),
        metrics.get('entropy', 0.0)
    ]]

    # 2. Run the prediction
    # We use warnings.catch_warnings() to suppress a harmless scikit-learn warning 
    # about missing feature names (because we are using a lightweight list instead of a Pandas DataFrame).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        # predict() returns an array with the binary class: [0] for Clean, [1] for Buggy
        prediction = model.predict(feature_values)[0]
        
        # predict_proba() returns the confidence percentages: [[Clean_Prob, Buggy_Prob]]
        probabilities = model.predict_proba(feature_values)[0]
        
        risk_probability = probabilities[1]
        clean_probability = probabilities[0]

    return {
        "is_buggy": bool(prediction == 1),
        "risk_score": float(round(risk_probability * 100, 2)), # Convert to a percentage (e.g., 75.5%)
        "clean_score": float(round(clean_probability * 100, 2))
    }

# --- Quick Local Test ---
if __name__ == "__main__":
    import joblib
    import os
    
    # Load the model locally just for this test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "jit_defect_model.pkl")
    
    if os.path.exists(model_path):
        test_model = joblib.load(model_path)
        
        # Simulate a high-risk commit (added 500 lines across 10 files with high entropy)
        risky_metrics = {'la': 500, 'ld': 10, 'nf': 10, 'entropy': 3.1}
        print("Testing Risky Commit:")
        print(predict_risk(test_model, risky_metrics))
        
        # Simulate a safe commit (changed 2 lines in 1 file)
        safe_metrics = {'la': 2, 'ld': 2, 'nf': 1, 'entropy': 0.0}
        print("\nTesting Safe Commit:")
        print(predict_risk(test_model, safe_metrics))
    else:
        print(f"Test skipped: Model not found at {model_path}")