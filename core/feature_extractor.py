import math

def extract_features(pr_files_data: list) -> dict:
    """
    Extracts 'la', 'ld', 'nf', and 'entropy' metrics from GitHub PR files.
    
    Args:
        pr_files_data (list): A list of dictionaries returned by the GitHub API.
        Example: [
            {"filename": "src/main.py", "additions": 25, "deletions": 5},
            {"filename": "src/utils.py", "additions": 10, "deletions": 0}
        ]
        
    Returns:
        dict: The exact 4 features required by the trained Random Forest model.
    """
    if not pr_files_data:
        # If the PR is empty, return zeros
        return {'la': 0, 'ld': 0, 'nf': 0, 'entropy': 0.0}

    # 1. Calculate base metrics
    la = sum(file_obj.get('additions', 0) for file_obj in pr_files_data)
    ld = sum(file_obj.get('deletions', 0) for file_obj in pr_files_data)
    nf = len(pr_files_data)
    
    total_changes = la + ld
    entropy = 0.0
    
    # 2. Calculate Shannon Entropy
    if total_changes > 0:
        for file_obj in pr_files_data:
            file_changes = file_obj.get('additions', 0) + file_obj.get('deletions', 0)
            
            if file_changes > 0:
                # p_i is the proportion of total changes that happened in this specific file
                probability = file_changes / total_changes
                entropy -= probability * math.log2(probability)
                
    return {
        'la': la,
        'ld': ld,
        'nf': nf,
        'entropy': round(entropy, 4) # Rounded for cleaner SHAP outputs later
    }

# --- Quick Local Test ---
if __name__ == "__main__":
    # Simulating a GitHub payload where a developer changed 3 files
    mock_github_payload = [
        {"filename": "app.py", "additions": 100, "deletions": 20},
        {"filename": "config.json", "additions": 5, "deletions": 1},
        {"filename": "README.md", "additions": 50, "deletions": 0}
    ]
    
    metrics = extract_features(mock_github_payload)
    print("Extracted Features for the AI:")
    print(metrics) 
    # Expected output: {'la': 155, 'ld': 21, 'nf': 3, 'entropy': 1.13...}