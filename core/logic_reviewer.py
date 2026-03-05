import requests
from radon.complexity import cc_visit

def review_code_logic(pr_files_data: list) -> list:
    """
    Downloads the modified Python files and calculates the cyclomatic complexity 
    of every function and class inside them using Abstract Syntax Trees (AST).
    """
    warnings = []
    
    for file_data in pr_files_data:
        filename = file_data.get("filename", "")
        
        # We only want to analyze Python files
        if not filename.endswith(".py"):
            continue
            
        # GitHub provides a raw URL to download the exact text of the file
        raw_url = file_data.get("raw_url")
        if not raw_url:
            continue
            
        try:
            # Fetch the raw code
            response = requests.get(raw_url)
            if response.status_code != 200:
                continue
                
            code_content = response.text
            
            # cc_visit parses the Python code into an Abstract Syntax Tree
            # and calculates the complexity of every block of code.
            blocks = cc_visit(code_content)
            
            # A complexity of 1-5 is good. Anything above 5 gets risky.
            # Let's flag anything >= 6 so it catches our junk code!
            for block in blocks:
                if block.complexity >= 6:
                    warnings.append({
                        "file": filename,
                        "name": block.name,
                        "complexity": block.complexity,
                        "line": block.lineno
                    })
        except Exception as e:
            print(f"⚠️ Could not parse logic for {filename}: {e}")
            
    return warnings

def generate_logic_report(warnings: list) -> str:
    """Formats the warnings into a clean Markdown section for GitHub."""
    if not warnings:
        return "### 🔬 Logic Review (Cyclomatic Complexity)\n✅ **Clean Code:** All functions in the modified Python files maintain an acceptable, simple logic flow."
        
    report = "### 🔬 Logic Review (Cyclomatic Complexity)\n"
    report += "⚠️ **Warning: High Logic Complexity Detected!**\n"
    report += "The following functions are too complex. Consider breaking them down into smaller functions to prevent bugs:\n\n"
    
    for w in warnings:
        report += f"* **Function/Class:** `{w['name']}` in `{w['file']}` (Line {w['line']}) -> **Complexity Score: {w['complexity']}**\n"
        
    return report

# --- Quick Local Test ---
if __name__ == "__main__":
    # A dummy payload pointing to a real file in a public repo
    test_payload = [{
        "filename": "test.py", 
        "raw_url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/fastapi/applications.py"
    }]
    warns = review_code_logic(test_payload)
    print(generate_logic_report(warns))