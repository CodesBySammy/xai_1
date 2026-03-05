import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define file paths
DATA_PATH = "../data/apachejit_train.csv"
MODEL_DIR = "../models"
MODEL_PATH = f"{MODEL_DIR}/jit_defect_model.pkl"

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    print("✅ Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # --- THE FIX: Normalize column names ---
    # 1. Convert all column names to lowercase and strip extra spaces
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Map variations of column names to our standard names
    column_mapping = {
        'buggy': 'bug',
        'is_bug': 'bug',
        'ent': 'entropy'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    print(f"🔍 Detected columns: {list(df.columns)}")
    # ---------------------------------------

    # Define the exact features we need
    features = ['la', 'ld', 'nf', 'entropy']
    target = 'bug'

    # Verify columns exist after normalization
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Still missing required columns: {missing_cols}")
        print("Please check the 'Detected columns' list above to see what names are actually in the CSV.")
        return

    # Prepare the data
    X = df[features].fillna(0)
    y = df[target]

    print("✅ Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    print("\n📊 Model Evaluation:")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n🚀 Success! Model saved locally to: {MODEL_PATH}")

if __name__ == "__main__":
    current_dir = os.path.basename(os.getcwd())
    if current_dir != "scripts":
        print("⚠️ Please run this script from inside the 'scripts/' directory.")
    else:
        train_and_save_model()