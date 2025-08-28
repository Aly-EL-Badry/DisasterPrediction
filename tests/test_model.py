import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# Paths
CTB_MODEL_PATH = "./Artifacts/ctb-model.pkl"
XG_MODEL_PATH = "./Artifacts/xgb-model.pkl"
TEST_DATA_PATH = "./data/test/test_data.csv"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(file_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix for {model_name} at {file_path}")

def evaluate_model(model, X, y, model_name):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds)
    
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    
    plot_confusion_matrix(cm, model_name)
    return acc

def test_models():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Load models
    ctb_model = load_model(CTB_MODEL_PATH)
    xg_model = load_model(XG_MODEL_PATH)

    # Evaluate both models
    acc_ctb = evaluate_model(ctb_model, X, y, "CatBoost_Model")
    acc_xg = evaluate_model(xg_model, X, y, "XGBoost_Model")

    # Basic assertion: accuracy > 0.5
    assert acc_ctb > 0.5, "CatBoost accuracy too low!"
    assert acc_xg > 0.5, "XGBoost accuracy too low!"

if __name__ == "__main__":
    test_models()
