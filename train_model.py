# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib, os, sys

CSV_PATH = "Iris.csv"  # keep your Kaggle filename here

def main():
    # 1) Load data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Couldn't find {CSV_PATH}. Put it next to this script.")
        sys.exit(1)

    print("Columns:", df.columns.tolist())

    # 2) Clean/prepare
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Expected columns for Kaggle Iris
    expected = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
    for col in expected:
        if col not in df.columns:
            print(f"❌ Missing column '{col}' in CSV.")
            sys.exit(1)

    X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    y = df["Species"]
    class_names = sorted(y.unique().tolist())

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Pipeline: scale + logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # 5) Train
    pipe.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    # 7) Save
    os.makedirs("artifacts", exist_ok=True)
    payload = {
        "model": pipe,
        "feature_names": X.columns.tolist(),
        "class_names": class_names
    }
    joblib.dump(payload, "artifacts/iris_model.joblib")
    print("💾 Saved model → artifacts/iris_model.joblib")

if __name__ == "__main__":
    main()
