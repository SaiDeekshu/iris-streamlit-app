# app.py
import streamlit as st
import numpy as np
import joblib
import os

MODEL_PATH = "artifacts/iris_model.joblib"

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements to predict the species.")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]
class_names = bundle["class_names"]

# Typical ranges from Iris dataset
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.5, 5.8, 0.1)
sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.35, 0.05)
petal_width  = st.slider("Petal Width (cm)",  0.0, 2.6, 1.30, 0.05)

if st.button("Predict"):
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Respect trained feature order
    # feature_names should be ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]

    st.subheader("Prediction")
    st.write(f"**Species:** {pred}")

    st.subheader("Confidence")
    prob_table = {cls: f"{p*100:.1f}%" for cls, p in zip(class_names, proba)}
    st.write(prob_table)

st.caption("Model: StandardScaler + LogisticRegression (scikit-learn)")
