# app.py
import streamlit as st
import pickle
import numpy as np

model  = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💧 Water Potability Predictor")
st.write("Enter physicochemical measurements to check if water is safe to drink.")

# Correct ranges derived from dataset statistics
feature_config = {
    "ph":               {"min": 0.0,    "max": 14.0,    "default": 7.0,     "step": 0.1},
    "Hardness":         {"min": 47.0,   "max": 323.0,   "default": 196.0,   "step": 1.0},
    "Solids":           {"min": 320.0,  "max": 61227.0, "default": 20927.0, "step": 100.0},
    "Chloramines":      {"min": 0.35,   "max": 13.13,   "default": 7.12,    "step": 0.1},
    "Sulfate":          {"min": 129.0,  "max": 481.0,   "default": 333.0,   "step": 1.0},
    "Conductivity":     {"min": 181.0,  "max": 753.0,   "default": 426.0,   "step": 1.0},
    "Organic_carbon":   {"min": 2.2,    "max": 28.3,    "default": 14.28,   "step": 0.1},
    "Trihalomethanes":  {"min": 0.738,  "max": 124.0,   "default": 66.4,    "step": 0.1},
    "Turbidity":        {"min": 1.45,   "max": 6.74,    "default": 3.97,    "step": 0.01},
}

inputs = []
for feature, cfg in feature_config.items():
    val = st.slider(
        feature,
        min_value=cfg["min"],
        max_value=cfg["max"],
        value=cfg["default"],
        step=cfg["step"]
    )
    inputs.append(val)

if st.button("Predict"):
    X_input = scaler.transform([inputs])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]
    if pred == 1:
        st.success(f"✅ Potable — Safe to drink ({prob:.1%} confidence)")
    else:
        st.error(f"❌ Not Potable — Unsafe ({1 - prob:.1%} confidence)")
