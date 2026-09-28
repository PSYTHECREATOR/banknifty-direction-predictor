
import gradio as gr
import pandas as pd
import numpy as np
import joblib

lin_reg = joblib.load("models/banknifty_linear_model.joblib")
log_reg = joblib.load("models/banknifty_logistic_model.joblib")
scaler  = joblib.load("models/banknifty_scaler.joblib")
feature_cols = joblib.load("models/banknifty_features.joblib")

df = pd.read_csv("data/banknifty_processed.csv")
latest = df.iloc[-1]

def predict(body, range_, ma_ratio, vol5, vol20, ret1, ret2, ret3, ret4, ret5):
    input_dict = {
        "Range": float(range_),
        "Body": float(body),
        "Return_lag1": float(ret1),
        "Range_lag1": float(latest["Range_lag1"]),
        "Return_lag2": float(ret2),
        "Range_lag2": float(latest["Range_lag2"]),
        "Return_lag3": float(ret3),
        "Range_lag3": float(latest["Range_lag3"]),
        "Return_lag4": float(ret4),
        "Range_lag4": float(latest["Range_lag4"]),
        "Return_lag5": float(ret5),
        "Range_lag5": float(latest["Range_lag5"]),
        "MA_ratio": float(ma_ratio),
        "Vol5": float(vol5),
        "Vol20": float(vol20)
    }
    X = pd.DataFrame([input_dict])[feature_cols]
    Xs = scaler.transform(X)
    
    exp_ret = float(lin_reg.predict(Xs)[0])
    proba  = float(log_reg.predict_proba(Xs)[0, 1])
    direction = "UP" if proba >= 0.5 else "DOWN"
    conf = max(proba, 1-proba)
    
    return f"""Direction Bias: {direction}
Probability of UP: {proba*100:.1f}%
Confidence: {conf*100:.1f}%
Expected Return: {exp_ret*100:.2f}%

Managerial Note: Transparent linear + logistic model.
Historical accuracy ~53%. Use only as one input among many."""

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Today Body %", value=float(latest["Body"])),
        gr.Number(label="Today Range %", value=float(latest["Range"])),
        gr.Number(label="Price / 20-day MA", value=float(latest["MA_ratio"])),
        gr.Number(label="5-day Volatility", value=float(latest["Vol5"])),
        gr.Number(label="20-day Volatility", value=float(latest["Vol20"])),
        gr.Number(label="Return lag 1", value=float(latest["Return_lag1"])),
        gr.Number(label="Return lag 2", value=float(latest["Return_lag2"])),
        gr.Number(label="Return lag 3", value=float(latest["Return_lag3"])),
        gr.Number(label="Return lag 4", value=float(latest["Return_lag4"])),
        gr.Number(label="Return lag 5", value=float(latest["Return_lag5"])),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=8),
    title="ABC Ltd – Bank Nifty Next-Day Direction Tool",
    description="Transparent Linear + Logistic Regression tool for managers"
)

if __name__ == "__main__":
    demo.launch()
    
