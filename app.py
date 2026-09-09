import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras


st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)


model = keras.models.load_model("model.keras")
scaler = joblib.load("scaler.pkl")
best_threshold = joblib.load("threshold.pkl")


st.title("💳 Credit Card Fraud Detector")
st.write("ANN-based credit card transaction fraud detection.")

st.divider()


st.subheader("Transaction Information")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0,
    step=10.0
)

time = st.number_input(
    "Transaction Time",
    min_value=0.0,
    value=0.0,
    step=1.0
)


st.divider()


if st.button("🔍 Analyze Transaction", use_container_width=True):

    v_features = {f"V{i}": 0.0 for i in range(1, 29)}

    transaction = {
        "Time": time,
        **v_features,
        "Amount": amount
    }

    input_df = pd.DataFrame([transaction])

    input_df[["Time", "Amount"]] = scaler.transform(
        input_df[["Time", "Amount"]]
    )

    probability = model.predict(
        input_df,
        verbose=0
    )[0][0]

    prediction = int(probability >= best_threshold)

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Potential Fraud Detected")
    else:
        st.success("✅ Transaction Appears Normal")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{best_threshold * 100:.2f}%"
        )

    st.progress(float(probability))

    with st.expander("Model Details"):
        st.write(f"Predicted probability: `{probability:.6f}`")
        st.write(f"Decision threshold: `{best_threshold:.6f}`")
        st.write(f"Prediction: `{'Fraud' if prediction else 'Normal'}`")