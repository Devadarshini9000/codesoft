import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("cscaler.pkl")

# Streamlit UI
st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraudulent or genuine.")

# User Inputs
amount = st.number_input("Transaction Amount ($)", min_value=0.01, step=0.01, value=10.00)

# Enter PCA components
st.markdown("### Enter PCA Feature Values")
pca_values = []
for i in range(1, 29):
    pca_values.append(st.number_input(f"V{i}", min_value=-10.0, max_value=10.0, value=0.0))

# Predict button
if st.button("Check Transaction"):
    # Normalize amount
    amount_scaled = scaler.transform([[amount]])[0][0]

    # Prepare input for model
    input_data = np.array([0] + pca_values + [amount_scaled]).reshape(1, -1)

    # Predict fraud or genuine
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error(" **Fraudulent Transaction Detected!**")
    else:
        st.success(" **Transaction is Genuine.**")
