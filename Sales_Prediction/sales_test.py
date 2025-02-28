import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("sales_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define advertising platforms and their impact multipliers
platforms = {
    "TV": [1, 0, 0],  # Only TV ads
    "Radio": [0, 1, 0],  # Only Radio ads
    "Newspaper": [0, 0, 1],  # Only Newspaper ads
    "TV + Radio": [0.5, 0.5, 0],  # Equal split between TV & Radio
    "TV + Newspaper": [0.5, 0, 0.5],  # Equal split between TV & Newspaper
    "Radio + Newspaper": [0, 0.5, 0.5],  # Equal split between Radio & Newspaper
    "All Platforms": [0.33, 0.33, 0.33],  # Even distribution
}

# Streamlit UI
st.title("📊 Sales Prediction Dashboard")

st.markdown("### Enter Advertising Expenditure")
budget = st.number_input("Total Advertising Budget ($)", min_value=0, step=1000, value=5000)

st.markdown("### Select Advertising Platform")
selected_platform = st.selectbox("Choose an Advertising Strategy", list(platforms.keys()))

# Predict sales when user clicks the button
if st.button("Predict Sales"):
    # Calculate actual spending distribution based on platform choice
    multipliers = platforms[selected_platform]
    ad_spends = np.array([budget * m for m in multipliers]).reshape(1, -1)

    # Scale the input data
    ad_spends_scaled = scaler.transform(ad_spends)

    # Predict sales
    predicted_sales = model.predict(ad_spends_scaled)[0]

    # Display result
    st.success(f"📈 **Predicted Sales: {predicted_sales:.2f} units**")

