import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title("💸 Fraud Detection App")
st.write("Enter transaction details to check if it's fraudulent.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0)
old_balance = st.number_input("Old Balance (Sender)", min_value=0.0)
new_balance = st.number_input("New Balance (Sender)", min_value=0.0)

# Predict fraud
if st.button("Check for Fraud"):
    input_data = np.array([[amount, old_balance, new_balance]])
    prediction = model.predict(input_data)
    result = "🔴 Fraud Detected!" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.subheader(result)

# Sidebar metrics (hardcoded)
st.sidebar.header("📊 Model Performance Metrics")
st.sidebar.write("**Precision:** 1.0")  # Replace with your actual precision
st.sidebar.write("**Accuracy:** 0.99")
st.sidebar.write("**Recall:** 0.99")
st.sidebar.write("**F1 Score:** 0.99")
