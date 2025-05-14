import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

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

# Load data to show metrics
@st.cache_data
def load_data():
    df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
    df["type"] = df["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
    df["isFraud"] = df["isFraud"].map({0: 0, 1: 1})
    X = df[["amount", "oldbalanceOrg", "newbalanceOrig"]]
    y = df["isFraud"]
    return X, y

# Sidebar metrics
X, y = load_data()
y_pred = model.predict(X)

precision = precision_score(y, y_pred)
accuracy = accuracy_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

st.sidebar.header("📊 Model Performance Metrics")
st.sidebar.write(f"**Precision:** {precision:.2f}")
st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
st.sidebar.write(f"**Recall:** {recall:.2f}")
st.sidebar.write(f"**F1 Score:** {f1:.2f}")
