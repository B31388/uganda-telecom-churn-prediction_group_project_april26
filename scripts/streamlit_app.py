# ============================================================
# STREAMLIT APP (FINAL CLEAN VERSION)
# ============================================================

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📱 Telecom Churn Risk Predictor")

# Load model
model = joblib.load('/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/churn_model.pkl')
THRESHOLD = joblib.load('/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/threshold.pkl')

# =========================
# INPUTS
# =========================
st.subheader("Customer Information")

tenure = st.slider("Tenure", 1, 72, 3)
monthly_charges = st.slider("Monthly Charges", 1000, 150000, 100000)
total_charges = st.number_input("Total Charges", 0.0, 20000000.0, 200000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Services
online_security = st.selectbox("Online Security", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])

# =========================
# PREDICT
# =========================
if st.button("🔮 Predict Risk"):

    input_df = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'InternetService': [internet_service],
        'PaymentMethod': [payment_method],
        'PaperlessBilling': [paperless_billing],
        'PhoneService': [1],
        'MultipleLines': [1 if multiple_lines == "Yes" else 0],
        'OnlineSecurity': [1 if online_security == "Yes" else 0],
        'OnlineBackup': [1 if online_backup == "Yes" else 0],
        'DeviceProtection': [1 if device_protection == "Yes" else 0],
        'TechSupport': [1 if tech_support == "Yes" else 0],
        'StreamingTV': [1 if streaming_tv == "Yes" else 0],
        'StreamingMovies': [1 if streaming_movies == "Yes" else 0],
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No']
    })

    # Feature engineering
    input_df['tenure'] = input_df['tenure'].replace(0, 1)
    input_df['Avg_Monthly_Spend'] = input_df['TotalCharges'] / input_df['tenure']
    input_df['Charge_per_Tenure'] = input_df['MonthlyCharges'] / input_df['tenure']

    service_cols = [
        'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]

    input_df['Service_Count'] = input_df[service_cols].sum(axis=1)

    # =========================
    # PREDICT (PIPELINE HANDLES EVERYTHING)
    # =========================
    prob = model.predict_proba(input_df)[0][1]

    # =========================
    # OUTPUT
    # =========================
    st.metric("Churn Probability", f"{prob:.2%}")
    st.write("Raw probability:", prob)

    if prob > THRESHOLD:
        st.error("🚨 HIGH RISK")
    else:
        st.success("✅ LOW RISK")

    # Risk bands
    if prob > 0.75:
        st.write("🔴 Very High Risk")
    elif prob > 0.55:
        st.write("🔴 High Risk")
    elif prob > 0.35:
        st.write("🟠 Medium Risk")
    else:
        st.write("🟢 Low Risk")