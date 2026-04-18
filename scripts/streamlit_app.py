import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Uganda Telecom Churn", layout="centered")
st.title("📱 Uganda Telecom Customer Churn Predictor")
st.markdown("**In-Class Demonstration**")

model = joblib.load('/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/churn_model.pkl')

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 4)
    monthly_charges = st.slider("Monthly Charges (UGX)", 1000, 150000, 105000, step=500)
    total_charges = st.number_input("Total Charges (UGX)", 0.0, 15000000.0, 420000.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"], index=2)
    payment_method = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)

st.subheader("Services")
col3, col4 = st.columns(2)
with col3:
    online_security = st.selectbox("Online Security", ["No", "Yes"], index=0)
    tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0)
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0)
with col4:
    device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=0)
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=0)

if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    
    input_df = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'InternetService': [internet_service],
        'PaymentMethod': [payment_method],
        'PaperlessBilling': [paperless_billing],
        'OnlineSecurity': [online_security],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'DeviceProtection': [device_protection],
        'MultipleLines': ["No"],
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'PhoneService': ['Yes'],
        'OnlineBackup': ['No']
    })

    # Feature Engineering
    input_df['Avg_Monthly_Spend'] = input_df['TotalCharges'] / input_df['tenure'].replace(0, 1)
    input_df['Tenure_to_Charge_Ratio'] = input_df['tenure'] / input_df['MonthlyCharges']
    input_df['Service_Count'] = input_df[['OnlineSecurity','TechSupport','StreamingTV',
                                          'StreamingMovies','DeviceProtection']].apply(
        lambda x: x.map({'Yes':1, 'No':0}).sum(), axis=1)

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align with model
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f"🚨 HIGH RISK — This customer is likely to CHURN")
        st.metric("Churn Probability", f"{prob:.1%}")
    else:
        st.success(f"✅ LOW RISK — This customer is likely to STAY")
        st.metric("Churn Probability", f"{prob:.1%}")

    risk = "🔴 High Risk" if prob > 0.70 else "🟠 Medium Risk" if prob > 0.45 else "🟢 Low Risk"
    st.info(f"**Risk Level:** {risk}")

    st.subheader("Why this prediction?")
    if prob < 0.45:
        st.success("Low risk factors detected.")
    else:
        st.warning("High risk factors:")
        if contract == "Month-to-month":
            st.write("• Month-to-month contract")
        if tenure < 12:
            st.write("• Very short tenure")
        if monthly_charges > 70000:
            st.write("• High monthly charges")
        if internet_service == "Fiber optic":
            st.write("• Fiber optic service")
        if payment_method == "Electronic check":
            st.write("• Electronic check payment method")

st.caption("Uganda Telecom Churn Prediction Project")