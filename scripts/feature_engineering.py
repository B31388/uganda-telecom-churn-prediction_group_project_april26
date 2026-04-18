# ============================================================
# FEATURE ENGINEERING (FINAL)
# ============================================================

import pandas as pd
import os

DATA_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/'
PROCESSED_PATH = DATA_PATH + 'processed/'

print("Loading dataset...")
df = pd.read_csv(PROCESSED_PATH + 'preprocessed_telecom_churn.csv')

# Avoid zero division
df['tenure'] = df['tenure'].replace(0, 1)

# New features
df['Avg_Monthly_Spend'] = df['TotalCharges'] / df['tenure']
df['Charge_per_Tenure'] = df['MonthlyCharges'] / df['tenure']

# Convert service columns to numeric
service_cols = [
    'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
]

for col in service_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

df['Service_Count'] = df[service_cols].sum(axis=1)

# Save
df.to_csv(PROCESSED_PATH + 'engineered_telecom_churn.csv', index=False)

print("✅ Feature engineering complete")