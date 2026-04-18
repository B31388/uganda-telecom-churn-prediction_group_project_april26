# Feature engineering and scaling.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# File paths - Corrected for your workspace
DATA_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/'
PROCESSED_PATH = DATA_PATH + 'processed/'
MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'

# Create models folder if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)

# Load preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv(PROCESSED_PATH + 'preprocessed_telecom_churn.csv')
print(f"Dataset shape on load: {df.shape}")


# Feature Engineering.

# 1. Average Monthly Spend
# Total charges divided by tenure (how much a customer spends per month on average)
# Avoid division by zero by replacing 0 tenure with 1
df['Avg_Monthly_Spend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

# 2. Tenure to Charge Ratio
# How long a customer has stayed relative to what they pay monthly
# Low ratio = high charges but short tenure = higher churn risk
df['Tenure_to_Charge_Ratio'] = df['tenure'] / df['MonthlyCharges']

# 3. Service Count
# Total number of services a customer has subscribed to
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Fix: Convert Yes/No columns to numeric before summing (prevents string concatenation)
for col in service_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

df['Service_Count'] = df[service_cols].sum(axis=1)

# Confirm new features
print("New Features Sample:")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges','Avg_Monthly_Spend', 'Tenure_to_Charge_Ratio','Service_Count']].head(10))

print("Shape after Feature Engineering:")
print(df.shape)

# Save the engineered dataset
df.to_csv(PROCESSED_PATH + 'engineered_telecom_churn.csv', index=False)
print(f"Engineered dataset saved to: {PROCESSED_PATH}engineered_telecom_churn.csv")

print("Feature Engineering completed successfully!")