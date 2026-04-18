# PREPROCESSING - TELECOM CUSTOMER CHURN

import pandas as pd
import numpy as np

# CONFIG
DATA_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/'

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH + 'telecom_churn_dataset.csv')

print(f"Dataset shape: {df.shape}")
print("First 5 rows:")
print(df.head())

# Basic information
print("Dataset Info:")
df.info()

print("Data Types:")
print(df.dtypes)

print("Missing Values:")
print(df.isnull().sum())

print("Duplicate Rows:", df.duplicated().sum())

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing values if any in TotalCharges
if df['TotalCharges'].isnull().sum() > 0:
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print("Missing Values after handling TotalCharges:")
print(df.isnull().sum())

# Drop customerID if exists (usually not useful for modeling)
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

print(f"Final shape after basic preprocessing: {df.shape}")

# Save the preprocessed data
PROCESSED_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/processed/'
import os
os.makedirs(PROCESSED_PATH, exist_ok=True)

df.to_csv(PROCESSED_PATH + 'preprocessed_telecom_churn.csv', index=False)
print(f"Preprocessed data saved to: {PROCESSED_PATH}preprocessed_telecom_churn.csv")

print("Preprocessing completed!")