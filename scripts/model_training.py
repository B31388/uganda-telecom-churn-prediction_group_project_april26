# ============================================================
# MODEL TRAINING - BULLETPROOF VERSION
# ============================================================

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/processed/'
MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'

os.makedirs(MODELS_PATH, exist_ok=True)

print("Loading engineered dataset...")
df = pd.read_csv(PROCESSED_PATH + 'engineered_telecom_churn.csv')

print(f"Original shape: {df.shape}")

# === CRITICAL: Convert Churn to numeric FIRST and remove it from encoding ===
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print("Converted Churn to numeric (0/1)")

# Now drop Churn so it is NOT encoded
y = df['Churn'].copy()
X = df.drop('Churn', axis=1)

# Encode only the remaining categorical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
if categorical_cols:
    print("Encoding categorical columns:", categorical_cols)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Shape after encoding: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
])

print("Training the model...")
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, MODELS_PATH + 'churn_model.pkl')
joblib.dump(list(X.columns), MODELS_PATH + 'feature_names.pkl')

print("\n✅ Model trained successfully!")
print(f"Training Accuracy : {pipeline.score(X_train, y_train):.4f}")
print(f"Test Accuracy     : {pipeline.score(X_test, y_test):.4f}")
print(f"Model saved as: churn_model.pkl")