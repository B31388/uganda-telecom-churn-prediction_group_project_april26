# ============================================================
# MODEL TRAINING (FINAL PIPELINE VERSION)
# ============================================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

PROCESSED_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/processed/'
MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'

os.makedirs(MODELS_PATH, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(PROCESSED_PATH + 'engineered_telecom_churn.csv')

# Target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

y = df['Churn']
X = df.drop('Churn', axis=1)

# Columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols)

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=2000))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
print("Training...")
pipeline.fit(X_train, y_train)

# Check probabilities
probs = pipeline.predict_proba(X_test)[:,1]
print("MIN:", probs.min())
print("MAX:", probs.max())
print("MEAN:", probs.mean())

# Save
THRESHOLD = 0.35

joblib.dump(pipeline, MODELS_PATH + 'churn_model.pkl')
joblib.dump(THRESHOLD, MODELS_PATH + 'threshold.pkl')

print("✅ Model saved successfully!")