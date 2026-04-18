# Hyper parameter tuning
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# File paths - CORRECTED for your workspace
MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'

# Create models folder if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)

# Load train and test data
print("Loading train and test data...")
X_train, X_test, y_train, y_test = joblib.load(MODELS_PATH + 'train_test_data.pkl')
print("Data loaded successfully!")

# Identify numerical columns for scaling (important for Logistic Regression)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Spend', 
                  'Tenure_to_Charge_Ratio', 'Service_Count']

# Create a pipeline with scaling + Logistic Regression
# This solves the ConvergenceWarning you saw earlier
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)

# Define parameter grid (C and penalty work well with scaled data)
param_grid = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logisticregression__penalty': ['l2'],           # 'l1' removed because liblinear + scaling is tricky
    'logisticregression__solver': ['lbfgs']
}

# GridSearchCV with Recall as primary scoring metric (good for churn)
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1
)

# Fit on training data
print("\nRunning GridSearchCV for Logistic Regression...")
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Recall Score:")
print(round(grid_search.best_score_, 4))

# Evaluate tuned model on test set
best_lr_model = grid_search.best_estimator_

y_pred_tuned = best_lr_model.predict(X_test)
y_prob_tuned = best_lr_model.predict_proba(X_test)[:, 1]

print("\nTuned Model Performance on Test Set:")
print(classification_report(y_test, y_pred_tuned, target_names=['No Churn', 'Churn']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_tuned):.4f}")

# Save the tuned model
joblib.dump(best_lr_model, MODELS_PATH + 'churn_model.pkl')
print("\nSaved: churn_model.pkl - Tuned Logistic Regression Model")

print("\n--- Hyperparameter Tuning Completed Successfully! ---")