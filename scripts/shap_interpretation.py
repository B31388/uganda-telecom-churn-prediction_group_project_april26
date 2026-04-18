# SHAP interpretation - Using Tuned Logistic Regression Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# File paths
MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'
FIGURES_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/reports/figures/'

# Create figures folder if it doesn't exist
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load the TUNED model (churn_model.pkl)
print("Loading tuned model and data...")
best_lr_model = joblib.load(MODELS_PATH + 'churn_model.pkl')
X_train, X_test, y_train, y_test = joblib.load(MODELS_PATH + 'train_test_data.pkl')
print("Tuned model and data loaded successfully!")

# Sample for speed
X_test_sample = X_test.sample(100, random_state=42)

# Create SHAP explainer for Logistic Regression (LinearExplainer is best & fastest)
print("Computing SHAP values...")
explainer = shap.LinearExplainer(best_lr_model.named_steps['logisticregression'], 
                                 best_lr_model.named_steps['standardscaler'].transform(X_train))

# Get SHAP values
shap_values = explainer.shap_values(best_lr_model.named_steps['standardscaler'].transform(X_test_sample))

print("SHAP values shape:", shap_values.shape)

# Summary Plot (Beeswarm)
print("Generating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test_sample, 
                  feature_names=X_train.columns.tolist(), 
                  show=False)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '11_shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 11_shap_summary_plot.png")

# Bar Plot (Global Feature Importance)
print("\nGenerating SHAP Bar Plot...")
shap.summary_plot(shap_values, X_test_sample, 
                  feature_names=X_train.columns.tolist(), 
                  plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '12_shap_bar_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 12_shap_bar_plot.png")

print("\nSHAP Interpretation using Tuned Model Completed Successfully!")