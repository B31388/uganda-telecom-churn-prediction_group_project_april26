# ============================================================
# SHAP INTERPRETATION (FINAL WORKING VERSION)
# ============================================================

import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'
DATA_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/processed/'
FIGURES_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/reports/figures/'

os.makedirs(FIGURES_PATH, exist_ok=True)

# ============================================================
# LOAD MODEL (FIRST!)
# ============================================================

print("Loading model...")
model = joblib.load(MODELS_PATH + 'churn_model.pkl')

# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH + 'engineered_telecom_churn.csv')

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)

# Sample for speed
X_sample = X.sample(50, random_state=42)

print("Sample shape:", X_sample.shape)

# ============================================================
# TRANSFORM DATA (NOW MODEL EXISTS)
# ============================================================

print("Transforming data...")

preprocessor = model.named_steps['preprocessor']
X_transformed = preprocessor.transform(X_sample)

# Convert sparse → dense if needed
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

print("Transformed shape:", X_transformed.shape)

# ============================================================
# SHAP EXPLAINER
# ============================================================

print("Creating SHAP explainer...")

lr_model = model.named_steps['model']

explainer = shap.Explainer(lr_model, X_transformed)

print("Computing SHAP values...")
shap_values = explainer(X_transformed)

print("SHAP computed successfully!")

# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()

# ============================================================
# PLOTS
# ============================================================

# Summary plot
shap.summary_plot(
    shap_values.values,
    X_transformed,
    feature_names=feature_names,
    show=False
)
plt.savefig(FIGURES_PATH + 'shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar plot
shap.summary_plot(
    shap_values.values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.savefig(FIGURES_PATH + 'shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# Waterfall plot
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig(FIGURES_PATH + 'shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ SHAP completed successfully!")