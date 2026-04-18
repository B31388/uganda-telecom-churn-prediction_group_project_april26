# Model Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score

# File paths - CORRECTED for your workspace
MODELS_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/models/'
FIGURES_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/reports/figures/'

# Create figures folder if it doesn't exist
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load models
lr_model = joblib.load(MODELS_PATH + 'lr_model.pkl')
rf_model = joblib.load(MODELS_PATH + 'rf_model.pkl')
xgb_model = joblib.load(MODELS_PATH + 'xgb_model.pkl')

# Load train and test data
X_train, X_test, y_train, y_test = joblib.load(MODELS_PATH + 'train_test_data.pkl')

print("All models and data loaded successfully!")

# Function to evaluate a model
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {name}', fontsize=13, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + f'09_confusion_matrix_{name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 09_confusion_matrix_{name.replace(' ', '_')}.png")

    return roc_auc_score(y_test, y_prob)

# Evaluate all three models
auc_lr = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
auc_rf = evaluate_model("Random Forest", rf_model, X_test, y_test)
auc_xgb = evaluate_model("XGBoost", xgb_model, X_test, y_test)

# Model Comparison summary
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
    }

summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    **{metric: [get_metrics(m, X_test, y_test)[metric] for m in [lr_model, rf_model, xgb_model]]
       for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']}
})

print("\nFull Model Comparison Summary:")
print(summary.to_string(index=False))

# Overfitting Checks
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

print("\nTraining vs Test Accuracy (Overfitting Check):")
for name, model in models.items():
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    diff = train_acc - test_acc

    print(f"{name}:")
    print(f"  Training Accuracy : {train_acc:.4f}")
    print(f"  Test Accuracy     : {test_acc:.4f}")
    print(f"  Difference        : {diff:.4f} {'Possible Overfitting' if diff > 0.05 else 'Good Generalization'}")
    print()

# Cross Validation Check
print("Cross Validation Scores (k=5, Recall):")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    print(f"{name}:")
    print(f"  CV Recall Scores  : {cv_scores.round(4)}")
    print(f"  Mean Recall       : {cv_scores.mean():.4f}")
    print(f"  Standard Dev      : {cv_scores.std():.4f}")
    print()

# ROC Curves for all models
plt.figure(figsize=(10, 6))

colors = ['steelblue', 'tomato', 'green']

for (name, model), color in zip(models.items(), colors):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_score:.4f})')

# Random classifier baseline
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Classifier')

plt.title('ROC Curve - All Models', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate (Recall)', fontsize=13)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '10_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 10_roc_curve.png")

print("\nModel Evaluation Completed Successfully.")