# EXPLORATORY DATA ANALYSIS - TELECOM CUSTOMER CHURN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Figures save path
FIGURES_PATH = '/workspaces/uganda-telecom-churn-prediction_group_project_april26/scripts/reports/figures'

# Load dataset
print("Loading dataset...")
df = pd.read_csv('/workspaces/uganda-telecom-churn-prediction_group_project_april26/data/telecom_churn_dataset.csv')

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

# Check for blank/whitespace in object columns
print("Whitespace/Blank Entries:")
for col in df.select_dtypes(include='object').columns:
    blank_count = df[df[col].str.strip() == ''].shape[0]
    if blank_count > 0:
        print(f"{col}: {blank_count} blank entries")
    else:
        print(f"{col}: No blanks")


# CHURN DISTRIBUTION - TARGET VARIABLE
print("Churn Value Counts")
print(df['Churn'].value_counts()) # counts the customers

print("Churn Percentage")
print(df['Churn'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%') # subject the counts to percentages

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#bar graph
df['Churn'].value_counts().plot(kind='bar', ax=axes[0],color=['steelblue', 'tomato'], edgecolor='black')
axes[0].set_title('Churn Count', fontsize=14)
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Number of Customers')
axes[0].tick_params(axis='x', rotation=0)

#pie chart
df['Churn'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['steelblue', 'tomato'],  startangle=90)
axes[1].set_title('Churn Proportion', fontsize=14)
axes[1].set_ylabel('')

plt.suptitle('Target Variable: Churn Distribution', fontsize=16, fontweight='bold') # big title for the image
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_churn_distribution.png', dpi=150, bbox_inches='tight') # saving the image
plt.close()
print("Saved: 01_churn_distribution.png")


# NUMERICAL FEATURES DISTRIBUTION
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges'] # create a list of 3 columns to analyze

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {col}', fontsize=13)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.suptitle('Numerical Features Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '02_numerical_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_numerical_distribution.png")

print("Summary Statistics for Numerical Features")
print(df[numerical_cols].describe().round(2))


# CATEGORICAL FEATURES DISTRIBUTION
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    df[col].astype(str).value_counts().plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)

plt.suptitle('Categorical Features Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '03_categorical_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_categorical_distribution.png")


# CHURN VS NUMERICAL FEATURES (BOXPLOTS)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for i, col in enumerate(numerical_cols):
    df.boxplot(column=col, by='Churn', ax=axes[i],boxprops=dict(color='steelblue'),medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(f'{col} by Churn', fontsize=13)
    axes[i].set_xlabel('Churn')
    axes[i].set_ylabel(col)

plt.suptitle('Churn vs Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '04_churn_vs_numerical.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_churn_vs_numerical.png")

print("Mean Values by Churn Group")
print(df.groupby('Churn')[numerical_cols].mean().round(2))


# CHURN VS CATEGORICAL FEATURES
fig, axes = plt.subplots(4, 4, figsize=(22, 18))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    churn_counts = df.groupby([col, 'Churn']).size().unstack(fill_value=0)
    churn_counts.plot(kind='bar', ax=axes[i],color=['steelblue', 'tomato'],edgecolor='black')
    axes[i].set_title(f'Churn by {col}', fontsize=11)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(['No Churn', 'Churn'], fontsize=8)

plt.suptitle('Churn vs Categorical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '05_churn_vs_categorical.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05_churn_vs_categorical.png")

print("\nChurn Rate per Category")
for col in categorical_cols:
    print(f"\n{col}:")
    churn_rate = df.groupby(col)['Churn'].value_counts(normalize=True).mul(100).round(2)
    print(churn_rate)


# CORRELATION HEATMAP

df_corr = df.copy()

binary_map = {'Yes': 1, 'No': 0}
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df_corr[col] = df_corr[col].map(binary_map)

service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df_corr[col] = df_corr[col].map({'Yes': 1, 'No': 0,'No internet service': 0, 'No phone service': 0})

df_corr['Contract'] = df_corr['Contract'].map({'Month-to-month': 0, 'One year': 1,'Two year': 2})

df_corr = df_corr.drop(columns=['customerID', 'gender', 'InternetService', 'PaymentMethod'])

plt.figure(figsize=(16, 12))
sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm',linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06_correlation_heatmap.png")

print("Feature Correlation with Churn")
print(df_corr.corr()['Churn'].sort_values(ascending=False).round(3))


# CHURN RATE BY TENURE GROUPS


df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months'])

tenure_churn = df.groupby('tenure_group', observed=True)['Churn'].value_counts(
    normalize=True).mul(100).round(2).unstack()

print("Churn Rate by Tenure Group")
print(tenure_churn)

tenure_churn.plot(kind='bar', figsize=(10, 6),color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Churn Rate by Tenure Group', fontsize=16, fontweight='bold')
plt.xlabel('Tenure Group')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig(FIGURES_PATH + '07_churn_by_tenure_group.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_churn_by_tenure_group.png")


# PAIRPLOT OF NUMERICAL FEATURES

sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']],hue='Churn', palette={'No': 'steelblue', 'Yes': 'tomato'},
  diag_kind='kde', plot_kws={'alpha': 0.5})

plt.suptitle('Pairplot of Numerical Features by Churn', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '08_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08_pairplot.png")

print("All EDA Figures Saved Successfully")