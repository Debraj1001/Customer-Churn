import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Define the code cells
cells = [
    # Cell 1: Imports
    """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import pickle
""",
    # Cell 2: Load Data
    """
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
print(df.info())
print(df.describe())
""",
    # Cell 3: Churn Distribution
    """
sns.set(style="darkgrid")
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Distribution of Customer Churn')
plt.show()
""",
    # Cell 4: Churn by Contract Type
    """
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.show()
""",
    # Cell 5: Churn by Tenure
    """
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Churn by Tenure')
plt.show()
""",
    # Cell 6: Churn by Monthly Charges
    """
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Churn by Monthly Charges')
plt.show()
""",
    # Cell 7: Handle Missing Values
    """
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_total_charges, inplace=True)
print(df.isnull().sum())
""",
    # Cell 8: Handle Duplicates
    """
print(f"Number of duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")
""",
    # Cell 9: Encode Categorical Features
    """
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_cols = df.select_dtypes(include=['object']).columns
if 'customerID' in categorical_cols:
    categorical_cols = categorical_cols.drop('customerID')
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
if 'customerID' in df_encoded.columns:
    df_encoded.drop('customerID', axis=1, inplace=True)
print(df_encoded.columns)
print(df_encoded.head())
""",
    # Cell 10: Scale Numerical Features
    """
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print(df_encoded.head())
""",
    # Cell 11: Split Data
    """
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
""",
    # Cell 12: Logistic Regression
    """
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_proba_log_reg = log_reg.predict_proba(X_test)[:, 1]
print("Logistic Regression Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_log_reg):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_log_reg):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_proba_log_reg)
""",
    # Cell 13: Random Forest
    """
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_pred_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
print("Random Forest Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
""",
    # Cell 14: XGBoost
    """
xgb_clf = xgb.XGBClassifier(random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
y_pred_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
print("XGBoost Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
""",
    # Cell 15: Plot ROC Curves
    """
plt.figure(figsize=(8, 6))
plt.plot(fpr_log_reg, tpr_log_reg, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
""",
    # Cell 16: Save Model and Scaler
    """
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Model and scaler saved successfully.")
"""
]

# Add the cells to the notebook
for cell_source in cells:
    nb['cells'].append(nbf.v4.new_code_cell(cell_source.strip()))

# Write the notebook to a file
with open('Customer_Churn_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
