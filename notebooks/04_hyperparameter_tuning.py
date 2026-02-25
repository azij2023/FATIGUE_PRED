import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load preprocessed splits
X_train = pd.read_csv("output/X_train.csv")
X_val = pd.read_csv("output/X_val.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()
y_val = pd.read_csv("output/y_val.csv").squeeze()

# Function to evaluate and collect metrics
def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:,1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Recall (Fatigue)": recall_score(y_val, y_pred, pos_label=1),
        "Precision (Fatigue)": precision_score(y_val, y_pred, pos_label=1),
        "F1 (Fatigue)": f1_score(y_val, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_val, y_proba)
    }

results = []

# -------------------------------
# Logistic Regression
# -------------------------------
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'),
                       param_grid_lr, scoring='roc_auc', cv=3, n_jobs=-1)
grid_lr.fit(X_train, y_train)
results.append(evaluate_model("Logistic Regression (Tuned)", grid_lr.best_estimator_, X_val, y_val))

# -------------------------------
# Random Forest
# -------------------------------
param_grid_rf = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                       param_grid_rf, scoring='roc_auc', cv=3, n_jobs=-1)
grid_rf.fit(X_train, y_train)
results.append(evaluate_model("Random Forest (Tuned)", grid_rf.best_estimator_, X_val, y_val))

# -------------------------------
# XGBoost
# -------------------------------
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_xgb = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'),
                        param_grid_xgb, scoring='roc_auc', cv=3, n_jobs=-1)
grid_xgb.fit(X_train, y_train)
results.append(evaluate_model("XGBoost (Tuned)", grid_xgb.best_estimator_, X_val, y_val))

# -------------------------------
# LightGBM
# -------------------------------
param_grid_lgbm = {
    'num_leaves': [31, 63],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500],
    'min_data_in_leaf': [20, 50]
}
grid_lgbm = GridSearchCV(LGBMClassifier(random_state=42, class_weight='balanced'),
                         param_grid_lgbm, scoring='roc_auc', cv=3, n_jobs=-1)
grid_lgbm.fit(X_train, y_train)
results.append(evaluate_model("LightGBM (Tuned)", grid_lgbm.best_estimator_, X_val, y_val))

# -------------------------------
# Save results
# -------------------------------
df_results = pd.DataFrame(results)

import os

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Save results
df_results.to_csv("reports/model_performance_tuned.csv", index=False)

df_results.to_csv("reports/model_performance_tuned.csv", index=False)
print(df_results)

