import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Load dataset ---
df = pd.read_csv("data/ott_train.csv")

# --- Drop identifier columns ---
df = df.drop("user_id", axis=1)

# --- One-hot encode categorical columns ---
df = pd.get_dummies(df, columns=["subscription_tier"], drop_first=True)

# --- Define features and target ---
X = df.drop("fatigue_label", axis=1)
y = df["fatigue_label"]

# --- Split dataset ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train base models ---
log_reg_bal = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
log_reg_bal.fit(X_train, y_train)

rf_bal = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_bal.fit(X_train, y_train)

lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
lgbm.fit(X_train, y_train)

# --- Evaluate base models ---
results = []
for model, name, threshold in [
    (log_reg_bal, "Logistic Regression (Balanced)", 0.50),
    (rf_bal, "Random Forest (Balanced)", 0.35),
    (lgbm, "LightGBM (Balanced)", 0.40)
]:
    proba = model.predict_proba(X_val)[:,1]
    y_pred = (proba >= threshold).astype(int)
    results.append({
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred, pos_label=1),
        "Precision": precision_score(y_val, y_pred, pos_label=1),
        "F1": f1_score(y_val, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_val, proba)
    })

# --- Blended model ---
proba_lr = log_reg_bal.predict_proba(X_val)[:,1]
proba_rf = rf_bal.predict_proba(X_val)[:,1]
proba_lgbm = lgbm.predict_proba(X_val)[:,1]

proba_blend = (proba_lr + proba_rf + proba_lgbm) / 3
threshold = 0.45
y_pred_blend = (proba_blend >= threshold).astype(int)

results.append({
    "Model": "Blended (LR + RF + LGBM)",
    "Threshold": threshold,
    "Accuracy": accuracy_score(y_val, y_pred_blend),
    "Recall": recall_score(y_val, y_pred_blend, pos_label=1),
    "Precision": precision_score(y_val, y_pred_blend, pos_label=1),
    "F1": f1_score(y_val, y_pred_blend, pos_label=1),
    "ROC-AUC": roc_auc_score(y_val, proba_blend)
})

# --- Save leaderboard ---
df_results = pd.DataFrame(results)
print(df_results)

os.makedirs("reports", exist_ok=True)
df_results.to_csv("reports/model_leaderboard.csv", index=False)
print("Leaderboard saved to reports/model_leaderboard.csv")

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(log_reg_bal, "models/log_reg_bal.pkl")
joblib.dump(rf_bal, "models/rf_bal.pkl")
joblib.dump(lgbm, "models/lgbm.pkl")
joblib.dump(X_train, "models/X_train.pkl")   # optional, for feature alignment
