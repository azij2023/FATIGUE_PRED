import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Load or prepare your training and validation data ---
# If data is saved from a previous step, load it here:
X_train, y_train, X_val, y_val = np.load('data_splits.npz').values()
# Or prepare the data using your preprocessing pipeline

# For now, ensure X_train, y_train, X_val, y_val are defined before proceeding
# Example placeholder - replace with actual data loading:
# X_train, y_train, X_val, y_val = load_your_data()

# --- Train base models on your existing splits ---
log_reg_bal = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
log_reg_bal.fit(X_train, y_train)

rf_bal = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_bal.fit(X_train, y_train)

lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
lgbm.fit(X_train, y_train)

# --- Get probabilities on validation set ---
proba_lr = log_reg_bal.predict_proba(X_val)[:,1]
proba_rf = rf_bal.predict_proba(X_val)[:,1]
proba_lgbm = lgbm.predict_proba(X_val)[:,1]

# --- Blend by averaging ---
proba_blend = (proba_lr + proba_rf + proba_lgbm) / 3
threshold = 0.45
y_pred_blend = (proba_blend >= threshold).astype(int)

# --- Evaluate blended model ---
results_blend = {
    "Model": "Blended (LR + RF + LGBM)",
    "Threshold": threshold,
    "Accuracy": accuracy_score(y_val, y_pred_blend),
    "Recall (Fatigue)": recall_score(y_val, y_pred_blend, pos_label=1),
    "Precision (Fatigue)": precision_score(y_val, y_pred_blend, pos_label=1),
    "F1 (Fatigue)": f1_score(y_val, y_pred_blend, pos_label=1),
    "ROC-AUC": roc_auc_score(y_val, proba_blend)
}

print(results_blend)

# --- Save results ---
os.makedirs("reports", exist_ok=True)
pd.DataFrame([results_blend]).to_csv("reports/blended_model_results.csv", index=False)

results = []

# Base models
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

# Blended model
results.append(results_blend)

df_results = pd.DataFrame(results)
print(df_results)

# Save leaderboard
df_results.to_csv("reports/model_leaderboard.csv", index=False)
