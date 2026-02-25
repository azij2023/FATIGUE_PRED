import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load preprocessed splits
X_train = pd.read_csv("output/X_train.csv")
X_val = pd.read_csv("output/X_val.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()
y_val = pd.read_csv("output/y_val.csv").squeeze()

# -------------------------------
# 1. Baseline Logistic Regression
# -------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_val)

print("\n=== Baseline Logistic Regression ===")
print(classification_report(y_val, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_val, log_reg.predict_proba(X_val)[:,1]))

# -------------------------------
# 2. Logistic Regression (Balanced)
# -------------------------------
log_reg_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_bal.fit(X_train, y_train)
y_pred_lr_bal = log_reg_bal.predict(X_val)

print("\n=== Logistic Regression (Balanced) ===")
print(classification_report(y_val, y_pred_lr_bal))
print("ROC-AUC:", roc_auc_score(y_val, log_reg_bal.predict_proba(X_val)[:,1]))

# -------------------------------
# 3. Random Forest (Balanced)
# -------------------------------
rf_bal = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_bal.fit(X_train, y_train)
y_pred_rf_bal = rf_bal.predict(X_val)

print("\n=== Random Forest (Balanced) ===")
print(classification_report(y_val, y_pred_rf_bal))
print("ROC-AUC:", roc_auc_score(y_val, rf_bal.predict_proba(X_val)[:,1]))

# -------------------------------
# 4. SMOTE + Logistic Regression
# -------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

log_reg_smote = LogisticRegression(max_iter=1000)
log_reg_smote.fit(X_train_res, y_train_res)
y_pred_lr_smote = log_reg_smote.predict(X_val)

print("\n=== Logistic Regression (SMOTE) ===")
print(classification_report(y_val, y_pred_lr_smote))
print("ROC-AUC:", roc_auc_score(y_val, log_reg_smote.predict_proba(X_val)[:,1]))

# -------------------------------
# 5. SMOTE + Random Forest
# -------------------------------
rf_smote = RandomForestClassifier(n_estimators=200, random_state=42)
rf_smote.fit(X_train_res, y_train_res)
y_pred_rf_smote = rf_smote.predict(X_val)

print("\n=== Random Forest (SMOTE) ===")
print(classification_report(y_val, y_pred_rf_smote))
print("ROC-AUC:", roc_auc_score(y_val, rf_smote.predict_proba(X_val)[:,1]))

# -------------------------------
# 6. XGBoost (Balanced)
# -------------------------------
xgb = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])))
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_val)

print("\n=== XGBoost (Balanced) ===")
print(classification_report(y_val, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1]))

# -------------------------------
# 7. LightGBM (Balanced)
# -------------------------------
lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_val)

print("\n=== LightGBM (Balanced) ===")
print(classification_report(y_val, y_pred_lgbm))
print("ROC-AUC:", roc_auc_score(y_val, lgbm.predict_proba(X_val)[:,1]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def tune_thresholds(model, X_val, y_val, model_name):
    y_proba = model.predict_proba(X_val)[:,1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "Model": model_name,
            "Threshold": t,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Recall (Fatigue)": recall_score(y_val, y_pred, pos_label=1),
            "Precision (Fatigue)": precision_score(y_val, y_pred, pos_label=1),
            "F1 (Fatigue)": f1_score(y_val, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_val, y_proba)  # threshold-independent
        })
    return pd.DataFrame(results)

df_lr = tune_thresholds(log_reg_bal, X_val, y_val, "Logistic Regression (Balanced)")
df_rf = tune_thresholds(rf_bal, X_val, y_val, "Random Forest (Balanced)")
df_xgb = tune_thresholds(xgb, X_val, y_val, "XGBoost (Balanced)")
df_lgbm = tune_thresholds(lgbm, X_val, y_val, "LightGBM (Balanced)")

df_all = pd.concat([df_lr, df_rf, df_xgb, df_lgbm])
print(df_all)
df_all.to_csv("reports/threshold_tuning.csv", index=False)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def best_threshold(model, X_val, y_val, model_name, metric="f1"):
    y_proba = model.predict_proba(X_val)[:,1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    scores = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        scores.append({
            "Model": model_name,
            "Threshold": t,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Recall (Fatigue)": recall_score(y_val, y_pred, pos_label=1),
            "Precision (Fatigue)": precision_score(y_val, y_pred, pos_label=1),
            "F1 (Fatigue)": f1_score(y_val, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_val, y_proba)
        })
    df = pd.DataFrame(scores)
    # Pick the row with the best chosen metric
    best_row = df.loc[df[metric.capitalize() + " (Fatigue)"].idxmax()]
    return best_row

# Collect best thresholds for all models
best_results = []
best_results.append(best_threshold(log_reg_bal, X_val, y_val, "Logistic Regression (Balanced)", metric="f1"))
best_results.append(best_threshold(rf_bal, X_val, y_val, "Random Forest (Balanced)", metric="f1"))
best_results.append(best_threshold(xgb, X_val, y_val, "XGBoost (Balanced)", metric="f1"))
best_results.append(best_threshold(lgbm, X_val, y_val, "LightGBM (Balanced)", metric="f1"))

df_best = pd.DataFrame(best_results)

# Ensure reports folder exists
import os
os.makedirs("reports", exist_ok=True)

# Save results
df_best.to_csv("reports/best_thresholds.csv", index=False)
print(df_best)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
import os

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

def plot_combined_pr(models, X_val, y_val, thresholds):
    plt.figure(figsize=(8,6))
    
    for model, name, best_t in models:
        # Get probabilities
        y_proba = model.predict_proba(X_val)[:,1]
        prec, rec, thr = precision_recall_curve(y_val, y_proba)
        
        # Plot curve
        plt.plot(rec, prec, label=f"{name}")
        
        # Mark best threshold
        idx = (np.abs(thr - best_t)).argmin()
        plt.scatter(rec[idx], prec[idx], s=70, marker='o',
                    label=f"{name} (t={best_t:.2f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision–Recall Curves")
    plt.legend()
    
    # Save combined figure
    filename = "reports/combined_PR_curves.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# Example usage with your tuned thresholds
models = [
    (log_reg_bal, "Logistic Regression (Balanced)", 0.50),
    (rf_bal, "Random Forest (Balanced)", 0.35),
    (xgb, "XGBoost (Balanced)", 0.45),
    (lgbm, "LightGBM (Balanced)", 0.40)
]

plot_combined_pr(models, X_val, y_val, thresholds=None)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import os

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

def plot_pr_roc_dashboard(model, X_val, y_val, model_name, best_threshold):
    # Get probabilities
    y_proba = model.predict_proba(X_val)[:,1]
    
    # Precision–Recall
    prec, rec, thr = precision_recall_curve(y_val, y_proba)
    idx = (np.abs(thr - best_threshold)).argmin()
    
    # ROC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    
    # Create side‑by‑side plots
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    
    # PR curve
    axes[0].plot(rec, prec, label=f"{model_name}")
    axes[0].scatter(rec[idx], prec[idx], color="red", s=80,
                    label=f"Best Threshold = {best_threshold:.2f}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"PR Curve: {model_name}")
    axes[0].legend()
    
    # ROC curve
    axes[1].plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    axes[1].plot([0,1], [0,1], 'k--', label="Random Guess")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title(f"ROC Curve: {model_name}")
    axes[1].legend()
    
    # Save figure
    filename = f"reports/{model_name.replace(' ', '_')}_PR_ROC_dashboard.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# Example usage with your tuned thresholds
plot_pr_roc_dashboard(log_reg_bal, X_val, y_val, "Logistic Regression (Balanced)", 0.50)
plot_pr_roc_dashboard(rf_bal, X_val, y_val, "Random Forest (Balanced)", 0.35)
plot_pr_roc_dashboard(xgb, X_val, y_val, "XGBoost (Balanced)", 0.45)
plot_pr_roc_dashboard(lgbm, X_val, y_val, "LightGBM (Balanced)", 0.40)
