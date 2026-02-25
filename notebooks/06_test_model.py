import joblib
import pandas as pd

# --- Load trained models ---
log_reg_bal = joblib.load("models/log_reg_bal.pkl")
rf_bal = joblib.load("models/rf_bal.pkl")
lgbm = joblib.load("models/lgbm.pkl")

# Load training feature columns (for alignment)
X_train = joblib.load("models/X_train.pkl")

# --- Load test dataset ---
df_test = pd.read_csv("data/ott_test.csv")
user_ids = df_test["user_id"]

# Drop identifier column
df_test = df_test.drop("user_id", axis=1)

# One-hot encode categorical columns
df_test = pd.get_dummies(df_test, columns=["subscription_tier"], drop_first=True)

# Align test features with training columns
df_test = df_test.reindex(columns=X_train.columns, fill_value=0)

# --- Predict probabilities ---
proba_test_lr = log_reg_bal.predict_proba(df_test)[:,1]
proba_test_rf = rf_bal.predict_proba(df_test)[:,1]
proba_test_lgbm = lgbm.predict_proba(df_test)[:,1]

# --- Blend by averaging ---
proba_test_blend = (proba_test_lr + proba_test_rf + proba_test_lgbm) / 3

# --- Convert to percentage ---
proba_test_blend_pct = proba_test_blend * 100

# --- Create prediction file ---
predictions = pd.DataFrame({
    "user_id": user_ids,
    "predicted_fatigue_probability": proba_test_blend_pct
})

# Save with required naming convention
predictions.to_csv("Qubits_Predictions.csv", index=False)
print("Prediction file saved as Qubits_Predictions.csv")

import matplotlib.pyplot as plt
import pandas as pd

# Example: predicted fatigue probabilities (0–1 scale)
proba_test_blend = [0.12, 0.34, 0.56, 0.78, 0.91, 0.45, 0.67]

# Convert to percentage
proba_pct = [p * 100 for p in proba_test_blend]

# Create DataFrame
df = pd.DataFrame({"probability_pct": proba_pct})

# Define bins and labels
bins = [0, 25, 50, 75, 100]
labels = ["0–25", "25–50", "50–75", "75–100"]

# Cut into categories
df["bucket"] = pd.cut(df["probability_pct"], bins=bins, labels=labels, include_lowest=True)

# Count per bucket
bucket_counts = df["bucket"].value_counts().sort_index()

# Define custom colors for each bucket
colors = ["green", "blue", "orange", "red"]

# --- Plot pie chart ---
plt.figure(figsize=(6,6))
plt.pie(bucket_counts, labels=bucket_counts.index, colors=colors,
        autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Predicted Fatigue Probabilities")

# Save the plot
plt.savefig("reports/fatigue_probability_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("Pie chart saved as reports/fatigue_probability_distribution.png")

