# StreamMax Fatigue Prediction Project

This repository contains my solution for the **StreamMax OTT Platform case competition**.  
# FATIGUE_PRED

A machine learning project for predicting user fatigue in OTT platforms, developed for the **StreamMax Case Competition**.

---

## 📌 Overview
This project aims to analyze user behavior on OTT platforms and predict fatigue levels.  
We use structured datasets (`ott_train.csv`, `ott_test.csv`) and apply advanced machine learning techniques to identify patterns that indicate user fatigue.  
The workflow is modular, with separate scripts for data exploration, preprocessing, modeling, hyperparameter tuning, blending, and testing.

---

## 🎯 Objectives
- Perform exploratory data analysis to understand fatigue drivers.
- Preprocess and clean raw OTT datasets for modeling.
- Train and evaluate multiple models (Logistic Regression, Random Forest, LightGBM).
- Blend models for improved accuracy and robustness.
- Generate final predictions for competition submission.
- Provide business insights and recommendations based on results.

---

## 📊 Data Exploration
| Metric | Visualization |
|--------|---------------|
| Subscription Tier | ![Subscription Tier](reports/bar_subscription_tier.png) |
| Completion Rate | ![Completion Rate](reports/box_avg_completion_rate.png) |
| Daily Minutes (7d) | ![Daily Minutes](reports/box_avg_daily_minutes_last_7d.png) |
| Binge Sessions | ![Binge Sessions](reports/box_binge_sessions_last_30d.png) |
| Fatigue Label Distribution | ![Fatigue Label](reports/hist_fatigue_label.png) |
| Recommendation Click Rate | ![Click Rate](reports/box_recommendation_click_rate.png) |

👉 Open the dashboard notebook in `notebooks/05_blending_model.py` or explore the saved figures in the `reports/` folder.

---

## 🤖 Model Visualization
We provide clear visualizations of:
- Feature importance (LightGBM, Random Forest).
- SHAP values for explainability.
- ROC and PR curves comparing models.
- Confusion matrices for classification performance.

Figures are saved in the `reports/` folder:
- `fatigue_probability_distribution.png`
- `fatigue_probability_histogram.png`
- `roc_curve.png`
- `pr_curve.png`

---

## 📂 Final Prediction
The final prediction file is:
- **`Qubits_Predictions.csv`** → Contains 2,000 rows of user IDs with predicted fatigue probabilities.

This file is competition-ready and can be directly submitted to the leaderboard.

---

## 📊 Visualizations
### Fatigue Risk Distribution (Pie Chart)
![Pie Chart](reports/fatigue_probability_distribution.png)

### ROC Curve
![ROC Curve](reports/combined_PR_curves.png)

### Precision-Recall Curve
![PR Curve](reports/pr_curve.png)

