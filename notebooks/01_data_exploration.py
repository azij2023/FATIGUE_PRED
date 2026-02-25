import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("data/ott_train.csv")

# 2. Inspect Dataset
print("Shape:", df.shape)
print(df.info())
print(df.head())

# 3. Summary Statistics
print(df.describe())
print("Target distribution:\n", df['fatigue_label'].value_counts(normalize=True))

# 4. Data Quality Checks
print("Missing values:\n", df.isnull().sum())

# Outlier check
sns.boxplot(x=df['avg_daily_minutes_last_7d'])
plt.title("Outliers in avg_daily_minutes_last_7d")
plt.show()

sns.boxplot(x=df['sessions_last_30d'])
plt.title("Outliers in sessions_last_30d")
plt.show()

# Consistency checks
print("Completion rate range:", df['avg_completion_rate'].min(), df['avg_completion_rate'].max())
print("Binge sessions check:", (df['binge_sessions_last_30d'] <= df['sessions_last_30d']).all())

invalid_count = (df['binge_sessions_last_30d'] > df['sessions_last_30d']).sum()
print("Rows with invalid binge sessions:", invalid_count)

# Correct inconsistent binge session values
df.loc[df['binge_sessions_last_30d'] > df['sessions_last_30d'], 'binge_sessions_last_30d'] = df['sessions_last_30d']

# Verify again
print("Corrected inconsistent binge session values and remaining invalid rows:", (df['binge_sessions_last_30d'] > df['sessions_last_30d']).sum())


# Save summary statistics
summary_stats = df.describe()
summary_stats.to_csv("output/reports/summary_statistics.csv")

# Save target distribution
target_dist = df['fatigue_label'].value_counts(normalize=True)
target_dist.to_csv("output/reports/target_distribution.csv")

# Save missing values report
missing_report = df.isnull().sum()
missing_report.to_csv("output/reports/missing_values.csv")

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"output/figures/hist_{col}.png")
    plt.close()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='fatigue_label', y=col, data=df)
    plt.title(f"{col} vs Fatigue Label")
    plt.savefig(f"output/figures/box_{col}.png")
    plt.close()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f"Count of {col}")
    plt.savefig(f"output/figures/bar_{col}.png")
    plt.close()

import pandas as pd
import numpy as np

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

outlier_report = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_report[col] = outliers

# Convert to DataFrame and save
outlier_df = pd.DataFrame.from_dict(outlier_report, orient='index', columns=['outlier_count'])
outlier_df.to_csv("output/reports/outlier_report.csv")
print(outlier_df)
