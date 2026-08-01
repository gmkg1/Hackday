import pandas as pd
import numpy as np
import os

# Ensure deterministic generation
np.random.seed(42)

# Generate synthetic HR Churn dataset
n_samples = 500

data = {
    'Employee_ID': [f"EMP_{i:04d}" for i in range(n_samples)],
    'Age': np.random.normal(35, 8, n_samples),
    'Monthly_Salary': np.random.normal(60000, 20000, n_samples),
    'Years_at_Company': np.random.exponential(4, n_samples),
    'Department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], n_samples, p=[0.3, 0.4, 0.1, 0.2]),
    'Satisfaction_Score': np.random.uniform(1, 10, n_samples),
}

# Create target variable 'Churn' (1 = Left, 0 = Stayed)
# Higher chance of churn if low salary, low satisfaction, or high years at company
prob_churn = (
    (70000 - data['Monthly_Salary']) / 50000 + 
    (10 - data['Satisfaction_Score']) / 10 +
    data['Years_at_Company'] / 10
) / 3
# Add some noise
prob_churn += np.random.normal(0, 0.1, n_samples)
prob_churn = np.clip(prob_churn, 0.05, 0.95)

data['Churn'] = np.random.binomial(1, prob_churn)

df = pd.DataFrame(data)

# Inject messy real-world data problems for the AI to fix

# 1. Missing Values
mask_age = np.random.rand(n_samples) < 0.15 # 15% missing age
df.loc[mask_age, 'Age'] = np.nan

mask_sat = np.random.rand(n_samples) < 0.05 # 5% missing satisfaction
df.loc[mask_sat, 'Satisfaction_Score'] = np.nan

# 2. Outliers (Extreme salaries)
outlier_indices = np.random.choice(n_samples, size=5, replace=False)
df.loc[outlier_indices, 'Monthly_Salary'] = df.loc[outlier_indices, 'Monthly_Salary'] * 5

# 3. Target Leakage candidate (highly correlated with Churn)
# Create a column that basically gives away the answer
df['Did_Exit_Interview'] = np.where(df['Churn'] == 1, 
                                    np.random.choice([1, 0], n_samples, p=[0.9, 0.1]), 
                                    np.random.choice([1, 0], n_samples, p=[0.05, 0.95]))

# Save to the frontend folder
output_path = os.path.join(os.path.dirname(__file__), "sample_dataset.csv")
df.to_csv(output_path, index=False)
print(f"Generated sample dataset at {output_path}")
