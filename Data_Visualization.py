"""
Adult (Census Income) Dataset - Exploratory Data Analysis (EDA)
UCI Dataset: https://archive.ics.uci.edu/dataset/2/adult
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# 1. DATA LOADING
# ============================================

# Local file path and fallback URLs
local_csv = "adult.csv"
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column names definition
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

if os.path.exists(local_csv):
    print(f"Loading dataset from local file: {local_csv}")
    df = pd.read_csv(local_csv, names=column_names, sep=r',\s*', engine='python')
else:
    print("Local file not found. Fetching dataset from UCI repository...")
    df_train = pd.read_csv(url_train, names=column_names, sep=r',\s*', engine='python')
    df_test = pd.read_csv(url_test, names=column_names, sep=r',\s*', skiprows=1, engine='python')
    df = pd.concat([df_train, df_test], ignore_index=True)

# Replace missing value placeholder ('?') with np.nan for accurate detection
df.replace('?', np.nan, inplace=True)

print(f"\nDataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")


# ============================================
# 2. INITIAL DATA INSPECTION
# ============================================

print("=" * 60)
print("FIRST 5 ROWS OF THE DATASET")
print("=" * 60)
print(df.head())
print("\n")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())
print("\n")

print("=" * 60)
print("DESCRIPTIVE STATISTICS FOR NUMERICAL FEATURES")
print("=" * 60)
pd.set_option('display.max_columns', None)
print(df.describe())
print("\n")


# ============================================
# 3. FEATURE TYPE CLASSIFICATION
# ============================================

print("=" * 60)
print("FEATURE TYPES")
print("=" * 60)

# Numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical Features ({len(numerical_features)}):")
print(numerical_features)

# Categorical features
categorical_features = df.select_dtypes(include=['object', 'string']).columns.tolist()
if 'income' in categorical_features:
    categorical_features.remove('income')  # Exclude target variable
print(f"\nCategorical Features ({len(categorical_features)}):")
print(categorical_features)

print("\nTarget Variable: income\n")


# ============================================
# 4. MISSING VALUES ANALYSIS
# ============================================

print("=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing Count': missing.values,
    'Percentage (%)': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values found!")
print("\n")

samples_with_missing = df.isnull().any(axis=1).sum()
pct_samples_with_missing = (samples_with_missing / len(df)) * 100
print(f"Percentage of samples with at least one missing value: {pct_samples_with_missing:.2f}%\n")


# ============================================
# 5. TARGET DISTRIBUTION ANALYSIS
# ============================================

print("=" * 60)
print("TARGET CLASS DISTRIBUTION (income)")
print("=" * 60)

# Clean target values (strip whitespace and trailing dots)
df['income'] = df['income'].astype(str).str.strip().str.rstrip('.')

target_dist = df['income'].value_counts()
target_pct = df['income'].value_counts(normalize=True) * 100

print("\nAbsolute Counts:")
print(target_dist)
print("\nPercentage Distribution:")
print(target_pct)
print("\n")


# ============================================
# 6. CATEGORICAL FEATURE CARDINALITY
# ============================================

print("=" * 60)
print("CATEGORICAL FEATURE CARDINALITY")
print("=" * 60)

for col in categorical_features:
    n_unique = df[col].nunique()
    print(f"{col:20s}: {n_unique:3d} unique values")
print("\n")


# Save processed DataFrame for downstream modules
df.to_pickle('data_cleaned.pkl')