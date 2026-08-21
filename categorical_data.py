"""
Adult Census Income Dataset - Categorical Data Encoding

This script loads the pre-cleaned dataset, encodes ordinal and nominal
categorical features, removes redundant columns, and exports the final
encoded dataset for machine learning models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================
# 1. SETUP & DATA LOADING
# ============================================
# Configure plot styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("STEP 1: Loading Cleaned Dataset")
print("-" * 50)

# Load dataset without missing values
df_clean = pd.read_pickle('data_clean.pkl')
print(f"Dataset successfully loaded. Initial shape: {df_clean.shape}")
print("\n")


# ============================================
# 2. ENCODING CATEGORICAL DATA
# ============================================
print("STEP 2: Encoding Categorical Features")
print("-" * 50)

# --- A. TARGET VARIABLE (INCOME) ---
# Map target labels ('<=50K', '>50K') to binary values (0, 1)
class_mapping = {
    label: idx for idx, label in enumerate(np.unique(df_clean['income']))
}
df_clean['income'] = df_clean['income'].map(class_mapping)

print("Target variable 'income' unique values:")
print(df_clean['income'].unique())
print("\n")


# --- B. ORDINAL FEATURE (EDUCATION) ---
# Map education levels to ordered integers
education_mapping = {
    'Preschool': 1,
    '1st-4th': 2,
    '5th-6th': 3,
    '7th-8th': 4,
    '9th': 5,
    '10th': 6,
    '11th': 7,
    '12th': 8,
    'HS-grad': 9,
    'Some-college': 10,
    'Assoc-voc': 11,
    'Assoc-acdm': 12,
    'Bachelors': 13,
    'Masters': 14,
    'Prof-school': 15,
    'Doctorate': 16,
}
df_clean['education'] = df_clean['education'].map(education_mapping)

# Note: 'education' becomes redundant as it conveys the exact same information as 'educational-num'.
# Dropping 'education' column to prevent multi-collinearity.
df_clean = df_clean.drop(columns=['education'])


# --- C. NOMINAL FEATURES (ONE-HOT ENCODING) ---
nominal_cols = [
    'workclass',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

# Apply One-Hot Encoding and drop the first category to avoid the dummy variable trap
df_encoded = pd.get_dummies(
    df_clean, columns=nominal_cols, drop_first=True, dtype=int
)


# ============================================
# 3. ANALYSIS OF NATIVE-COUNTRY DISTRIBUTION
# ============================================
print("STEP 3: Native Country Feature Distribution")
print("-" * 50)

# Identify One-Hot Encoded columns for 'native-country'
country_cols = [
    c for c in df_encoded.columns if c.startswith("native-country_")
]
country_counts = df_encoded[country_cols].sum().sort_values(ascending=False)

print("Sample counts per native country (One-Hot Columns):")
print(country_counts)
print("\n")


# ============================================
# 4. REORDERING COLUMNS & EXPORT
# ============================================
print("STEP 4: Reordering Columns and Saving Final Dataset")
print("-" * 50)

# Move the target column ('income') to the first position
df_encoded.insert(0, 'income', df_encoded.pop('income'))

print(
    f"Final position of target column 'income': {df_encoded.columns.get_loc('income')}"
)

# Export Encoded Dataset
df_encoded.to_pickle('data_encoded.pkl')
print("\nEncoded dataset successfully saved to 'data_encoded.pkl'")
print(
    f"Final Dataset Dimensions: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns"
)