"""
Adult (Census Income) Dataset - Missing Value Handling & Feature Identification

This module handles missing data imputation using mean strategies for numerical
features and mode (most frequent) strategies for categorical features. It also
provides verification and summary reporting of dataset feature types.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer


def handle_missing_values(
    df: pd.DataFrame, target_column: str = "income"
) -> Tuple[pd.DataFrame, List[str], List[str]]:
  """Imputes missing values for numerical and categorical features.

  Args:
      df (pd.DataFrame): Input raw DataFrame.
      target_column (str): Target column name to exclude from categorical
        feature list.

  Returns:
      Tuple[pd.DataFrame, List[str], List[str]]:
          - Cleaned DataFrame without missing values.
          - List of numerical feature names.
          - List of categorical feature names (excluding target).
  """
  # Configure visualization settings
  sns.set_style("whitegrid")
  plt.rcParams["figure.figsize"] = (14, 6)

  print("============================================")
  print("STEP 1: MISSING VALUE IMPUTATION")
  print("============================================")

  df_clean = df.copy()

  # Identify numerical and categorical columns
  numeric_cols = [
      "age",
      "fnlwgt",
      "education-num",
      "capital-gain",
      "capital-loss",
      "hours-per-week",
  ]
  # Filter numeric columns to only those actually present in the DataFrame
  numeric_cols = [col for col in numeric_cols if col in df_clean.columns]

  categorical_cols = df_clean.select_dtypes(
      include=["object", "string"]
  ).columns.tolist()

  # Define imputers
  imr_num = SimpleImputer(missing_values=np.nan, strategy="mean")
  imr_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

  # Fit and transform features
  if numeric_cols:
    df_clean[numeric_cols] = imr_num.fit_transform(df_clean[numeric_cols])

  if categorical_cols:
    df_clean[categorical_cols] = imr_cat.fit_transform(
        df_clean[categorical_cols]
    )

  # Save cleaned dataset
  df_clean.to_pickle("data_clean.pkl")
  print("\nCleaned dataset successfully saved to 'data_clean.pkl'\n")

  # ============================================
  # 2. MISSING VALUES VERIFICATION
  # ============================================
  print("============================================")
  print("STEP 2: VERIFYING IMPUTATION RESULTS")
  print("============================================")

  missing_counts = df_clean.isnull().sum()
  missing_percentages = (missing_counts / len(df_clean)) * 100

  missing_df = pd.DataFrame({
      "Feature": missing_counts.index,
      "Missing Count": missing_counts.values,
      "Percentage (%)": missing_percentages.values,
  })

  missing_remaining = missing_df[missing_df["Missing Count"] > 0].sort_values(
      by="Missing Count", ascending=False
  )

  if not missing_remaining.empty:
    print("Remaining Missing Values Detected:")
    print(missing_remaining.to_string(index=False))
  else:
    print("Verification Passed: No missing values remaining in the dataset.")

  print("\n")

  # ============================================
  # 3. FEATURE TYPE CLASSIFICATION
  # ============================================
  print("============================================")
  print("STEP 3: FEATURE TYPE IDENTIFICATION")
  print("============================================")

  numerical_features = df_clean.select_dtypes(
      include=[np.number]
  ).columns.tolist()
  print(f"Numerical Features ({len(numerical_features)}):")
  print(numerical_features)

  # Select categorical features and safely exclude target column
  categorical_features = [
      col
      for col in df_clean.select_dtypes(
          include=["object", "string"]
      ).columns.tolist()
      if col != target_column
  ]

  print(f"\nCategorical Features ({len(categorical_features)}):")
  print(categorical_features)
  print("\n")

  return df_clean, numerical_features, categorical_features


if __name__ == "__main__":
  # Load dataset
  dataset_path = "data.pkl"
  try:
    raw_df = pd.read_pickle(dataset_path)
    df_cleaned, num_cols, cat_cols = handle_missing_values(
        raw_df, target_column="income"
    )
  except FileNotFoundError:
    print(f"Error: Dataset pickle file '{dataset_path}' not found.")