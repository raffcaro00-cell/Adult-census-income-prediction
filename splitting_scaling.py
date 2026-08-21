"""
Adult (Census Income) Dataset - Train/Test Split & Feature Scaling

This module handles stratified train-test splitting and feature standardization
using StandardScaler to prevent data leakage during model training.
"""

import pickle
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale_data(
    input_pickle_path: str = "data_encoded.pkl",
    target_column: str = "income",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
  """Loads encoded dataset, performs stratified train-test split, and standardizes features.

  Args:
      input_pickle_path (str): Path to input pickled Dataframe.
      target_column (str): Name of the target variable column.
      test_size (float): Proportion of the dataset to include in the test split
        (default: 0.2).
      random_state (int): Controls the shuffling applied to the data before
        splitting.

  Returns:
      Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
          - Standardized training features (X_train_std).
          - Standardized testing features (X_test_std).
          - Training target array (y_train).
          - Testing target array (y_test).
          - Fitted StandardScaler object.
  """
  # Load encoded dataset
  df_split = pd.read_pickle(input_pickle_path)

  # Separate features and target label
  X = df_split.drop(columns=[target_column]).values
  y = df_split[target_column].values

  # ============================================
  # 1. TRAINING-TEST DATA SPLITTING
  # ============================================
  print("============================================")
  print("STEP 1: STRATIFIED TRAIN-TEST SPLITTING")
  print("============================================")

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state, stratify=y
  )

  print("Dataset Dimensions:")
  print(f"  X_train: {X_train.shape}")
  print(f"  X_test:  {X_test.shape}")
  print(f"  y_train: {y_train.shape}")
  print(f"  y_test:  {y_test.shape}\n")

  # ============================================
  # 2. FEATURE STANDARDIZATION
  # ============================================
  print("============================================")
  print("STEP 2: FEATURE STANDARDIZATION (SCALING)")
  print("============================================")

  scaler = StandardScaler()
  # Fit only on training data to prevent data leakage
  X_train_std = scaler.fit_transform(X_train)
  X_test_std = scaler.transform(X_test)

  print("Standardized Feature Shapes:")
  print(f"  X_train_std: {X_train_std.shape}")
  print(f"  X_test_std:  {X_test_std.shape}\n")

  # ============================================
  # 3. EXPORTING PROCESSED ARRAYS & SCALER
  # ============================================
  print("============================================")
  print("STEP 3: SAVING PROCESSED ARRAYS")
  print("============================================")

  np.save("X_train_std.npy", X_train_std)
  np.save("X_test_std.npy", X_test_std)
  np.save("y_train.npy", y_train)
  np.save("y_test.npy", y_test)

  with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

  print("Artifacts successfully saved:")
  print("  - X_train_std.npy")
  print("  - X_test_std.npy")
  print("  - y_train.npy")
  print("  - y_test.npy")
  print("  - scaler.pkl\n")

  return X_train_std, X_test_std, y_train, y_test, scaler


if __name__ == "__main__":
  try:
    X_train_std, X_test_std, y_train, y_test, fitted_scaler = (
        split_and_scale_data(
            input_pickle_path="data_encoded.pkl", target_column="income"
        )
    )
  except FileNotFoundError:
    print(
        "Error: Input file 'data_encoded.pkl' not found. Ensure previous preprocessing steps are complete."
    )