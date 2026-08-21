import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path: str, target_column: str = 'income', test_size: float = 0.2, random_state: int = 42):
    """
    Loads, cleans, imputes missing values, encodes categorical features,
    and standardizes the Adult Census dataset.
    """
    # 1. Load raw dataset
    if file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    else:
        df = pd.read_csv(file_path)

    # Clean whitespace strings in target column if object type
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype(str).str.strip()

    # Replace missing value string placeholders
    df.replace('?', np.nan, inplace=True)

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])

    # Drop redundant ordinal feature if both exist
    if 'education' in df.columns and 'education-num' in df.columns:
        df.drop(columns=['education'], inplace=True)

    # Encode target column safely to binary integers (0, 1)
    # Checks if '>50K' is present in the string (handles '>50K.', '>50K', etc.)
    y_raw = df[target_column].astype(str)
    y = np.where(y_raw.str.contains('>50K'), 1, 0)

    # Separate features
    X = df.drop(columns=[target_column])

    # 2. Train / Test Split (Stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Missing Value Imputation
    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # Select numeric columns that actually exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    categorical_cols = list(set(X.columns) - set(numeric_cols))

    imr_num = SimpleImputer(strategy="mean")
    imr_cat = SimpleImputer(strategy="most_frequent")

    X_train_clean = X_train_raw.copy()
    X_test_clean = X_test_raw.copy()

    if numeric_cols:
        X_train_clean[numeric_cols] = imr_num.fit_transform(X_train_raw[numeric_cols])
        X_test_clean[numeric_cols] = imr_num.transform(X_test_raw[numeric_cols])

    if categorical_cols:
        X_train_clean[categorical_cols] = imr_cat.fit_transform(X_train_raw[categorical_cols])
        X_test_clean[categorical_cols] = imr_cat.transform(X_test_raw[categorical_cols])

    # 4. One-Hot Encoding and Alignment
    X_train_encoded = pd.get_dummies(X_train_clean, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_clean, columns=categorical_cols, drop_first=True)

    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_encoded)
    X_test_std = scaler.transform(X_test_encoded)

    return X_train_std, X_test_std, y_train, y_test, scaler