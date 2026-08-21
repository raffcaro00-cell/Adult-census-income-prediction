"""
Adult (Census Income) Dataset - Hyperparameter Optimization via Grid Search
Section: SVM Hyperparameter Tuning (Kernel, C, Gamma) using PCA-Transformed Features
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Visualization styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================
# 1. DATA LOADING
# ============================================
print("STEP 1: Loading Preprocessed & PCA-Transformed Datasets")
print("-" * 70)

X_train_pca = np.load('X_train_pca.npy')
X_test_pca = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"X_train_pca shape: {X_train_pca.shape}")
print(f"X_test_pca shape:  {X_test_pca.shape}")
print(f"y_train shape:     {y_train.shape}")
print(f"y_test shape:      {y_test.shape}\n")


# ============================================
# 2. HYPERPARAMETER OPTIMIZATION (GRID SEARCH)
# ============================================
print("STEP 2: Initializing Grid Search Cross-Validation")
print("-" * 70)

# Define logarithmic search ranges for C and gamma hyperparameters
param_range = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

# Construct parameter grid across different kernel functions
param_grid = [
    {
        'kernel': ['linear'],
        'C': param_range
    },
    {
        'kernel': ['rbf'],
        'C': param_range,
        'gamma': param_range
    },
    {
        'kernel': ['sigmoid'],
        'C': param_range,
        'gamma': param_range
    }
]

# Configure GridSearchCV with Stratified 5-Fold Cross-Validation
gs = GridSearchCV(
    estimator=SVC(random_state=0),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2
)

print("Starting Grid Search execution across hyperparameter space...")
start_time = time.time()
gs.fit(X_train_pca, y_train)
elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("GRID SEARCH COMPLETED")
print("=" * 70)
print(f"Execution Time: {elapsed_time:.2f} seconds")
print(f"Best Cross-Validation Score (Mean Accuracy): {gs.best_score_:.6f}")
print(f"Optimal Hyperparameter Combination: {gs.best_params_}")
print(f"Best Estimator Architecture: {gs.best_estimator_}\n")


# ============================================
# 3. EVALUATION ON HOLD-OUT TEST SET
# ============================================
print("STEP 3: Evaluating Optimized Model on Hold-Out Test Data")
print("-" * 70)

# Best model is automatically refitted on full training data by GridSearchCV
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test_pca)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Hold-out Test Accuracy Score: {test_accuracy:.6f}\n")

print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))