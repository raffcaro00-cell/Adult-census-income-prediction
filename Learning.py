"""
Adult (Census Income) Dataset - Model Training & Selection Module
Provides functions for SVM baseline training, hyperparameter cross-validation,
and validation curve plotting for model diagnostics.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    validation_curve,
)
from sklearn.svm import SVC

# Plotting configurations
sns.set_style("whitegrid")


def train_svm_baseline(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """Trains a baseline SVM model with a linear kernel."""
    print("Training Baseline SVM model (Linear kernel)...")
    svm_baseline = SVC(kernel='linear', C=1.0, random_state=42)
    svm_baseline.fit(X_train, y_train)
    return svm_baseline


def optimize_svm_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray, cv_splits: int = 5
) -> SVC:
    """Performs Stratified K-Fold cross-validation and hyperparameter optimization

    using GridSearchCV on an RBF kernel SVM.
    """
    print("Executing GridSearchCV with StratifiedKFold for SVM optimization...")
    cv_strategy = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=42
    )

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf'],
    }

    grid_search = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def plot_validation_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_name: str = "gamma",
    param_range: np.ndarray = None,
    save_fig: bool = True,
):
    """Computes and plots the validation curve for a given hyperparameter."""
    if param_range is None:
        param_range = np.logspace(-5, 2, 8)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_scores, cv_scores = validation_curve(
        estimator=SVC(kernel='rbf', C=1.0, random_state=42),
        X=X_train,
        y=y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    cv_mean = np.mean(cv_scores, axis=1)
    cv_std = np.std(cv_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.semilogx(
        param_range,
        train_mean,
        label="Training Score",
        color="darkorange",
        lw=2,
    )
    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="darkorange",
    )

    plt.semilogx(
        param_range,
        cv_mean,
        label="Cross-Validation Score",
        color="navy",
        lw=2,
    )
    plt.fill_between(
        param_range,
        cv_mean - cv_std,
        cv_mean + cv_std,
        alpha=0.2,
        color="navy",
    )

    plt.title(f"Validation Curve for SVM ({param_name.upper()})")
    plt.xlabel(f"Hyperparameter '{param_name}'")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"svm_{param_name}_validation_curve.png")
        print(
            f"Validation curve plot saved as 'svm_{param_name}_validation_curve.png'."
        )

    plt.show()